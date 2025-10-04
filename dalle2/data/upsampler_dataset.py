"""
upsampler_dataset.py: Dataset for training the DALL·E 2 upsampler.

Description:
    * Returns low-res (64x64), high-res (128x128), and CLIP embeddings
    * CLIP embeddings are computed from the 128x128 target images
    * Supports precomputed embeddings for faster training
    * Adds noise to high-res images during training
    
Author:
    * Spencer Karofsky (https://github.com/spencer-karofsky)
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import random
from pathlib import Path
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from typing import Tuple
from urllib.parse import urlparse
import boto3

from dalle2.models.clip_encoding import CLIPEncoder
from dalle2.sampling.noise_scheduler import NoiseScheduler


def is_s3_uri(p: str) -> bool:
    return isinstance(p, str) and p.startswith("s3://")


def split_s3_uri(uri: str):
    pr = urlparse(uri)
    return pr.netloc, pr.path.lstrip("/")


class S3Cache:
    """Downloads S3 objects to a local cache and returns the local path."""
    def __init__(self, cache_dir: str = "/tmp/dalle2_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.s3 = boto3.client("s3")

    def get_local_path(self, bucket: str, key: str) -> str:
        local_path = self.cache_dir / bucket / key
        if not local_path.exists():
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self.s3.download_file(bucket, key, str(local_path))
        return str(local_path)

    def get_csv_as_local(self, bucket: str, key: str) -> str:
        return self.get_local_path(bucket, key)


class MidJourneyUpsamplerDataset(Dataset):
    """
    Dataset for training the upsampler.
    
    Returns:
        - high_res: 128x128 target image (with noise added during training)
        - low_res: 64x64 conditioning image
        - clip_embedding: CLIP embedding from the 128x128 image
    """
    
    def __init__(
        self,
        metadata_path: str,
        images_dir: str,
        device: torch.device,
        noise_scheduler: NoiseScheduler,
        low_res_size: int = 64,
        high_res_size: int = 128,
        n_repeat: int = 1,
        cache_dir: str = "/tmp/dalle2_cache",
        precomputed_embeddings_path: str = None,
    ):
        """
        Initialize upsampler dataset.
        
        Args:
            metadata_path: Path to metadata.csv
            images_dir: Directory containing images
            device: PyTorch device
            noise_scheduler: NoiseScheduler for adding noise during training
            low_res_size: Low-resolution image size (64)
            high_res_size: High-resolution image size (128)
            n_repeat: Number of times to repeat dataset per epoch
            cache_dir: Cache directory for S3 downloads
            precomputed_embeddings_path: Path to precomputed CLIP embeddings
        """
        super().__init__()
        self.device = device
        self.noise_scheduler = noise_scheduler
        self.images_dir = images_dir
        self.low_res_size = low_res_size
        self.high_res_size = high_res_size
        self.n_repeat = n_repeat
        
        # S3 support
        self.using_s3 = is_s3_uri(images_dir)
        self.s3_cache = S3Cache(cache_dir) if self.using_s3 or is_s3_uri(metadata_path) else None
        self.s3_bucket = None
        self.s3_prefix = None
        if self.using_s3:
            self.s3_bucket, prefix = split_s3_uri(images_dir)
            self.s3_prefix = prefix.rstrip('/')
        
        # Load metadata
        if is_s3_uri(metadata_path):
            bucket, key = split_s3_uri(metadata_path)
            local_csv = self.s3_cache.get_csv_as_local(bucket, key)
            self.df = pd.read_csv(local_csv)
        else:
            if not os.path.isfile(metadata_path):
                raise FileNotFoundError(f'metadata.csv not found at {metadata_path}')
            self.df = pd.read_csv(metadata_path)
        
        # Load precomputed embeddings or initialize CLIP
        if precomputed_embeddings_path:
            print(f"[UpsamplerDataset] Loading precomputed embeddings from {precomputed_embeddings_path}")
            embedding_data = torch.load(precomputed_embeddings_path)
            self.precomputed_embeddings = embedding_data['embeddings']  # From 128x128 images
            self.use_precomputed = True
            print(f"[UpsamplerDataset] Loaded {len(self.precomputed_embeddings)} precomputed embeddings")
        else:
            print("[UpsamplerDataset] Warning: No precomputed embeddings, will encode on-the-fly")
            self.clip_encoder = CLIPEncoder().to(device).eval()
            self.use_precomputed = False
        
        # Transforms for low-res and high-res
        self.transform_low_res = transforms.Compose([
            transforms.Resize((low_res_size, low_res_size), antialias=True),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1)  # [-1, 1]
        ])
        
        self.transform_high_res = transforms.Compose([
            transforms.Resize((high_res_size, high_res_size), antialias=True),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1)  # [-1, 1]
        ])
        
        self.T = len(noise_scheduler.alpha_bar_t)
        self.total_len = len(self.df) * n_repeat
    
    def __len__(self):
        return self.total_len
    
    def _resolve_img_path(self, img_path_csv: str) -> str:
        """Resolve image path for local or S3 storage."""
        if self.using_s3:
            fname = os.path.basename(img_path_csv)
            key = f"{self.s3_prefix}/{fname}" if self.s3_prefix else fname
            return self.s3_cache.get_local_path(self.s3_bucket, key)
        else:
            if os.path.isabs(img_path_csv):
                return img_path_csv
            return os.path.join(self.images_dir, os.path.basename(img_path_csv))
    
    def __getitem__(self, idx):
        """
        Returns a training sample.
        
        Returns dict with:
            - high_res: (3, 128, 128) clean high-res image
            - low_res: (3, 64, 64) low-res conditioning image
            - clip_embedding: (512,) CLIP embedding from high-res image
        """
        row_idx = idx % len(self.df)
        row = self.df.iloc[row_idx]
        img_path = self._resolve_img_path(str(row['image_path']))
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Create low-res and high-res versions
        low_res = self.transform_low_res(image).to(self.device)  # (3, 64, 64)
        high_res = self.transform_high_res(image).to(self.device)  # (3, 128, 128)
        
        # Get CLIP embedding (from high-res image)
        if self.use_precomputed:
            clip_emb = self.precomputed_embeddings[row_idx].to(self.device)
        else:
            with torch.no_grad():
                high_res_batch = high_res.unsqueeze(0)  # (1, 3, 128, 128)
                clip_emb = self.clip_encoder.encode_image(high_res_batch).squeeze(0)
                clip_emb = F.normalize(clip_emb, dim=-1)
        
        return {
            'high_res': high_res,      # (3, 128, 128)
            'low_res': low_res,        # (3, 64, 64)
            'clip_embedding': clip_emb  # (512,)
        }
    
    def get_random_sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a random sample for visualization.
        
        Returns:
            (high_res, low_res, clip_embedding)
        """
        idx = random.randint(0, len(self.df) - 1)
        row = self.df.iloc[idx]
        img_path = self._resolve_img_path(str(row['image_path']))
        
        image = Image.open(img_path).convert('RGB')
        
        low_res = self.transform_low_res(image).unsqueeze(0).to(self.device)
        high_res = self.transform_high_res(image).unsqueeze(0).to(self.device)
        
        if self.use_precomputed:
            clip_emb = self.precomputed_embeddings[idx].unsqueeze(0).to(self.device)
        else:
            with torch.no_grad():
                clip_emb = self.clip_encoder.encode_image(high_res)
                clip_emb = F.normalize(clip_emb, dim=-1)
        
        return high_res, low_res, clip_emb


class SingleImageUpsamplerOverfitDataset(Dataset):
    """
    Dataset that returns the same image repeatedly for overfitting tests.
    Useful for debugging the upsampler.
    """
    
    def __init__(
        self,
        image_path: str,
        z_img: torch.Tensor,
        device: torch.device,
        noise_scheduler: NoiseScheduler,
        low_res_size: int = 64,
        high_res_size: int = 128,
        steps_per_epoch: int = 500
    ):
        super().__init__()
        self.device = device
        self.noise_scheduler = noise_scheduler
        self.T = len(noise_scheduler.alpha_bar_t)
        self.steps_per_epoch = steps_per_epoch
        
        # Load and preprocess the single image
        image = Image.open(image_path).convert('RGB')
        
        # Transforms
        self.transform_low_res = transforms.Compose([
            transforms.Resize((low_res_size, low_res_size), antialias=True),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1)
        ])
        
        self.transform_high_res = transforms.Compose([
            transforms.Resize((high_res_size, high_res_size), antialias=True),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1)
        ])
        
        self.low_res = self.transform_low_res(image).unsqueeze(0).to(device)
        self.high_res = self.transform_high_res(image).unsqueeze(0).to(device)
        self.z_img = z_img.to(device)
    
    def __len__(self):
        return self.steps_per_epoch
    
    def __getitem__(self, idx):
        return {
            'high_res': self.high_res.squeeze(0),
            'low_res': self.low_res.squeeze(0),
            'clip_embedding': self.z_img.squeeze(0)
        }
    
    def get_random_sample(self):
        return self.high_res, self.low_res, self.z_img.unsqueeze(0)