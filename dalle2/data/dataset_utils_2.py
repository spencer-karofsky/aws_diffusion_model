"""
dataset_utils_2.py: Contains Improved Dataset Curation and Retrieval.

Description:
    * Prior:
        - Training:
            - Inputs (to the Denoising Decoder-only Transformer):
                1. Timestep vector, t (positionally-encoded and MLP-projected), of shape (B, 512)
                    - Timesteps are uniform-randomly sampled
                2. CLIP text embeddings, z_txt, of shape (B, 512)
                3. CLIP image embeddings, z_img, of shape (B, 512)
                    - The DDPM process adds noise to these embeddings, resulting in the true image noise in the CLIP image embeddings.
            - Predicts:
                - Noise in the CLIP image embeddings for the target images it seeks to generate, of shape (B, 512)
            - Loss: MSE(predicted noise, true noise)
            - Objective: Train the decoder-only transformer to predict the noise in the image embeddings
        - Inference:
            - Inputs (to the trained Transformer):
                1. Timestep vector, t (positionally-encoded and MLP-projected), of shape (B, 512)
                    - Timesteps are uniformly distributed from steps T to 0
                2. CLIP text embeddings, of shape (B, 512)
                3. Pure Gaussian noise Tensor, of shape (B, 512)
            - Predicts:
                - Noise in the image embeddings, of shape (B, 512)
                    - Once we know the noise, we can deterministically remove it to get the clean image embeddings
    * Decoder:
        - Training:
            - Inputs (to the Denoising U-Net):
                1. Timestep vector, t (positionally-encoded and MLP-projected), of shape (B, 512)
                    - Timesteps are uniform-randomly sampled
                2. True CLIP image embeddings (output of the CLIP encoder), of shape (B, 512)
                3. Noisy images (added according to the DDPM forward noising process), of shape (B, 3, H, W)
            - Predicts: 
                - Noise that is corrupting the clean images, of shape (B, 3, H, W)
            - Loss: MSE(predicted image noise, true image noise)
            - Objective: Train the U-Net to predict the noise in the clean images
        - Inference:
            - Inputs (to the trained U-Net):
                1. Timestep vector, t (positionally-encoded and MLP-projected), of shape (B, 512)
                    - Timesteps are uniformly distributed from steps T to 0
                2. Predicted image embeddings (output of the trained prior), of shape (B, 512)
                3. Pure Gaussian noise images, of shape (B, 3, H, W)
            - Predicts:
                - Noise in the images, of shape (B, 3, H, W)
                    - Once we know the predicted noise, denoising the image is a determinstic process that results in the generated images.

Author:
    * Spencer Karofsky (https://github.com/spencer-karofsky)
"""
from __future__ import annotations
from torch.utils.data import Dataset
import torch
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from dalle2.models.clip_encoding import CLIPEncoder
from dalle2.sampling.noise_scheduler import NoiseScheduler
import random
from typing import Tuple

import os, random, pandas as pd, torch
from pathlib import Path
from typing import Tuple
from urllib.parse import urlparse

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F

from dalle2.models.clip_encoding import CLIPEncoder
from dalle2.sampling.noise_scheduler import NoiseScheduler


# datasets/midjourney_dropin.py
from torch.utils.data import Dataset
import torch, os, random, pandas as pd
from PIL import Image
from torchvision import transforms
from typing import Tuple
from dalle2.models.clip_encoding import CLIPEncoder
from dalle2.sampling.noise_scheduler import NoiseScheduler
import io
import boto3
from urllib.parse import urlparse
from pathlib import Path

def is_s3_uri(p: str) -> bool:
    return isinstance(p, str) and p.startswith("s3://")

def split_s3_uri(uri: str):
    pr = urlparse(uri)
    return pr.netloc, pr.path.lstrip('/')  # bucket, key

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
        # same as get_local_path, but explicit name for clarity
        return self.get_local_path(bucket, key)



def _resolve_path(p: str, images_dir: str) -> str:
    """If CSV path is absolute, return as-is; else join with images_dir (by basename)."""
    if os.path.isabs(p):
        return p
    if images_dir is None:
        return p
    return os.path.join(images_dir, os.path.basename(p))



# ─────────────────────────────────────────────────────────────────────────────
# small helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_s3_uri(p: str) -> bool:
    return isinstance(p, str) and p.startswith("s3://")


def split_s3_uri(uri: str):
    pr = urlparse(uri)
    return pr.netloc, pr.path.lstrip("/")  # bucket, key


class S3Cache:
    """Downloads S3 objects to a local cache and returns the local path."""

    def __init__(self, cache_dir: str = "/tmp/dalle2_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        import boto3

        self.s3 = boto3.client("s3")

    def get_local_path(self, bucket: str, key: str) -> str:
        local_path = self.cache_dir / bucket / key
        if not local_path.exists():
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self.s3.download_file(bucket, key, str(local_path))
        return str(local_path)

    def get_csv_as_local(self, bucket: str, key: str) -> str:  # alias
        return self.get_local_path(bucket, key)

class MidJourneyPriorDataset(Dataset):
    """Dataset that yields noisy-image-embedding triplets for training the prior.

    Each item is a **dict** containing:
        z_txt        : (512,)  clean, unit-norm text embedding
        t            : ()      timestep integer
        z_img_noisy  : (512,)  noisy image embedding at timestep *t*
        eps_img      : (512,)  the ground-truth noise added to z_img
        z_img        : (512,)  clean, unit-norm image embedding (for metrics)
    """

    def __init__(
        self,
        metadata_path: str,
        images_dir: str,
        batch_size: int,
        device: torch.device,
        noise_scheduler: NoiseScheduler,
        resize_size: int = 128,
        n_repeat: int = 1,
        seed: int = 42,
        cache_dir: str = "/tmp/dalle2_cache",
        precomputed_embeddings_path: str = None,
    ):
        super().__init__()
        self.B = batch_size
        self.device = device
        self.noise_scheduler = noise_scheduler
        self.T = noise_scheduler.alpha_bar_t.shape[0]
        self.images_dir = images_dir
        self.n_repeat = n_repeat
        self.seed = seed

        # ─── optional S3 support ────────────────────────────────────────────
        self.using_s3 = is_s3_uri(images_dir)
        self.s3_cache = (
            S3Cache(cache_dir) if self.using_s3 or is_s3_uri(metadata_path) else None
        )
        if self.using_s3:
            self.s3_bucket, prefix = split_s3_uri(images_dir)
            self.s3_prefix = prefix.rstrip("/")
        else:
            self.s3_bucket, self.s3_prefix = None, None

        # ─── load metadata CSV ──────────────────────────────────────────────
        if is_s3_uri(metadata_path):
            bucket, key = split_s3_uri(metadata_path)
            local_csv = self.s3_cache.get_csv_as_local(bucket, key)
            self.df = pd.read_csv(local_csv)
        else:
            if not os.path.isfile(metadata_path):
                raise FileNotFoundError(f"metadata.csv not found at {metadata_path}")
            self.df = pd.read_csv(metadata_path)

        # ─── Load precomputed embeddings or initialize CLIP ─────────────────
        if precomputed_embeddings_path:
            print(f"[PriorDataset] Loading precomputed embeddings from {precomputed_embeddings_path}")
            embedding_data = torch.load(precomputed_embeddings_path)
            self.image_embeddings = embedding_data['image_embeddings']
            self.text_embeddings = embedding_data['text_embeddings']
            self.use_precomputed = True
            print(f"[PriorDataset] Loaded {len(self.image_embeddings)} precomputed embedding pairs")
        else:
            # Fallback to on-the-fly encoding (slow)
            print("[PriorDataset] Warning: No precomputed embeddings provided, will encode on-the-fly")
            self.transform = transforms.Compose([
                transforms.Resize((resize_size, resize_size), antialias=True),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * 2 - 1),
            ])
            self.clip_encoder = CLIPEncoder().to(device).eval()
            self.use_precomputed = False

        self.total_len = len(self.df) * n_repeat

    def __len__(self):
        return self.total_len

    def _resolve_img_path(self, img_path_csv: str) -> str:
        if self.using_s3:
            fname = os.path.basename(img_path_csv)
            key = f"{self.s3_prefix}/{fname}" if self.s3_prefix else fname
            return self.s3_cache.get_local_path(self.s3_bucket, key)
        if os.path.isabs(img_path_csv):
            return img_path_csv
        return os.path.join(self.images_dir, os.path.basename(img_path_csv))

    def __getitem__(self, idx):
        row_idx = idx % len(self.df)
        
        if self.use_precomputed:
            # Fast path: load precomputed embeddings
            z_img = self.image_embeddings[row_idx].to(self.device)
            z_txt = self.text_embeddings[row_idx].to(self.device)
        else:
            # Slow path: encode on-the-fly
            row = self.df.iloc[row_idx]
            caption = row["caption"]
            img_path = self._resolve_img_path(str(row["image_path"]))
            
            image = Image.open(img_path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                z_img = self.clip_encoder.encode_image(image_tensor).squeeze(0).to(self.device)
                z_txt = self.clip_encoder.encode_text([caption]).squeeze(0).to(self.device)
                
                z_img = F.normalize(z_img, dim=-1)
                z_txt = F.normalize(z_txt, dim=-1)
        
        # Sample diffusion timestep and noise
        g = torch.Generator(device=self.device).manual_seed(self.seed + idx)
        t = torch.randint(0, self.T, (1,), generator=g, device=self.device)
        eps = torch.randn(1, z_img.shape[0], generator=g, device=self.device)
        z_img_noisy = self.noise_scheduler.q_sample(z_img.unsqueeze(0), t, eps)

        return {
            "z_txt": z_txt,
            "t": t.squeeze(0),
            "z_img_noisy": z_img_noisy.squeeze(0),
            "eps_img": eps.squeeze(0),
            "z_img": z_img,
        }

    @torch.no_grad()
    def get_random_text_and_embedding(self) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = random.randint(0, len(self.df) - 1)
        
        if self.use_precomputed:
            z_txt = self.text_embeddings[idx].to(self.device)
            z_img = self.image_embeddings[idx].to(self.device)
        else:
            row = self.df.iloc[idx]
            caption = row["caption"]
            img_path = self._resolve_img_path(str(row["image_path"]))
            
            image = Image.open(img_path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            z_img = self.clip_encoder.encode_image(image_tensor).squeeze(0).to(self.device)
            z_txt = self.clip_encoder.encode_text([caption]).squeeze(0).to(self.device)
            
            z_img = F.normalize(z_img, dim=-1)
            z_txt = F.normalize(z_txt, dim=-1)
        
        return z_txt, z_img

class MidJourneyDecoderDataset(Dataset):
    def __init__(
        self,
        metadata_path: str,
        images_dir: str,
        device: torch.device,
        noise_scheduler: NoiseScheduler,
        resize_size: int = 64,
        n_repeat: int = 1,
        cache_dir: str = "/tmp/dalle2_cache",
        precomputed_embeddings_path: str = None,
    ):
        super().__init__()
        self.device = device
        self.noise_scheduler = noise_scheduler
        self.images_dir = images_dir

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
        if precomputed_embeddings_path:
            embedding_data = torch.load(precomputed_embeddings_path)
            self.precomputed_embeddings = embedding_data['embeddings']
            self.use_precomputed = True
            print(f"[Dataset] Loaded {len(self.precomputed_embeddings)} pre-computed CLIP embeddings")
        else:
            self.clip_encoder = CLIPEncoder().to(device).eval()
            self.use_precomputed = False
      
        self.transform = transforms.Compose([
            transforms.Resize((resize_size, resize_size), antialias=True),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1)
        ])

        self.clip_encoder = CLIPEncoder().to(device).eval()
        self.noise_scheduler.alpha_bar_t = self.noise_scheduler.alpha_bar_t.to(device)
        self.T = self.noise_scheduler.alpha_bar_t.shape[0]

        self.n_repeat = n_repeat
        self.total_len = len(self.df) * n_repeat

    def __len__(self):
        return self.total_len

    def _resolve_img_path(self, img_path_csv: str) -> str:
        if self.using_s3:
            fname = os.path.basename(img_path_csv)
            key = f"{self.s3_prefix}/{fname}" if self.s3_prefix else fname
            return self.s3_cache.get_local_path(self.s3_bucket, key)
        else:
            if os.path.isabs(img_path_csv):
                return img_path_csv
            return os.path.join(self.images_dir, os.path.basename(img_path_csv))

    def __getitem__(self, idx):
        row_idx = idx % len(self.df)
        row = self.df.iloc[row_idx]
        img_path = self._resolve_img_path(str(row['image_path']))

        # Load and preprocess clean image
        image = Image.open(img_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)  # (1, 3, H, W)

        # Use pre-computed embeddings or encode on-the-fly
        if self.use_precomputed:
            z_img = self.precomputed_embeddings[row_idx].to(self.device).unsqueeze(0)
        else:
            with torch.no_grad():
                z_img = self.clip_encoder.encode_image(image_tensor).to(self.device)
                z_img = z_img / z_img.norm(dim=-1, keepdim=True)
        
        # Sample timestep and noise
        t = torch.randint(0, self.T, (1,), device=self.device).long()
        eps_img = torch.randn_like(image_tensor)
        x_t = self.noise_scheduler.q_sample(image_tensor, t, eps_img)

        return {
            'x_t': x_t.squeeze(0), # Noisy image
            'z_img': z_img.squeeze(0), # CLIP image embedding
            't': t.squeeze(0), # Timestep
            'eps_img': eps_img.squeeze(0), # True noise
            'x0': image_tensor.squeeze(0), # Clean image
        }


    def get_random_clean_image_and_embedding(self):
        idx = random.randint(0, len(self.df) - 1)
        row = self.df.iloc[idx]

        fname = os.path.basename(row["image_path"])
        local_path = self._resolve_img(row["image_path"])

        # Load low-res image (same as training)
        img = self._load_image(local_path)
        img = img.unsqueeze(0)   # (1,3,H,W)

        # Get CLIP embedding
        z_img = self.z_img_list[idx]
        z_img = z_img.unsqueeze(0)  # (1,512)  ✔️ FIX

        return img, z_img

    

class SingleImageOverfitDataset(Dataset):
    """Dataset that returns the same image repeatedly for overfitting tests."""
    
    def __init__(
        self,
        image_path: str,
        z_img: torch.Tensor,
        device: torch.device,
        noise_scheduler: NoiseScheduler,
        resize_size: int = 64,
        steps_per_epoch: int = 500
    ):
        super().__init__()
        self.device = device
        self.noise_scheduler = noise_scheduler
        self.T = len(noise_scheduler.alpha_bar_t)
        self.steps_per_epoch = steps_per_epoch
        
        # Load and preprocess the single image
        image = Image.open(image_path).convert('RGB')
        self.transform = transforms.Compose([
            transforms.Resize((resize_size, resize_size), antialias=True),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1)
        ])
        self.image_tensor = self.transform(image).unsqueeze(0).to(device)
        self.z_img = z_img.to(device)
        
    def __len__(self):
        return self.steps_per_epoch
    
    def __getitem__(self, idx):
        t = torch.randint(0, self.T, (1,), device=self.device).long()
        
        eps_img = torch.randn_like(self.image_tensor)
        x_t = self.noise_scheduler.q_sample(self.image_tensor, t, eps_img)
        
        return {
            'x_t': x_t.squeeze(0), # Noisy image
            'z_img': self.z_img.squeeze(0), # CLIP image embedding
            't': t.squeeze(0), # Timestep
            'eps_img': eps_img.squeeze(0), # True noise
            'x0': self.image_tensor.squeeze(0), # Clean image (ground truth)
        }

    
    def get_random_clean_image_and_embedding(self):
        return self.image_tensor, self.z_img.unsqueeze(0)