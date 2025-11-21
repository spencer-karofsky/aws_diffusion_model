from __future__ import annotations
import os, random
import torch
import pandas as pd
from pathlib import Path
from typing import Tuple
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from urllib.parse import urlparse

from dalle2.models.clip_encoding import CLIPEncoder
from dalle2.sampling.noise_scheduler import NoiseScheduler


### --- S3 helpers ---
def is_s3_uri(p: str):
    return isinstance(p, str) and p.startswith("s3://")

def split_s3_uri(uri: str):
    pr = urlparse(uri)
    return pr.netloc, pr.path.lstrip("/")

class S3Cache:
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



# ======================================================================
# 1) PRIOR DATASET  (caption → z_txt, image → z_img)
# ======================================================================

class BostonPriorDataset(Dataset):
    """
    Provides:
        z_txt        (512,)
        z_img_noisy  (512,)
        eps_img      (512,)
        t            ()
        z_img_clean  (512,)
    """

    def __init__(
        self,
        metadata_csv: str,
        images_dir: str,
        device: torch.device,
        noise_scheduler: NoiseScheduler,
        precomputed_path: str | None = None,
        resize: int = 128,
        n_repeat: int = 1,
        cache_dir: str = "/tmp/dalle2_cache"
    ):
        super().__init__()
        self.device = device
        self.noise_scheduler = noise_scheduler
        self.T = noise_scheduler.alpha_bar_t.shape[0]
        self.n_repeat = n_repeat

        # -------------------------
        # Metadata loading
        # -------------------------
        self.using_s3 = is_s3_uri(images_dir)
        if self.using_s3:
            self.s3_bucket, prefix = split_s3_uri(images_dir)
            self.s3_prefix = prefix.rstrip("/")
            self.s3_cache = S3Cache(cache_dir)
        else:
            self.images_dir = images_dir
            self.s3_cache = None

        self.df = pd.read_csv(metadata_csv)

        # -----------------------------
        # Precomputed CLIP embeddings
        # -----------------------------
        if precomputed_path:
            data = torch.load(precomputed_path, map_location="cpu")
            self.z_img_list = data["image_embeddings"]
            self.z_txt_list = data["text_embeddings"]
            self.use_precomputed = True
            print(f"[BostonPriorDataset] Loaded {len(self.z_img_list)} precomputed embeddings.")
        else:
            self.use_precomputed = False
            self.clip = CLIPEncoder().to(device).eval()
            self.transform = transforms.Compose([
                transforms.Resize((resize, resize), antialias=True),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x*2 - 1)
            ])

        self.total_len = len(self.df) * n_repeat


    def __len__(self):
        return self.total_len


    def _resolve_img_path(self, rel_path: str) -> str:
        fname = os.path.basename(rel_path)
        if not self.using_s3:
            return os.path.join(self.images_dir, fname)
        key = f"{self.s3_prefix}/{fname}" if self.s3_prefix else fname
        return self.s3_cache.get_local_path(self.s3_bucket, key)


    def __getitem__(self, idx):
        i = idx % len(self.df)
        row = self.df.iloc[i]
        caption = row["caption"]
        img_path = self._resolve_img_path(row["image_path"])

        if self.use_precomputed:
            z_img = self.z_img_list[i].to(self.device)
            z_txt = self.z_txt_list[i].to(self.device)
        else:
            # load + encode image
            image = Image.open(img_path).convert("RGB")
            x = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                z_img = self.clip.encode_image(x).squeeze(0)
                z_txt = self.clip.encode_text([caption]).squeeze(0)

            z_img = F.normalize(z_img, dim=-1)
            z_txt = F.normalize(z_txt, dim=-1)

        # diffusion
        t = torch.randint(0, self.T, (1,), device=self.device)
        eps = torch.randn_like(z_img)
        z_noisy = self.noise_scheduler.q_sample(z_img.unsqueeze(0), t, eps.unsqueeze(0))

        return {
            "z_txt": z_txt,
            "z_img_noisy": z_noisy.squeeze(0),
            "eps_img": eps,
            "t": t.squeeze(0),
            "z_img": z_img,
        }


class BostonDecoder32Dataset(Dataset):
    """
    Provides:
        x_t      : noisy 32×32 image
        z_img    : CLIP embedding (512,)
        t        : timestep
        eps_img  : noise
        x0       : clean 32×32
    """

    def __init__(
        self,
        metadata_csv: str,
        images_dir: str,
        precomputed_embeddings: str,
        device: torch.device,
        noise_scheduler: NoiseScheduler,
        n_repeat: int = 1,
        cache_dir: str = "/tmp/dalle2_cache",
        lowres: int = 32,
    ):
        super().__init__()
        self.device = device
        self.noise_scheduler = noise_scheduler
        self.T = noise_scheduler.alpha_bar_t.shape[0]
        self.n_repeat = n_repeat
        self.lowres = lowres

        # Metadata
        self.df = pd.read_csv(metadata_csv)

        # Precomputed CLIP embeddings (only need z_img)
        data = torch.load(precomputed_embeddings, map_location="cpu")
        self.z_img_list = data["image_embeddings"]
        print(f"[Decoder32] Loaded {len(self.z_img_list)} embeddings.")

        # S3/local
        self.using_s3 = is_s3_uri(images_dir)
        if self.using_s3:
            self.s3_bucket, prefix = split_s3_uri(images_dir)
            self.s3_prefix = prefix.rstrip("/")
            self.s3_cache = S3Cache(cache_dir)
        else:
            self.images_dir = images_dir

        # 32×32 transform
        self.transform = transforms.Compose([
            transforms.Resize((lowres, lowres), antialias=True),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x*2 - 1)
        ])

        self.total_len = len(self.df) * n_repeat


    def __len__(self):
        return self.total_len


    def _resolve_img(self, path: str):
        fname = os.path.basename(path)
        if not self.using_s3:
            return os.path.join(self.images_dir, fname)
        key = f"{self.s3_prefix}/{fname}" if self.s3_prefix else fname
        return self.s3_cache.get_local_path(self.s3_bucket, key)


    def __getitem__(self, idx):
        i = idx % len(self.df)
        row = self.df.iloc[i]

        img_path = self._resolve_img(row["image_path"])
        z_img = self.z_img_list[i].to(self.device)

        # load 32×32
        x0 = self.transform(Image.open(img_path).convert("RGB")).to(self.device)

        # diffusion
        t = torch.randint(0, self.T, (1,), device=self.device)
        eps = torch.randn_like(x0)
        x_t = self.noise_scheduler.q_sample(x0.unsqueeze(0), t, eps.unsqueeze(0)).squeeze(0)

        return {
            "x_t": x_t,
            "z_img": z_img,
            "eps_img": eps,
            "t": t.squeeze(0),
            "x0": x0,
        }

class BostonUpsamplerDataset(Dataset):
    """
    Provides:
        low_res      : 32×32
        high_res     : 128×128
        t            : diffusion timestep (if used)
        eps_img      : noise
        x_t          : noisy 128×128 (if diffusion is used)
    """

    def __init__(
        self,
        metadata_csv: str,
        images_dir: str,
        device: torch.device,
        noise_scheduler: NoiseScheduler | None = None,
        lowres: int = 32,
        highres: int = 128,
        n_repeat: int = 1,
        cache_dir: str = "/tmp/dalle2_cache",
        diffusion: bool = True
    ):
        super().__init__()
        self.device = device
        self.noise_scheduler = noise_scheduler
        self.diffusion = diffusion
        self.lowres = lowres
        self.highres = highres
        self.n_repeat = n_repeat

        self.df = pd.read_csv(metadata_csv)

        # S3/local logic
        self.using_s3 = is_s3_uri(images_dir)
        if self.using_s3:
            self.s3_bucket, prefix = split_s3_uri(images_dir)
            self.s3_prefix = prefix.rstrip("/")
            self.s3_cache = S3Cache(cache_dir)
        else:
            self.images_dir = images_dir

        self.low_transform = transforms.Compose([
            transforms.Resize((lowres, lowres), antialias=True),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x*2 - 1),
        ])

        self.high_transform = transforms.Compose([
            transforms.Resize((highres, highres), antialias=True),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x*2 - 1),
        ])

        if diffusion:
            self.T = noise_scheduler.alpha_bar_t.shape[0]

        self.total_len = len(self.df) * n_repeat


    def __len__(self):
        return self.total_len


    def _resolve_img(self, p: str):
        fname = os.path.basename(p)
        if not self.using_s3:
            return os.path.join(self.images_dir, fname)
        key = f"{self.s3_prefix}/{fname}" if self.s3_prefix else fname
        return self.s3_cache.get_local_path(self.s3_bucket, key)


    def __getitem__(self, idx):
        i = idx % len(self.df)
        row = self.df.iloc[i]

        path = self._resolve_img(row["image_path"])
        img = Image.open(path).convert("RGB")

        low = self.low_transform(img).to(self.device)
        high = self.high_transform(img).to(self.device)

        if not self.diffusion:
            return {"low": low, "high": high}

        # diffusion on 128×128
        t = torch.randint(0, self.T, (1,), device=self.device)
        eps = torch.randn_like(high)
        x_t = self.noise_scheduler.q_sample(high.unsqueeze(0), t, eps.unsqueeze(0)).squeeze(0)

        return {
            "low": low,
            "high": high,
            "x_t": x_t,
            "eps_img": eps,
            "t": t.squeeze(0),
        }
