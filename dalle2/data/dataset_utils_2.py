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
from torch.utils.data import Dataset
import torch
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from dalle2.models.clip_encoding import CLIPEncoder
from dalle2.sampling.noise_scheduler import NoiseScheduler

class COCOPriorDataset(Dataset):
    def __init__(
        self,
        metadata_path: str,
        images_dir: str,
        batch_size: int,
        device: torch.device,
        resize_size: int = 128,
    ):
        super().__init__()
        self.B = batch_size
        self.device = device

        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(f"metadata.csv not found at {metadata_path}")
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"Image directory not found at {images_dir}")

        self.df = pd.read_csv(metadata_path)
        self.images_dir = images_dir
        self.resize_size = resize_size

        self.transform = transforms.Compose([
            transforms.Resize((resize_size, resize_size), antialias=True),
            transforms.ToTensor(),
        ])

        self.clip_encoder = CLIPEncoder().to(device)
        self.noise_scheduler = NoiseScheduler()
        self.noise_scheduler.alpha_bar_t = self.noise_scheduler.alpha_bar_t.to(device)
        self.T = self.noise_scheduler.alpha_bar_t.shape[0]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        caption = row["caption"]
        img_path = row["image_path"]

        if not os.path.isabs(img_path):
            img_path = os.path.join(self.images_dir, os.path.basename(img_path))
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image not found at {img_path}")

        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device) # shape: (1, 3, H, W)

        with torch.no_grad():
            z_txt = self.clip_encoder.encode_text_ensemble([caption], n=3).to(self.device) # shape: (1, 3, 512)
            z_img = self.clip_encoder.encode_image(image_tensor).to(self.device) # shape: (1, 512)

        # Sample timestep
        t = torch.randint(0, self.T, (1,), device=self.device).long()

        # Add noise
        z_img_noisy, eps_img = self.noise_scheduler.add_noise(z_img, t)

        return t.squeeze(0), z_txt.squeeze(0), z_img_noisy.squeeze(0), eps_img.squeeze(0)
    
class COCODecoderDataset(Dataset):
    def __init__(
        self,
        metadata_path: str,
        images_dir: str,
        device: torch.device,
        resize_size: int = 64,  # should match decoder output resolution
    ):
        super().__init__()
        self.device = device

        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(f"metadata.csv not found at {metadata_path}")
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"Image directory not found at {images_dir}")

        self.df = pd.read_csv(metadata_path)
        self.images_dir = images_dir
        self.resize_size = resize_size

        self.transform = transforms.Compose([
            transforms.Resize((resize_size, resize_size), antialias=True),
            transforms.ToTensor(),
        ])

        self.clip_encoder = CLIPEncoder().to(device)
        self.noise_scheduler = NoiseScheduler().to(device)
        self.T = self.noise_scheduler.alpha_bar_t.shape[0]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img_path = row["image_path"]

        if not os.path.isabs(img_path):
            img_path = os.path.join(self.images_dir, os.path.basename(img_path))
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image not found at {img_path}")

        image = Image.open(img_path).convert("RGB")
        x_0 = self.transform(image).to(self.device).unsqueeze(0)  # (1, 3, H, W)

        with torch.no_grad():
            z_img = self.clip_encoder.encode_image(x_0).to(self.device)  # (1, 512)

        # Sample timestep
        t = torch.randint(0, self.T, (1,), device=self.device).long()

        # Forward diffusion: Add noise to the image
        x_t, eps_img = self.noise_scheduler.add_noise(x_0, t)  # each: (1, 3, H, W)

        return t.squeeze(0), z_img.squeeze(0), x_t.squeeze(0), eps_img.squeeze(0)

