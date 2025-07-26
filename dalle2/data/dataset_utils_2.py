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
        noise_scheduler: NoiseScheduler,
        resize_size: int = 128,
        n_repeat: int = 64,  # number of samples
        seed: int = 42       # for reproducibility
    ):
        super().__init__()
        self.B = batch_size
        self.device = device
        self.noise_scheduler = noise_scheduler
        self.T = noise_scheduler.alpha_bar_t.shape[0]
        self.n_repeat = n_repeat

        # Deterministic sampling
        g = torch.Generator(device=device).manual_seed(seed)
        self.timesteps = torch.randint(0, self.T, (n_repeat,), generator=g, device=device)

        # Load and encode image
        self.df = pd.read_csv(metadata_path)
        self.images_dir = images_dir
        row = self.df.iloc[0]
        self.caption = row['caption']

        img_path = os.path.join(self.images_dir, os.path.basename(row['image_path']))
        image = Image.open(img_path).convert("RGB")

        self.transform = transforms.Compose([
            transforms.Resize((resize_size, resize_size), antialias=True),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1)
        ])

        self.clip_encoder = CLIPEncoder().to(device)
        image_tensor = self.transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            self.z_img = self.clip_encoder.encode_image(image_tensor).squeeze(0).to(device)  # (512,)
            self.z_txt = self.clip_encoder.encode_text([self.caption]).squeeze(0).to(device)  # (512,)

        # Pre-sample all noise vectors
        self.noises = torch.randn(n_repeat, self.z_img.shape[0], generator=g, device=device)

    def __len__(self):
        return self.n_repeat

    def __getitem__(self, idx):
        t = torch.randint(0, self.T, (1,), device=self.device)
        eps = torch.randn(1, self.z_img.shape[0], device=self.device)
        z_img = self.z_img.unsqueeze(0)
        z_img_noisy = self.noise_scheduler.q_sample(z_img, t, eps)

        return {
            'z_txt': self.z_txt,                    # (512,)
            't': t.squeeze(0),                      # ()
            'z_img_noisy': z_img_noisy.squeeze(0),  # (512,)
            'eps_img': eps.squeeze(0),              # (512,)
            'z_img': self.z_img                     # (512,)
        }


class COCODecoderDataset(Dataset):
    def __init__(
        self,
        metadata_path: str,
        images_dir: str,
        device: torch.device,
        noise_scheduler: NoiseScheduler,
        resize_size: int = 64,
        n_repeat: int = 64 # number of times to repeat the single image
    ):
        super().__init__()
        self.device = device
        self.n_repeat = n_repeat
        self.noise_scheduler = noise_scheduler

        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(f'metadata.csv not found at {metadata_path}')
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f'Image directory not found at {images_dir}')

        self.df = pd.read_csv(metadata_path)
        self.images_dir = images_dir
        self.resize_size = resize_size

        self.transform = transforms.Compose([
            transforms.Resize((resize_size, resize_size), antialias=True),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1)
        ])

        self.clip_encoder = CLIPEncoder().to(device)
        self.noise_scheduler.alpha_bar_t = self.noise_scheduler.alpha_bar_t.to(device)
        self.T = self.noise_scheduler.alpha_bar_t.shape[0]

        # Only use the first image
        self.image_tensor = self._load_image_tensor(self.df.iloc[0]['image_path'])

        with torch.no_grad():
            self.z_img = self.clip_encoder.encode_image(self.image_tensor).to(self.device)
            self.z_img = self.z_img / self.z_img.norm(dim=-1, keepdim=True)

    def _load_image_tensor(self, img_path):
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.images_dir, os.path.basename(img_path))
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f'Image not found at {img_path}')
        image = Image.open(img_path).convert('RGB')
        return self.transform(image).to(self.device).unsqueeze(0) # (1, 3, H, W)

    def __len__(self):
        return self.n_repeat

    def __getitem__(self, idx):
        batch_size = self.image_tensor.shape[0]
        t = torch.randint(0, self.T, (batch_size,), device=self.device).long()

        x_t, eps_img = self.noise_scheduler.add_noise(self.image_tensor, t)

        with torch.no_grad():
            a_bar = self.noise_scheduler.get_alpha_bar(t).view(-1,1,1,1)
            x_t_re = (a_bar.sqrt() * self.image_tensor +
                    (1 - a_bar).sqrt() * eps_img)
            assert torch.allclose(x_t, x_t_re, atol=1e-6), 'Mismatch between x_t and eps_img!'

        return {
            'x_t': x_t.squeeze(0),
            'z_img': self.z_img.squeeze(0),
            't': t.squeeze(0),
            'eps_img': eps_img.squeeze(0)
        }

