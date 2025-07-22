import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
from torchvision import transforms
from dalle2.models.clip_encoding import CLIPEncoder
from dalle2.sampling.noise_scheduler import NoiseScheduler

class BaseCOCODataset(Dataset):
    def __init__(self, metadata_path=None, images_dir=None, resize_size=128):
        # Resolve default paths if not provided
        current_dir = os.path.dirname(os.path.abspath(__file__))
        default_data_root = os.path.abspath(os.path.join(current_dir, '..', 'data', 'local_datasets'))
        
        metadata_path = metadata_path or os.path.join(default_data_root, 'metadata.csv')
        images_dir = images_dir or os.path.join(default_data_root, 'val2017')

        # Verify file existence
        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(f"metadata.csv not found at {metadata_path}")
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"Image directory not found at {images_dir}")

        # Load metadata and store paths
        self.df = pd.read_csv(metadata_path)
        self.images_dir = images_dir
        self.resize_size = resize_size

        # Shared modules
        self.clip_encoder = CLIPEncoder()
        self.noise_scheduler = NoiseScheduler()

        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize((resize_size, resize_size), antialias=True),
            transforms.ToTensor(),
        ])

    def load_image_and_caption(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.images_dir, os.path.basename(img_path))
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image not found at {img_path}")
        image = Image.open(img_path).convert("RGB")
        caption = row['caption']
        return self.transform(image), caption

    def encode(self, image, caption):
        z_img = self.clip_encoder.encode_image(image.unsqueeze(0)).squeeze(0)
        z_txt = self.clip_encoder.encode_text(caption)
        return z_img, z_txt

    def sample_timestep_and_noise(self, image: torch.Tensor):
        """
        Samples a timestep and returns the noised image and the noise used.

        Args:
            image: a tensor of shape (C, H, W)

        Returns:
            t: scalar timestep (int tensor)
            noise: Gaussian noise used, shape (C, H, W)
            x_t: noised image, shape (C, H, W)
        """
        device = image.device
        T = self.noise_scheduler.alpha_bar_t.shape[0]

        t = torch.randint(0, T, (1,), device=device).long()  # (1,)
        noise = torch.randn_like(image, device=device)

        alpha_bar_t = self.noise_scheduler.get_alpha_bar(t).view(-1, 1, 1).to(device)  # (1, 1, 1)
        x_t = torch.sqrt(alpha_bar_t) * image + torch.sqrt(1 - alpha_bar_t) * noise

        return t.squeeze(0), noise, x_t


    def __len__(self):
        return 128
        # return len(self.df)
    
class PriorCOCODataset(BaseCOCODataset):
    def __getitem__(self, idx):
        image, caption = self.load_image_and_caption(idx)
        z_img, z_txt = self.encode(image, caption)
        t, noise, z_T = self.sample_timestep_and_noise(z_img)
        return z_T, z_txt, t, noise

class DecoderCOCODataset(BaseCOCODataset):
    def __getitem__(self, idx):
        image, caption = self.load_image_and_caption(idx)
        z_img, _ = self.encode(image, caption)
        t, noise, x_t = self.sample_timestep_and_noise(image)
        return x_t, z_img, t, noise

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from dalle2.models.clip_encoding import CLIPEncoder
from dalle2.sampling.noise_scheduler import NoiseScheduler
import torch
class SingleImagePriorDataset(Dataset):
    def __init__(self, image_path: str, caption: str, resize_size=128, device='cpu'):
        self.device = device
        self.caption = caption
        self.resize_size = resize_size

        # Load and transform the image
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((resize_size, resize_size), antialias=True),
            transforms.ToTensor(),
        ])
        self.image_tensor = transform(image).to(device)  # [C, H, W]

        # Shared modules
        self.clip_encoder = CLIPEncoder()
        self.clip_encoder.eval()
        self.noise_scheduler = NoiseScheduler()

        with torch.no_grad():
            self.z_img = self.clip_encoder.encode_image(self.image_tensor.unsqueeze(0)).squeeze(0).to(device)
            self.z_txt = self.clip_encoder.encode_text(self.caption).to(device)  # [1, 512] or [512]

    def __len__(self):
        return 64  # Overfit for 64 batches

    def __getitem__(self, idx):
        T = self.noise_scheduler.alpha_bar_t.shape[0]
        t = torch.randint(0, T, (1,), device=self.device).long()
        noise = torch.randn_like(self.z_img)
        alpha_bar_t = self.noise_scheduler.get_alpha_bar(t).view(-1, 1).to(self.device)
        z_T = torch.sqrt(alpha_bar_t) * self.z_img + torch.sqrt(1 - alpha_bar_t) * noise

        return z_T, self.z_txt.clone(), t.squeeze(0), noise

class SingleImageDecoderDataset(Dataset):
    def __init__(self, image_path: str, caption: str, resize_size=128, device='cpu'):
        self.device = device
        self.caption = caption
        self.resize_size = resize_size

        # Load and transform the image
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((resize_size, resize_size), antialias=True),
            transforms.ToTensor(),
        ])
        self.image_tensor = transform(image).to(device)  # [C, H, W]

        # Shared modules
        self.clip_encoder = CLIPEncoder()
        self.clip_encoder.eval()
        self.noise_scheduler = NoiseScheduler()

        with torch.no_grad():
            self.z_img = self.clip_encoder.encode_image(self.image_tensor.unsqueeze(0)).squeeze(0).to(device)

    def __len__(self):
        return 64  # Overfit loop

    def __getitem__(self, idx):
        # Sample a random timestep
        T = self.noise_scheduler.alpha_bar_t.shape[0]
        t = torch.randint(0, T, (1,), device=self.device).long()

        # Sample noise and apply forward process manually
        noise = torch.randn_like(self.image_tensor)
        alpha_bar_t = self.noise_scheduler.get_alpha_bar(t).view(-1, 1, 1).to(self.device)
        x_t = torch.sqrt(alpha_bar_t) * self.image_tensor + torch.sqrt(1 - alpha_bar_t) * noise

        # Return x_t, z_img, t, and the noise that was used (target = eps_t)
        return x_t, self.z_img.clone(), t.squeeze(0), noise
