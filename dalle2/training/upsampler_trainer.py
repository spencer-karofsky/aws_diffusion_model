"""
train_upsampler.py: Training script for the DALL·E 2 upsampler.

Description:
    * Trains the upsampler to generate 128x128 images from 64x64 decoder outputs
    * Uses DDPM forward process to add noise to high-res images
    * Conditions on low-res images and CLIP embeddings
    * Predicts noise using MSE loss
    * Saves intermediate outputs and checkpoints to S3 (if on_aws=True)
    
Usage:
    from train_upsampler import UpsamplerTrainer
    trainer = UpsamplerTrainer(...)
    trainer.train(...)
    
Author:
    * Spencer Karofsky (https://github.com/spencer-karofsky)
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

# Standard library imports
import os
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import csv
import tempfile
import math
from copy import deepcopy

# AWS imports
import boto3

# Module imports
from dalle2.models.upsampler import Upsampler
from dalle2.sampling.noise_scheduler import NoiseScheduler
from dalle2.sampling.upsampler_ddim_sampling import UpsamplerDDIMSampler

from dalle2.data.boston_dataset_utils import BostonUpsamplerDataset

# Plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torchvision.utils as vutils


def cosine_lr(step: int, total_steps: int, lr_max: float, lr_min: float, warmup_steps: int = 0) -> float:
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return lr_max * (step + 1) / max(1, warmup_steps)
    t = step - warmup_steps
    T = max(1, total_steps - warmup_steps)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * t / T))


class UpsamplerTrainer:
    def __init__(
        self,
        train_module: Upsampler,
        optimizer: torch.optim.Optimizer,
        noise_scheduler: NoiseScheduler,
        dataset: Dataset,
        batch_size: int,
        model_save_name: str = "upsampler",
        shuffle: bool = True,
        on_aws: bool = False,
        debug: bool = False,
        use_amp: bool = False
    ):
        """
        Initialize the upsampler trainer.
        
        Args:
            train_module: the Upsampler model
            optimizer: optimizer for training
            noise_scheduler: NoiseScheduler instance
            dataset: PyTorch dataset with 'high_res', 'low_res', 'clip_embedding'
            batch_size: training batch size
            model_save_name: name for saving checkpoints
            shuffle: whether to shuffle dataloader
            on_aws: whether to upload artifacts to S3
            debug: enable debug printing
            use_amp: use automatic mixed precision
        """
        # Save params
        self.train_module = train_module
        self.optimizer = optimizer
        self.noise_scheduler = noise_scheduler
        self.model_save_name = model_save_name
        self.on_aws = on_aws
        self.debug = debug
        self.use_amp = use_amp
        self.scaler = GradScaler('cuda') if use_amp else None
        
        # Learning rate schedule params
        self.use_cosine_lr = True
        self.lr_max = 1e-4
        self.lr_min = 2e-5
        self.warmup_frac = 0.05
        
        # Create DataLoader
        num_workers = min(4, os.cpu_count() or 1)

        

        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0  # Set to 0 for debugging, increase for production
        )
        self.steps_per_epoch = len(self.dataloader)
        self.total_steps = 0  # Set when training starts
        
        # Device setup
        self.device = (
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )
        self.train_module.to(self.device)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.train_losses = []
        self.val_losses = []
        
        # EMA model
        self.ema_model = deepcopy(self.train_module)
        self.ema_decay = 0.999
        for p in self.ema_model.parameters():
            p.requires_grad = False
        
        # AWS configuration
        if self.on_aws:
            self._configure_for_aws()
        
        if self.debug:
            print(f'[UpsamplerTrainer] Training module: Upsampler')
            print(f'[UpsamplerTrainer] Device: {self.device}')
            print(f'[UpsamplerTrainer] Number of training batches: {len(self.dataloader)}')
            print(f'[UpsamplerTrainer] Model parameters: {sum(p.numel() for p in self.train_module.parameters()):,}')

    def _configure_for_aws(self) -> None:
        """Prepare S3 for artifact uploads."""
        self.outputs_bucket = 'dalle2-outputs'
        self.models_bucket = 'dalle2-models'
        self.s3 = boto3.client('s3')

    def _s3_put(self, local_path: str, key: str):
        """Upload file to S3."""
        try:
            if not self.on_aws:
                return
            if not hasattr(self, "s3"):
                self.s3 = boto3.client("s3")
            
            # Route by key
            is_checkpoint = (
                key.startswith("upsampler/checkpoints/")
                or key.endswith(".pth")
            )
            bucket = self.models_bucket if is_checkpoint else self.outputs_bucket
            self.s3.upload_file(local_path, bucket, key)
        except Exception as e:
            print(f"[WARN] S3 upload failed for key '{key}': {e}")

    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor) -> tuple:
        """
        Add noise to clean images using DDPM forward process.
        
        Args:
            x_0: clean high-res images, shape (B, 3, 128, 128)
            t: timesteps, shape (B,)
            
        Returns:
            Tuple of (x_t, eps) where x_t is noisy image and eps is the noise
        """
        B = x_0.size(0)
        
        # Sample noise
        eps = torch.randn_like(x_0)
        
        # Get alpha_bar for timestep t (correct attribute name)
        alpha_bar_t = self.noise_scheduler.alpha_bar_t[t].view(B, 1, 1, 1)
        
        # Add noise: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * eps
        
        return x_t, eps

    def train_step(self, high_res: torch.Tensor, low_res: torch.Tensor, clip_emb: torch.Tensor):
        """
        Perform one training step.
        
        Args:
            high_res: clean high-res images, shape (B, 3, 128, 128)
            low_res: low-res conditioning images, shape (B, 3, 64, 64)
            clip_emb: CLIP embeddings, shape (B, 512)
            
        Returns:
            Loss value
        """
        B = high_res.size(0)
        
        # Sample random timesteps
        t = torch.randint(0, self.noise_scheduler.T, (B,), device=self.device, dtype=torch.long)
        
        # Add noise to high-res images
        x_t, eps_true = self.add_noise(high_res, t)
        
        # Mixed precision training
        if self.use_amp:
            with autocast('cuda'):
                # Predict noise
                eps_pred = self.train_module(
                    x_t=x_t,
                    low_res_img=low_res,
                    z_img=clip_emb,
                    t=t
                )
                
                # Compute loss
                loss = self.criterion(eps_pred, eps_true)
        else:
            # Predict noise
            eps_pred = self.train_module(
                x_t=x_t,
                low_res_img=low_res,
                z_img=clip_emb,
                t=t
            )
            
            # Compute loss
            loss = self.criterion(eps_pred, eps_true)
        
        # Compute cosine similarity for monitoring
        cos_eps = F.cosine_similarity(
            eps_pred.flatten(1), eps_true.flatten(1), dim=1
        ).mean().item()
        
        # Backward pass
        self.optimizer.zero_grad()
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        
        # Update EMA model
        with torch.no_grad():
            for ema_p, p in zip(self.ema_model.parameters(), self.train_module.parameters()):
                ema_p.data.mul_(self.ema_decay).add_(p.data, alpha=1.0 - self.ema_decay)
        
        return loss.item(), cos_eps

    def _save_current_model(self, epoch: int, batch_i: int) -> None:
        """Save model checkpoint."""
        fname = f'epoch{epoch}_batch{batch_i}.pth'
        checkpoint = {
            "epoch": epoch,
            "batch": batch_i,
            "train_module": self.train_module.state_dict(),
            "ema_model": self.ema_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
        }
        
        if self.on_aws:
            with tempfile.TemporaryDirectory() as td:
                local_path = os.path.join(td, fname)
                torch.save(checkpoint, local_path)
                key = f"upsampler/checkpoints/{fname}"
                self._s3_put(local_path, key)
        else:
            os.makedirs(f"dalle2/checkpoints/upsampler", exist_ok=True)
            save_path = f"dalle2/checkpoints/upsampler/{fname}"
            torch.save(checkpoint, save_path)
            print(f"[UpsamplerTrainer] Checkpoint saved: {save_path}")

    def _save_final_model(self) -> None:
        """Save final trained model."""
        fname = "final_trained_model.pth"
        checkpoint = {
            "train_module": self.train_module.state_dict(),
            "ema_model": self.ema_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
        }
        
        if self.on_aws:
            with tempfile.TemporaryDirectory() as td:
                local_path = os.path.join(td, fname)
                torch.save(checkpoint, local_path)
                key = f"upsampler/checkpoints/{fname}"
                self._s3_put(local_path, key)
        else:
            os.makedirs(f"dalle2/checkpoints/upsampler", exist_ok=True)
            save_path = f"dalle2/checkpoints/upsampler/{fname}"
            torch.save(checkpoint, save_path)
            print(f"[UpsamplerTrainer] Final model saved: {save_path}")

    def _log_loss(self, epoch: int, batch: int, mse_loss: float, plot_interval: int = 500) -> None:
        """Log loss to CSV and generate plots."""
        log_dir = 'dalle2/checkpoints/logs'
        plot_dir = 'dalle2/checkpoints/plots'
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)
        
        csv_path = os.path.join(log_dir, 'upsampler_batch_losses.csv')
        plot_path = os.path.join(plot_dir, 'upsampler_loss_curve.png')
        
        # Append to CSV
        write_header = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['epoch', 'batch', 'mse_loss'])
            writer.writerow([epoch, batch, float(mse_loss.item() if hasattr(mse_loss, "item") else mse_loss)])
        
        # Mirror CSV to S3
        if self.on_aws:
            try:
                self._s3_put(csv_path, "upsampler/logs/upsampler_batch_losses.csv")
            except Exception as e:
                print(f"[WARN] S3 upload (CSV) failed: {e}")
        
        # Generate plots periodically
        if (batch + 1) % plot_interval == 0:
            epochs, batches, losses = [], [], []
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    epochs.append(int(row['epoch']))
                    batches.append(int(row['batch']))
                    losses.append(float(row['mse_loss']))
            
            plt.figure(figsize=(14, 6))
            
            # Linear scale
            plt.subplot(1, 2, 1)
            plt.plot(range(len(losses)), losses, label='Batch MSE Loss', linewidth=1.5)
            plt.xlabel('Batch Index')
            plt.ylabel('MSE Loss')
            plt.title('Upsampler Training Loss (Linear Scale)')
            plt.grid(True)
            plt.legend()
            
            # Log scale
            plt.subplot(1, 2, 2)
            plt.plot(range(len(losses)), losses, label='Batch MSE Loss (Log)', linewidth=1.5)
            plt.xlabel('Batch Index')
            plt.ylabel('MSE Loss (log)')
            plt.title('Upsampler Training Loss (Log Scale)')
            plt.yscale('log')
            plt.grid(True, which='both')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            
            # Mirror plot to S3
            if self.on_aws:
                try:
                    self._s3_put(plot_path, "upsampler/plots/upsampler_loss_curve.png")
                except Exception as e:
                    print(f"[WARN] S3 upload (plot) failed: {e}")

    @torch.no_grad()
    def _run_intermediate_upsampler_preview(
        self,
        epoch: int,
        batch: int,
        loss: float,
        steps: int = 50,
        n_img: int = 3
    ) -> None:
        """
        Generate and save upsampled images using DDIM sampling.
        
        Args:
            epoch: current epoch
            batch: current batch
            loss: current loss
            steps: number of DDIM steps
            n_img: number of images to generate
        """
        plt.close()
        fig, ax = plt.subplots(3, n_img, figsize=(n_img * 3, 9), sharex=True, sharey=True)
        
        # Build sampler
        sampler = UpsamplerDDIMSampler(
            noise_scheduler=self.noise_scheduler,
            num_inference_steps=steps,
            eta=0.0,
            device=self.device
        )
        
        self.ema_model.eval()
        
        for i in range(n_img):
            # Get random sample from dataset
            # Assumes dataset has method: get_random_sample() -> (high_res, low_res, clip_emb)
            if hasattr(self.dataloader.dataset, 'get_random_sample'):
                high_res_true, low_res, clip_emb = self.dataloader.dataset.get_random_sample()
            else:
                # Fallback: grab from first batch
                sample = next(iter(self.dataloader))
                high_res_true = sample['high_res'][0:1]
                low_res = sample['low_res'][0:1]
                clip_emb = sample['clip_embedding'][0:1]
            
            low_res = low_res.to(self.device)
            clip_emb = clip_emb.to(self.device)
            
            # Generate high-res image
            high_res_gen = sampler.sample(
                model=self.ema_model,
                z_img=clip_emb,
                low_res_img=low_res,
                image_size=(128, 128)
            )
            
            # Clamp and rescale [-1, 1] -> [0, 1]
            high_res_gen = high_res_gen.clamp(-1, 1)
            high_res_gen = (high_res_gen + 1) / 2
            high_res_gen_np = high_res_gen.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            # Process ground truth and low-res
            high_res_true_np = (high_res_true.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2
            low_res_np = (low_res.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2
            
            # Plot
            ax[0, i].imshow(low_res_np)
            ax[0, i].axis('off')
            ax[0, i].set_title('Low-res (64x64)')
            
            ax[1, i].imshow(high_res_true_np)
            ax[1, i].axis('off')
            ax[1, i].set_title('Ground Truth (128x128)')
            
            ax[2, i].imshow(high_res_gen_np)
            ax[2, i].axis('off')
            ax[2, i].set_title('Generated (128x128)')
        
        # Save figure
        output_dir = 'dalle2/checkpoints/upsampler_intermediate_outputs'
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f'epoch_{epoch}_batch_{batch}_output.png')
        
        plt.suptitle(f'Upsampler - Epoch: {epoch}, Batch: {batch}, Loss: {loss:.6f}')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        print(f"[UpsamplerTrainer] Saved preview: {save_path}")
        
        # Mirror to S3
        if self.on_aws:
            key = f"upsampler/intermediate_outputs/epoch_{epoch}_batch_{batch}_output.png"
            self._s3_put(save_path, key)
        
        self.train_module.train()

    def train(
        self,
        num_epochs: int,
        save_intermediate_output: int = 100,
        save_intermediate_model: int = 1000,
        resume_checkpoint_name: str = None
    ) -> None:
        """
        Train the upsampler.
        
        Args:
            num_epochs: number of training epochs
            save_intermediate_output: batch interval for saving previews
            save_intermediate_model: batch interval for saving checkpoints
            resume_checkpoint_name: optional checkpoint to resume from
        """
        self.train_module.train()
        self.total_steps = num_epochs * self.steps_per_epoch
        warmup_steps = int(self.warmup_frac * self.total_steps)
        
        # Resume from checkpoint if provided
        start_epoch = 0
        if resume_checkpoint_name is not None:
            checkpoint_path = os.path.join("dalle2", "checkpoints", "upsampler", resume_checkpoint_name)
            if os.path.isfile(checkpoint_path):
                print(f'[UpsamplerTrainer] Resuming from checkpoint: {checkpoint_path}')
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                if isinstance(checkpoint, dict) and "train_module" in checkpoint:
                    self.train_module.load_state_dict(checkpoint["train_module"])
                    self.ema_model.load_state_dict(checkpoint["ema_model"])
                    if "optimizer" in checkpoint:
                        self.optimizer.load_state_dict(checkpoint["optimizer"])
                    if self.scaler and checkpoint.get("scaler") is not None:
                        self.scaler.load_state_dict(checkpoint["scaler"])
                    start_epoch = checkpoint.get("epoch", 0)
                    print(f"[UpsamplerTrainer] Resumed at epoch {start_epoch}")
        
        # Training loop
        for epoch in range(start_epoch, num_epochs):
            print(f'\n[UpsamplerTrainer] Epoch {epoch + 1}/{num_epochs}')
            epoch_loss = 0.0
            
            for batch_i, batch in enumerate(tqdm(self.dataloader, desc='Training', leave=True)):
                # Prepare batch
                high_res = batch['high_res'].to(self.device)
                low_res = batch['low_res'].to(self.device)
                clip_emb = batch['clip_embedding'].to(self.device)
                
                # Training step
                loss, cos_eps = self.train_step(high_res, low_res, clip_emb)
                epoch_loss += loss
                
                # Learning rate schedule
                global_step = epoch * self.steps_per_epoch + batch_i
                if self.use_cosine_lr:
                    lr = cosine_lr(
                        step=global_step,
                        total_steps=self.total_steps,
                        lr_max=self.lr_max,
                        lr_min=self.lr_min,
                        warmup_steps=warmup_steps
                    )
                    for pg in self.optimizer.param_groups:
                        pg['lr'] = lr
                
                # Logging
                if batch_i % save_intermediate_output == 0:
                    print(f"[Epoch {epoch + 1}, Batch {batch_i + 1}] Loss: {loss:.6f}, Cos: {cos_eps:.4f}")
                
                # Log loss
                self._log_loss(epoch + 1, batch_i + 1, loss, plot_interval=save_intermediate_output)
                
                # Save intermediate preview
                if (batch_i + 1) % save_intermediate_output == 0:
                    self._run_intermediate_upsampler_preview(epoch + 1, batch_i + 1, loss, steps=50, n_img=3)
                
                # Save intermediate model
                if (batch_i + 1) % save_intermediate_model == 0:
                    self._save_current_model(epoch + 1, batch_i + 1)
                
                # Save on very good loss
                if loss < 1e-3:
                    fname = f"epoch{epoch+1}_batch{batch_i+1}_goodloss.pth"
                    print(f"[UpsamplerTrainer] Loss {loss:.6f} < 1e-3 - saving: {fname}")
                    self._save_current_model(epoch + 1, batch_i + 1)
                    self._run_intermediate_upsampler_preview(epoch + 1, batch_i + 1, loss, steps=50, n_img=3)
                
                self.global_step += 1
            
            avg_loss = epoch_loss / len(self.dataloader)
            print(f'[UpsamplerTrainer] Average Epoch Loss: {avg_loss:.4f}')
            
            # Save at end of epoch
            self._save_current_model(epoch + 1, len(self.dataloader))
        
        # Save final model
        self._save_final_model()
        print("[UpsamplerTrainer] Training complete!")


def main():
    """Example usage of UpsamplerTrainer."""
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize components
    upsampler = Upsampler(
        device=device,
        T=250,
        num_inference_steps=30,
        low_res_size=64,
        high_res_size=128
    )
    
    noise_scheduler = NoiseScheduler(T=250)
    
    optimizer = torch.optim.AdamW(
        upsampler.parameters(),
        lr=1e-4,
        weight_decay=0.01
    )
    
    dataset = BostonUpsamplerDataset(
        metadata_csv="/home/ec2-user/data/train_img/metadata.csv",
        images_dir="/home/ec2-user/data/train_img",
        precomputed_embeddings = "/home/ec2-user/aws_diffusion_model/dalle2/data/precomputed_embeddings.pth",
        device=device,
        noise_scheduler=noise_scheduler,
        lowres=64,
        highres=128,
        n_repeat=1,
    )

    trainer = UpsamplerTrainer(
        train_module=upsampler,
        optimizer=optimizer,
        noise_scheduler=noise_scheduler,
        dataset=dataset,
        batch_size=16,
        on_aws=True,         # or False
        use_amp=True
    )

    # Train
    trainer.train(
        num_epochs=100,
        save_intermediate_output=100,
        save_intermediate_model=1000
    )


if __name__ == "__main__":
    main()