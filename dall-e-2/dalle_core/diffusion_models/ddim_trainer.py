"""
trainer: Trains a DDIM according to the DDIM and DDPM papers.

Description:
    - Trains a Denoising Diffusion Implicit Model (DDIM).
        - U-Net is the model that powers the DDIM architecture.
        - The loss is the Mean Square Error (MSE) betwen the true noise and the U-Net's predicted noise.
    - Training Process:
        1. Sample random timesteps
            - Clarification: We do not train the model at all timesteps during training, just one per every image in the current batch.
                - Training all timesteps would be extremely inefficient/impossibly-slow without much added benefit.
                - However, if we train millions of (text, image)-pairs on one timestep each, every time step will be trained thousands of times in each epoch, while being 1000x faster.
        2. Sample random noise, epsilon = (N(0, I)).
            - In the forward noising process samples from more meaningful means and variances, but it adds some random noise.
        3. Compute noised image.
            - q(x_t | x_{t-1} := N(x_t; sqrt(1 - beta_t) * x_{t-1}, beta_t * I) 
            - Interpretation: We add noise to an image by sampling from a Gaussian distribution, where we use a noising schedule to control the amount of added noise.
            - Clarification: While the paper describes the noising process as if it's one image, this formula corresponds to the whole batch.
        4. Pass x_t and t into the U-Net model
            - Inputs: The noisy images, of shape [B, C, H, W], and the timesteps, [B]
            - Outputs: The prediction of the per-image noise (not the clean images) as the same shape as the noisy images input.
        5. Feed this predicted noise into the loss function.
            - Loss = MSE(epsilon, predicted noise)
                - epsilon is the actual noise we added to an otherwise deterministic process.
        6. Given the loss, perform backprop and update the weights.
            - Since we train using a U-Net, backprop is updating the filter weights and biases, attention weights, linear weights, etc..

Usage:
    from ddim_trainer import DDIMTrainer
    trainer = DDIMTrainer(...)
    trainer.train(...)

Classes:
    - DDIMTrainer: Trains the DDIM.

References:
    - DDPM Paper: https://arxiv.org/pdf/2006.11239
    - My DDPM Notes: https://github.com/spencer-karofsky/aws_diffusion_model/blob/main/dall-e-2/research_notes/DDPM%202020.pdf or /dall-e-2/research_notes/DDPM 2020.pdf
    - DDIM Paper: https://arxiv.org/pdf/2010.02502
    - My DDIM Notes: https://github.com/spencer-karofsky/aws_diffusion_model/blob/main/dall-e-2/research_notes/DDIM%202021.pdf or /dall-e-2/research_notes/DDIM 2021.pdf

Author:
    - Spencer Karofsky (https://github.com/spencer-karofsky)
"""
import torch
import torch.nn as nn
from dalle_core.unet.unet import UNetDenoiser
from dalle_core.diffusion_models.noise_scheduler import NoiseScheduler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

class DDIMTrainer:
    def __init__(
            self,
            unet_model: UNetDenoiser,
            noise_scheduler: NoiseScheduler,
            dataset: Dataset,
            optimizer: torch.optim.Optimizer,
            batch_size: int = 32,
            epochs: int = 10
        ):
        """Initializes the DDIM training model
        Args:
            unet_model: the U-Net model we will use for training
            noise_scheduler: the noise scheduler
            dataset: the dataset the DDIM will train on
            optimizer: the PyTorch optimizer 
            batch_size: the batch size
            epochs: number of epochs
        """
        self.unet = unet_model
        self.scheduler = noise_scheduler

        # Initialize device
        # Ideally use CUDA (when using AWS), then MPS if CUDA is unavaliable, otherwise default to the CPU
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        
        self.optimizer = optimizer

        self.unet.to(self.device)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        self.num_epochs = epochs
    
    def _compute_loss(
            self,
            x_0: torch.Tensor,
            text_embed: torch.Tensor
        ) -> torch.Tensor:
        """Computes the loss of the DDIM model conditioned on the text
        Args:
            x_0: clean images
            text_embed: the CLIP-embedded text
        Returns:
            Loss between predicted noise and true noise
        """
        B = x_0.size(0)
        T = self.scheduler.T

        # Move input to correct device (for faster PyTorch GPU training if a GPU is available)
        x_0 = x_0.to(self.device)

        # Sample random timesteps for each image in the batch
        t = torch.randint(1, T, (B,), device=self.device)

        # Sample noise from N(0, I)
        epsilon = torch.randn_like(x_0)

        # Get alpha_bar_t for each t
        # Using alpha_bar is a tick that speeds up the forward noising process, but it is mathematically the same.
        alpha_bar_t = self.scheduler.alpha_bars[t].view(B, 1, 1, 1)  # Reshape for broadcasting

        # Apply forward process (generate x_t from x_0 and epsilon)
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * epsilon

        # Predict the noise using the U-Net
        epsilon_pred = self.unet.forward(x_t, t, text_embed)

        # Return mean squared error (MSE) loss between the predicted and actual noise
        return nn.functional.mse_loss(epsilon_pred, epsilon)
    
    def train(self, aws_config: bool = False) -> None:
        """Train the UNet (core model of the DDIM architecture)
        Args:
            aws_config: when enabled, it trains through AWS functionalities, not locally. TODO
        """
        if aws_config:
            raise NotImplementedError('AWS training pipeline not yet implemented.')

        self.unet.train()
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            pbar = tqdm(self.dataloader, desc=f'Epoch {epoch+1}/{self.num_epochs}')

            for batch in pbar:
                batch_crops, batch_text = batch # batch = [(image (CLIP-embedded), text (CLIP-embedded)), ...]
                # Turns the list of images into a tensor (that PyTorch expects) and enables GPU-acceleration if enabled
                x_0 = torch.cat(batch_crops, dim=0).to(self.device)
                
                # Compute the loss
                loss = self._compute_loss(x_0, batch_text)

                # Reset the weights, perform backprop, and update weights
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix(loss=total_loss / (pbar.n + 1))

            print(f'[Epoch {epoch+1}] Avg Loss: {total_loss / len(self.dataloader):.4f}')
