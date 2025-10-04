"""
upsampler_ddim_sampling.py: Implements DDIM sampling for the upsampler.

Description:
    * Modified DDIM sampler that conditions on low-resolution images
    * Used during inference to generate high-resolution images from low-resolution inputs
    * Similar to DecoderDDIMSampler but includes low_res_img in the forward pass

Classes:
    * UpsamplerDDIMSampler: DDIM sampling specifically for the upsampler module.

References:
    * DDIM Paper: https://arxiv.org/pdf/2010.02502
    * DALL·E 2 Paper: https://cdn.openai.com/papers/dall-e-2.pdf

Author:
    * Spencer Karofsky (https://github.com/spencer-karofsky)
"""
# PyTorch imports
import torch
import torch.nn as nn

# Other imports
from typing import Tuple


class UpsamplerDDIMSampler:
    def __init__(
            self,
            noise_scheduler,
            num_inference_steps: int = 30,
            eta: float = 0.0,
            device: torch.device = None,
            debug: bool = False
    ):
        """
        Initialize the DDIM sampler for upsampling.

        Args:
            noise_scheduler: NoiseScheduler instance for computing alphas, betas, etc.
            num_inference_steps: number of sampling steps (fewer = faster, but lower quality)
            eta: controls stochasticity (0.0 = deterministic DDIM, 1.0 = DDPM)
            device: PyTorch device
            debug: enable debug printing
        """
        self.noise_scheduler = noise_scheduler
        self.num_inference_steps = num_inference_steps
        self.eta = eta
        self.device = device
        self.debug = debug

        # Compute timestep schedule
        self.timesteps = self._get_timestep_schedule()

    def _get_timestep_schedule(self) -> torch.Tensor:
        """
        Generate uniformly spaced timesteps for DDIM sampling.
        
        Returns:
            Timesteps tensor of shape (num_inference_steps,)
        """
        T = self.noise_scheduler.T
        step_size = T // self.num_inference_steps
        timesteps = torch.arange(0, T, step_size, device=self.device)
        return timesteps.flip(0)  # Start from T and go to 0

    @torch.no_grad()
    def sample(
            self,
            model: nn.Module,
            z_img: torch.Tensor,
            low_res_img: torch.Tensor,
            image_size: Tuple[int, int] = (128, 128)
    ) -> torch.Tensor:
        """
        Generate high-resolution image using DDIM sampling.

        Args:
            model: the upsampler model
            z_img: CLIP image embedding, shape (B, 512)
            low_res_img: low-resolution conditioning image, shape (B, 3, 64, 64)
            image_size: target image size (H, W)

        Returns:
            Generated high-resolution image, shape (B, 3, H, W)
        """
        B = z_img.size(0)
        H, W = image_size

        if self.debug:
            print(f'[UpsamplerDDIMSampler] Starting sampling with {self.num_inference_steps} steps')
            print(f'[UpsamplerDDIMSampler] Image size: {image_size}')
            print(f'[UpsamplerDDIMSampler] Low-res image shape: {low_res_img.shape}')

        # Initialize with pure Gaussian noise
        x_t = torch.randn(B, 3, H, W, device=self.device)

        if self.debug:
            print(f'[UpsamplerDDIMSampler] Initial noise: mean={x_t.mean():.4f}, std={x_t.std():.4f}')

        # Reverse diffusion process
        for i, t in enumerate(self.timesteps):
            if self.debug and i % 5 == 0:
                print(f'[UpsamplerDDIMSampler] Step {i}/{self.num_inference_steps}, t={t.item()}')

            # Create batch of timesteps
            t_batch = torch.full((B,), t, device=self.device, dtype=torch.long)

            # Predict noise using the model
            eps_pred = model(
                x_t=x_t,
                low_res_img=low_res_img,
                z_img=z_img,
                t=t_batch
            )

            if self.debug and i % 5 == 0:
                print(f'[UpsamplerDDIMSampler] eps_pred: mean={eps_pred.mean():.4f}, std={eps_pred.std():.4f}')

            # Get next timestep
            t_next = self.timesteps[i + 1] if i < len(self.timesteps) - 1 else torch.tensor(0, device=self.device)

            # DDIM update step
            x_t = self._ddim_step(x_t, eps_pred, t, t_next)

            if self.debug and i % 5 == 0:
                print(f'[UpsamplerDDIMSampler] x_t after step: mean={x_t.mean():.4f}, std={x_t.std():.4f}')

        # Final denoised image
        x_0 = x_t

        if self.debug:
            print(f'[UpsamplerDDIMSampler] Final image: mean={x_0.mean():.4f}, std={x_0.std():.4f}')
            print(f'[UpsamplerDDIMSampler] Final image range: ({x_0.min():.4f}, {x_0.max():.4f})')

        return x_0

    def _ddim_step(
            self,
            x_t: torch.Tensor,
            eps_pred: torch.Tensor,
            t: torch.Tensor,
            t_next: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform one DDIM denoising step.

        Args:
            x_t: current noisy image, shape (B, 3, H, W)
            eps_pred: predicted noise, shape (B, 3, H, W)
            t: current timestep
            t_next: next timestep

        Returns:
            x_{t-1}: denoised image at next timestep, shape (B, 3, H, W)
        """
        # Get alpha values
        alpha_t = self.noise_scheduler.alpha_bar_t[t]
        alpha_t_next = self.noise_scheduler.alpha_bar_t[t_next]

        # Reshape alphas for broadcasting
        alpha_t = alpha_t.view(-1, 1, 1, 1)
        alpha_t_next = alpha_t_next.view(-1, 1, 1, 1)

        # Predict x_0 from x_t and eps_pred
        x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * eps_pred) / torch.sqrt(alpha_t)

        # Compute direction pointing to x_t
        sigma_t = self.eta * torch.sqrt((1 - alpha_t_next) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_next)

        # Compute x_{t-1}
        dir_xt = torch.sqrt(1 - alpha_t_next - sigma_t ** 2) * eps_pred

        # Add noise if eta > 0 and not at final step
        if self.eta > 0 and t_next > 0:
            noise = torch.randn_like(x_t)
            x_t_next = torch.sqrt(alpha_t_next) * x_0_pred + dir_xt + sigma_t * noise
        else:
            x_t_next = torch.sqrt(alpha_t_next) * x_0_pred + dir_xt

        return x_t_next