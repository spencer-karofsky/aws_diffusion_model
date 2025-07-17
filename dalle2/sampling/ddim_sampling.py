"""
ddim_sampling.py: Defines the DDIM-based sampling process used to iteratively denoise latent or image tensors during inference.

Description:
    * Performs image generation by iteratively denoising pure Gaussian noise into a final output.
    * This generation process can be deterministic or stochastic.
        - When eta is set to 0, this process is fully-deterministic (the same prompt leads to the same output).
        - Values of eta greater than 0 lead to more generation diversity.
        - For experimentation, try different values of eta and observe generation quality as eta increases.

Classes:
    * DDIMSampler: Performs reverse denoising using DDIM to generate samples from noise.

References:
    * DDPM Paper: https://arxiv.org/pdf/2006.11239
    * DDIM Paper: https://arxiv.org/pdf/2006.11239
    * My DDPM Notes: https://github.com/spencer-karofsky/aws_diffusion_model/blob/main/dalle2/research_notes/DDPM%202020.pdf or /dalle2/research_notes/DDPM 2020.pdf
    * My DDIM Notes: https://github.com/spencer-karofsky/aws_diffusion_model/blob/main/dalle2/research_notes/DDIM%202021.pdf or /dalle2/research_notes/DDIM 2021.pdf
    
Author:
    * Spencer Karofsky (https://github.com/spencer-karofsky)
"""
# PyTorch imports
import torch
import torch.nn as nn

# Module imports
from dalle2.sampling.noise_scheduler import NoiseScheduler

# Other imports
from typing import Tuple

class DDIMSampler:
    def __init__(
            self,
            noise_scheduler: NoiseScheduler,
            num_inference_steps: int,
            eta: float = 0.0
    ):
        """
        Initializes DDIM Sampler, which takes in pure noise and generates actual samples given a fixed number of steps.

        Example Usage:
            from noise_scheduler import NoiseScheduler
            from ddim_sampling import DDIMSampler

            scheduler = NoiseScheduler(1000)
            sampler = DDIMSampler(noise_scheduler, 50, 0.1)

            model = UNetModel(...)
            model.load_state_dict(torch.load("path_to_model.pt"))
            model.eval().cuda()

            z_cond = clip_model.encode_text(["a cat wearing sunglasses"]).cuda() # [B, D]
            samples = sampler.sample(model=model, z_cond=z_cond, shape=(1, 3, 64, 64))
        
        Args:
            noise_scheduler: contains beta, alpha, and alpha-bar schedules
            num_inference_steps: number of steps to use in DDIM sampling (e.g., 50)
            eta: amount of noise to inject at each step (0.0 = deterministic DDIM, >0 = stochastic variant)
        """
        self.scheduler = noise_scheduler
        self.num_inference_steps = num_inference_steps
        self.eta = eta

        self.timesteps = self._get_ddim_timesteps()

    def _get_ddim_timesteps(self) -> torch.Tensor:
        """
        Returns a tensor of T inference timesteps (evenly spaced over T total timesteps)
        """
        total_steps = len(self.scheduler.alpha_bar_t)
        return torch.linspace(total_steps - 1, 0, self.num_inference_steps, dtype=torch.long)
    
    def _predict_x0(
                self,
                x_t: torch.Tensor,
                eps_pred: torch.Tensor,
                t: torch.Tensor
    ) -> torch.Tensor:
        """
        Predicts x_0 from the noisy sample x_t and the predicted noise eps_t.

        Args:
            x_t: current noisy sample, of shape (B, C, H, W)
            eps_pred: predicted noise from model, of shape (B, C, H, W)
            t: current timestep indices, of shape (B)

        Returns:
            reconstructed clean sample x_0, of shape (B, C, H, W)
        """
        alpha_bar_t = self.scheduler.get_alpha_bar(t).view(-1, 1, 1, 1)
        sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar_t)
        return (x_t - sqrt_one_minus_alpha_bar * eps_pred) / sqrt_alpha_bar
    
    def _ddim_step(
            self,
            x_t: torch.Tensor,
            x0_pred: torch.Tensor,
            eps_pred: torch.Tensor,
            t: torch.Tensor,
            t_prev: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes x_{t-1} using the DDIM update rule.

        Args:
            x_t: Current noisy sample, of shape (B, C, H, W)
            x0_pred: Predicted clean sample, of shape (B, C, H, W)
            eps_pred: Predicted noise, of shape (B, C, H, W)
            t: Current timestep, of shape (B)
            t_prev: previous timestep, of shape (B)

        Returns:
            x_{t-1} sample, of shape (B, C, H, W)
        """
        alpha_bar_t     = self.scheduler.get_alpha_bar(t).view(-1, 1, 1, 1)
        alpha_bar_prev  = self.scheduler.get_alpha_bar(t_prev).view(-1, 1, 1, 1)

        sigma = self.eta * torch.sqrt(
            (1 - alpha_bar_t) / (1 - alpha_bar_prev) * (1 - alpha_bar_prev / alpha_bar_t)
        )

        noise = torch.randn_like(x_t) if self.eta > 0 else torch.zeros_like(x_t)

        x_prev = (
            torch.sqrt(alpha_bar_prev) * x0_pred +
            torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps_pred +
            sigma * noise
        )
        return x_prev
    
    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        z_cond: torch.Tensor,
        shape: Tuple[int, int, int, int],
        steps: int = None
    ) -> torch.Tensor:
        """
        Performs DDIM sampling using the given model and conditioning.

        Args:
            model: the trained denoising model
            z_cond: conditioning vector, of shape (B, D)
            shape: Shape of output (B, C, H, W)
            steps: number of inference steps

        Returns:
            final denoised sample x_0, of shape (B, C, H, W)
        """
        effective_steps = steps if steps is not None else self.num_inference_steps
        timesteps = torch.linspace(
            len(self.scheduler.alpha_bar_t) - 1, 0, effective_steps, dtype=torch.long, device=z_cond.device
        )

        device = z_cond.device
        B = shape[0]
        x_t = torch.randn(shape, device=device)

        for i, t_i in enumerate(timesteps):
            t = torch.full((B,), t_i.item(), device=device, dtype=torch.long)

            eps_pred = model(x_t, t, z_cond)
            x0_pred = self._predict_x0(x_t, eps_pred, t)

            if i == len(timesteps) - 1:
                return x0_pred

            t_prev = torch.full((B,), timesteps[i + 1].item(), device=device, dtype=torch.long)
            x_t = self._ddim_step(x_t, x0_pred, eps_pred, t, t_prev)

        return x_t  # Should never hit this unless num_inference_steps == 1