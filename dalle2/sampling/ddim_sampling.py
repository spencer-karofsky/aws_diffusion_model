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
        alpha_bar_t = self.scheduler.get_alpha_bar(t)
        while len(alpha_bar_t.shape) < len(x_t.shape):
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)
        sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar_t)
        return (x_t - sqrt_one_minus_alpha_bar * eps_pred) / sqrt_alpha_bar
    
    def _ddim_step(
            self,
            x_t: torch.Tensor,
            x0_pred: torch.Tensor,
            eps_pred: torch.Tensor,
            t_current: torch.Tensor,
            t_next: torch.Tensor,
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
        # Check for NaNs
        if torch.isnan(t_current).any():
            raise Exception('[DDIMSampler._ddim_step] NaNs in t')
        if torch.isnan(t_next).any():
            raise Exception('[DDIMSampler._ddim_step] NaNs in t_prev')
        
        # Get alpha, alpha-bar values
        alpha_bar_prev = self.scheduler.get_alpha_bar(t_current)
        alpha_bar_t = self.scheduler.get_alpha_bar(t_next)

        # Check for NaNs
        if torch.isnan(alpha_bar_t).any():
            raise Exception('[DDIMSampler._ddim_step] NaNs in alpha_bar_t. Check NoiseScheduler.get_alpha_bar')
        if torch.isnan(alpha_bar_prev).any():
            raise Exception('[DDIMSampler._ddim_step] NaNs in alpha_bar_prev. Check NoiseScheduler.get_alpha_bar')

        while len(alpha_bar_t.shape) < len(x_t.shape):
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)
            alpha_bar_prev = alpha_bar_prev.unsqueeze(-1)

        # Compute sigma
        sigma = self.eta * torch.sqrt(
            (1 - alpha_bar_t) / (1 - alpha_bar_prev) * (1 - alpha_bar_prev / alpha_bar_t)
        )
        # Check for NaNs
        if torch.isnan(sigma).any():
            raise Exception('[DDIMSampler._ddim_step] NaNs in sigma. Check for division by 0, division by near-0, or sqrt of negative value(s)')
        
        noise = torch.randn_like(x_t) if self.eta > 0 else torch.zeros_like(x_t)
        # Check for NaNs
        if torch.isnan(noise).any():
            raise Exception('[DDIMSampler._ddim_step] NaNs in noise')

        x_prev = (
            torch.sqrt(alpha_bar_prev) * x0_pred +
            torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps_pred +
            sigma * noise
        )
        # Check for NaNs
        if torch.isnan(x_prev).any():
            raise Exception('[DDIMSampler._ddim_step] NaNs in x_prev')
        
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
            z_cond: conditioning vector, of shape (B, 512)
            shape: Shape of output (B, C, H, W)
            steps: number of inference steps

        Returns:
            final denoised sample x_0, of shape (B, C, H, W)
        """
        steps = steps or self.num_inference_steps
        timesteps = torch.linspace(
            len(self.scheduler.alpha_bar_t) - 1,
            0,
            steps,
            dtype=torch.long,
            device=z_cond.device
        )

        B = shape[0]
        x_t = torch.randn(shape, device=z_cond.device)

        for i, t_i in enumerate(timesteps):
            t = torch.full((B,), t_i.item(), device=z_cond.device, dtype=torch.long)

            if hasattr(model, 'debug') and model.debug:
                print(f'[DDIMSampler.sample] Step {i}/{steps} — t = {t_i.item()}')

            if model.__class__.__name__ == 'Prior':
                eps_pred = model(
                    z_txt=z_cond,
                    t=t,
                    z_T=x_t
                )
            elif model.__class__.__name__ == 'Decoder':
                eps_pred = model(
                    x_t=x_t,
                    z_img=z_cond,
                    t=t,   
                )

            x0_pred = self._predict_x0(x_t, eps_pred, t)

            if i == steps - 1:
                return x0_pred

            t_next = torch.full((B,), timesteps[i + 1].item(), device=z_cond.device, dtype=torch.long)
            x_t = self._ddim_step(x_t, x0_pred, eps_pred, t, t_next)

        return x_t