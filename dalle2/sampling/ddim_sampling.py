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
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dalle2.sampling.noise_scheduler import NoiseScheduler

def _make_timesteps(total_steps: int, num_inference_steps: int, device: torch.device) -> torch.Tensor:
    """
    Return exactly num_inference_steps unique, strictly descending timesteps starting at total_steps-1 and ending at 0.
    """
    if num_inference_steps < 2:
        raise ValueError('num_inference_steps must be >= 2 so that we include both (T-1) and 0')

    ts = torch.linspace(total_steps - 1, 0, num_inference_steps, device=device)
    ts = torch.round(ts).long()

    # Remove duplicates caused by rounding and force strict descending order
    ts = torch.flip(torch.unique(torch.flip(ts, dims=[0])), dims=[0])

    # Always make sure t=0 is present as the last element
    if ts[-1] != 0:
        ts = torch.cat([ts, ts.new_tensor([0])])

    # If we now have more than requested due to the duplicate removal + t0 addition, drop the earliest redundant timesteps (those at the highest noise end).
    if ts.numel() > num_inference_steps:
        ts = ts[:num_inference_steps]

    # print(f"[DEBUG] Timesteps for sampling: {ts}")
    # print(f"[DEBUG] Number of unique timesteps: {ts.numel()}")

    assert ts.numel() == num_inference_steps, "Timesteps calculation failed to produce the required length"
    return ts


def _extract_alpha_bar(
        scheduler: NoiseScheduler,
        t: torch.Tensor,
        target_ndim: int
) -> torch.Tensor:
    """
    Return alpha_bar_t with shape broadcastable to target_ndim
    """
    alpha_bar = scheduler.get_alpha_bar(t)
    while alpha_bar.ndim < target_ndim:
        alpha_bar = alpha_bar.unsqueeze(-1)
    return alpha_bar

class PriorDDIMSampler:
    def __init__(
        self,
        noise_scheduler: NoiseScheduler,
        num_inference_steps: int,
        device = 'mps',
        eta: float = 0.0,
    ) -> None:
        self.scheduler = noise_scheduler
        T_total = len(self.scheduler.alpha_bar_t)

        ab = self.scheduler.alpha_bar_t

        self.num_inference_steps = num_inference_steps
        self.eta = float(eta)
        self.timesteps = _make_timesteps(T_total, num_inference_steps, ab.device)

    def _predict_x0(
            self,
            x_t: torch.Tensor,
            eps: torch.Tensor,
            t: torch.Tensor
    ) -> torch.Tensor:
        alpha_bar = _extract_alpha_bar(self.scheduler, t, x_t.ndim)
        return (x_t - torch.sqrt(1 - alpha_bar) * eps) / torch.sqrt(alpha_bar)

    def _ddim_step(
            self,
            x0: torch.Tensor,
            eps: torch.Tensor,
            t_cur: torch.Tensor,
            t_next: torch.Tensor
    ) -> torch.Tensor:
        alpha_bar_cur = _extract_alpha_bar(self.scheduler, t_cur, x0.ndim)
        alpha_bar_next = _extract_alpha_bar(self.scheduler, t_next, x0.ndim)

        if self.eta == 0.0:
            sigma = 0.0
            noise = 0.0
        else:
            sigma = self.eta * torch.sqrt((1 - alpha_bar_next) / (1 - alpha_bar_cur) * (1 - alpha_bar_cur / alpha_bar_next))
            noise = torch.randn_like(eps)

        return (
            torch.sqrt(alpha_bar_next) * x0 +
            torch.sqrt(1 - alpha_bar_next - sigma ** 2) * eps +
            sigma * noise
         )
    
    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        z_txt: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate a CLIP image embedding from a CLIP text embedding using the learned prior.
        """
        model.eval()

        B = z_txt.size(0)
        device = z_txt.device
        x_t = torch.randn(B, 512, device=device)

        for i, t_i in enumerate(self.timesteps):
            t = torch.full((B,), t_i, dtype=torch.long, device=device)
            eps = model(z_txt=z_txt, t=t, z_T=x_t)  # ε_θ
            x0 = self._predict_x0(x_t, eps, t)

            if i == len(self.timesteps) - 1:
                return F.normalize(x0, dim=-1) # already in embedding space, no range clamp necessary

            t_next = torch.full((B,), self.timesteps[i + 1], dtype=torch.long, device=device)
            x_t = self._ddim_step(x0, eps, t, t_next)

class DecoderDDIMSampler:
    def __init__(
        self,
        noise_scheduler: NoiseScheduler,
        num_inference_steps: int,
        eta: float = 0.0,
        guidance_scale: float = 1.0,
        renorm_each_step: bool = False,
    ) -> None:
        self.scheduler = noise_scheduler
        self.num_inference_steps = num_inference_steps
        self.eta = float(eta)
        self.renorm_each_step = renorm_each_step
        self.timesteps = _make_timesteps(len(noise_scheduler.alpha_bar_t), num_inference_steps, noise_scheduler.alpha_bar_t.device)
        self.guidance_scale = guidance_scale

    def _predict_x0(
            self,
            x_t: torch.Tensor,
            eps: torch.Tensor,
            t: torch.Tensor
    ) -> torch.Tensor:
        alpha_bar = _extract_alpha_bar(self.scheduler, t, x_t.ndim)
        return (x_t - torch.sqrt(1 - alpha_bar) * eps) / torch.sqrt(alpha_bar)

    def _ddim_step(
            self,
            x0: torch.Tensor,
            eps: torch.Tensor,
            t_cur: torch.Tensor,
            t_next: torch.Tensor
        ) -> torch.Tensor:
        alpha_bar_cur = _extract_alpha_bar(self.scheduler, t_cur, x0.ndim)
        alpha_bar_next = _extract_alpha_bar(self.scheduler, t_next, x0.ndim)

        if self.eta == 0.0:
            sigma = 0.0
            noise = 0.0
        else:
            sigma = self.eta * torch.sqrt((1 - alpha_bar_next) / (1 - alpha_bar_cur) * (1 - alpha_bar_cur / alpha_bar_next))
            noise = torch.randn_like(eps)

        x_prev = (
            torch.sqrt(alpha_bar_next) * x0 +
            torch.sqrt(1 - alpha_bar_next - sigma ** 2) * eps +
            sigma * noise
        )
        return x_prev

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        z_img: torch.Tensor,
        image_size: Tuple[int, int] = (64, 64),
    ) -> torch.Tensor:
        B, device = z_img.size(0), z_img.device
        H, W = image_size
        x_t = torch.randn(B, 3, H, W, device=device)

        for i, t_i in enumerate(self.timesteps):
            t = torch.full((B,), t_i, dtype=torch.long, device=device)

            # Only use guidance if scale != 1.0 AND model was trained for it
            if self.guidance_scale == 1.0:
                eps = model(x_t=x_t, z_img=z_img, t=t)
            else:
                eps_cond = model(x_t=x_t, z_img=z_img, t=t)
                eps_uncond = model(x_t=x_t, z_img=torch.zeros_like(z_img), t=t)
                eps = eps_uncond + self.guidance_scale * (eps_cond - eps_uncond)
            
            x0 = self._predict_x0(x_t, eps, t)

            if i == len(self.timesteps) - 1:
                return x0.clamp(-1, 1)

            t_next = torch.full((B,), self.timesteps[i + 1], dtype=torch.long, device=device)
            x_t = self._ddim_step(x0, eps, t, t_next)