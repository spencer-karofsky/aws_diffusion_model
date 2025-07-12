"""
ddim_forward_diffusion: Defines the forward diffusion process described in the DDIM paper.

Description and Purpose:
    - The DDPM and DDIM model classes add and remove noise according to a noise schedule:
        - For the DDPM and DDIM, the authors chose a linear schedule over T time steps, increasing from beta=10^{-4} to 0.02.
    - Given beta, we can compute alpha, and alpha_bar:
        - alpha_t := 1 - beta_t
        - alpha_bar_t := alpha_1 * alpha_2 * ... * alpha_t
    - The forward process is used the same in both papers:
        - Given:
            - q: the true data distribution
            - x_t: an image with Gaussian noise
            - x_{t-1}: x_t's slightly less noisy image
        - q(x_t | x_{t-1}) := N(x_t; sqrt(1 - beta_t) * x_{t-1}, beta_t * I)
            - Interpretation: A noisy image is derived from adding controlled/scheduled Gaussian noise to it's less noisy image.

Usage:
    from ddim_forward_diffusion import ForwardDiffuser
    noiser = ForwardDiffuser(...)


Classes:
    - DDIM: Contains functions to sample from the q (forward distribution).

References:
    - DDPM Paper: https://arxiv.org/pdf/2006.11239
    - DDIM Paper: https://arxiv.org/pdf/2010.02502
    - My DDPM Notes: https://github.com/spencer-karofsky/aws_diffusion_model/blob/main/dall-e-2/research_notes/DDPM%202020.pdf or /dall-e-2/research_notes/DDPM 2020.pdf
    - MY DDIM Notes: https://github.com/spencer-karofsky/aws_diffusion_model/blob/main/dall-e-2/research_notes/DDIM%202021.pdf or /dall-e-2/research_notes/DDIM 2021.pdf

Author:
    - Spencer Karofsky (https://github.com/spencer-karofsky)
"""
import torch
import torch.nn as nn
#from dalle_core.unet.unet_decoder import UNetDenoiser
from dalle_core.diffusion.noise_scheduler import NoiseScheduler
import math

class ForwardDiffuser:
    def __init__(
            self,
            noise_scheduler: NoiseScheduler
        ):
        """Initialize the Forward Diffusion Process
        Args:
            noise_scheduler: a noise scheduler object
        """
        self.scheduler = noise_scheduler

    # The forward diffusion trick in the paper makes this function irrelevant
    # def _add_noise_step(
    #         self,
    #         x_t: torch.Tensor,
    #         ts: torch.Tensor
    #     ) -> torch.Tensor:
    #     """Adds Gaussian noise to an image over one time step according to a predefined noise schedule.
    #     - The papers write q(x_t | x_{t-1}). This function achieves the same effect, but my code implies q(x_{t+1} | x_t) for clearer naming in the code.
    #         - Otherwise, the variable names would be confusing.
    #     Args:
    #         x_t: the current image that we will add Gaussian noise to
    #         ts: the current time steps for each of the images (we are generating the image at x_{t+1})
    #     Returns:
    #         the more noisy image.
    #     """
    #     ts = ts.long()  # Ensure ts is of integer type for indexing
    #     beta_ts = self.scheduler.get_beta(ts).view(-1, 1, 1, 1)  # (B, 1, 1, 1)
    #     mean_t = torch.sqrt(1 - beta_ts) * x_t
    #     noise = torch.randn_like(x_t)
    #     noisy_image = mean_t + torch.sqrt(beta_ts) * noise
    #     return noisy_image
    
    def forward_process(
            self,
            x_0: torch.Tensor
        ) -> torch.Tensor:
        """Applies to full forward noising process
        Args:
            x_0: all of the original images this function corrupts with Gaussian noise
        Returns:
            x_T: the final images of pure noise
        """
        B = x_0.size(0)
        device = x_0.device

        # Sample random timesteps for each image in the batch
        ts = torch.randint(0, self.scheduler.T, (B,), device=device)

        # Get alpha_bar_t for each sampled timestep
        alpha_bars = self.scheduler.get_alpha_bar(ts).view(B, 1, 1, 1)

        # Sample Gaussian noise
        eps = torch.randn_like(x_0)

        # Compute x_t directly
        x_t = torch.sqrt(alpha_bars) * x_0 + torch.sqrt(1 - alpha_bars) * eps

        return x_t, ts, eps
        
