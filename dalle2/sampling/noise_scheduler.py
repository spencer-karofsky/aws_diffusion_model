"""
noise_scheduler.py: Defines the noise schedule (betas, alphas, alpha-bars) for forward and reverse diffusion processes in DDPM/DDIM.

Description:
    * Precomputes and stores the noise schedule coefficients required for the diffusion process.
    * These include:
        - beta_t: noise variance at each step
        - alpha_t: noise retention at each step
        - alpha_bar_t: cumulative product of alphas, used for both sampling and denoising

Classes:
    * NoiseScheduler: Computes noise coefficients for the diffusion process.

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

class NoiseScheduler:
    def __init__(
            self,
            T: int = 1000
    ):
        """
        Pre-computes beta, alpha, and alpha_bar.

        Example Usage:
            from noise_scheduler import NoiseScheduler
            scheduler = NoiseScheduler(1000)
            t = torch.randint(0, 1000, (32,))
            scheduler.get_alphas(t)
        
        Args:
            T: the number of timesteps
        """
        # Compute beta_t
        BETA_START, BETA_END = 1e-4, .02
        self.beta_t = torch.linspace(BETA_START, BETA_END, T)

        # Compute alpha_t
        self.alpha_t = 1.0 - self.beta_t

        # Compute alpha_bar_t
        self.alpha_bar_t = torch.cumprod(self.alpha_t, dim=0)
    
    def get_beta(
            self,
            t: torch.Tensor
    
    ) -> torch.Tensor:
        """
        Retrieves beta values for the given time steps.

        Args:
            t: tensor of shape (B) containing time indices

        Returns:
            tensor of shape (B) with beta values.
        """
        return self.beta_t[t]
    
    def get_alpha(
            self,
            t: torch.Tensor
    
    ) -> torch.Tensor:
        """
        Retrieves alpha values for the given time steps.

        Args:
            t: tensor of shape (B) containing time indices

        Returns:
            tensor of shape (B) with alpha values.
        """
        return self.alpha_t[t]

    def get_alpha_bar(
            self,
            t: torch.Tensor
    
    ) -> torch.Tensor:
        """
        Retrieves alpha-bar values for the given time steps.

        Args:
            t: tensor of shape (B) containing time indices

        Returns:
            tensor of shape (B) with alpha-bar values.
        """
        return self.alpha_bar_t[t]
    
    def add_noise(
            self,
            clean_input: torch.Tensor,
            t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Adds noise to clean_input at timestep t, returning noised input and the noise used.

        Args:
            clean_input: Original (clean) data, e.g. CLIP image embeddings, shape (B, D)
            t: Timestep indices, shape (B,)

        Returns:
            z_t: Noised input, shape (B, D)
            noise: The Gaussian noise that was added, shape (B, D)
        """
        device = clean_input.device
        noise = torch.randn_like(clean_input)
        alpha_bar = self.get_alpha_bar(t).to(device).view(-1, 1)  # (B, 1)

        z_t = torch.sqrt(alpha_bar) * clean_input + torch.sqrt(1 - alpha_bar) * noise
        return z_t, noise
