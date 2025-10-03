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
            T: int = 1000,
            schedule_type: str = 'cosine'
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
            schedule_type: linear or cosine
        """
        self.T = T
        # Initialize device (ideally using GPU)
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'
        self.debug = False

        # Compute beta_t
        if schedule_type == 'linear':
            BETA_START, BETA_END = 1e-4, .02

            self.beta_t = torch.linspace(BETA_START, BETA_END, T).to(self.device)
        elif schedule_type == 'cosine':
            self.beta_t = self._make_cosine_beta_schedule(T)

        # Compute alpha_t
        self.alpha_t = (1.0 - self.beta_t).to(self.device)

        # Compute alpha_bar_t
        self.alpha_bar_t = torch.cumprod(self.alpha_t, dim=0).to(self.device)

        print(f"[NoiseScheduler] T={T}, schedule={schedule_type}")
        print(f"[NoiseScheduler] beta_t range: [{self.beta_t.min():.6f}, {self.beta_t.max():.6f}]")
        print(f"[NoiseScheduler] alpha_bar_t range: [{self.alpha_bar_t.min():.6f}, {self.alpha_bar_t.max():.6f}]")
        print(f"[NoiseScheduler] alpha_bar_t[0]={self.alpha_bar_t[0]:.6f}, alpha_bar_t[-1]={self.alpha_bar_t[-1]:.6f}")

    def _make_cosine_beta_schedule(
            self,
            T: int,
            s: float = 0.008,
            final_alpha_bar: float = 1e-3
    ) -> torch.Tensor:
        """
        Computes cosine beta schedule as in DDIM/Stable Diffusion.
        """
        steps = torch.linspace(0, T, T + 1, device=self.device) / T
        alpha_bar = torch.cos(((steps + s) / (1 + s)) * torch.pi * 0.5) ** 2

        # Rescale to ensure alpha_bar starts at 1 and ends at final_alpha_bar
        alpha_bar = alpha_bar / alpha_bar[0]
        alpha_bar = alpha_bar * (1 - final_alpha_bar) + final_alpha_bar

        beta = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        return beta.clamp(min=1e-5, max=0.999)
    
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
        return self.beta_t.to(t.device)[t]
        # return self.beta_t[t].to(t.device)[t]
    
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
        return self.alpha_t.to(t.device)[t]
        # return self.alpha_t[t].to(t.device)[t]

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
        return self.alpha_bar_t.to(t.device)[t]
    
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
        alpha_bar = self.get_alpha_bar(t).to(device).view(-1, 1) # (B, 1)

        z_t = torch.sqrt(alpha_bar) * clean_input + torch.sqrt(1 - alpha_bar) * noise
        if self.debug:
            print(f'[NoiseScheduler] x_t shape: {z_t.shape}')
        return z_t, noise

    def get_eps_pred_from_x0(self, x0_pred, x_t, t):
        alpha_bar_t = self.get_alpha_bar(t)
        while len(alpha_bar_t.shape) < len(x_t.shape):
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)
        sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar_t)
        return (x_t - sqrt_alpha_bar * x0_pred) / sqrt_one_minus_alpha_bar
    
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward diffusion process:
            x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise

        Args:
            x0: the original clean input (B, C, H, W) or (B, D)
            t: timestep indices (B,)
            noise: sampled Gaussian noise (same shape as x0)

        Returns:
            x_t: the noised version of x0
        """
        alpha_bar_t = self.get_alpha_bar(t).view(-1, *[1] * (x0.ndim - 1))
        return (alpha_bar_t.sqrt() * x0) + ((1 - alpha_bar_t).sqrt() * noise)