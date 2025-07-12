"""
noise_scheduler: Defines noise schedules used by DDPM and DDIM models.

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
    - The noise in the reverse process is used differently:
        - Given:
            - p_{theta}: predicted/learned reverse distribution
            - mu_{theta}(x_t, t): predicted mean given a noisy image and its timestep, learned by a NN
            - Sigma_{theta}(x_t, t): predicted variance of an image given a noisy image and its timestep, learned by a NN
            - f_{theta}^{(t)}(x_t): estimate of a clean image at a given timestep, t (defined below)
            - p_{theta}^{(t)}(x_{t-1} | x_t): predicted next (less noisy) image given the current partially-noisy image
            - epsilon_{theta}^{(t)}(x_t): predicted noise at that timestep, t
        - DDPM:
            - p_{theta}(x_{t_1} | x_t) := N(x_{t-1}; mu_{theta}(x_t, t), Sigma_{theta}(x_t, t))
                - Interpretation: Noise is removed from an image by sampling from a controlled/scheduled Gaussian distribution, where the mean and variance are predicted by a neural network.
        - DDIM:
            - A := x_t - sqrt(1 - alpha_t) * epsilon_{theta}^{(t)}(x_t)
            - B := sqrt(alpha_t)

            - f_{theta}^{(t)}(x_t) := A / B
                - Interpretation: Noise is removed from an image determinstically, by subtracting the predicted/learned noise from the current noisy image.

            - p_{theta}^{(t)}(x_{t-1} | x_t) :=
                - N(f_{theta}^{(1)}(x_1), sigma_1^2 * I) if t=1
                - q_{sigma}(x_{t-1} | x_t, f_{theta}^{(t)}(x_t)) otherwise
            
                - Interpretation:
                    - If we are at the final denoising timestep, we add a small Gaussian centered at the prediction of the clean image.
                    - Otherwise, we use the clean image estimate to deterministically guide the reverse process.

Usage:
    from noise_scheduler import NoiseScheduler
    ns = NoiseScheduler()
    # get schedules at 6th timestep
    print(ns.get_beta(6))
    print(ns.get_alpha(6))
    print(ns.get_alpha_bar(6))

Classes:
    - NoiseScheduler: Precomputes all values of alpha, beta, and alpha_bar for fast retrieval during training.

References:
    - DDPM Paper: https://arxiv.org/pdf/2006.11239
    - DDIM Paper: https://arxiv.org/pdf/2010.02502
    - My DDPM Notes: https://github.com/spencer-karofsky/aws_diffusion_model/blob/main/dall-e-2/research_notes/DDPM%202020.pdf or /dall-e-2/research_notes/DDPM 2020.pdf
    - MY DDIM Notes: https://github.com/spencer-karofsky/aws_diffusion_model/blob/main/dall-e-2/research_notes/DDIM%202021.pdf or /dall-e-2/research_notes/DDIM 2021.pdf

Author:
    - Spencer Karofsky (https://github.com/spencer-karofsky)
"""
import torch

class NoiseScheduler:
    def __init__(
            self,
            T: int = 1000,
            beta_low: float = 1e-4,
            beta_high: float = .02
        ):
        """Initializes and computes all values for alpha, alpha_bar, and beta for fast retrieval
        Args:
            T: the number of forward/reverse timesteps
            beta_low: the beta value for t=0
            beta_high: the beta value for t=T
        """
        self.T = T

        # Define all betas
        self.betas = torch.linspace(beta_low, beta_high, T)

        # Define all alphas
        self.alphas = 1.0 - self.betas

        # Define all alpha_bars, using vectorized approach for speed
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
    
    def get_beta(self, t: torch.Tensor) -> torch.Tensor:
        """Looks-up beta value at timestep t
        Args:
            t: the timesteps
        Returns:
            beta_ts
        """
        return self.betas[t]
    
    def get_alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Looks-up alpha value at timestep t
        Args:
            t: the timesteps
        Returns:
            alpha_ts
        """
        return self.alphas[t]
    
    def get_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """Looks-up alpha_bar value at timestep t
        Args:
            t: the timesteps
        Returns:
            alpha_bar_ts
        """
        return self.alpha_bars[t]
