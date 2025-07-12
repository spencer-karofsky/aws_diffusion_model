"""
ddim_sampler: Implements the DDIM sampling procedure from the paper "Denoising Diffusion Implicit Models"

Description:
    - DDIM enables efficient image generation using a non-Markovian deterministic or stochastic process.
    - This implementation supports both deterministic (η=0) and stochastic (η>0) sampling, giving the user full control over the generation process.
    - The model denoises from x_T to x_0 by repeatedly applying the learned reverse transitions using the U-Net and noise scheduler.

Usage:
    from ddim_sampler import DDIMSampler
    sampler = DDIMSampler(unet_model, scheduler)
    x_0 = sampler(x_T, t=T, deterministic=True)

Classes:
    - DDIMSampler: Performs a single denoising step (x_t → x_{t-1}) using the DDIM formulation.

References:
    - DDIM Paper: https://arxiv.org/pdf/2010.02502
    - My DDIM Notes: https://github.com/spencer-karofsky/aws_diffusion_model/blob/main/dall-e-2/research_notes/DDIM%202021.pdf or /dall-e-2/research_notes/DDIM 2021.pdf

Author:
    - Spencer Karofsky (https://github.com/spencer-karofsky)
"""
import torch
import torch.nn as nn
from dalle_core.unet.unet import UNetDenoiser
from dalle_core.diffusion_models.noise_scheduler import NoiseScheduler
import math

class DDIMSampler(nn.Module):
    def __init__(
            self,
            unet_model: UNetDenoiser,
            scheduler: NoiseScheduler,
        ):
        """Initialize a Denoising Diffusion Implicit Model
        - 
        Args:
            unet_model: the U-Net model that estimates the images
            scheduler: the noise scheduler
        """
        super().__init__()

        self.unet = unet_model
        self.scheduler = scheduler

    def forward(
            self,
            x_t: torch.Tensor,
            t: torch.Tensor,
            clip_embed: torch.Tensor,
            deterministic: bool = False
        ) -> torch.Tensor:
        """Denoise image over one timestep (goes from x_t -> x_{t-1})
        Args:
            x_t: the current image we seek to remove some noise from
            t: the time steps of the current image
            clip_embed: the CLIP image embedding
            deterministic: makes the inference process deterministic (same output) when enabled
        Returns:
            x_{t-1}, the more denoised image, at timestep t-1
        """
        ETA = 0.5

        # ----- Predict clean image (scaled) -----
        # Get values for alpha_t and alpha_{t - 1} from the noise scheduler
        alpha_t = self.scheduler.get_alpha(t).view(-1, 1, 1, 1)
        alpha_prev = self.scheduler.get_alpha(t - 1).view(-1, 1, 1, 1)

        # Get U-Net's current prediction of the noise for the current image
        #total_noise_pred = self.unet.forward(x_t, t)
        total_noise_pred = self.unet(x_t, t, clip_embed)
        total_noise_scaled = total_noise_pred * torch.sqrt(1 - alpha_t)

        # The clean image is the noise at a time step subtracted from the noisy image at that time step
        # We only subtract a fraction of the total predicted noise
        # After slightly denoising at each time steps, all noise will be removed
        clean_img_pred = (x_t - total_noise_scaled)

        # We divide by sqrt(alpha_t), because of the equation for x_hat_0, our prediction for the clean image
        clean_img_pred_scaled = clean_img_pred / torch.sqrt(alpha_t)

        # Because we later incorporate another component to x_{t-1}, we multiply by the sqrt of alpha_{t-1}
        clean_img_pred_scaled = clean_img_pred_scaled * torch.sqrt(alpha_prev)

        # If we set sigma_t to 0, the inference process becomes fully deterministic (the DDIM will generate the same image when the prompt is unchanged)
        if deterministic:
            sigma_t = torch.zeros_like(alpha_t)
        else:
            sigma_t = ETA * torch.sqrt((1 - alpha_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_prev)

        # Next, we add the direction pointing to x_t, the more noisy image
        # This addition encourages the denoised image to follow the true denoising trajectory, by adding some noise in the direction it came from
        # Intuition: "I know how to get to the bottom of the hill, but I can't just jump down – I have to follow a marked path (the true reverse distribution)"
        # Because we scaled the cleaned image by the sqrt of alpha_{t-1} and we scale random noise by sigma_t, we scale this by the sqrt of 1 - alpha_{t-1} - sigma_t^2
        term = (1 - alpha_prev - sigma_t ** 2).clamp(min=0.0)
        true_reverse_distribution_vector_scaled = total_noise_pred * torch.sqrt(term)

        # Lastly, we add random Gaussian noise, with mean=0 and standard variance
        random_noise = torch.randn_like(x_t)
        # We scale by sigma_t, so the three components' coefficients add to 1
        random_noise_scaled = random_noise * sigma_t

        # The denoised image is the sum of the three components
        denoised_image = clean_img_pred_scaled + true_reverse_distribution_vector_scaled + random_noise_scaled
        
        return denoised_image