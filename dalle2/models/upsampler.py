"""
upsampler.py: Implements the upsampler used by DALL·E 2.

Description:
    * The upsampler learns to upsample low-resolution images (64x64) to high-resolution images (128x128).
    * Similar to the decoder, it predicts noise conditioned on:
        - Low-resolution image (bilinearly upsampled to target resolution)
        - CLIP image embedding
        - Timestep embedding
    * Training:
        - Takes a clean high-res image, adds noise via DDPM forward process
        - Concatenates noisy high-res image with low-res image (upsampled to match spatial dims)
        - U-Net predicts the noise
        - Loss: MSE between expected and predicted noise
    * Inference:
        - Takes generated 64x64 image from decoder
        - Uses DDIM sampling to generate 128x128 version
        - Conditioned on the same CLIP embedding used in decoder

Classes:
    * Upsampler(nn.Module): A modular DALL·E 2 upsampler.

References:
    * DALL·E 2 Paper: https://cdn.openai.com/papers/dall-e-2.pdf
    * Cascaded Diffusion Models Paper: https://arxiv.org/abs/2106.15282

Author:
    * Spencer Karofsky (https://github.com/spencer-karofsky)
"""
# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Module imports
from dalle2.models.upsampler_unet import UpsamplerUNet
from dalle2.models.timestep_embedding import TimestepEmbedder
from dalle2.sampling.ddim_sampling import DecoderDDIMSampler


class Upsampler(nn.Module):
    def __init__(
            self,
            device: torch.device,
            T: int = 1000,
            num_inference_steps: int = 30,
            low_res_size: int = 64,
            high_res_size: int = 128,
            on_aws: bool = False,
            debug: bool = False
    ):
        """
        Initialize the upsampler.

        Example Usage:
            from upsampler import Upsampler

            upsampler_model = Upsampler(...)
            high_res_pred = upsampler_model.forward(...)  # Training
            high_res_pred = upsampler_model.sample(...)   # Inference
        
        Args:
            device: the PyTorch device (CUDA, Metal (MPS), or CPU)
            T: the total number of noising/denoising timesteps
            num_inference_steps: the number of inference steps for DDIM sampling
            low_res_size: input low-resolution image size (64)
            high_res_size: output high-resolution image size (128)
            on_aws: configures for AWS (TODO Configure script for AWS)
            debug: outputs relevant information (useful for debugging)
        """
        super().__init__()

        # Save params
        self.device = device
        self.T = T
        self.low_res_size = low_res_size
        self.high_res_size = high_res_size
        self.debug = debug

        # Timestep embedding network
        self.TIMESTEP_DIM = 512
        self.timestep_embedder = TimestepEmbedder(
            dim=self.TIMESTEP_DIM,
            module='decoder',
            T=self.T
        )

        # Define U-Net hyperparameters and instantiate the U-Net
        self.IN_CHANNELS = 3  # RGB channels
        self.LOW_RES_CHANNELS = 3  # Low-res image channels (concatenated with noisy high-res)
        self.COND_EMB_DIM = 512  # CLIP embedding dimension
        self.BASE_CHANNELS = 64
        self.CHANNEL_MULTS = (1, 2, 4, 8)
        self.RESIDUAL_BLOCKS = 2
        self.ATTENTION_RESOLUTIONS = (32, 16)  # Apply attention at multiple scales

        self.upsampler_unet = UpsamplerUNet(
            channel_multipliers=self.CHANNEL_MULTS,
            attention_resolutions=self.ATTENTION_RESOLUTIONS,
            image_size=self.high_res_size,
            in_channels=self.IN_CHANNELS + self.LOW_RES_CHANNELS,  # Concatenate noisy + low-res
            out_channels=self.IN_CHANNELS,
            conditional_embedding_dim=self.COND_EMB_DIM,
            base_channels=self.BASE_CHANNELS,
            residual_blocks=self.RESIDUAL_BLOCKS,
            device=self.device,
            debug=self.debug
        )

        self.sampler = None

        # AWS configuration
        if on_aws:
            raise NotImplementedError

    def _prepare_low_res_input(self, low_res_img: torch.Tensor) -> torch.Tensor:
        """
        Bilinearly upsample low-res image to match high-res spatial dimensions.
        
        Args:
            low_res_img: low-resolution image of shape (B, 3, low_res_size, low_res_size)
            
        Returns:
            Upsampled image of shape (B, 3, high_res_size, high_res_size)
        """
        return F.interpolate(
            low_res_img,
            size=(self.high_res_size, self.high_res_size),
            mode='bilinear',
            align_corners=False
        )

    def forward(
            self,
            x_t: torch.Tensor,
            low_res_img: torch.Tensor,
            z_img: torch.Tensor,
            t: torch.Tensor
    ) -> torch.Tensor:
        """
        Defines one step of the forward pass of the upsampler.
        
        Args:
            x_t: noisy high-res images at timestep t, shape (B, 3, high_res_size, high_res_size)
            low_res_img: low-resolution conditioning image, shape (B, 3, low_res_size, low_res_size)
            z_img: clean CLIP image embedding, shape (B, 512)
            t: timesteps, shape (B,)
            
        Returns:
            Predicted noise (epsilon), eps_theta, shape (B, 3, high_res_size, high_res_size)
        """
        B = x_t.size(0)

        assert x_t.dim() == 4, f'[Upsampler] x_t must be 4D (B, 3, H, W), got {x_t.shape}'
        assert x_t.shape[2:] == (self.high_res_size, self.high_res_size), \
            f'[Upsampler] x_t spatial dims must be ({self.high_res_size}, {self.high_res_size}), got {x_t.shape[2:]}'
        assert low_res_img.shape == (B, 3, self.low_res_size, self.low_res_size), \
            f'[Upsampler] low_res_img must have shape ({B}, 3, {self.low_res_size}, {self.low_res_size}), got {low_res_img.shape}'
        assert z_img.shape == (B, 512), f'[Upsampler] z_img must have shape ({B}, 512), got {z_img.shape}'
        assert t.shape == (B,), f'[Upsampler] t must have shape ({B},), got {t.shape}'

        if self.debug:
            print(f'[Upsampler] x_t shape: {x_t.shape}')
            print(f'[Upsampler] low_res_img shape: {low_res_img.shape}')
            print(f'[Upsampler] z_img shape: {z_img.shape}')
            print(f'[Upsampler] t shape: {t.shape}')
            print(f'[Upsampler] CLIP z_img norm: {z_img.norm(dim=-1).mean():.4f}')

        # Upsample low-res image to match high-res spatial dimensions
        low_res_upsampled = self._prepare_low_res_input(low_res_img)
        
        if self.debug:
            print(f'[Upsampler] low_res_upsampled shape: {low_res_upsampled.shape}')

        # Concatenate noisy high-res with upsampled low-res
        x_concat = torch.cat([x_t, low_res_upsampled], dim=1)
        
        if self.debug:
            print(f'[Upsampler] x_concat shape: {x_concat.shape} (expected: ({B}, 6, {self.high_res_size}, {self.high_res_size}))')

        # Generate timestep embeddings
        t_emb = self.timestep_embedder(t)
        
        if self.debug:
            print(f'[Upsampler] t_emb shape: {t_emb.shape}, mean: {t_emb.mean():.4f}, std: {t_emb.std():.4f}')

        # Forward through U-Net
        eps_pred = self.upsampler_unet(
            x_t=x_concat,
            z_img=z_img,
            t_emb=t_emb
        )

        if self.debug:
            print(f'[Upsampler] eps_pred shape: {eps_pred.shape}, mean: {eps_pred.mean():.4f}, std: {eps_pred.std():.4f}')

        return eps_pred

    @torch.no_grad()
    def sample(
            self,
            low_res_img: torch.Tensor,
            z_img: torch.Tensor,
            steps: int = None,
            sampler: DecoderDDIMSampler = None
    ) -> torch.Tensor:
        """
        Uses DDIM sampling to upsample low-res image to high-res.
        
        Args:
            low_res_img: low-resolution image from decoder, shape (B, 3, low_res_size, low_res_size)
            z_img: predicted CLIP image embedding from Prior, shape (B, 512)
            steps: number of inference steps (optional override)
            sampler: optional DDIMSampler instance
            
        Returns:
            x0_pred: generated high-res image, shape (B, 3, high_res_size, high_res_size)
        """
        assert z_img.dim() == 2 and z_img.size(1) == 512, \
            f'[Upsampler.sample] z_img must be (B, 512), got {z_img.shape}'
        assert low_res_img.shape[2:] == (self.low_res_size, self.low_res_size), \
            f'[Upsampler.sample] low_res_img spatial dims must be ({self.low_res_size}, {self.low_res_size}), got {low_res_img.shape[2:]}'
        
        B = z_img.size(0)

        if self.debug:
            print(f'[Upsampler.sample] low_res_img shape: {low_res_img.shape}')
            print(f'[Upsampler.sample] z_img shape: {z_img.shape}')
            print(f'[Upsampler.sample] Using {steps or self.sampler.num_inference_steps} inference steps')

        sampler = sampler or self.sampler

        # Custom sampling loop that includes low_res_img conditioning
        return sampler.sample(
            model=self,
            z_img=z_img,
            image_size=(self.high_res_size, self.high_res_size),
            low_res_img=low_res_img  # Pass low-res for conditioning
        )