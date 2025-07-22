"""
decoder.py: Implements the decoder used by DALL·E 2.

Description:
    * The decoder learns to reconstruct an image from random noise (conditioned on a CLIP image embedding and timestep embedding).
    * Training:
        - During training, we train a U-Net that takes in three inputs:
            1. Noisy images at a timestep, t, according to the DDPM's forward noising process.
            2. Clean CLIP image embeddings, z_img.
            3. MLP-Projected and Sinusoidally-encoded timestep embeddings, t_emb.
        - The U-Net learns to predict the batch of clean images, x-hat_0.
        - Loss: Computed as the MSE between x-hat_0 and the true images, x_0.
    * Inference:
        - Trained U-Net inputs:
            1. Pure Gaussian Noise, x_T.
            2. Predicted Clean CLIP image embedding, z-hat_img (output of the prior).
            3. MLP-Projected and Sinusoidally-encoded timestep embeddings, t_emb.
        - Output of the U-Net: the predicted/generated clean image, x-hat_0, conditioned on the original text captions.

Classes:
    * Decoder(nn.Module): A modular DALL·E 2 decoder.

References:
    * DALL·E 2 Paper: https://cdn.openai.com/papers/dall-e-2.pdf
    * DDPM Paper: https://arxiv.org/pdf/2006.11239
    * DDIM Paper: https://arxiv.org/pdf/2006.11239
    * U-Net Paper: https://arxiv.org/pdf/1505.04597
    * My DALL·E 2 Notes: https://github.com/spencer-karofsky/aws_diffusion_model/blob/main/dalle2/research_notes/DALL-E-2%202022.pdf or /dalle2/research_notes/DALL-E-2 2022.pdf
    * My DDPM Notes: https://github.com/spencer-karofsky/aws_diffusion_model/blob/main/dalle2/research_notes/DDPM%202020.pdf or /dalle2/research_notes/DDPM 2020.pdf
    * My U-Net Notes: https://github.com/spencer-karofsky/aws_diffusion_model/blob/main/dalle2/research_notes/U-net%202015.pdf or /dalle2/research_notes/U-net 2015.pdf
    * My DDIM Notes: https://github.com/spencer-karofsky/aws_diffusion_model/blob/main/dalle2/research_notes/DDIM%202021.pdf or /dalle2/research_notes/DDIM 2021.pdf

Author:
    * Spencer Karofsky (https://github.com/spencer-karofsky)
"""
# PyTorch imports
import torch
import torch.nn as nn

# Module imports
from dalle2.models.decoder_unet import DecoderUNet
from dalle2.models.timestep_embedding import TimestepEmbedder

from dalle2.sampling.noise_scheduler import NoiseScheduler
from dalle2.sampling.ddim_sampling import DDIMSampler

class Decoder(nn.Module):
    def __init__(
            self,
            device: torch.device,
            T: int = 1000,
            num_inference_steps: int = 30,
            on_aws: bool = False,
            debug: bool = False 
    ):
        """
        Initialize the decoder.

        Example Usage:
            from decoder import Decoder

            decoder_model = Decoder(...)
            img_pred = decoder_model.forward(...) # Feed img_pred into loss computation
            img_pred = decoder_model.sample(...)
        
        Args:
            device: the PyTorch device (CUDA, Metal (MPS), or CPU)
            T: the total number of noising/denoising timesteps (total for each individually, not total of both combined)
            num_inference_steps: the number of inference steps (to uniformly sample from T total timesteps)
            on_aws: configures for AWS (TODO Configure script for AWS)
            debug: debug: outputs relevant information (useful for debugging)
        """
        super().__init__()

        # Save params
        self.device = device
        self.T = T
        self.debug = debug
        self.debug = False

        # Timestep embedding network
        self.TIMESTEP_DIM = 512 # Keep at 512 (or update self.TIMESTEP_DIM in dalle2.DALLe2.generate so they match)
        self.timestep_embedder = TimestepEmbedder(dim=self.TIMESTEP_DIM)

        # Define U-Net hyperparameters and instantiate the U-Net
        self.IMG_SIZE = 128 # pixels
        self.IN_CHANNELS = 3 # 3 color channels (RGB)
        self.COND_EMB_DIM = 512 # embedding dimensionality of the conditioning vector you inject into the U-Net
        self.BASE_CHANNELS = 64 # base channels in the first convolution layer
        self.CHANNEL_MULTS = (1, 2, 4, 8,) # base channel multiplier at each level
        self.RESIDUAL_BLOCKS = 2 # number of residual blocks at each level (use 2 for now)
        self.ATTENTION_RESOLUTIONS = (16,) # Resolutions at which we apply attention (e.g., (16,) means we apply attention when the image size is 16x16)

        self.decoder_unet = DecoderUNet(
            channel_multipliers=self.CHANNEL_MULTS,
            attention_resolutions=self.ATTENTION_RESOLUTIONS,
            image_size=self.IMG_SIZE,
            in_channels=self.IN_CHANNELS,
            conditional_embedding_dim=self.COND_EMB_DIM,
            base_channels=self.BASE_CHANNELS,
            residual_blocks=self.RESIDUAL_BLOCKS,
            device=self.device,
            debug=self.debug
        )

        # Initialize DDIM Sampler (DDIMSampler.sample is a fully-deterministic process)
        noise_scheduler = NoiseScheduler(T=T)
        self.sampler = DDIMSampler(
                noise_scheduler=noise_scheduler,
                num_inference_steps=num_inference_steps,
                eta=0.0 # Fully-deterministic
        )

        # AWS configuration, if training on AWS
        if on_aws:
            raise NotImplementedError
        
    def forward(
         self,
         x_t: torch.Tensor,
         z_img: torch.Tensor,
         t: torch.Tensor   
    ) -> torch.Tensor:
        """
        Defines one step of the forward pass of the decoder.
        - The forward steps runs inference on the decoder's U-Net.

        Args:
            x_t: noisy images generated by the DDPM forward diffusion process at timestep t, of shape (B, 3, H, W)
            z_img: clean (true) CLIP image embedding conditioned on the text, of shape (B, 512)
            t: the timesteps (to be projected), of shape (B,)

        Returns:
            Predicted noise (epsilon), eps_theta, of shape (B, 3, H, W)
        """
        B = x_t.size(0)

        assert x_t.dim() == 4, f'[Decoder] x_t must be 4D (B, 3, H, W), got {x_t.shape}'
        assert z_img.shape == (B, 512), f'[Decoder] z_img must have shape ({B}, 512), got {z_img.shape}'
        assert t.shape == (B,), f'[Decoder] t must have shape ({B},), got {t.shape}'

        if self.debug:
            print(f'[Decoder] x_t shape: {x_t.shape} (expected: ({B}, 3, {self.IMG_SIZE}, {self.IMG_SIZE}))')
            print(f'[Decoder] z_img shape: {z_img.shape} (expected: ({B}, 512))')
            print(f'[Decoder] t shape before embedding: {t.shape} (expected: ({B},))')

        t_emb = self.timestep_embedder(t)

        if self.debug:
            print(f'[Decoder] t_emb shape: {t_emb.shape} (expected: ({B}, {self.TIMESTEP_DIM}))')

        # Forward through U-Net
        eps_pred = self.decoder_unet(
            x_t=x_t,
            z_img=z_img,
            t_emb=t_emb
        )

        if self.debug:
            print(f'[Decoder] eps_pred shape: {eps_pred.shape} (expected: ({B}, 3, {self.IMG_SIZE}, {self.IMG_SIZE}))')

        return eps_pred
    
    def sample(
            self,
            z_img: torch.Tensor,
            sampler: DDIMSampler = None,
            steps: int = None
    ) -> torch.Tensor:
        """
        Defines the full forward pass of the decoder (used during inference only)

        Args:
            z_img: the CLIP image embeddings, of shape (B, 512) (B=batch size)
            sampler: the DDIM sampler
            steps: number of inference steps
        
        Returns:
            z_hat_img: the predicted CLIP image embeddings, of shape (B, 3, H, W)
        """
        assert z_img.dim() == 2, f'[Decoder.sample] z_img must be 2D (B, 512), got shape {z_img.shape}'
        B = z_img.size(0)
        assert z_img.shape[1] == 512, f'[Decoder.sample] z_img must have dim 512, got {z_img.shape[1]}'

        if self.debug:
            print(f'[Decoder.sample] z_img shape: {z_img.shape} (expected: ({B}, 512))')
            print(f'[Decoder.sample] Using {steps or self.sampler.num_inference_steps} inference steps')

        sampler = sampler or self.sampler

        # Call DDIMSampler.sample
        z_hat_img = sampler.sample(
            model=self,
            z_cond=z_img,
            shape=(B, 3, self.IMG_SIZE, self.IMG_SIZE),
            steps=steps
        )

        if self.debug:
            print(f'[Decoder.sample] z_hat_img shape: {z_hat_img.shape} (expected: ({B}, 3, {self.IMG_SIZE}, {self.IMG_SIZE}))')

        return z_hat_img
    
    def predict_eps(self, x_t, z_cond, t):
        return self.forward(x_t=x_t, z_img=z_cond, t=t)
