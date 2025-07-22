"""
prior.py: Implements the prior used by DALL·E 2.

Description:
    * The prior learns to generate image embeddings conditioned on the text embedding and timestep.
    * Training:
        - During training, we train a decoder-only Transformer with causal attention (prohibits looking ahead to future tokens).
        - Inputs:
            1. Noisy CLIP image embedding (real image embedding with noise added as defined according to the DDPM forward noising process)
            2. Clean CLIP text embedding.
            3. Timestep (sinusoidally-embedded, then projected) that is randomly sampled from (0, T]
        - The Transformer learns to predict the noise in the CLIP image embedding, eps_theta (makes more sense for computing loss).
        - Loss: the MSE of the predicted noise (output of the Transformer) and the actual noise.
            - The actual noise is iteratively added to the image embedding according to a noise schedule.
                - This process is fully-defined in the DDPM forward noising process.
    * Inference:
        - Trained Transformer Inputs:
            1. Pure Gaussian noise tensor (the fully-noised image embedding we seek to denoise).
            2. CLIP text embedding (known from the CLIP encoder).
            3. Timestep (sinusoidally-embedded, then projected) that is randomly sampled from (0, T]
        - Output: Predicted CLIP image embedding, z-hat_img (makes more sense as input to the decoder).
            - The training output is the predicted noise, eps_theta, but it's easy (and fully-deterministic) to get one from the other:
                - z-hat_img -> eps_theta: Use forward DDPM noising equation (in my DDPM notes).
                - eps_theta -> z-hat_img: Use reverse DDPM denoising equation (also in my DDPM notes).

Classes:
    * Prior(nn.Module): A modular DALL·E 2 decoder.

References:
    * DALL·E 2 Paper: https://cdn.openai.com/papers/dall-e-2.pdf
    * DDPM Paper: https://arxiv.org/pdf/2006.11239
    * My DALL·E 2 Notes: https://github.com/spencer-karofsky/aws_diffusion_model/blob/main/dalle2/research_notes/DALL-E-2%202022.pdf or /dalle2/research_notes/DALL-E-2 2022.pdf
    * My DDPM Notes: https://github.com/spencer-karofsky/aws_diffusion_model/blob/main/dalle2/research_notes/DDPM%202020.pdf or /dalle2/research_notes/DDPM 2020.pdf
    
Author:
    * Spencer Karofsky (https://github.com/spencer-karofsky)
"""
# PyTorch imports
import torch
import torch.nn as nn

# Module imports
from dalle2.models.prior_transformer import PriorTransformer
from dalle2.models.timestep_embedding import TimestepEmbedder

from dalle2.sampling.noise_scheduler import NoiseScheduler
from dalle2.sampling.ddim_sampling import DDIMSampler

class Prior(nn.Module):
    def __init__(
        self,
        device: torch.device,
        T: int = 1000,
        num_inference_steps: int = 30,
        on_aws: bool = False,
        debug: bool = False 
    ):
        """
        Initializes the prior, as described in the DALL·E 2 paper.

        Example Usage:
            from prior import Prior
            
            prior_model = Prior(...)
            eps_theta = prior_model.forward(...) # Use predicted noise to compute loss
            img_emb_pred = prior_model.sample(...)

        Args:
            device: the PyTorch device (tries to use the GPU)
            T: the total number of noising/denoising timesteps (total for each individually, not total of both combined)
            num_inference_steps: number of inference steps (to uniformly sample from T total timesteps)
            on_aws: configures for AWS (TODO Configure script for AWS)
            debug: debug: outputs relevant information (useful for debugging)
        """
        super().__init__()

        # Save params
        self.device = device
        self.T = T
        self.debug = debug
        # self.debug = False

        # Timestep embedding network
        self.TIMESTEP_DIM = 512 # Keep at 512 (or update self.TIMESTEP_DIM in dalle2.DALLe2.generate so they match)
        self.timestep_embedder = TimestepEmbedder(dim=self.TIMESTEP_DIM)

        # Instantiate Transformer
        DIM, BLOCKS, HEADS = 512, 6, 8 # Optionally reduce to speed-up prior training
        self.prior_transformer = PriorTransformer(
            dim=DIM,
            transformer_blocks=BLOCKS,
            attention_heads=HEADS,
            debug=self.debug
        )

        # Initialize DDIM Sampler (DDIMSampler.sample is a fully-deterministic process)
        noise_scheduler = NoiseScheduler(T=T)
        self.sampler = DDIMSampler(
            noise_scheduler=noise_scheduler,
            num_inference_steps=num_inference_steps,
            eta=0.0  # Fully-deterministic DDIM
        )

        # AWS configuration, if training on AWS
        if on_aws:
            raise NotImplementedError
        
    def forward(
            self,
            z_txt: torch.Tensor,
            t: torch.Tensor,
            z_T: torch.Tensor
    ) -> torch.Tensor:
        """
        Defines one step of the forward pass of the prior.
        - The forward steps runs inference on the prior's Transformer.

        Args:
            z_txt: the CLIP text embeddings, of shape (B, 512) (B=batch size)
            t: the timesteps (to be projected), of shape (B,)
            z_T: the noisy CLIP image embeddings we seek to denoise (noised by the DDPM forward diffusion process), of shape (B, 512)
        
        Returns:
            the predicted noise of the CLIP image embedding, eps_theta, of shape (B, 512)
        """
        assert t.dim() == 1, f'[Prior.forward] Expected shape (B,), got {t.shape}'
        t_emb = self.timestep_embedder(t)

        if not self.training and z_txt.dim() == 2 and z_txt.size(1) == 512:
            if self.debug:
                print(f'[Prior.forward] In eval mode: repeating z_txt (B, 512) to (B, 3, 512)')
            z_txt = z_txt.unsqueeze(1).expand(-1, 3, 512)

        # Squeeze any extra dims
        extra_dims = t.dim() - 2
        for _ in range(extra_dims):
           t = t.squeeze(1)
        
        extra_dims = z_T.dim() - 2
        for _ in range(extra_dims):
           z_T = z_T.squeeze(1)

        # Validate shapes
        assert t.dim() == 1, f'[Prior] t should have shape (B,), got {t.shape}'
        assert z_T.dim() == 2 and z_T.size(1) == 512, f'[Prior] z_T expected shape (B, 512), got {z_T.shape}'

        # Project timesteps
        t_emb = self.timestep_embedder(t)

        # Reshape z_txt here before passing it to PriorTransformer
        B = z_txt.size(0)

        if z_txt.dim() == 2 and z_txt.size(1) == 512 * 3:
        # Flattened text embeddings (3 tokens x 512)
            if self.debug:
                print(f'[Prior] Reshaping z_txt from (B, 1536) to (B, 3, 512)')
            z_txt = z_txt.view(B, 3, 512)

        elif z_txt.dim() == 4 and z_txt.size(1) == 1:
        # Comes in as (B, 1, 3, 512), squeeze to (B, 3, 512)
            if self.debug:
                print(f'[Prior] Reshaping z_txt from {z_txt.shape} to (B, 3, 512)')
            z_txt = z_txt.view(B, z_txt.shape[2], z_txt.shape[3])
        
        elif z_txt.dim() == 3 and z_txt.shape[1] == 1:
            # Comes in as (B, 1, 512) → repeat 3 times or error
            if self.debug:
                print(f'[Prior] z_txt came in as (B, 1, 512); repeating or broadcasting to (B, 3, 512)')
            z_txt = z_txt.expand(B, 3, 512)

        # Final shape check
        assert z_txt.dim() == 3 and z_txt.shape[1:] == (3, 512), f'[Prior] z_txt must be (B, 3, 512), got {z_txt.shape}'

        if self.debug:
            print(f'[Prior] Final z_txt shape: {z_txt.shape}')
            print(f'[Prior] t_emb shape: {t_emb.shape}')
            print(f'[Prior] z_T shape: {z_T.shape}\n')

        # Call PriorTransformer.forward
        return self.prior_transformer.forward(
            z_txt=z_txt,
            z_T=z_T,
            t_emb=t_emb,
            device=self.device
        )

    @torch.no_grad()
    def sample(
            self,
            z_txt: torch.Tensor,
            sampler: DDIMSampler = None,
            steps: int = None,
    ) -> torch.Tensor:
        """
        Defines the full forward pass of the prior (used during inference only)

        Args:
            z_txt: the CLIP text embeddings, of shape (B, 3, 512) (B=batch size)
            sampler: the DDIM sampleer
            steps: number of inference steps
        
        Returns:
            z_hat_img: the predicted CLIP image embeddings, of shape (B, 512)
        """
        B = z_txt.size(0)

        # Normalize input shape to (B, 3, 512)
        if z_txt.dim() != 2 or z_txt.size(0) != B or z_txt.size(1) != 512:
            if z_txt.dim() == 2 and z_txt.size(1) == 3 * 512:
                z_txt = z_txt.view(B, 3, 512)
            elif z_txt.dim() == 3 and z_txt.size(1) == 1:
                z_txt = z_txt.expand(B, 3, 512)
            elif z_txt.dim() == 4 and z_txt.size(1) == 1:
                z_txt = z_txt.view(B, z_txt.shape[2], z_txt.shape[3])
            elif z_txt.dim() == 2:
                z_txt = z_txt.view(B, 3, 512)
            
            assert z_txt.shape == (B, 3, 512), f'[Prior.sample] z_txt must be shape (B, 3, 512), got {z_txt.shape}'
        else:
            raise Exception(f'[Prior.sample]z_txt must be (B, 3, 512), got {z_txt.shape}')

        if self.debug:
            print(f'[Prior.sample] z_txt final shape: {z_txt.shape}')
            print(f'[Prior.sample] steps: {steps}')

        # Default to self.sampler if not provided
        sampler = sampler or self.sampler

        # Call DDIM sampler
        return sampler.sample(
            model=self,
            z_cond=z_txt,
            shape=(B, 512),
            steps=steps
        )
    
    def predict_eps(self, x_t, z_cond, t):
        return self.forward(z_txt=z_cond, t=t, z_T=x_t)

