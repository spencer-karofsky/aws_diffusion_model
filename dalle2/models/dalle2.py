"""
dalle2.py: Runs inference on the DALL·E 2 architecture, used for generating images conditioned on text prompts.

Description:
    * DALL·E 2 Consists of two Primary Components:
        1. The Prior:
            - A trained Transformer that takes in three inputs:
                1. A pure Gaussian noise vector,
                2. A CLIP text embedding conditioned on the input caption.
                3. The time step we are sampling from (decrementing from T to 0), sinusoidally-embedded who's output is passed through a (trained) neural network.
            - The prior learns to transform random noise into a semantically-meaningful image embedding via a diffusion process.
        2. The Decoder:
            - A trained U-Net that takes in three inputs:
                1. Pure Gaussian noise.
                2. Predicted image embedding (output of the prior).
                3. The timestep (embedded), sinusoidally-embedded who's output is passed through a (trained) neural network (a different NN than from the prior).

Classes:
    * DALLe2: Runs inference on a trained prior and decoder to generate image(s) conditioned on text prompts.

References:
    * DALL·E 2 Paper: https://cdn.openai.com/papers/dall-e-2.pdf
    * My DALL·E 2 Notes: https://github.com/spencer-karofsky/aws_diffusion_model/blob/main/dalle2/research_notes/DALL-E-2%202022.pdf or /dalle2/research_notes/DALL-E-2 2022.pdf

Author:
    * Spencer Karofsky (https://github.com/spencer-karofsky)
"""
# PyTorch imports
import torch

# Module imports
from dalle2.models.prior import Prior
from dalle2.models.decoder import Decoder
from dalle2.models.clip_encoding import CLIPEncoder

from dalle2.sampling.ddim_sampling import DDIMSampler

# Other imports
import os
from typing import Union, List

class DALLe2:
    def __init__(
            self,
            prior_path: str,
            decoder_path: str,
            clip_encoder: CLIPEncoder,
            prior_sampler: DDIMSampler,
            decoder_sampler: DDIMSampler,
            H: int,
            W: int,
            T: int = 1000,
            num_inference_timesteps: int = 30,
            on_aws: bool = False,
            debug: bool = False
    ):
        """
        Runs inference on the trained DALL·E 2 prior and decoder.

        Example Usage:
            from dalle2 import DALLe2
            dalle = DALLe2(...)
            prompts = [
                'a plaid sports car',
                'panda juggling chairs',
                'a glowing fox made of stardust leaping through a nebula'
            ]
            generated_images = dalle.generate(prompts)

        Args:
            prior_path: path to the trained prior
            decoder_path: path to the trained decoder
            clip_encoder: instantiated CLIP text and image encoder
            prior_sampler: performs DDIM-based denoising of the latent CLIP image embedding during inference.
            decoder_sampler: performs DDIM-based denoising of noisy images into final RGB outputs, conditioned on the CLIP image embedding.
            H: image height (pixels)
            W: image width (pixels)
            T: total amount of noising/denoising timesteps
            num_inference_timesteps: choose somewhere between 20 and 50 for the optimal quality/speed tradeoff
            on_aws: configures everything for AWS (TODO need to define this functionality out more when migrating training to AWS)
            debug: outputs relevant information (useful for debugging)
        """
        # Initialize device (ideally use CUDA; secondly, Metal; worst-case, use the CPU)
        self.device = (
            'cuda' if torch.cuda.is_available() 
            else 'mps' if torch.backends.mps.is_available() 
            else 'cpu'
        )

        # Try to load trainer prior and decoder
        try:
            if not os.path.exists(prior_path):
                raise FileNotFoundError(f'Prior checkpoint not found at: {prior_path}')
            if not os.path.exists(decoder_path):
                raise FileNotFoundError(f'Decoder checkpoint not found at: {decoder_path}')

            self.prior = Prior(device=self.device, debug=debug)
            self.prior.load_state_dict(torch.load(prior_path, map_location=self.device), strict=False)
            self.prior.to(self.device)
            self.prior.eval()
            if debug:
                print('Successfully loaded the prior and switched to mode "eval".')

            self.decoder = Decoder(device=self.device, debug=debug)
            self.decoder.load_state_dict(torch.load(decoder_path, map_location=self.device), strict=False)
            self.decoder.to(self.device)
            self.decoder.eval()
            if debug:
                print('Successfully loaded the decoder and switched to mode "eval".')

        except FileNotFoundError as e:
            print('[ERROR] failed to load either the trained prior or decoder weights.')
            raise e

        except RuntimeError as e:
            raise RuntimeError(f'Model loading failed — checkpoint might not match model architecture:\n{e}')

        # Save params
        self.clip_encoder = clip_encoder
        self.prior_sampler = prior_sampler
        self.decoder_sampler = decoder_sampler
        self.H = H
        self.W = W
        self.T = T
        self.debug = debug

        # Timesteps configuration
        self.TIMESTEP_DIM = 512 # Use same dimenstionality as for the other inputs (not a strict requirement, but maintains consistency)
        self.t = torch.linspace(T - 1, 0, steps=num_inference_timesteps, dtype=torch.long)
        
        # AWS configuration
        if on_aws:
            raise NotImplementedError
    
    def generate(
            self,
            captions: Union[str, List[str]]
    ) -> torch.Tensor:
        """
        Generate images conditioned on text captions/prompts, x-hat_0.

        Args:
            captions: either one (str) or multiple (list-form) text prompts to condition DALL·E 2 (maximum of 77 tokens per prompt)
        
        Returns:
            img_generations: the images conditioned on the text prompts (of shape (len(captions), 3, H, W), where H and W are the image height and width (px))
        """
        if isinstance(captions, str):
            captions = [captions]

        # Step 1: Encode text prompts into multi-layer CLIP embeddings
        text_emb = self.clip_encoder.encode_text_multilayer(captions)  # shape: (B, 3, 512)
        assert text_emb.dim() == 3 and text_emb.shape[1:] == (3, 512), f'[Error] Expected z_txt shape (B, 3, 512), got {text_emb.shape}'
        
        if self.debug:
            print(f'[DALLe2.generate] z_txt shape before reduction: {text_emb.shape}')

        # Step 2: Reduce 3-layer CLIP embeddings to single (B, 512)
        # text_emb = text_emb.mean(dim=1)  # shape: (B, 512)

        if self.debug:
            print(f'[DALLe2.generate] z_txt shape after reduction: {text_emb.shape}')

        # Step 3: Sample predicted image embeddings using the prior
        B = text_emb.size(0)

        image_embeddings_pred = self.prior.sample(
            z_txt=text_emb,
            sampler=self.prior_sampler,
            steps=len(self.t)
        )
        # HERE! z_img = self.clip_encoder.encode_image(image.unsqueeze(0)).squeeze(0).to(device)


        assert image_embeddings_pred.dim() == 2 and image_embeddings_pred.shape[1] == 512, f'[Error] Prior output z_img must be shape (B, 512), got {image_embeddings_pred.shape}'

        if self.debug:
            print(f'[DALLe2.generate] z_img (image embeddings) shape: {image_embeddings_pred.shape}')

        # Step 4: Decode image embeddings into full-resolution images
        generated_images = self.decoder.sample(z_img=image_embeddings_pred)
        assert generated_images.shape[1:] == (3, self.H, self.W), \
            f'[Error] Generated image shape should be (B, 3, {self.H}, {self.W}), got {generated_images.shape}'

        if self.debug:
            print(f'[DALLe2.generate] generated_images shape: {generated_images.shape}')

        return generated_images