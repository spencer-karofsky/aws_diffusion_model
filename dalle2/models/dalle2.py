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
from prior import Prior
from decoder import Decoder
from clip_encoding import CLIPEncoder

from sampling.ddim_sampling import DDIMSampler

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

            self.prior = Prior(device=self.device)
            self.prior.load_state_dict(torch.load(prior_path, map_location=self.device))
            self.prior.eval()
            if debug:
                print('Successfully loaded the prior and switched to mode "eval".')

            self.decoder = Decoder(device=self.device)
            self.decoder.load_state_dict(torch.load(decoder_path, map_location=self.device))
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
        # Encode text captions with CLIP, resulting in the text embeddings
        if isinstance(captions, str):
            captions = [captions]
        
        text_emb = self.clip_encoder.encode_text(captions)

        # Generate the image embeddings (using the trained prior) given the text embeddings
        image_embeddings_pred = self.prior.sample(
            z_txt=text_emb,
            sampler=self.prior_sampler,
            steps=len(self.t)
        )

        # Generate the clean images (using the decoder) given the image embeddings
        generated_images = self.decoder.sample(z_emb_pred=image_embeddings_pred)

        if self.debug:
            raise NotImplementedError

        return generated_images