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

from dalle2.sampling.ddim_sampling import PriorDDIMSampler, DecoderDDIMSampler

# Other imports
import os
from typing import Union, List

class DALLe2:
    def __init__(self, prior_path, decoder_path, clip_encoder,
                 prior_sampler, decoder_sampler, H, W,
                 T=1000, num_inference_timesteps=30,
                 on_aws=False, debug=False):
        self.device = (
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )
        self.debug = debug
        self.H, self.W = H, W
        self.T = T
        self.num_inference_timesteps = num_inference_timesteps

        # Load models
        self.prior = Prior(device=self.device, debug=debug)
        self.prior.load_state_dict(torch.load(prior_path, map_location=self.device), strict=False)
        self.prior.to(self.device).eval()

        self.decoder = Decoder(device=self.device, debug=debug)
        self.decoder.load_state_dict(torch.load(decoder_path, map_location=self.device), strict=False)
        self.decoder.to(self.device).eval()

        # Samplers (ensure same T/schedule as training when you constructed them)
        self.prior_sampler = prior_sampler
        self.decoder_sampler = decoder_sampler
        # Also set them on the modules so .sample() without explicit arg works.
        self.prior.sampler = prior_sampler
        self.decoder.sampler = decoder_sampler

        # CLIP encoder
        self.clip_encoder = clip_encoder
        try:
            self.clip_encoder.to(self.device)  # if your CLIPEncoder implements .to()
        except AttributeError:
            pass

    @torch.no_grad()
    def generate(self, captions: Union[str, List[str]]) -> torch.Tensor:
        if isinstance(captions, str):
            captions = [captions]

        # 1) Text → CLIP text embedding (B,512)
        z_txt = self.clip_encoder.encode_text(captions)  # may return CPU tensor
        z_txt = z_txt.to(self.device)

        if self.debug:
            print(f'[DALLe2.generate] z_txt: {z_txt.shape} device={z_txt.device}')

        # 2) Prior: DDIM denoise to predicted CLIP image embedding (B,512)
        #    Prior.sample will expand to (B,3,512) internally.
        z_img_hat = self.prior.sample(
            z_txt=z_txt,
            steps=self.num_inference_timesteps,
            sampler=self.prior_sampler
        )  # (B,512)

        # Normalize to match decoder training distribution
        z_img_hat = z_img_hat / (z_img_hat.norm(dim=-1, keepdim=True) + 1e-8)

        if self.debug:
            m, s = z_img_hat.mean().item(), z_img_hat.std().item()
            print(f'[DALLe2.generate] z_img_hat: {z_img_hat.shape} mean={m:.4f} std={s:.4f}')

        # 3) Decoder: DDIM denoise to image (B,3,H,W)
        imgs = self.decoder.sample(
            z_img=z_img_hat,
            steps=self.num_inference_timesteps,
            sampler=self.decoder_sampler
        )

        if self.debug:
            print(f'[DALLe2.generate] imgs: {imgs.shape} '
                  f'range=({imgs.min().item():.3f},{imgs.max().item():.3f})')

        return imgs
