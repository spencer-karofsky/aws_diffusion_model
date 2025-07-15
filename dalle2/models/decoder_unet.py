"""
decoder_unet.py: Implements the decoder's U-Net model.

Description:
    * The U-Net is a type of CNN that was originally introduced for SOTA segmentation on limited training data.
    * In DALL·E 2 (and DDPM), the U-Net is used as the model/backbone of the decoder.
        - Whereas the decoder provides a high-level diffusion process, the the U-Net is the actual model.
    * High-level U-Net overview:
        - In the paper, the input is a batch of images, represented by a Tensor of shape (B, in_channels, image_size (H), image_size (W))
        - Output is the predicted noise, eps_theta, represented by the same Tensor shape (B, in_channels, image_size (H), image_size (W))
        - Training:
            - In my implementation, we pass in three components as the inputs to the U-Net:
                1. Noisy images, where the noise has been added according to the DDPM forward noising process, of shape (B, in_channels, image_size (H), image_size (W)).
                2. MLP-Projected and Sinusoidally-encoded timestep embedding (uniform-randomly sampled), of final shape (B, 512)
                3. The CLIP Image Embeddings, of shape (B, 512).
                - However, we pass these three inputs into different stages of the U-Net.
                    - The initial input is the noisy images batch.
                    - The image and timestep embeddings are combined via a MLP and then injected into residual blocks.
            - The output of the U-Net is a predicted clean image (B, in_channels, image_size (H), image_size (W)).
            - The loss (not computed in this class) is computed as the MSE between the predicted clean image and ground truth clean image.
        - Inference:
            - Inputs:
                1. We replace the DPPM-added noise with pure Gaussian Noise images.
                2. We replace the true image embeddings with the prior's predicted image embeddings (conditioned on the text prompt).
                3. We replace the intial randomly-sampled timestep vector with defined timesteps (according to how the DDIM samples timesteps).
            - Output: The B target/generated clean images (conditioned on the text embedding), of shape (B, in_channels, image_size (H), image_size (W)).

Classes:
    * DecoderUNet(nn.Module): U-Net implementation used by the decoder.

References:
    * U-Net Paper: https://arxiv.org/pdf/1505.04597
    * Transformer Paper: https://arxiv.org/pdf/1706.03762
    * My U-Net Notes: https://github.com/spencer-karofsky/aws_diffusion_model/blob/main/dalle2/research_notes/U-net%202015.pdf or /dalle2_new/research_notes/U-net 2015.pdf
    * My Transformer Notes: https://github.com/spencer-karofsky/aws_diffusion_model/blob/main/dalle2/research_notes/Attention%20is%20All%20You%20Need.pdf or /dalle2_new/research_notes/Attention is All You Need.pdf
    
Author:
    * Spencer Karofsky (https://github.com/spencer-karofsky)
"""
# PyTorch imports
import torch
import torch.nn as nn

# Module imports
from attention import SelfAttention2d

# Other imports
from typing import Tuple


class DecoderUNet(nn.Module):
    def __init__(
            self,
            channel_multipliers: Tuple[int],
            attention_resolutions: Tuple[int],
            device: torch.device,
            image_size: int = 128,
            in_channels: int = 3,
            conditional_embedding_dim: int = 512,
            base_channels: int = 64,
            residual_blocks: int = 2,
            on_aws: bool = False,
            debug: bool = False 
    ):
        """
        Initializes a U-Net architecture utilizing causal attention.

        Example Usage:
            from decoder_unet import DecoderUNet
            unet = DecoderUNet(channel_multipliers=(1, 2, 4,), attention_resolutions=(16,))
            clean_img_pred = unet.forward(...)
        
        Args:
            channel_multipliers: scales the number of feature channels at each U-Net resolution level to control model capacity
            attention_resolutions: downsampled image resolutions where causal attention is applied
            device: the PyTorch device (CUDA, Metal (MPS), or CPU)
            image_size: spatial resolution at the U-Net bottleneck (must be power of 2)
            in_channels: number of input channels (use 3 for RGB, or 1 for grayscale)
            conditional_embedding_dim: the dimensionality of the conditioning vector injected into the U-Net
            base_channels: number of (base) filters in the first convolution layer (channel_multipliers is the multiplier of base_channels)
            residual_blocks: number of residual blocks per level (more = deeper network = more computationally expensive)
            on_aws: configures for AWS (TODO Configure script for AWS)
            debug: debug: outputs relevant information (useful for debugging)
        """
        super().__init__()

        # Save params
        self.device = device
        self.attention_resolutions = attention_resolutions
        self.channel_multipliers = channel_multipliers
        self.image_size = image_size
        self.in_channels = in_channels
        self.conditional_embedding_dim = conditional_embedding_dim
        self.base_channels = base_channels
        self.residual_blocks = residual_blocks
        self.debug = debug

        # Define 

        # AWS configuration, if training on AWS
        if on_aws:
            raise NotImplementedError
        
    
    def forward(
            self,
            x_t: torch.Tensor,
            z_img: torch.Tensor,
            t_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Defines the forward propagation through the U-Net.

        Args:
            x_t: the noisy image at timestep t, of shape (B, in_channels, image_size (H), image_size (W))
            z_img: the true image embeddings, of shape (B, 512)
            t_emb: the true timestep embeddings, of shape (B, 512)
        
        Returns:
            the predicted batch of clean images, x-hat_0
        """
        pass

