"""
attention.py: Implements the two types of attention used by the prior's Transformer and the decoder's U-Net.

Description:
    * TODO

Classes:
    * CausalSelfAttention(nn.Module): Enables tokens to attend only to themselves and earlier tokens for autoregressive modeling, used by the prior.
    * SelfAttention2d(nn.Module): Allows each pixel to attend to all other spatial locations for global image context, used by the decoder.

References:
    * Transformer Paper: https://arxiv.org/pdf/1706.03762
    * My Transformer Notes: https://github.com/spencer-karofsky/aws_diffusion_model/blob/main/dalle2/research_notes/Attention%20is%20All%20You%20Need.pdf or /dalle2/research_notes/Attention is All You Need.pdf

Author:
    * Spencer Karofsky (https://github.com/spencer-karofsky)
"""
# PyTorch imports
import torch
import torch.nn as nn

# Module imports


# Other imports


class CausalSelfAttention(nn.Module):
    def __init__(
            self,

    ):
        """

        """
        super().__init__()
        pass

    def forward(
            self,

    ) -> torch.Tensor:
        """

        """
        pass

class SelfAttention2d(nn.Module):
    def __init__(
            self,

    ):
        """

        """
        super().__init__()
        pass

    def forward(
            self,

    ) -> torch.Tensor:
        """

        """
        pass