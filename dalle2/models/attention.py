"""
attention.py: Implements the two types of attention used by the prior's Transformer and the decoder's U-Net.

Description:
    * TODO

Classes:
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

class SelfAttention2d(nn.Module):
    def __init__(
            self,
            channels: int
    ):
        """
        Initializes Self-Attention module used in the decoder's U-Net.

        Example Usage:
            from attention import SelfAttention2d

            attn = SelfAttention2d(channels=128)
        
        Args:
            channels: number of input and output channels for the attention layer
        """
        super().__init__()
        self.channels = channels
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.scale = channels ** -0.5

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward Pass of SelfAttention2d

        Args:
            x: input tensor, of shape (B, C, H, W)
        
        Returns:
            output tensor, of shape (B, C, H, W), where each spatial location has been updated based on attention across all other locations
        """
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, C, H * W).permute(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Transpose q and k to shape (B, HW, C)
        q = q.permute(0, 2, 1)
        k = k.permute(0, 2, 1)
        v = v.permute(0, 2, 1)

        # Compute attention scores
        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)

        # Attend to values: (B, HW, C)
        out = attn @ v
        out = out.permute(0, 2, 1).reshape(B, C, H, W)

        return self.proj(out)
