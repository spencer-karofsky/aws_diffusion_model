"""
timestep_embedding.py: Implements the decoder used by DALL·E 2.

Description:
    * Encodes the timestep into a learned embedding.

Classes:
    * TimestepEmbedder(nn.Module): Sinusoidally-encodes a timestep vector, which is then passed through a Neural Network and outputs the Projected Timestep Embedding.
    
Author:
    * Spencer Karofsky (https://github.com/spencer-karofsky)
"""
# PyTorch imports
import torch
import torch.nn as nn

# Other imports
import math

class TimestepEmbedder(nn.Module):
    def __init__(
            self,
            dim: int
    ):
        """
        Encodes the timestep into a learned embedding.

        Example Usage:
            from timestep_embedding import TimestepEmbedder
            t = TimestepEmbedder(512)
            timesteps = torch.randn(32,)
            t_emb = time_emb.forward(timesteps)

        Args:
            dim: embedding dimension (512)
        """
        super().__init__()
        self.dim = dim

        # Projection after sinusoidally-encoding
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(
            self,
            timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of Timestep Embedder:
            1. Sinusoidally-encode timesteps.
            2. Pass through MLP

        Args:
            timesteps: the timesteps vector, of shape (B)
        
        Returns:
            the sinusoidally-encoded and projected timesteps Tensor, of shape (B, self.dim)
        """
        # Get sinusoidal encoding
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        timesteps = timesteps.view(-1)
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)


        # Concat Sine and Cosine components
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        return self.mlp(emb)