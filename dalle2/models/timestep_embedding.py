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
        assert timesteps.dim() == 1, f'[TimestepEmbedder] Expected 1D timesteps, got shape: {timesteps.shape}'
        self.debug = False

        if self.debug:
            print(f'[TimestepEmbedder] Input timesteps shape: {timesteps.shape}')

        timesteps = timesteps.float() / 1000.0 # Normalize assuming T=1000
        timesteps = torch.clamp(timesteps, min=1e-5, max=1.0)

        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb_scale)

        if self.debug:
            print(f'[TimestepEmbedder] freqs shape: {freqs.shape}')

        # Broadcast and compute sinusoidal embedding
        angles = timesteps.unsqueeze(1) * freqs.unsqueeze(0) # (B, half_dim)

        if self.debug:
            print(f'[TimestepEmbedder] angles shape: {angles.shape}')

        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1) # (B, dim)

        if self.debug:
            print(f'[TimestepEmbedder] sinusoidal emb shape: {emb.shape}')

        projected = self.mlp(emb)

        if self.debug:
            print(f'[TimestepEmbedder] projected emb shape: {projected.shape}')

        return projected