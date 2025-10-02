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
        dim: int,
        module: str,
        T: int = 1000
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
            module: the prior needs a layer norm, while it breaks the decoder's output
            T: number of timesteps
        """
        super().__init__()
        self.dim = dim
        self.T = T  # Store T for normalization

        # Projection after sinusoidally-encoding
        if module == 'prior':
            self.mlp = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.SiLU(),
                nn.Linear(dim * 4, dim),
                nn.LayerNorm(dim)
            )
        elif module == 'decoder':
            self.mlp = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.SiLU(),
                nn.Linear(dim * 4, dim)
            )
        
        self.module = module

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
        assert timesteps.dim() == 1
        
        # Use fixed normalization for decoder (like the old code)
        if self.module == 'decoder':
            t = timesteps.float() / 999.0
        else:
            t = timesteps.float() / max(self.T - 1, 1)
        t = t.clamp_(1e-5, 1.0)
        
        half = self.dim // 2
        # Standard sinusoid base
        freqs = torch.exp(torch.arange(half, device=t.device) * -(math.log(10000.0) / (half - 1)))
        angles = t.unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([angles.sin(), angles.cos()], dim=1)
        
        return self.mlp(emb)