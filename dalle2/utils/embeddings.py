# dalle2/utils/embeddings.py
import torch
import math

def get_timestep_embedding(timesteps: torch.Tensor, dim: int = 512) -> torch.Tensor:
    """
    Sinusoidal timestep embeddings (like in transformers & diffusion models).

    Args:
        timesteps: (B,) or (B, 1) tensor of timestep indices
        dim: embedding dimension

    Returns:
        (B, dim) tensor of embeddings
    """
    half_dim = dim // 2
    exponent = -math.log(10000) / (half_dim - 1)
    freqs = torch.exp(torch.arange(half_dim, dtype=torch.float32) * exponent).to(timesteps.device)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb
