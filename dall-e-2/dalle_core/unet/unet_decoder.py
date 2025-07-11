"""
unet: Implements U-Net architecture, modified from the 2015 paper: https://arxiv.org/abs/1505.04597

Description and Purpose:
    - The DDPM and DDIM are both implemented using U-Net (and these diffusion models are used in DALL·E 2):
    - The U-Net learns to predict noise for both models.
        - Takes in a noisy image and time step (embedded)
        - Outputs predicted noise
    - The U-Net is Called Once per Time Step for both Models.
    - U-Net Notes: https://github.com/spencer-karofsky/aws_diffusion_model/blob/main/dall-e-2/notes/U-net%202015.pdf

Usage:
    from models.decoder.unet import UNetDenoiser
    model = UNetDenoiser(...)
    predicted_noise = model(x_t, t)

Classes:
    * Utility
        - Time Embedding: Applies Time Step Embedding
            - SinusodalPosEmb: Applies Sinusodal Postional Embedding
        - ResBlock: Defines a ResBlock from ResNet
        - AttentionBlock: Includes Attention Mechanism
    * Core
        - UNetDownsamplingBlock: Performs the downsampling process of the U-Net
            - Downsample: Individual downsampling operation
        - UNetBottleneckBlock: Performs the bottleneck process of the U-Net
        - UNetUpsamplingBlock: Performs the upsampling process of the U-Net
            - Upsample: Individual upsampling operation
    * Main U-Net Denoiser
        - UNetDenoiser: Combines all of the core blocks for one unified U-Net Denoising process

Author:
    - Spencer Karofsky (https://github.com/spencer-karofsky)
"""
import math
import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F

# ----------------------------------------
# Utility Blocks
# ----------------------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int = 512):
        """Defines Sinusoidal Positional Embedding
        This process is explained in detail in my notes on Positional Encoding:
            - https://github.com/spencer-karofsky/aws_diffusion_model/blob/main/dall-e-2/notes/Attention%20is%20All%20You%20Need.pdf
        Args:
            dim: the dimensionality of the embedding (use 512 for DALL·E 2)
        """
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embeds a time step t into a high-dimensional vector
        Args:
            t: Tensor of shape [B], timesteps
        Returns:
            Tensor of shape [B, dim] — sinusoidal embeddings
        """
        # Leverage PyTorch's vectorized methods for faster computation
        device = t.device
        half_dim = self.dim // 2

        # Compute the exponents
        index = torch.arange(half_dim, device=device)

        # Compute the scaling factors: 10000^(−2i/dim)
        exponent = -math.log(10000) * index / half_dim
        freqs = torch.exp(exponent)  # [half_dim]

        # Ensures broadcasting works
        t = t[:, None]

        # Outer product
        angles = t * freqs

        # Create final interleaved embedding
        emb = torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1)
        emb = emb.view(t.shape[0], -1)
        return emb

class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        """Wraps Sinusoidal embedding + MLP projection for timestep conditioning.
        Process is described in Position-Wise FFN section:
            -https://github.com/spencer-karofsky/aws_diffusion_model/blob/main/dall-e-2/notes/Attention%20is%20All%20You%20Need.pdf
        Args:
            dim: the dimensionality of the embedding (use 512 for DALL·E 2)
        """
        super().__init__()
        self.embed = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),  # Uses SiLU instead of ReLU, as SiLU is slightly better for Diffusion
            nn.Linear(dim * 4, dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Performs embedding and runs it through the MLP
        Args:
            t: Tensor of shape [B], timesteps
        Returns:
            Tensor of shape [B, dim] timestep embeddings
        """
        x = self.embed(t)
        x = self.mlp(x)

        return x

class ResBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            time_emb_dim: int
        ):
        """Initializes ResNet block
        Args:
            in_channels: input channels
            out_channels: output channels
            time_emb_dim: time embedding dimension
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(32, in_channels)
        self.activation = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Projects time embedding to channel dim
        self.time_proj = nn.Linear(time_emb_dim, out_channels)

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # 1×1 conv to match channels if needed
        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] input feature map
            t_emb: [B, time_emb_dim] timestep embedding
        Returns:
            [B, out_channels, H, W] residual output
        """
        # First block
        h = self.norm1(x)
        h = self.activation(h)
        h = self.conv1(h)

        # Inject time embedding
        t = self.time_proj(t_emb)  # [B, out_channels]
        h = h + t[:, :, None, None]  # broadcast over H, W

        # Second block
        h = self.norm2(h)
        h = self.activation(h)
        h = self.conv2(h)

        # Residual connection
        return h + self.residual_conv(x)

class AttentionBlock(nn.Module):
    def __init__(
            self,
            channels: int,
            num_heads: int = 1
        ):
        """Applies spatial self-attention over [H, W] locations for each channel.

        Args:
            channels: number of input and output channels.
            num_heads: number of attention heads. Default is 1 (can be increased).
        """
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(num_groups=32, num_channels=channels)

        # 1x1 convolutions for Q, K, V projections
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Tensor of shape [B, C, H, W] after spatial self-attention
        """
        B, C, H, W = x.shape
        h = self.norm(x)

        # Project to Q, K, V
        qkv = self.qkv(h)  # [B, 3C, H, W]
        q, k, v = torch.chunk(qkv, chunks=3, dim=1)  # Each is [B, C, H, W]

        # Flatten spatial dims for attention
        q = q.reshape(B, self.num_heads, C // self.num_heads, H * W)  # [B, heads, C//heads, HW]
        k = k.reshape(B, self.num_heads, C // self.num_heads, H * W)
        v = v.reshape(B, self.num_heads, C // self.num_heads, H * W)

        # Compute attention
        attn_scores = torch.einsum("bhdn,bhdm->bhnm", q, k) / (C // self.num_heads) ** 0.5
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.einsum("bhnm,bhdm->bhdn", attn_weights, v)

        # Reshape back to [B, C, H, W]
        out = attn_output.reshape(B, C, H, W)
        out = self.proj_out(out)

        # Residual connection
        return x + out
    
# ----------------------------------------
# Core Blocks
# ----------------------------------------
class Downsample(nn.Module):
    def __init__(self, channels: int):
        """Defines a learnable downsampling operation that reduces the
        spatial resolution by a factor of 2 using a strided convolution.

        Args:
            channels: number of input and output channels
        """
        self.op = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            padding=1
        )

    def forward(self, x_tensor: torch.Tensor) -> torch.Tensor:
        """Performs the Downsample operation
        Args:
            x_tensor: input tensor of shape [B, C, H, W]
        Returns:
            output tensor of shape [B, C, H/2, W/2]
        """
        return self.op(x_tensor)

class UNetDownsamplingBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            base_channels: int,
            channel_multipliers: List[int],
            time_emb_dim: int,
            attention_resolutions: List[int],
            image_size: int = 128,
            num_res_blocks: int = 2
        ):
        """Initializes the Downsampling portion of the U-Net Architecture
        Explained in detail in my notes: https://github.com/spencer-karofsky/aws_diffusion_model/blob/main/dall-e-2/notes/U-net%202015.pdf
        
        Args:
            in_channels: input channels of the image (3 for RGB)
            base_channels: base channels (e.g., 128)
            channel_multipliers: defines how wide each level is (2015 paper uses [1, 2, 4, 8])
            time_embed_dim: dimensionality of the time step embedding vector
            attention_resolutions: resolutions which to apply attention
            img_size: the size of the input image
            num_res_blocks: number of Residual blocks
            
        """
        super().__init__()
        
        self.blocks = nn.ModuleList()
        self.resolutions = []

        curr_res = image_size
        in_ch = in_channels

        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.blocks.append(ResBlock(in_ch, out_ch, time_emb_dim))
                in_ch = out_ch

                if curr_res in attention_resolutions:
                    self.blocks.append(AttentionBlock(out_ch))

            # Only add Downsample if not the final level (bottleneck comes after)
            if i != len(channel_multipliers) - 1:
                self.blocks.append(Downsample(out_ch))
                curr_res //= 2
            
            self.resolutions.append(curr_res)

    def forward(
            self,
            x: torch.Tensor,
            t_emb: torch.Tensor
        ) -> list[torch.Tensor]:
        """Runs input through all ResBlocks, Attention, and Downsample layers.
        Args:
            x: input tensor of shape [B, C, H, W]
            t_emb: time embedding tensor of shape [B, D], used to condition each ResBlock
        Returns:
            list of tensors captured after each ResBlock, to be passed to the corresponding upsampling stages as skip connections
        """
        skips = []
        for block in self.blocks:
            if isinstance(block, ResBlock):
                x = block(x, t_emb)
                skips.append(x)
            elif isinstance(block, AttentionBlock):
                x = block(x)
            else:  # Downsample
                x = block(x)
        return skips, x

class UNetBottleneckBlock(nn.Module):
    def __init__(
            self,
            channels: int,
            time_emb_dim: int,
            use_attention: bool = True
        ):
        """Initializes the bottleneck
        Args:
            channels: number of input/output channels (needs to match out_channels from UNetDownsamplingBlock)
            time_emb_dim: dimensionality of time step embedding
            use_attention: uses attention between the two ResBlocks
        """
        super().__init__()

        self.resblock1 = ResBlock(channels, channels, time_emb_dim)
        self.attn = AttentionBlock(channels) if use_attention else nn.Identity()
        self.resblock2 = ResBlock(channels, channels, time_emb_dim)

    def forward(
            self,
            x: torch.Tensor,
            t_emb: torch.Tensor
        ) -> torch.Tensor:
        """Forward pass through the U-Net bottleneck.

        Applies two residual blocks with an optional attention layer in between.
        The output is passed to the upsampling path of the U-Net.

        Args:
            x: input tensor of shape [B, C, H, W]
            t_emb: time embedding tensor of shape [B, D]
        """
        x = self.resblock1(x, t_emb)
        x = self.attn(x)
        x = self.resblock2(x, t_emb)
        return x
    
class Upsample(nn.Module):
    def __init__(self, channels: int):
        """Defines a learnable upsampling operation that doubles spatial resolution.

        Args:
            channels: number of input and output channels
        """
        super().__init__()
        self.op = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  # Fast and stable
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)  # Learnable smoothing
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the upsampling
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W]
        Returns:
            torch.Tensor: Output tensor of shape [B, C, 2H, 2W]
        """
        return self.op(x)

class UNetUpsamplingBlock(nn.Module):
    def __init__(
        self,
        base_channels: int,
        channel_multipliers: List[int],
        time_emb_dim: int,
        attention_resolutions: List[int],
        image_size: int = 128,
        num_res_blocks: int = 2
    ):
        """Initializes upsampling block in the U-Net
        Args:
            base_channels: base number of channels for scaling with multipliers
            channel_multipliers: multipliers to control the width at each resolution level (e.g., [1, 2, 4, 8])
            time_emb_dim: dimensionality of the time embedding vector used to condition each ResBlock
            attention_resolutions: resolutions (e.g., 16, 32) where attention should be applied
            image_sizs: initial image resolution. Default is 128
            num_res_blocks: number of ResBlocks per resolution level. Default is 2
        """
        super().__init__()

        self.blocks = nn.ModuleList()
        self.resolutions = []

        curr_res = image_size // (2 ** (len(channel_multipliers) - 1))
        in_ch = base_channels * channel_multipliers[-1]

        for i, mult in reversed(list(enumerate(channel_multipliers))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):  # +1 for skip connection concat
                self.blocks.append(ResBlock(in_ch, out_ch, time_emb_dim))
                in_ch = out_ch

                if curr_res in attention_resolutions:
                    self.blocks.append(AttentionBlock(out_ch))

            # Only add Upsample if not final level
            if i != 0:
                self.blocks.append(Upsample(out_ch))
                curr_res *= 2

            self.resolutions.append(curr_res)

    def forward(
            self,
            x: torch.Tensor,
            skips: List[torch.Tensor],
            t_emb: torch.Tensor
        ) -> torch.Tensor:
        """
        Runs input through all ResBlocks, Attention, and Upsample layers.
        Combines with skip connections at each level.

        Args:
            x: input tensor from bottleneck [B, C, H, W]
            skips: list of skip tensors from encoder
            t_emb: time embedding tensor [B, D]

        Returns:
            Tensor of shape [B, out_channels, H, W] after upsampling
        """
        for block in self.blocks:
            if isinstance(block, ResBlock):
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)
                x = block(x, t_emb)
            elif isinstance(block, AttentionBlock):
                x = block(x)
            else:  # Upsample
                x = block(x)

        return x

# ----------------------------------------
# Main U-Net Denoiser
# ----------------------------------------
class UNetDenoiser(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_multipliers: List[int] = [1, 2, 4, 8],
        time_emb_dim: int = 512,
        attention_resolutions: List[int] = [16],
        image_size: int = 128,
        num_res_blocks: int = 2
    ):
        """Initializes the full U-Net denoiser used in diffusion models.

        Args:
            in_channels: Number of channels in the input image (e.g., 3 for RGB)
            out_channels: Number of channels in the predicted output (e.g., 3 for RGB noise)
            base_channels: Number of base channels used to scale the model width
            channel_multipliers: List of multipliers for each resolution level
            time_emb_dim: Dimensionality of the time embedding vector
            attention_resolutions: Resolutions at which to apply attention
            image_size: Size of the input image (assumed square)
            num_res_blocks: Number of residual blocks per resolution level
        """
        super().__init__()

        self.time_embedding = TimeEmbedding(time_emb_dim)

        self.down = UNetDownsamplingBlock(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            time_emb_dim=time_emb_dim,
            attention_resolutions=attention_resolutions,
            image_size=image_size,
            num_res_blocks=num_res_blocks
        )

        self.bottleneck = UNetBottleneckBlock(
            channels=base_channels * channel_multipliers[-1],
            time_emb_dim=time_emb_dim,
            use_attention=True
        )

        self.up = UNetUpsamplingBlock(
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            time_emb_dim=time_emb_dim,
            attention_resolutions=attention_resolutions,
            image_size=image_size,
            num_res_blocks=num_res_blocks
        )

        # Final output layer to predict noise (same shape as input image)
        self.final_norm = nn.GroupNorm(32, base_channels)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Noisy input image [B, C, H, W]
            t: Timestep [B]
        Returns:
            Predicted noise [B, C, H, W]
        """
        t_emb = self.time_embedding(t) # gets the time embedding [B, time_emb_dim]
        skips, x = self.down(x, t_emb) # downsample x
        x = self.bottleneck(x, t_emb)
        x = self.up(x, skips[::-1], t_emb)

        x = self.final_norm(x)
        x = self.final_act(x)
        x = self.final_conv(x)

        return x
