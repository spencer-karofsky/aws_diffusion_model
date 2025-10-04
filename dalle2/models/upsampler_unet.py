"""
upsampler_unet.py: Implements the upsampler's U-Net model.

Description:
    * Similar to DecoderUNet but designed for super-resolution tasks
    * Takes concatenated input: [noisy high-res image, upsampled low-res image]
    * Conditioned on CLIP embeddings and timestep embeddings
    * Predicts noise to be removed from high-res image

Classes:
    * UpsamplerUNet(nn.Module): U-Net implementation for the upsampler.

References:
    * DALL·E 2 Paper: https://cdn.openai.com/papers/dall-e-2.pdf
    * Cascaded Diffusion Models: https://arxiv.org/abs/2106.15282
    * U-Net Paper: https://arxiv.org/pdf/1505.04597

Author:
    * Spencer Karofsky (https://github.com/spencer-karofsky)
"""
# PyTorch imports
import torch
import torch.nn as nn

# Module imports
from dalle2.models.attention import SelfAttention2d
from dalle2.models.shared_modules import DownsampleBlock, UpsampleBlock, ResidualBlock, ConditioningProjector

# Other imports
from typing import Tuple


class UpsamplerUNet(nn.Module):
    def __init__(
            self,
            channel_multipliers: Tuple[int],
            attention_resolutions: Tuple[int],
            device: torch.device,
            image_size: int = 128,
            in_channels: int = 6,  # 3 (noisy) + 3 (low-res) = 6
            out_channels: int = 3,
            conditional_embedding_dim: int = 512,
            base_channels: int = 64,
            residual_blocks: int = 2,
            on_aws: bool = False,
            debug: bool = False
    ):
        """
        Initializes a U-Net architecture for upsampling.

        Example Usage:
            from upsampler_unet import UpsamplerUNet
            unet = UpsamplerUNet(channel_multipliers=(1, 2, 4, 8), attention_resolutions=(32, 16))
            noise_pred = unet.forward(...)
        
        Args:
            channel_multipliers: scales feature channels at each resolution level
            attention_resolutions: resolutions where self-attention is applied
            device: PyTorch device (CUDA, Metal (MPS), or CPU)
            image_size: spatial resolution (128 for upsampler)
            in_channels: input channels (6 = 3 noisy + 3 low-res)
            out_channels: output channels (3 for RGB noise prediction)
            conditional_embedding_dim: dimensionality of conditioning vector
            base_channels: base number of filters in first conv layer
            residual_blocks: number of residual blocks per level
            on_aws: configures for AWS
            debug: outputs debug information
        """
        super().__init__()

        # Save params
        self.device = device
        self.attention_resolutions = attention_resolutions
        self.channel_multipliers = channel_multipliers
        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conditional_embedding_dim = conditional_embedding_dim
        self.base_channels = base_channels
        self.residual_blocks = residual_blocks
        self.debug = debug

        # Define Conditioning Projector
        self.conditioning_projector = ConditioningProjector(
            input_dim=self.conditional_embedding_dim,
            hidden_dim=self.conditional_embedding_dim
        )

        # Define Input Projection - takes concatenated input
        self.input_proj = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.base_channels * self.channel_multipliers[0],
            kernel_size=3,
            padding=1
        )

        # Define U-Net downsampling layers
        self.downsample_layers = nn.ModuleList()
        for i in range(len(self.channel_multipliers) - 1):
            self.downsample_layers.append(
                DownsampleBlock(
                    in_channels=base_channels * self.channel_multipliers[i],
                    out_channels=base_channels * self.channel_multipliers[i + 1]
                )
            )

        # Define downsampling residual blocks
        self.encoder_res_blocks = nn.ModuleList()
        for i in range(len(self.channel_multipliers) - 1):
            blocks = nn.ModuleList()
            for _ in range(self.residual_blocks):
                blocks.append(ResidualBlock(
                    in_channels=base_channels * self.channel_multipliers[i],
                    out_channels=base_channels * self.channel_multipliers[i],
                    cond_dim=self.conditional_embedding_dim,
                    debug=self.debug
                ))
            self.encoder_res_blocks.append(blocks)

        # Define bottleneck layers
        bottleneck_dim = self.base_channels * self.channel_multipliers[-1]
        self.bottleneck_blocks = nn.ModuleList([
            ResidualBlock(
                in_channels=bottleneck_dim,
                out_channels=bottleneck_dim,
                cond_dim=self.conditional_embedding_dim,
                debug=self.debug
            ),
            SelfAttention2d(bottleneck_dim),
            ResidualBlock(
                in_channels=bottleneck_dim,
                out_channels=bottleneck_dim,
                cond_dim=self.conditional_embedding_dim,
                debug=self.debug
            )
        ])

        # Define U-Net upsampling layers (reverse order)
        self.upsample_layers = nn.ModuleList()
        for i in reversed(range(len(self.channel_multipliers) - 1)):
            self.upsample_layers.append(
                UpsampleBlock(
                    in_channels=base_channels * self.channel_multipliers[i + 1],
                    out_channels=base_channels * self.channel_multipliers[i],
                    debug=self.debug
                )
            )

        # Define upsampling residual blocks
        self.decoder_res_blocks = nn.ModuleList()
        for i in reversed(range(len(self.channel_multipliers) - 1)):
            blocks = nn.ModuleList()
            for _ in range(self.residual_blocks):
                blocks.append(ResidualBlock(
                    in_channels=base_channels * self.channel_multipliers[i],
                    out_channels=base_channels * self.channel_multipliers[i],
                    cond_dim=self.conditional_embedding_dim,
                    debug=self.debug
                ))
            self.decoder_res_blocks.append(blocks)

        # Define Output Projection
        self.output_proj = nn.Conv2d(
            in_channels=self.base_channels * self.channel_multipliers[0],
            out_channels=self.out_channels,
            kernel_size=3,
            padding=1
        )

        # Initialize output layer
        nn.init.xavier_uniform_(self.output_proj.weight, gain=1.0)
        nn.init.zeros_(self.output_proj.bias)

        # Add attention capabilities
        self.encoder_attn_blocks = nn.ModuleList()
        self.decoder_attn_blocks = nn.ModuleList()

        # Track spatial resolution at each level
        resolution = image_size
        for i in range(len(self.channel_multipliers) - 1):
            channels = base_channels * channel_multipliers[i]

            if resolution in attention_resolutions:
                self.encoder_attn_blocks.append(SelfAttention2d(channels))
            else:
                self.encoder_attn_blocks.append(None)

            resolution //= 2  # halve resolution after each downsample

        # Do the same for upsampling (reverse)
        for i in reversed(range(len(self.channel_multipliers) - 1)):
            channels = base_channels * channel_multipliers[i]

            if resolution in attention_resolutions:
                self.decoder_attn_blocks.append(SelfAttention2d(channels))
            else:
                self.decoder_attn_blocks.append(None)

            resolution *= 2  # double resolution after each upsample

        # AWS configuration
        if on_aws:
            raise NotImplementedError

    def forward(
            self,
            x_t: torch.Tensor,
            z_img: torch.Tensor,
            t_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward propagation through the upsampler U-Net.
        
        Args:
            x_t: concatenated [noisy high-res, upsampled low-res], shape (B, 6, H, W)
            z_img: CLIP image embeddings, shape (B, 512)
            t_emb: timestep embeddings, shape (B, 512)
        
        Returns:
            Predicted noise, shape (B, 3, H, W)
        """
        B, C, H, W = x_t.shape

        assert x_t.dim() == 4, f'[UpsamplerUNet] Expected x_t to be 4D (B, C, H, W), got {x_t.shape}'
        assert C == self.in_channels, f'[UpsamplerUNet] Expected {self.in_channels} channels, got {C}'
        assert z_img.shape == (B, 512), f'[UpsamplerUNet] z_img shape mismatch: expected ({B}, 512), got {z_img.shape}'
        assert t_emb.shape == (B, 512), f'[UpsamplerUNet] t_emb shape mismatch: expected ({B}, 512), got {t_emb.shape}'

        if self.debug:
            print(f'[UpsamplerUNet] x_t shape: {x_t.shape}')
            print(f'[UpsamplerUNet] z_img shape: {z_img.shape}')
            print(f'[UpsamplerUNet] t_emb shape: {t_emb.shape}')

        # Step 1. Fuse conditioning vectors
        cond_emb = self.conditioning_projector(
            t_emb=t_emb,
            z_img=z_img
        )
        if self.debug:
            print(f'[UpsamplerUNet] cond_emb shape: {cond_emb.shape}')

        # Step 2. Project the concatenated input
        x = self.input_proj(x_t)
        if self.debug:
            print(f'[UpsamplerUNet] After input_proj: {x.shape}')
            print(f'[UpsamplerUNet] input_proj x range: ({x.min().item():.4f}, {x.max().item():.4f})')

        # Step 3. Downsampling
        skip_connections = []
        for i, down in enumerate(self.downsample_layers):
            for res_block in self.encoder_res_blocks[i]:
                x = res_block(x, cond_emb)
                if self.encoder_attn_blocks[i] is not None:
                    x = self.encoder_attn_blocks[i](x)
            skip_connections.append(x)
            if self.debug:
                print(f'[UpsamplerUNet] After DownsampleBlock {i}, shape: {x.shape}')
            x = down(x)
            if self.debug:
                print(f'[UpsamplerUNet] After downsample {i}, range: ({x.min().item():.4f}, {x.max().item():.4f})')

        # Step 4. Bottleneck
        x = self.bottleneck_blocks[0](x, cond_emb)  # ResidualBlock
        x = self.bottleneck_blocks[1](x)  # SelfAttention2d
        x = self.bottleneck_blocks[2](x, cond_emb)  # ResidualBlock
        if self.debug:
            print(f'[UpsamplerUNet] Bottleneck range: ({x.min().item():.4f}, {x.max().item():.4f})')

        # Step 5. Upsampling
        for i, up in enumerate(self.upsample_layers):
            skip = skip_connections.pop()
            x = up(x, skip)
            if self.debug:
                print(f'[UpsamplerUNet] After upsample {i}, range: ({x.min().item():.4f}, {x.max().item():.4f})')
            for res_block in self.decoder_res_blocks[i]:
                x = res_block(x, cond_emb)
                if self.decoder_attn_blocks[i] is not None:
                    x = self.decoder_attn_blocks[i](x)

        # Step 6. Output projection
        out = self.output_proj(x)
        assert out.shape == (B, self.out_channels, H, W), \
            f'[UpsamplerUNet] Output shape mismatch: expected ({B}, {self.out_channels}, {H}, {W}), got {out.shape}'

        if self.debug:
            print(f'[UpsamplerUNet] Output range: ({out.min().item():.4f}, {out.max().item():.4f})')
            print(f'[UpsamplerUNet] Output mean: {out.mean().item():.4f}, std: {out.std().item():.4f}')

        return out