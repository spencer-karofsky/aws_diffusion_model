"""
shared_modules.py: Contains low-level modules used by the U-Net and Transformer. 

Description:
    * TODO

Classes:
    * ResidualBlock(nn.Module): Core U-Net block that enables gradient flow via skip connections.
    * DownsampleBlock(nn.Module): Reduces spatial resolution of feature maps.
    * UpsampleBlock(nn.Module): Increases spatial resolution of feature maps.
    * ConditioningProjector(nn.Module): Projects the conditioning inputs (e.g., the timestep and image embedding) into a shared latent space. 

References:
    * U-Net Paper: https://arxiv.org/pdf/1505.04597
    * Transformer Paper: https://arxiv.org/pdf/1706.03762
    * DALL·E 2 Paper: https://cdn.openai.com/papers/dall-e-2.pdf
    * DDPM Paper: https://arxiv.org/pdf/2006.11239
    * My U-Net Notes: https://github.com/spencer-karofsky/aws_diffusion_model/blob/main/dalle2/research_notes/U-net%202015.pdf or /dalle2_new/research_notes/U-net 2015.pdf
    * My Transformer Notes: https://github.com/spencer-karofsky/aws_diffusion_model/blob/main/dalle2/research_notes/Attention%20is%20All%20You%20Need.pdf or /dalle2_new/research_notes/Attention is All You Need.pdf
    * My DALL·E 2 Notes: https://github.com/spencer-karofsky/aws_diffusion_model/blob/main/dalle2/research_notes/DALL-E-2%202022.pdf or /dalle2/research_notes/DALL-E-2 2022.pdf
    * My DDPM Notes: https://github.com/spencer-karofsky/aws_diffusion_model/blob/main/dalle2/research_notes/DDPM%202020.pdf or /dalle2/research_notes/DDPM 2020.pdf
    
Author:
    * Spencer Karofsky (https://github.com/spencer-karofsky)
"""
# PyTorch imports
import torch
import torch.nn as nn

# Module imports


# Other imports


class ConditioningProjector(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int
    ):
        """
        Projects the Conditioning Vector.

        Example Usage:
            from shared_modules import ConditioningProjector

            t_emb = torch.randn(32, 512) # sinusoidal + MLP projected timestep embedding
            z_img = torch.randn(32, 512) # CLIP image embedding

            projector = ConditioningProjector(
                input_dim=512, # for each embedding
                hidden_dim=512 # output dim of the projector
            )

            cond = projector.forward(t_emb, z_img)  # (32, 512)

        Args:
            input_dim: the input vector dimensionality
            hidden_dim: output dimensionality of the projector
        """
        super().__init__()
        pass

    def forward(
            self,
            t_emb: torch.Tensor,
            z_img: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward Pass of ConditioningProjector:
            * Combines and transforms the timestep embedding and the CLIP image embedding.
        
        Args:
            t_emb: timestep embedding of shape (B, D), typically sinusoidal + MLP encoded.
            z_img: CLIP image embedding of shape (B, D), representing semantic content.
        
        Returns:
            a fused conditioning vector of shape (B, cond_dim), where cond_dim is the projection output dimensionality.
        """
        pass

class ResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            cond_dim: int
    ):
        """
        Defines the Residual Block, which Enables Gradient Flow via Skip Connections.

        Example Usage:
            from shared_modules import ResidualBlock

            x = torch.randn(32, 128, 32, 32) # (B, C, H, W)
            cond_emb = torch.randn(32, 512) # (B, D) conditioning vector

            res_block = ResidualBlock(
                in_channels=128,
                out_channels=128,
                cond_dim=512 # matches conditioning vector's dim
            )

            out = res_block.forward(x, cond_emb) # (32, 128, 32, 32)
        
        Args:
            in_channels: number of feature channels in the input
            out_channels: number of feature channels in the output
            cond_dim: the dimensionality of the conditioning vector
        """
        super().__init__()
        pass

    def forward(
            self,
            x: torch.Tensor,
            cond_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the ResidualBlock:
            * Applies convolution operations to the input feature map.
            * Injects conditioning information.
            * Adds a residual/skip connection from the input
            * Projects the conditioning vector with ConditioningProjector

        Args:
            x: the primary input, of shape (B, in_channels, H, W)
            cond_emb: the conditioning vector of shape (B, cond_dim)

        Returns:
            the output feature map, of shape (B, out_channels, H, W)
        """
        pass

class DownsampleBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int
    ):
        """
        Downsamples a latent space from one resolution to another.

        Example Usage:
            from shared_modules import DownsampleBlock

            x = torch.randn(32, 128, 64, 64) # (B, C, H, W)

            down = DownsampleBlock(
                in_channels=128,
                out_channels=256
            )

            x_down = down.forward(x) # (32, 256, 32, 32)
        
        Args:
            in_channels: the number of input channels
            out_channels: the number of output channels
        """
        super().__init__()
        pass

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward Pass of DownsampleBlock:
            * Reduces spatial resolution of the input feature map.
            * Increases or maintains the number of channels.
        
        Args:
            x: The input feature map of shape (B, in_channels, H, W)

        Returns:
            the downsampled feature map of shape (B, out_channels, H/2, W/2)
        """
        pass


class UpsampleBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int
    ):
        """
        Initializes the UpsampleBlock.

        Example Usage:
            from shared_modules import UpsampleBlock

            x = torch.randn(32, 256, 32, 32)  # (B, in_channels, H, W)

            skip = torch.randn(32, 256, 32, 32)

            upsample_block = UpsampleBlock(
                in_channels=256, # channels of x
                out_channels=128, # desired output channels after upsampling
            )

            out = upsample_block(x, skip) # (32, 128, 64, 64)

        Args:
            in_channels: number of channels in the input decoder feature map
            out_channels: number of channels in the upsampled output feature map
            method: Upsampling method; options include 'nearest' (followed by Conv) or 'transpose' (ConvTranspose2d).
        """
        super().__init__()
        pass

    def forward(
            self,
            x: torch.Tensor,
            skip: torch.Tensor

    ) -> torch.Tensor:
        """
        Forward pass of the UpsampleBlock:
            * Increases spatial resolution of the input feature map.
            * Fuse with skip connection from encoder path using ResidualBlock.
        
        Args:
            x: The decoder feature map to be upsampled, of shape (B, in_channels, H, W)
            skip: The encoder feature map to be fused, of shape (B, in_channels, H, W)
        
        Returns:
            the fused feature map of shape (B, out_channels, H*2, W*2), ready for residual processing
        """
        pass