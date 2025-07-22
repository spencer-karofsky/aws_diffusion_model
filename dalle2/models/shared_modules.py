"""
shared_modules.py: Contains low-level modules used by the U-Net and Transformer. 

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

class ConditioningProjector(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            debug: bool = False
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
            debug: optionally prints expected vs. actual Tensor shapes; useful for debugging
        """
        super().__init__()

        self.debug = debug

        # Initialize MLP
        self.projection_mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Store useful debugging dimensions for ConditioningProjector.forward
        if self.debug:
            self.hidden_dim = hidden_dim

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
        # Ensure t_emb and z_img have the same B and D dims
        B_t, D_t = t_emb.size(0), t_emb.size(-1)
        B_z, D_z = z_img.size(0), z_img.size(-1)
        assert B_t == B_z and D_t == D_z, f't_emb ({t_emb.size()}) != z_img ({z_img.size})'

        # Squeeze any extra dims
        extra_dims = t_emb.dim() - 2
        for _ in range(extra_dims):
           t_emb = t_emb.squeeze(1)
        
        extra_dims = z_img.dim() - 2
        for _ in range(extra_dims):
           z_img = z_img.squeeze(1)

        if self.debug:
            print(f'[ConditioningProjector] t_emb actual shape: {t_emb.size()}')
            print(f'[ConditioningProjector] t_emb expected shape: [{B_t}, {D_t}]\n')
            print(f'[ConditioningProjector] z_img actual shape: {z_img.size()}')
            print(f'[ConditioningProjector] z_img expected shape: [{B_z}, {D_z}]\n')

        # Concat t_emb and z_img, resulting in shape (B, 2D)
        x = torch.cat([t_emb, z_img], dim=-1)
        # Expected output shape: (B, 2D)
        if self.debug:
            print(f'[ConditioningProjector] x actual shape: {x.size()}')
            print(f'[ConditioningProjector] x expected shape: [{B_t}, {D_t * 2}]\n')

        x_proj = self.projection_mlp(x)
        # Expected output shape: (B, hidden_dim)
        if self.debug:
            print(f'[ConditioningProjector] x actual shape: {x_proj.size()}')
            print(f'[ConditioningProjector] x expected shape: [{B_t}, {self.hidden_dim}]\n')

        return x_proj

class ResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            cond_dim: int,
            debug: bool = False
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
            debug: optionally prints expected vs. actual Tensor shapes; useful for debugging
        """
        super().__init__()

        # Save params for debugging
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_dim = cond_dim
        self.debug = debug
        
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.cond_proj = nn.Linear(cond_dim, out_channels)

        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

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
        # Validate params against expected inputs
        assert x.dim() == 4, f'[ResidualBlock] x ({x.size()}) expected to have 4 dims: (B, in_channels, H, W)'
        assert cond_emb.dim() == 2, f'[ResidualBlock] cond_emb ({cond_emb.size()}) expected to have 2 dims: (B, cond_dim)'
        
        x_B, x_in_ch, x_H, x_W = x.size(0), x.size(1), x.size(2), x.size(3)
        cond_emb_B, cond_emb_dim = cond_emb.size(0), cond_emb.size(1)

        assert x_B == cond_emb_B, f'[ResidualBlock] x and cond_emb have different batch sizes ({x.size(0)} vs. {cond_emb.size(0)})'
        assert x_in_ch == self.in_channels, f'[ResidualBlock] input channels differ between class defintion ({self.in_channels}) and passed in ({x_in_ch})'
        assert cond_emb_dim == self.cond_dim, f'[ResidualBlock] conditional embedding dim differs between class defintion ({self.cond_dim}) and passed in ({cond_emb_dim})'
        

        if self.debug:
            x_size = x.size()
            print(f'[ResidualBlock] x input size: {x_size}\n')

        residual = self.skip(x)

        if self.debug:
            print(f'[ResidualBlock] output residual actual size: {residual.size()}')
            print(f'[ResidualBlock] output residual expected size: {x_size}\n')

        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv1(x)

        if self.debug:
            print(f'[ResidualBlock] output x actual size: {x.size()}')
            print(f'[ResidualBlock] output x expected size: {x_size}\n')

        # Inject conditioning (broadcasted)
        cond = self.cond_proj(cond_emb) # (B, out_channels)
        if self.debug:
            print(f'[ResidualBlock] conditional projection actual output size: {cond.size()}')
            print(f'[ResidualBlock] conditional projection expected output size: ({x_B}, {self.out_channels})\n')

        # Dynamically squeeze extra dims that would otherwise cause a shape mismatch error
        extra_dims = cond.dim() - 2
        for _ in range(extra_dims):
            cond = cond.squeeze(1)
        
        while cond.dim() < x.dim():
            cond = cond.unsqueeze(-1)

        x = x + cond
        if self.debug:
            print(f'[ResidualBlock] x actual output size: {x.size()}')
            print(f'[ResidualBlock] x expected output size: ({x_B, self.out_channels}, {x_H}, {x_W})\n')

        x = self.norm2(x)
        x = self.act2(x)
        x = self.conv2(x)

        if self.debug:
            print(f'[ResidualBlock] x actual output size: {x.size()}')
            print(f'[ResidualBlock] x expected output size: ({x_B, self.out_channels}, {x_H}, {x_W})\n')

        out = x + residual

        if self.debug:
            print(f'[ResidualBlock] x actual output size: {x.size()}')
            print(f'[ResidualBlock] x expected output size: ({x_B, self.out_channels}, {x_H}, {x_W})\n')

        return out

class DownsampleBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            debug: bool = False
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
            debug: optionally prints expected vs. actual Tensor shapes; useful for debugging
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.debug = debug
        
        self.downsample = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1
        )

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
        # self.debug = False
        # Validate dimensions
        assert x.dim() == 4, f'[DownsampleBlock] Expected 4D input (B, C, H, W), got {x.shape}'
        B, C, H, W = x.shape
        assert C == self.in_channels, f'[DownsampleBlock] Input channel mismatch: expected {self.in_channels}, got {C}'

        if self.debug:
            print(f'[DownsampleBlock] Input shape: {x.shape}')

        out = self.downsample(x)

        if self.debug:
            print(f'[DownsampleBlock] Output shape: {out.shape}')
            print(f'[DownsampleBlock] Expected output shape: ({B}, {self.out_channels}, {H//2}, {W//2})\n')

        return out

class UpsampleBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            debug: bool = False
    ):
        """
        Initializes the UpsampleBlock.

        Example Usage:
            from shared_modules import UpsampleBlock

            x = torch.randn(32, 256, 32, 32) # (B, in_channels, H, W)

            skip = torch.randn(32, 256, 32, 32)

            upsample_block = UpsampleBlock(
                in_channels=256, # channels of x
                out_channels=128, # desired output channels after upsampling
            )

            out = upsample_block(x, skip) # (32, 128, 64, 64)

        Args:
            in_channels: number of channels in the input decoder feature map
            out_channels: number of channels in the upsampled output feature map
            debug: optionally prints expected vs. actual Tensor shapes; useful for debugging
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.debug = debug

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

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
        # self.debug = False
        assert x.dim() == 4, f'[UpsampleBlock] x expected shape (B, C, H, W), got {x.shape}'
        assert skip.dim() == 4, f'[UpsampleBlock] skip expected shape (B, C, H, W), got {skip.shape}'
        B, C_in, H, W = x.shape

        x_up = self.upsample(x)

        assert x_up.shape == skip.shape, (
            f'[UpsampleBlock] shape mismatch after upsampling:\n'
            f'Upsampled x: {x_up.shape}\n'
            f'Skip       : {skip.shape}'
        )

        if self.debug:
            print(f'[UpsampleBlock] x input shape       : {x.shape}')
            print(f'[UpsampleBlock] x upsampled shape   : {x_up.shape}')
            print(f'[UpsampleBlock] skip connection shape: {skip.shape}')
            print(f'[UpsampleBlock] fused output shape  : {x_up.shape}\n')

        return x_up + skip