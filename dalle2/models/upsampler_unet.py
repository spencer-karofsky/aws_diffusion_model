import torch
import torch.nn as nn
import torch.nn.functional as F
from dalle2.models.decoder_unet import DecoderUNet

class UpsamplerUNet(nn.Module):
    """
    U-Net for 64→128 upsampling diffusion.
    Conditions on a low-res (64×64) image and optional CLIP embedding.
    """
    def __init__(
        self,
        device: torch.device,
        image_size: int = 128,
        in_channels: int = 3,
        conditional_embedding_dim: int = 512,
        base_channels: int = 64,
        channel_multipliers=(1, 2, 4),
        attention_resolutions=(16,),
        residual_blocks: int = 2,
        use_text_condition: bool = True,
        debug: bool = False
    ):
        super().__init__()
        self.device = device
        self.use_text_condition = use_text_condition
        self.debug = debug

        # Low-res conditioning encoder
        self.lowres_encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU()
        )

        # Reuse the existing DecoderUNet for 128×128 processing
        # but with extra input channels to fuse lowres features.
        self.unet = DecoderUNet(
            channel_multipliers=channel_multipliers,
            attention_resolutions=attention_resolutions,
            device=device,
            image_size=image_size,
            in_channels=in_channels + base_channels,   # concat lowres features
            conditional_embedding_dim=conditional_embedding_dim,
            base_channels=base_channels,
            residual_blocks=residual_blocks,
            debug=debug
        )

    def forward(self, x_t, x_low, z_img, t_emb):
        """
        Args:
            x_t: Noisy 128×128 image (B,3,128,128)
            x_low: Conditioning 64×64 image (B,3,64,64)
            z_img: CLIP image embedding (B,512)
            t_emb: Time embedding (B,512)
        """
        # Upsample low-res conditioning to match target resolution
        x_low_up = F.interpolate(x_low, size=x_t.shape[-2:], mode='bilinear', align_corners=False)
        low_feat = self.lowres_encoder(x_low_up)

        # Concatenate low-res features and noisy high-res image
        x_in = torch.cat([x_t, low_feat], dim=1)

        # Forward through base U-Net
        return self.unet(x_in, z_img=z_img, t_emb=t_emb)
