"""
prior_transformer.py: Implements the prior's Transformer functionality.

Description:
    * The prior is trained by a decoder-only Transformer with causal attention.
        - A trained CLIP model encodes the text and images into latent space, so we only need a decoder.
        - Causal attention: attention mechanism that uses masking to block the transformer from attending to future tokens (when predicting future tokens).

Classes:
    * PriorTransformer(nn.Module): Decoder-only Transformer with Causal Attention.

References:
    * Transformer Paper: https://arxiv.org/pdf/1706.03762
    * My Transformer Notes: https://github.com/spencer-karofsky/aws_diffusion_model/blob/main/dalle2/research_notes/Attention%20is%20All%20You%20Need.pdf or /dalle2_new/research_notes/Attention is All You Need.pdf
    
Author:
    * Spencer Karofsky (https://github.com/spencer-karofsky)
"""
# PyTorch imports
import torch
import torch.nn as nn

class PriorTransformer(nn.Module):
    def __init__(
            self,
            dim: int,
            transformer_blocks: int,
            attention_heads: int
    ):
        """
        Initializes the Decoder-Only Transformer for the Prior.

        Example Usage:
            from prior_transformer import PriorTransformer
            
            transformer = PriorTransformer(512, 6, 8)
            transformer.forward(
                z_txt=z_txt,
                z_T=z_T,
                t_emb=t_emb,
                device=self.device
            )
        
        Args:
            dim: the Transformer dimensionality
            transformer_blocks: number of transformer blocks to stack in a module
            attention_heads: the number of attention heads
        """
        super().__init__()

        # Positional encoding
        self.pos_emb = nn.Parameter(torch.randn(1, 2, dim))

        # Transformer decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=attention_heads,
            dim_feedforward=dim * 4,
            activation='gelu',
            batch_first=True,
        )

        # Stack decoder layers
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=transformer_blocks
        )

        # Optional final linear projection back to dim
        self.output_proj = nn.Linear(dim, dim)
    
    def forward(
            self,
            z_txt: torch.Tensor,
            z_T: torch.Tensor,
            t_emb: torch.Tensor,
            device: torch.device

    ) -> torch.Tensor:
        """
        Forward Pass of PriorTransformer:
            1. Add timestep embedding to the image embedding, z_T.
            2. Stack query (z_T + t_emb) and key/value (z_txt) with positional encodings.
            3. Decode using TransformerDecoder.
            4. Project output back to latent dimension.
        
        Args:
            z_txt: the CLIP text embedding
            z_T: the noisy CLIP image embeddings we seek to denoise (noised by the DDPM forward diffusion process), of shape (B, 512)
            t_emb: the MLP-projected and sinusoidally-encoded timestep Tensor
            device: the PyTorch device (ideally GPU-accelerated)
        
        Returns:
            the predicted noise of the CLIP image embedding, eps_theta, of shape (B, 512)
        """
        # Add timestep embedding to the image embedding, z_T.
        query = z_T + t_emb

        # Stack query (z_T + t_emb) and key/value (z_txt) with positional encodings.
        query = query.unsqueeze(1) + self.pos_emb[:, 0, :]
        memory = z_txt.unsqueeze(1) + self.pos_emb[:, 1, :]

        # Decode using TransformerDecoder.
        decoded = self.decoder(tgt=query, memory=memory)

        # Project output back to latent dimension.
        eps_pred = self.output_proj(decoded.squeeze(1))

        return eps_pred