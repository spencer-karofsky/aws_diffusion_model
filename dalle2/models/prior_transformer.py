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
            attention_heads: int,
            debug: bool = False
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
            debug: debug: outputs relevant information (useful for debugging)
        """
        super().__init__()

        self.debug = debug

        # Positional encoding
        self.pos_emb = nn.Parameter(torch.zeros(1, 4, dim))
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.05)

        self.txt_proj   = nn.Linear(dim, dim)
        self.pre_ln_q   = nn.LayerNorm(dim)
        self.pre_ln_m   = nn.LayerNorm(dim)
        self.query_mlp  = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.gamma_direct = nn.Parameter(torch.zeros(1))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=attention_heads,
            dim_feedforward=dim * 4,
            activation='gelu',
            batch_first=True,
            dropout=0.0,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=transformer_blocks)

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
            z_txt: the CLIP text embedding, of shape (B, 512)
            z_T: the noisy CLIP image embeddings we seek to denoise (noised by the DDPM forward diffusion process), of shape (B, 512)
            t_emb: the MLP-projected and sinusoidally-encoded timestep Tensor, of shape (B, 512)
            device: the PyTorch device (ideally GPU-accelerated)
        
        Returns:
            the predicted noise of the CLIP image embedding, eps_theta, of shape (B, 512)
        """
        # 1) Build query with SUM conditioning (z_T + t_emb + text)
        txt_pooled = z_txt.mean(dim=1)
        txt_cond  = self.txt_proj(txt_pooled)

        query_vec = z_T + t_emb + txt_cond
        query = query_vec.unsqueeze(1)

        # Memory from tokens
        if z_txt.dim() == 2:
            memory = z_txt.unsqueeze(1)
            expected_seq_len = 1
            pos_slice = self.pos_emb[:, 1:2, :]
        else:
            memory = z_txt
            expected_seq_len = z_txt.size(1)
            pos_slice = self.pos_emb[:, 1:1 + expected_seq_len, :]

        # Positional encodings
        query  = query  + self.pos_emb[:, 0:1, :]
        memory = memory + pos_slice

        # Pre-norm for stability
        query  = self.pre_ln_q(query)
        memory = self.pre_ln_m(memory)

        # 2) Cross-attention refinement
        decoded = self.transformer_decoder(tgt=query, memory=memory)
        decoded = decoded.squeeze(1)

        # 3) Gated direct residual path to prevent variance collapse
        eps_pred = self.output_proj(decoded) + self.gamma_direct * self.query_mlp(query_vec)

        return eps_pred
