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
        self.pos_emb = nn.Parameter(torch.randn(1, 4, dim))

        # Transformer decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=attention_heads,
            dim_feedforward=dim * 4,
            activation='gelu',
            batch_first=True,
        )

        # Stack decoder layers
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=transformer_blocks
        )

        # Optional final linear projection back to dim
        self.output_proj = nn.Linear(dim, dim)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_m = nn.LayerNorm(dim)
    
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
        if torch.isnan(z_T).any():
            raise Exception('[PriorTransformer] NaNs in z_T')
        if torch.isnan(z_txt).any():
            raise Exception('[PriorTransformer] NaNs in z_txt')
        if torch.isnan(t_emb).any():
            raise Exception('[PriorTransformer] NaNs in t_emb')

        assert t_emb.size(0) == z_T.size(0), f'[PriorTransformer] t_emb ({t_emb.size(0)}) has different batch size than z_T ({z_T.size(0)})'
        B = t_emb.size(0)

        # Before concatenating z_T and t_emb, validate they both have the same shape
        if self.debug:
            print(f'[PriorTransformer] z_T actual shape: {z_T.shape}')
            print(f'[PriorTransformer] z_T expected shape: ({B}, 512)')
            print(f'[PriorTransformer] t_emb actual shape: {t_emb.shape}')
            print(f'[PriorTransformer] t_emb expected shape: ({B}, 512)')

        # Add timestep embedding to image embedding
        query = z_T + t_emb

        # Add sequence dimension to query (batch, 1, dim)
        query = query.unsqueeze(1) # (B, 1, 512)

        if z_txt.dim() == 2:
            # Single token, add seq dim
            memory = z_txt.unsqueeze(1)  # (B, 1, 512)
            pos_slice = self.pos_emb[:, 1:2, :]
            expected_seq_len = 1
        else:
            # Multiple tokens, no unsqueeze
            memory = z_txt # (B, seq_len, 512)
            expected_seq_len = z_txt.size(1)
            pos_slice = self.pos_emb[:, 1:1 + expected_seq_len, :]
        
        assert query.shape == (B, 1, 512), f'[PriorTransformer] query shape {query.shape} must be ({B}, 1, 512)'
        assert memory.shape == (B, expected_seq_len, 512), f'[PriorTransformer] memory shape {memory.shape} must be ({B}, {expected_seq_len}, 512)'
        assert pos_slice.shape == (1, expected_seq_len, 512), f'[PriorTransformer] pos_slice shape {pos_slice.shape} must match memory shape {memory.shape}'

        # Add positional encodings
        query = query + self.pos_emb[:, 0:1, :]
        memory = memory + pos_slice

        if self.debug:
            print(f'[PriorTransformer] query actual shape: {query.shape}')
            print(f'[PriorTransformer] query expected shape: ({B}, 1, 512)')
    
            print(f'[PriorTransformer] memory actual shape: {memory.shape}')
            print(f'[PriorTransformer] memory expected shape: ({B}, {expected_seq_len}, 512)')

        # For debugging
        query_nan = torch.isnan(query).any()
        if query_nan:
            raise Exception('[PriorTransformer] NaNs in query')
        memory_nan = torch.isnan(memory).any()
        if memory_nan:
            raise Exception('[PriorTransformer] NaNs in memory')

        query = self.norm_q(query)
        memory = self.norm_m(memory)

        # For debugging
        if not query_nan and torch.isnan(query).any():
            raise Exception('[PriorTransformer] NaNs in query after applying nn.LayerNorm')
        if not memory_nan and torch.isnan(memory).any():
            raise Exception('[PriorTransformer] NaNs in memory after applying nn.LayerNorm')

        # Decode (generate) the image embeddings
        decoded = self.transformer_decoder(tgt=query, memory=memory)

        # For debugging
        decoded_nan = torch.isnan(decoded).any()
        if decoded_nan:
            raise Exception('[PriorTransformer] NaNs in decoded after applying nn.TransformerDecoder')

        # Project output
        if decoded.dim() == 4:
            # e.g., (B, 1, 3, 512) → (B, 3, 512)
            if self.debug:
                print(f'[PriorTransformer] Decoded had 4 dims, squeezing and averaging across token dim: {decoded.shape}')
            decoded = decoded.squeeze(1).mean(dim=1)  # (B, 3, 512) → (B, 512)
        elif decoded.dim() == 3 and decoded.shape[1] == 1:
            # (B, 1, 512) → (B, 512)
            decoded = decoded.squeeze(1)

        eps_pred = self.output_proj(decoded)
        
        if self.debug:
            print(f'[PriorTransformer] eps_pred shape: {eps_pred.shape}')

        return eps_pred
