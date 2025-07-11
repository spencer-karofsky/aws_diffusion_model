"""
clip_embedder: Encapsulates a pretrained CLIP model from HuggingFace in a class.

Description and Purpose:
    - We need a way to convert raw images and text captions into a format that DALL·E 2 can understand.
    - The diffusion prior needs two tensors from CLIP (among other things):
        1. a [B, 77, 512]-Tensor CLIP embedding of the text caption (token-level) to capture fine-grained details
        2. a [B, 512]-Tensor CLIP embedding of the text caption (entire caption) to capture the global context
    - Given a list of text captions or images, the CLIP model (introduced by OpenAI) converts them into a numerical vector.

Usage:
    from clip_embedder import CLIPEmbedder
    clip_embedder = CLIPEmbedder(...)
    text_embedding = clip_embedder.encode_text(['dog', 'cat', 'airplane'])
    image_embedding = clip_embedder.encode_image([img])
    # compute cosine similarity of image_embedding with each text class and pass resulting vector through softmax

Class:
    - CLIPEmbedder: embeds images and text using a pretrained CLIP model

Author:
    - Spencer Karofsky (https://github.com/spencer-karofsky)
"""
import open_clip
import torch
from PIL import Image
from typing import List, Union

class CLIPEmbedder:
    def __init__(
            self,
            model_name: str = "ViT-B-32",
            pretrained: str = "laion2b_s34b_b79k",
            device: Union[str, torch.device] = None
        ):
        """Initializes CLIP
        Docs:
            https://huggingface.co/docs/hub/en/open_clip
        Args:
            model_name: model architecture (e.g. "ViT-B-32")
            pretrained: which pretrained weights (e.g. "laion2b_s34b_b79k")
            device: computation device (defaults to GPU if available)
        """
        if not device:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.to(self.device).eval()
    
    def encode_text_tokens(self, texts: List[str]) -> torch.Tensor:
        """Converts a prompt to a per-token CLIP embedding.
        - Embeds fine-grained details of the input caption.
        - Input (1) to the diffusion prior.
        Args:
            texts: list of text strings (prompts) (e.g., ['a dog sitting on a beach', 'an astronaut riding a horse'])
        Returns:
            [B, 77, D] tensor of text embeddings (one 512-dimension vector per token).
        """
        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            # Run the transformer encoder directly
            x = self.model.token_embedding(tokens)  # [B, 77, D]
            x = x + self.model.positional_embedding  # Add positional embeddings
            x = self.model.transformer(x)  # Run through transformer
        return x
    
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Converts a prompt to CLIP text embedding.
        - Embeds global context of the input caption.
        - Input (2) to the diffusion prior.
        Args:
            texts: list of text strings (prompts) (e.g., ['a dog sitting on a beach', 'an astronaut riding a horse'])
        Returns:
            [B, D] tensor of text embeddings (one 512-dimension vector per prompt).
        """
        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def encode_image(self, images: List[Image.Image]) -> torch.Tensor:
        """Converts images to CLIP image embeddings.
        Args:
            images: list of PIL images
        Returns:
            [B, D] tensor of image embeddings
        """
        batch = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(batch)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features