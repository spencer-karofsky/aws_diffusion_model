"""
clip_text_embedder: Uses HuggingFace's pretrained CLIP model

Description and Purpose:
    - 

Usage:
    

Class:
    - CLIPEmbedder: embeds images and text
    

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
    
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Converts prompts to CLIP text embeddings.
        Args:
            texts: list of strings (prompts)
        Returns:
            [B, D] tensor of text embeddings
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