"""
clip_encoding.py: Runs inference on OpenCLIP's trained CLIP image and text embedder.

Description:
    * We need a way to convert raw images and text captions into a format that DALL·E 2 can understand.
    * The diffusion prior needs two tensors from CLIP (among other things):
        1. a [B, 77, 512]-Tensor CLIP embedding of the text caption (token-level) to capture fine-grained details
        2. a [B, 512]-Tensor CLIP embedding of the text caption (entire caption) to capture the global context
    * Given a list of text captions or images, the CLIP model (introduced by OpenAI) converts them into a numerical vector.

Classes:
    * CLIPEncoder: Encodes text captions and images.

References:
    * CLIP Paper: https://arxiv.org/pdf/2103.00020
    * My CLIP Notes: https://github.com/spencer-karofsky/aws_diffusion_model/blob/main/dalle2/research_notes/CLIP%202021.pdf or /dalle2/research_notes/CLIP 2021.pdf

Author:
    * Spencer Karofsky (https://github.com/spencer-karofsky)
"""
# PyTorch imports
import torch
import torch.nn as nn
from torchvision.transforms.functional import to_pil_image

# Other imports
import open_clip
from PIL import Image
from typing import List, Union

class CLIPEncoder(nn.Module):
    def __init__(
            self,
            model_name: str = "ViT-B-32",
            pretrained: str = "laion2b_s34b_b79k",
            device: Union[str, torch.device] = None
    ):
        """
        Initializes CLIP Encoder.

        Example Usage:
            from clip_embedder import CLIPEmbedder
            clip_embedder = CLIPEmbedder(...)
            text_embedding = clip_embedder.encode_text(['dog', 'cat', 'airplane'])
            image_embedding = clip_embedder.encode_image([img])

        Docs:
            https://huggingface.co/docs/hub/en/open_clip

        Args:
            model_name: model architecture (e.g. "ViT-B-32")
            pretrained: which pretrained weights (e.g. "laion2b_s34b_b79k")
            device: computation device (defaults to GPU if available)
        """
        super().__init__()
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
    
    def encode_text_tokens(
                self,
                texts: List[str]
    ) -> torch.Tensor:
        """
        Converts a prompt to a per-token CLIP embedding.
            * Embeds fine-grained details of the input caption.
            * Input (1) to the diffusion prior.

        Args:
            texts: list of text strings (prompts) (e.g., ['a dog sitting on a beach', 'an astronaut riding a horse'])

        Returns:
            [B, 77, 512] tensor of text embeddings (one 512-dimension vector per token).
        """
        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            # Run the transformer encoder directly
            x = self.model.token_embedding(tokens)
            x = x + self.model.positional_embedding
            x = self.model.transformer(x)
        return x
    
    def encode_text(
                self,
                texts: List[str]
    ) -> torch.Tensor:
        """
        Converts a prompt to CLIP text embedding.
            * Embeds global context of the input caption.
            * Input (2) to the diffusion prior.

        Args:
            texts: list of text strings (prompts) (e.g., ['a dog sitting on a beach', 'an astronaut riding a horse'])
        
        Returns:
            [B, 512] tensor of text embeddings (one 512-dimension vector per prompt)
        """
        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def encode_image(
                self,
                images: Union[List[Image.Image], torch.Tensor]
    ) -> torch.Tensor:
        """
        Converts images to CLIP image embeddings.

        Args:
            images: either a list of PIL images or a batch tensor

        Returns:
            [B, 512] CLIP image embeddings
        """
        if isinstance(images, torch.Tensor):
            # Convert batched tensor [B, 3, H, W] into list of PIL Images
            images = [to_pil_image(img) for img in images]

        batch = torch.stack([self.preprocess(img) for img in images]).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(batch)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        return image_features
    
    def encode_text_multilayer(
        self,
        texts: List[str],
        layers: List[int] = [-5, -3, -1]
    ) -> torch.Tensor:
        """
        Returns 3-layer text embedding from CLIP for each prompt.

        Args:
            texts: list of text strings (e.g., ['a cat', 'a dog'])
            layers: which transformer layers to extract (default = [-5, -3, -1])

        Returns:
            Tensor of shape [B, 3, 512]
        """
        # raise Exception('Check clip_encoding.py and likely use other method!')
        token_embeddings = self.encode_text_tokens(texts)  # [B, 77, 512]
        assert token_embeddings.dim() == 3 and token_embeddings.shape[1:] == (77, 512)

        start = token_embeddings[:, 0] # (B, 512)
        mean = token_embeddings.mean(dim=1) # (B, 512)
        end = token_embeddings[:, -1] # (B, 512)

        combined = torch.stack([start, mean, end], dim=1) # (B, 3, 512)
        return combined
    
    def encode_text_ensemble(
        self,
        texts: List[str],
        n: int = 3
    ) -> torch.Tensor:
        """
        Returns multiple stochastic CLIP text embeddings per prompt using dropout.

        Args:
            texts: list of text prompts
            n: number of stochastic embeddings per prompt

        Returns:
            Tensor of shape [B, n, 512]
        """
        tokens = self.tokenizer(texts).to(self.device)
        embeddings = []

        self.model.train() # Enables dropout for stochastic encoding
        for _ in range(n):
            with torch.no_grad():
                features = self.model.encode_text(tokens) # (B, 512)
                features /= features.norm(dim=-1, keepdim=True)
            embeddings.append(features)

        self.model.eval() # Reset back to eval mode
        return torch.stack(embeddings, dim=1) # (B, n, 512)

