"""
test_clip_encoding.py: Tests CLIPEncoder

Usage (CLI, from Root Project Directory):
    PYTHONPATH=. python -m dalle2.tests.test_clip_encoding

Author:
    * Spencer Karofsky (https://github.com/spencer-karofsky)
"""
# PyTorch imports
import torch
from torchvision.transforms import ToPILImage

# Module imports
from dalle2.models.clip_encoding import CLIPEncoder

# Other imports
import unittest

class CLIPEncoderTest(unittest.TestCase):
    def setUp(self):
        """
        Set up mocked CLIP Encoder
        """
        self.L, self.D = 77, 512
        self.captions = [
            'A cat wearing a spacesuit floating through a neon galaxy',
            'An abandoned castle overgrown with glowing mushrooms',
            'A rainy New York street at night, reflected in a puddle',
            'A robot playing chess with a child in a sunlit park',
            'A desert with giant floating jellyfish in the sky'
        ]

        self.C, self.H, self.W = 3, 32, 32
        to_pil = ToPILImage()
        self.images = [to_pil(torch.rand(self.C, self.H, self.W)) for _ in range(len(self.captions))]

        self.clip_encoder = CLIPEncoder()

        self.captions_clip = self.clip_encoder.encode_text(self.captions)
        self.captions_clip_tokens = self.clip_encoder.encode_text_tokens(self.captions)
        self.images_clip = self.clip_encoder.encode_image(self.images)

    def test_output_shape_singular(self):
        """
        Validate output shape (B, 512)
        """
        self.assertEqual(self.captions_clip.shape, (len(self.captions), self.D))

    def test_output_shape_tokens(self):
        """
        Validate output shape (B, 77, 512)
        """
        self.assertEqual(self.captions_clip_tokens.shape, (len(self.captions), self.L, self.D))
    
    def test_output_shape_images(self):
        """
        Validate output shape (B, 512)
        """
        self.assertEqual(self.captions_clip.shape, (len(self.images), self.D))
    
    def test_no_nan_inf(self):
        """
        Test neither the CLIP text or image embeddings contain any NaNs or infs
        """
        self.assertFalse(torch.isnan(self.captions_clip).any())
        self.assertFalse(torch.isnan(self.captions_clip_tokens).any())
        self.assertFalse(torch.isnan(self.images_clip).any())
    
if __name__ == '__main__':
    unittest.main()