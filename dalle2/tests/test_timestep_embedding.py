"""
test_timestep_embedding.py: Tests TimestepEmbedder

Usage (CLI, from Root Project Directory):
    PYTHONPATH=. python -m dalle2.tests.test_timestep_embedding

Author:
    * Spencer Karofsky (https://github.com/spencer-karofsky)
"""
# PyTorch imports
import torch
from torchvision.transforms import ToPILImage

# Module imports
from dalle2.models.timestep_embedding import TimestepEmbedder

# Other imports
import unittest

class TimestepEmbedderTest(unittest.TestCase):
    def setUp(self):
        """
        Set up mocked Timestep Embedder
        """
        self.batch_size = 8
        self.embed_dim = 512
        self.embedder = TimestepEmbedder(dim=self.embed_dim)
        self.timesteps = torch.randint(low=0, high=1000, size=(self.batch_size,), dtype=torch.int64)
        self.emb = self.embedder(self.timesteps)

    def test_output_shape(self):
        """
        Validate output shape (B, D)
        """
        self.assertEqual(self.emb.shape, (self.batch_size, self.embed_dim))

    def test_output_type(self):
        """
        Validate output is a torch.Tensor
        """
        self.assertIsInstance(self.emb, torch.Tensor)

    def test_requires_grad(self):
        """
        Validate output supports autograd
        """
        self.timesteps = self.timesteps.clone().float().requires_grad_()  # Simulate learnable input
        emb = self.embedder(self.timesteps)
        self.assertTrue(emb.requires_grad)

    def test_no_nan_or_inf(self):
        """
        Ensure no NaNs or Infs in output
        """
        self.assertFalse(torch.isnan(self.emb).any(), "Output contains NaNs")
        self.assertFalse(torch.isinf(self.emb).any(), "Output contains Infs")

    def test_extreme_timesteps(self):
        """
        Test timestep normalization for edge values
        """
        timesteps = torch.tensor([0.0, 1.0, 500.0, 999.0])
        emb = self.embedder(timesteps)
        self.assertEqual(emb.shape, (4, self.embed_dim))
    
if __name__ == '__main__':
    unittest.main()