"""
test_prior_transformer.py: Tests PriorTransformer

Usage (CLI, from Root Project Directory):
    PYTHONPATH=. python -m dalle2.tests.test_prior_transformer

Author:
    * Spencer Karofsky (https://github.com/spencer-karofsky)
"""
import torch
import unittest
from dalle2.models.prior_transformer import PriorTransformer

class PriorTransformerTest(unittest.TestCase):
    def setUp(self):
        """
        Set up mocked Prior Transformer
        """
        self.B = 4
        self.D = 512
        self.device = torch.device('cpu')

        self.z_T = torch.randn(self.B, self.D)
        self.t_emb = torch.randn(self.B, self.D)
        self.z_txt_single = torch.randn(self.B, self.D) # Single-token case
        self.z_txt_multi = torch.randn(self.B, 3, self.D) # Multi-token case

        self.model = PriorTransformer(
            dim=self.D,
            transformer_blocks=2,
            attention_heads=8,
            debug=False
        ).to(self.device)

    def test_output_shape_single_token(self):
        """
        Validate the correct output shape of the Transformer (B, D)
        """
        out = self.model(self.z_txt_single, self.z_T, self.t_emb, self.device)
        self.assertEqual(out.shape, (self.B, self.D))

    def test_output_shape_multi_token(self):
        """
        Validate the correct output shape of the Transformer (B, D)
        """
        out = self.model(self.z_txt_multi, self.z_T, self.t_emb, self.device)
        self.assertEqual(out.shape, (self.B, self.D))

    def test_output_type(self):
        """
        Validate the output is a PyTorch Tensor
        """
        out = self.model(self.z_txt_single, self.z_T, self.t_emb, self.device)
        self.assertIsInstance(out, torch.Tensor)

    def test_requires_grad(self):
        """
        Validate the output uses PyTorch's automatic differentiation
        """
        z_T = torch.randn(self.B, self.D, requires_grad=True)
        t_emb = torch.randn(self.B, self.D, requires_grad=True)
        z_txt = torch.randn(self.B, 3, self.D, requires_grad=True)
        out = self.model(z_txt, z_T, t_emb, self.device)
        self.assertTrue(out.requires_grad)

    def test_no_nan_or_inf(self):
        """
        Validate no NaNs or infs
        """
        out = self.model(self.z_txt_single, self.z_T, self.t_emb, self.device)
        self.assertFalse(torch.isnan(out).any(), "Output contains NaNs")
        self.assertFalse(torch.isinf(out).any(), "Output contains Infs")

    def test_shape_mismatch_raises(self):
        """
        Validate shape mismatches raise exceptions 
        """
        bad_z_T = torch.randn(self.B + 1, self.D)
        with self.assertRaises(AssertionError):
            self.model(self.z_txt_single, bad_z_T, self.t_emb, self.device)

if __name__ == '__main__':
    unittest.main()
