"""
test_decoder.py: Tests Decoder

Usage (CLI, from Root Project Directory):
    PYTHONPATH=. python -m dalle2.tests.test_decoder

Author:
    * Spencer Karofsky (https://github.com/spencer-karofsky)
"""
import torch
import unittest
from dalle2.models.decoder import Decoder

class DecoderTest(unittest.TestCase):
    def setUp(self):
        """
        Set up mocked Decoder
        """
        self.B = 2
        self.D = 512
        self.H = 128
        self.W = 128
        self.device = torch.device('mps')

        self.decoder = Decoder(device=self.device, debug=False)

        self.x_t = torch.randn(self.B, 3, self.H, self.W)
        self.z_img = torch.randn(self.B, self.D)
        self.t = torch.randint(low=1, high=1000, size=(self.B,), dtype=torch.int64)

    def test_forward_shape(self):
        """
        Validate correct output shape (B, 3, H, W)
        """
        eps_pred = self.decoder(self.x_t, self.z_img, self.t)
        self.assertEqual(eps_pred.shape, (self.B, 3, self.H, self.W))

    def test_forward_type(self):
        """
        Validate output type is torch.Tensor
        """
        eps_pred = self.decoder(self.x_t, self.z_img, self.t)
        self.assertIsInstance(eps_pred, torch.Tensor)

    def test_forward_requires_grad(self):
        """
        Validate PyTorch's automatic differentiation works correctly
        """
        x_t = self.x_t.clone().detach().requires_grad_()
        z_img = self.z_img.clone().detach().requires_grad_()
        eps_pred = self.decoder(x_t, z_img, self.t)
        self.assertTrue(eps_pred.requires_grad)

    def test_forward_no_nan_inf(self):
        """
        Validate no NaNs or infs in the forward pass output
        """
        eps_pred = self.decoder(self.x_t, self.z_img, self.t)
        self.assertFalse(torch.isnan(eps_pred).any(), 'Output contains NaNs')
        self.assertFalse(torch.isinf(eps_pred).any(), 'Output contains infs')

    @torch.no_grad()
    def test_sample_shape(self):
        """
        Validate sample method returns correct output shape (B, 3, H, W)
        """
        z_img = torch.randn(self.B, self.D)
        sample_img = self.decoder.sample(z_img)
        self.assertEqual(sample_img.shape, (self.B, 3, self.H, self.W))

    @torch.no_grad()
    def test_sample_no_nan_inf(self):
        """
        Validate no NaNs or infs after sampling
        """
        z_img = torch.randn(self.B, self.D)
        sample_img = self.decoder.sample(z_img)
        self.assertFalse(torch.isnan(sample_img).any(), 'Sample output contains NaNs')
        self.assertFalse(torch.isinf(sample_img).any(), 'Sample output contains infs')


if __name__ == '__main__':
    unittest.main()
