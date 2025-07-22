"""
test_decoder_unet.py: Tests DecoderUNet

Usage (CLI, from Root Project Directory):
    PYTHONPATH=. python -m dalle2.tests.test_decoder_unet

Author:
    * Spencer Karofsky (https://github.com/spencer-karofsky)
"""
import torch
import unittest
from dalle2.models.decoder_unet import DecoderUNet

class DecoderUNetTest(unittest.TestCase):
    def setUp(self):
        """
        Set up a minimal UNet for testing (fast to run)
        """
        self.B, self.C, self.H, self.W = 2, 3, 32, 32
        self.cond_dim = 512
        self.device = torch.device('cpu')

        self.model = DecoderUNet(
            channel_multipliers=(1, 2),
            attention_resolutions=(8,),
            device=self.device,
            image_size=self.H,
            in_channels=self.C,
            conditional_embedding_dim=self.cond_dim,
            base_channels=16,
            residual_blocks=1,
            debug=False
        ).to(self.device)

        self.x_t = torch.randn(self.B, self.C, self.H, self.W).to(self.device)
        self.z_img = torch.randn(self.B, self.cond_dim).to(self.device)
        self.t_emb = torch.randn(self.B, self.cond_dim).to(self.device)

        self.out = self.model(self.x_t, self.z_img, self.t_emb)

    def test_output_shape(self):
        """
        Validate output shape is (B, C, H, W)
        """
        self.assertEqual(self.out.shape, (self.B, self.C, self.H, self.W))

    def test_output_type(self):
        """
        Validate output is a torch.Tensor
        """
        self.assertIsInstance(self.out, torch.Tensor)

    def test_requires_grad(self):
        """
        Validate autograd support
        """
        x_t = self.x_t.clone().detach().requires_grad_()
        out = self.model(x_t, self.z_img, self.t_emb)
        self.assertTrue(out.requires_grad)

    def test_no_nan_or_inf(self):
        """
        Ensure no NaNs or Infs in the output
        """
        self.assertFalse(torch.isnan(self.out).any(), "Output contains NaNs")
        self.assertFalse(torch.isinf(self.out).any(), "Output contains Infs")

if __name__ == '__main__':
    unittest.main()
