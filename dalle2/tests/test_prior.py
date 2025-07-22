"""
test_prior.py: Tests Prior

Usage (CLI, from Root Project Directory):
    PYTHONPATH=. python -m dalle2.tests.test_prior

Author:
    * Spencer Karofsky (https://github.com/spencer-karofsky)
"""
import torch
import unittest
from dalle2.models.prior import Prior

class PriorTest(unittest.TestCase):
    def setUp(self):
        """
        Set up mocked Prior
        """
        self.B = 4
        self.D = 512
        self.device = torch.device('cpu')

        self.prior = Prior(device=self.device, debug=False)

        self.z_txt = torch.randn(self.B, 3, self.D)
        self.z_T = torch.randn(self.B, self.D)
        self.t = torch.randint(low=1, high=1000, size=(self.B,), dtype=torch.int64)

    def test_forward_shape(self):
        """
        Validate correct output shape (B, D)
        """
        eps_pred = self.prior(self.z_txt, self.t, self.z_T)
        self.assertEqual(eps_pred.shape, (self.B, self.D))

    def test_forward_type(self):
        """
        Validate correct output type (torch.Tensor)
        """
        eps_pred = self.prior(self.z_txt, self.t, self.z_T)
        self.assertIsInstance(eps_pred, torch.Tensor)

    def test_forward_requires_grad(self):
        """
        Validate PyTorch's automatic differentiation works correctly
        """
        z_T = self.z_T.clone().detach().requires_grad_()
        z_txt = self.z_txt.clone().detach().requires_grad_()
        eps_pred = self.prior(z_txt, self.t, z_T)
        self.assertTrue(eps_pred.requires_grad)

    def test_forward_no_nan_inf(self):
        """
        Validate no NaNs or infs
        """
        eps_pred = self.prior(self.z_txt, self.t, self.z_T)
        self.assertFalse(torch.isnan(eps_pred).any(), 'Output contains NaNs')
        self.assertFalse(torch.isinf(eps_pred).any(), 'Output contains Infs')

    def test_forward_flexible_z_txt_shapes(self):
        """
        Test forward with different shapes
        """
        # Test (B, 1, 512)
        eps_pred_3d = self.prior(self.z_txt[:, :1, :], self.t, self.z_T)
        self.assertEqual(eps_pred_3d.shape, (self.B, self.D))

        # Test (B, 1, 3, 512)
        eps_pred_4d = self.prior(self.z_txt.unsqueeze(1), self.t, self.z_T)
        self.assertEqual(eps_pred_4d.shape, (self.B, self.D))

    @torch.no_grad()
    def test_sample_shape(self):
        """
        Validate sample shape (B, D)
        """
        z_txt = torch.randn(self.B, 3, self.D) # fresh input
        z_hat_img = self.prior.sample(z_txt)
        self.assertEqual(z_hat_img.shape, (self.B, self.D))

    @torch.no_grad()
    def test_sample_no_nan_inf(self):
        """
        Validate no NaNs or infs after Prior.sample
        """
        z_txt = torch.randn(self.B, 3, self.D) # fresh input
        z_hat_img = self.prior.sample(z_txt)
        self.assertFalse(torch.isnan(z_hat_img).any(), 'Sample output contains NaNs')
        self.assertFalse(torch.isinf(z_hat_img).any(), 'Sample output contains Infs')


if __name__ == '__main__':
    unittest.main()