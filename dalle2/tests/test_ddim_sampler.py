"""
test_ddim_sampling.py: Unit tests for DDIMSampler.

Usage (CLI, from Root Project Directory):
    PYTHONPATH=. python -m dalle2.tests.test_ddim_sampler

Author:
    * Spencer Karofsky (https://github.com/spencer-karofsky)
"""
import unittest
import torch
from dalle2.sampling.ddim_sampling import DDIMSampler
from dalle2.sampling.noise_scheduler import NoiseScheduler

class DummyDenoiser(torch.nn.Module):
    def forward(self, x_t, z_cond, t):
        # Just return scaled noise for testing
        return torch.randn_like(x_t) * 0.5

class DDIMSamplerTest(unittest.TestCase):
    def setUp(self):
        """
        Initializes a DDIMSampler with dummy model and scheduler.
        """
        self.device = torch.device('mps' if torch.mps.is_available() else 'cpu')
        self.scheduler = NoiseScheduler(T=1000)
        self.sampler = DDIMSampler(
            noise_scheduler=self.scheduler,
            num_inference_steps=10,
            eta=0.0
        )
        self.model = DummyDenoiser().to(self.device)

        # Input conditioning and shape
        self.B = 4
        self.D = 512
        self.shape = (self.B, self.D)
        self.z_cond = torch.randn((self.B, self.D), device=self.device)

    def test_sample_shape(self):
        """
        Validate output sample shape.
        """
        samples = self.sampler.sample(
            model=self.model,
            z_cond=self.z_cond,
            shape=self.shape
        )
        self.assertEqual(samples.shape, self.shape, msg=f"Expected shape {self.shape}, got {samples.shape}")

    def test_sample_no_nan_inf(self):
        """
        Ensure output contains no NaN or Inf values.
        """
        samples = self.sampler.sample(
            model=self.model,
            z_cond=self.z_cond,
            shape=self.shape
        )
        self.assertFalse(torch.isnan(samples).any(), msg="Output contains NaNs")
        self.assertFalse(torch.isinf(samples).any(), msg="Output contains Infs")

    def test_predict_x0_shape(self):
        """
        Validate shape of x0 prediction from noisy sample.
        """
        x_t = torch.randn(self.shape, device=self.device)
        eps_pred = torch.randn(self.shape, device=self.device)
        t = torch.randint(0, 1000, (self.B,), device=self.device)
        x0_pred = self.sampler._predict_x0(x_t, eps_pred, t)
        self.assertEqual(x0_pred.shape, self.shape, msg=f"Expected x0 shape {self.shape}, got {x0_pred.shape}")

if __name__ == '__main__':
    unittest.main()
