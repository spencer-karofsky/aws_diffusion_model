"""
test_dalle2.py: Tests DALLe2 generate method

Usage (CLI, from Root Project Directory):
    PYTHONPATH=. python -m dalle2.tests.test_dalle2

Author:
    * Spencer Karofsky (https://github.com/spencer-karofsky)
"""
import torch
import unittest
from unittest.mock import MagicMock
from dalle2.models.dalle2 import DALLe2
from dalle2.models.prior import Prior
from dalle2.models.decoder import Decoder
from dalle2.models.clip_encoding import CLIPEncoder
from dalle2.sampling.ddim_sampling import DDIMSampler

class DALLe2Test(unittest.TestCase):
    def setUp(self):
        """
        Set up mocked DALLe2 with dummy prior, decoder, clip encoder, and samplers
        """
        self.B = 2
        self.D = 512
        self.H = 128
        self.W = 128
        self.device = torch.device('cpu')

        # Mock CLIPEncoder with encode_text_multilayer
        self.mock_clip_encoder = MagicMock(spec=CLIPEncoder)
        self.mock_clip_encoder.encode_text_multilayer.return_value = torch.randn(self.B, 3, self.D)

        # Mock Prior.sample to return (B, 512) tensor
        self.mock_prior = MagicMock(spec=Prior)
        self.mock_prior.sample.return_value = torch.randn(self.B, self.D)

        # Mock Decoder.sample to return (B, 3, H, W) tensor
        self.mock_decoder = MagicMock(spec=Decoder)
        self.mock_decoder.sample.return_value = torch.randn(self.B, 3, self.H, self.W)

        # Dummy samplers (not used since prior.sample and decoder.sample are mocked)
        self.mock_prior_sampler = MagicMock(spec=DDIMSampler)
        self.mock_decoder_sampler = MagicMock(spec=DDIMSampler)

        # Instantiate DALLe2 with mocks
        self.dalle = DALLe2.__new__(DALLe2)  # bypass __init__
        self.dalle.clip_encoder = self.mock_clip_encoder
        self.dalle.prior = self.mock_prior
        self.dalle.decoder = self.mock_decoder
        self.dalle.prior_sampler = self.mock_prior_sampler
        self.dalle.decoder_sampler = self.mock_decoder_sampler
        self.dalle.H = self.H
        self.dalle.W = self.W
        self.dalle.t = torch.linspace(999, 0, steps=30, dtype=torch.long)
        self.dalle.debug = False

    def test_generate_output_shape(self):
        """
        Test that generate returns images with shape (B, 3, H, W)
        """
        captions = ["a cat", "a dog"]

        generated_images = self.dalle.generate(captions)

        # Assert mocks called correctly
        self.mock_clip_encoder.encode_text_multilayer.assert_called_once_with(captions)
        self.mock_prior.sample.assert_called_once_with(
            z_txt=self.mock_clip_encoder.encode_text_multilayer.return_value,
            sampler=self.mock_prior_sampler,
            steps=len(self.dalle.t)
        )
        self.mock_decoder.sample.assert_called_once_with(
            z_img=self.mock_prior.sample.return_value
        )

        # Assert output shape
        self.assertIsInstance(generated_images, torch.Tensor)
        self.assertEqual(generated_images.shape, (self.B, 3, self.H, self.W))


if __name__ == '__main__':
    unittest.main()
