"""
test_shared_modules.py: Tests ConditioningProjector, ResidualBlock, DownsampleBlock and UpsampleBlock

Usage (CLI, from Root Project Directory):
    PYTHONPATH=. python -m dalle2.tests.test_shared_modules

Author:
    * Spencer Karofsky (https://github.com/spencer-karofsky)
"""
# PyTorch imports
import torch
from torch import nn

# Module imports
from dalle2.models.shared_modules import ConditioningProjector, ResidualBlock, DownsampleBlock, UpsampleBlock

# Other imports
import unittest

class ConditioningProjectorTest(unittest.TestCase):
    def setUp(self):
        """
        Set up mocked Conditioning Projection
        """
        self.batch_size = 8
        self.input_dim = 512
        self.hidden_dim = 512
        self.projector = ConditioningProjector(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            debug=False
        )
        t_emb = torch.randn(self.batch_size, self.input_dim)
        z_img = torch.randn(self.batch_size, self.input_dim)
        self.output = self.projector(t_emb, z_img)

    def test_output_shape(self):
        """
        Validate correct output shape (B, D)
        """
        self.assertEqual(self.output.shape, (self.batch_size, self.hidden_dim))
    
    def test_output_type(self):
        """
        Validate output datatype is a PyTorch Tensor
        """
        self.assertIsInstance(self.output, torch.Tensor)
    
    def test_requires_grad(self):
        """
        Validate the output uses PyTorch's automatic differentiation
        """
        t_emb = torch.randn(self.batch_size, self.input_dim, requires_grad=True)
        z_img = torch.randn(self.batch_size, self.input_dim, requires_grad=True)
        output = self.projector(t_emb, z_img)
        self.assertTrue(output.requires_grad)
    
    def test_no_nan_inf(self):
        """
        Validate no NaN's inf values
        """
        t_emb = torch.randn(self.batch_size, self.input_dim)
        z_img = torch.randn(self.batch_size, self.input_dim)
        output = self.projector(t_emb, z_img)
        self.assertFalse(torch.isnan(output).any(), 'Output contains NaNs')
        self.assertFalse(torch.isinf(output).any(), 'Output contains Infs')
    
    def test_shape_mismatch_raises(self):
        """
        Validate any shape mismatches raise an error
        """
        t_emb = torch.randn(self.batch_size, self.input_dim)
        z_img = torch.randn(self.batch_size + 1, self.input_dim)
        with self.assertRaises(AssertionError):
            self.projector(t_emb, z_img)
    
    def test_unsqueeze_extra_dims_handled(self):
        """
        Validate ConditioningProjector.forward gracefully squeezes extra uneccessary dimensions
        """
        t_emb = torch.randn(self.batch_size, 1, self.input_dim)
        z_img = torch.randn(self.batch_size, 1, self.input_dim)
        output = self.projector(t_emb, z_img)
        self.assertEqual(output.shape, (self.batch_size, self.hidden_dim))

class ResidualBlockTest(unittest.TestCase):
    def setUp(self):
        """
        Set up mocked Residual Block
        """
        self.B, self.C, self.H, self.W, self.D = 32, 128, 32, 32, 512

        self.x = torch.randn(self.B, self.C, self.H, self.W)
        self.cond_emb = torch.randn(self.B, self.D)

        self.res_block = ResidualBlock(
            in_channels=self.C,
            out_channels=self.C,
            cond_dim=self.D # matches conditioning vector's dim
        )

        self.out = self.res_block.forward(self.x, self.cond_emb)

    def test_output_shape(self):
        """
        Validate correct output shape (B, C, H, W)
        """
        self.assertEqual(self.out.shape, (self.B, self.C, self.H, self.W))
    
    def test_output_type(self):
        """
        Validate the output type is a PyTorch Tensor
        """
        self.assertIsInstance(self.out, torch.Tensor)
    
    def test_requires_grad(self):
        """
        Validate the output uses PyTorch's automatic differentiation
        """
        x = torch.randn(self.B, self.C, self.H, self.W, requires_grad=True)
        cond_emb = torch.randn(self.B, self.D, requires_grad=True)
        res_block = ResidualBlock(
            in_channels=self.C,
            out_channels=self.C,
            cond_dim=self.D # matches conditioning vector's dim
        )
        out = res_block.forward(x, cond_emb)
        self.assertTrue(out.requires_grad)
    
    def test_no_nan_inf(self):
        """
        Validate no NaN's or infinite values
        """
        self.assertFalse(torch.isnan(self.out).any(), 'Output contains NaNs')
        self.assertFalse(torch.isinf(self.out).any(), 'Output contains Infs')
    
    def test_shape_mismatch_raises(self):
        """
        Validate any shape mismatches raise an error
        """
        x = torch.randn(self.B + 1, self.C, self.H, self.W)
        res_block = ResidualBlock(
            in_channels=self.C,
            out_channels=self.C,
            cond_dim=self.D
        )
        with self.assertRaises(AssertionError):
            res_block(x, self.cond_emb)

class DownsampleBlockTest(unittest.TestCase):
    def setUp(self):
        """
        Set up mocked Downsample Block
        """
        self.B, self.H, self.W = 32, 32, 32
        self.in_channels, self.out_channels = 128, 256

        self.x = torch.randn(self.B, self.in_channels, self.H, self.W)
        self.down_block = DownsampleBlock(
            in_channels=self.in_channels,
            out_channels=self.out_channels
        )

        self.x_down = self.down_block.forward(self.x) # (B, out_channels, H/2, W/2)
        

    def test_output_shape(self):
        """
        Validate correct output shape (B, out_channels, H/2, W/2)
        """
        self.assertEqual(self.x_down.shape, (self.B, self.out_channels, self.H // 2, self.W // 2))
    
    def test_output_type(self):
        """
        Validate the output type is a PyTorch Tensor
        """
        self.assertIsInstance(self.x_down, torch.Tensor)
    
    def test_requires_grad(self):
        """
        Validate the output uses PyTorch's automatic differentiation
        """
        x = torch.randn(self.B, self.in_channels, self.H, self.W, requires_grad=True)
        down_block = DownsampleBlock(
            in_channels=self.in_channels,
            out_channels=self.out_channels
        )
        out = down_block.forward(x)
        self.assertTrue(out.requires_grad)
    
    def test_no_nan_inf(self):
        """
        Validate no NaN's or infinite values
        """
        self.assertFalse(torch.isnan(self.x_down).any(), 'Output contains NaNs')
        self.assertFalse(torch.isinf(self.x_down).any(), 'Output contains Infs')
    
    def test_shape_mismatch_raises(self):
        """
        Validate any shape mismatches raise an error
        """
        x = torch.randn(self.B, self.in_channels, self.H, self.W, requires_grad=True)
        down_block = DownsampleBlock(
            in_channels=self.in_channels + 1,
            out_channels=self.out_channels
        )
        with self.assertRaises(AssertionError):
            down_block(x)

class UpsampleBlockTest(unittest.TestCase):
    def setUp(self):
        """
        Set up mocked Upsample Block Manager
        """
        self.B, self.H, self.W = 2, 32, 32
        self.in_channels, self.out_channels = 128, 256

        self.x = torch.randn(self.B, self.in_channels, self.H, self.W)
        self.skip = torch.randn(self.B, self.out_channels, self.H * 2, self.W * 2)

        self.upsample_block = UpsampleBlock(
            in_channels=self.in_channels,
            out_channels=self.out_channels
        )

        self.out = self.upsample_block(self.x, self.skip)
        
    def test_output_shape(self):
        """
        Validate correct output shape (B, out_channels, H*2, W*2)
        """
        self.assertEqual(self.out.shape, (self.B, self.out_channels, self.H * 2, self.W * 2))
    
    def test_output_type(self):
        """
        Validate the output type is a PyTorch Tensor
        """
        self.assertIsInstance(self.out, torch.Tensor)
    
    def test_requires_grad(self):
        """
        Validate the output uses PyTorch's automatic differentiation
        """
        x = torch.randn(self.B, self.in_channels, self.H, self.W, requires_grad=True)
        skip = torch.randn(self.B, self.out_channels, self.H * 2, self.W * 2, requires_grad=True)
        up_block = UpsampleBlock(self.in_channels, self.out_channels)
        out = up_block(x, skip)
        self.assertTrue(out.requires_grad)
    
    def test_no_nan_inf(self):
        """
        Validate no NaN's or infinite values
        """
        self.assertFalse(torch.isnan(self.out).any(), "Output contains NaNs")
        self.assertFalse(torch.isinf(self.out).any(), "Output contains Infs")

    def test_shape_mismatch_raises(self):
        """
        Validate any shape mismatches raise an error
        """
        # Intentionally create shape mismatch in skip connection
        bad_skip = torch.randn(self.B, self.out_channels, self.H, self.W) # wrong H/W
        with self.assertRaises(AssertionError):
            self.upsample_block(self.x, bad_skip)

if __name__ == '__main__':
    unittest.main()