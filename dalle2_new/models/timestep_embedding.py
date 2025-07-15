"""
timestep_embedding.py: Implements the decoder used by DALL·E 2.

Description:
    * TODO

Classes:
    * TimestepEmbedder(nn.Module): Sinusoidally-encodes a timestep vector, which is then passed through a Neural Network and outputs the Projected Timestep Embedding.

References:
    * TODO
    
Author:
    * Spencer Karofsky (https://github.com/spencer-karofsky)
"""
# PyTorch imports
import torch
import torch.nn as nn

# Module imports
from decoder_unet import DecoderUNet

# Other imports

class TimestepEmbedder(nn.Module):
    pass