"""
ddim_model: an improvement of the DDPM image generation architecture, introduced in the paper: https://arxiv.org/pdf/2010.02502

Description and Purpose:
    - Unlike the DDPM, the Denoising Diffusion Implicit Models (DDIM) architecture introduces a non-Markovian process for image prediction.
        - The Denoising Diffusion Probabilistic Models (DDPM) architecture uses a Markov chain over many (1000's of) timesteps.
            - Learns (with a U-Net) to predict noise that it can then remove, step-by-step, to generate a new sample (image) along the same distribution.
            - Consists of a forward process that gradually adds Gaussian noise; then a reverse process, which step-by-step removes the noise the U-Net learns to predict.
            - Loss is defined as how well the DDPM model predicts noise.
        - The DDIM architecture uses a non-Markovian process.
            - Denoises an image using both the previous (more noisy) image and a prediction of the clean image, which breaks the Markov chain.
            - Unlike the DDPM, sampling is completely deterministic during inference. 

Usage:
    TODO

Classes:
    TODO

Author:
    - Spencer Karofsky (https://github.com/spencer-karofsky)
"""
import torch
import torch.nn as nn