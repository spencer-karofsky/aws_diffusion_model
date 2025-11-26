# End-to-End DALL·E 2 Implementation from Scratch

A complete implementation of OpenAI's DALL·E 2 text-to-image generation system, built entirely from first principles. This project includes a CLIP text encoder, cascaded diffusion decoder (64x64 → 128x128), and modular AWS infrastructure for scalable training and inference.

<img width="874" height="566" alt="image" src="https://github.com/user-attachments/assets/53ab7308-ce42-4552-9383-277d3eb925e6" />


## Overview

This implementation was built to deeply understand the computational science underlying modern generative models—how architectural choices, optimization dynamics, and numerical methods converge to create emergent intelligence.

**Key Features:**
- Full CLIP + cascaded diffusion pipeline (~9,500 lines)
- Systematic debugging workflow: single-image overfitting → full dataset training
- Custom AWS training infrastructure with Infrastructure as Code (IaC)
- Achieved 10× training speedup through architectural optimization
- Complete system trained for under $60 on AWS

**Technical Highlights:**
- Cascaded generation: 64x64 base decoder + 128x128 upsampler
- Mixed-precision training, EMA weights, classifier-free guidance
- Cosine noise scheduling for improved convergence
- Modular architecture with comprehensive unit testing
- Complete AWS deployment pipeline (S3, SageMaker, CloudWatch, IAM, VPC)

## Architecture

The system consists of three main components:

1. **CLIP Text Encoder**: Converts text prompts into semantic embeddings
2. **Prior Model**: Transformer that maps CLIP text embeddings to image embeddings
3. **Cascaded Diffusion Decoder**: 
   - Base 64x64 decoder generates low-resolution images
   - Upsampler refines to 128x128 resolution

## Key Implementation Details

**Debugging Process:**
- Initial 15,000-line implementation failed to converge
- Complete redesign with systematic approach: PowerPoint architecture diagrams, documented tensor shapes, isolated component testing
- Validated architecture by deliberately overfitting on single images before scaling to full dataset

**Optimization Improvements:**
- Restructured from direct high-resolution generation to cascaded approach
- Implemented mixed-precision training for memory efficiency
- Added exponential moving average (EMA) of model weights
- Tuned noise schedules and learning rate strategies

## Training

Training was conducted on AWS using custom infrastructure:
- SageMaker GPU instances (g4dn.xlarge)
- S3 for dataset storage and model checkpoints
- CloudWatch for monitoring training metrics
- Complete training cost: <$60

## Acknowledgments

This project uses [OpenCLIP](https://github.com/mlfoundations/open_clip) for the CLIP text encoder implementation.

**Citation:**
```
Ilharco et al., "OpenCLIP", 2021. 
DOI: 10.5281/zenodo.5143773
```

## Notice

This project is a personal research implementation developed for learning and educational purposes. The code is provided as-is for exploration and study. No guarantees of functionality, support, or maintenance are provided.
