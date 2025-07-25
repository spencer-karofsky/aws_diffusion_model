import torch
from dalle2.models.dalle2 import DALLe2
from dalle2.models.clip_encoding import CLIPEncoder
from dalle2.sampling.noise_scheduler import NoiseScheduler
from dalle2.sampling.ddim_sampling import PriorDDIMSampler, DecoderDDIMSampler

# Constants
H, W = 128, 128
T = 1000
num_steps = 30  # Inference steps — good default for DDIM

# Load components
clip_encoder = CLIPEncoder()

prior_scheduler = NoiseScheduler(T=T)
decoder_scheduler = NoiseScheduler(T=T)

prior_sampler = PriorDDIMSampler(
    noise_scheduler=prior_scheduler,
    num_inference_steps=num_steps,
    eta=0.0  # deterministic DDIM
)

decoder_sampler = DecoderDDIMSampler(
    noise_scheduler=decoder_scheduler,
    num_inference_steps=num_steps,
    eta=0.0
)

# Initialize DALLe2
dalle = DALLe2(
    prior_path='dalle2/checkpoints/prior/final_trained_model.pth',
    decoder_path='dalle2/checkpoints/decoder/epoch240_batch1.pth',
    clip_encoder=clip_encoder,
    prior_sampler=prior_sampler,
    decoder_sampler=decoder_sampler,
    H=H,
    W=W,
    T=T,
    num_inference_timesteps=num_steps,
    debug=False
)

state_dict = torch.load('dalle2/checkpoints/prior/final_trained_model.pth')
print("Prior keys:", list(state_dict.keys())[:5])
state_dict = torch.load('dalle2/checkpoints/decoder/final_trained_model.pth')
print("Decoder keys:", list(state_dict.keys())[:5])


# Prompt
prompts = [
    'a dog and cat lying together on an orange couch.',
    'a dog and cat lying together on an orange couch.',
]

# Generate images
with torch.no_grad():
    images = dalle.generate(prompts) # shape [B, 3, H, W]

import torchvision.utils as vutils
import matplotlib.pyplot as plt

# Assuming images ∈ [0, 1]
grid = vutils.make_grid(images, nrow=3)
plt.figure(figsize=(12, 6))
plt.imshow(grid.permute(1, 2, 0).cpu())
plt.axis('off')
plt.show()