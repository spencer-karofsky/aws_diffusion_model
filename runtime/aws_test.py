# test_decoder_aws.py

"""
Sanity check for decoder model on AWS SageMaker.

- Verifies CUDA availability
- Loads a small batch from the MidJourneyDecoderDataset
- Performs one forward pass through the Decoder
"""

import torch
import torch.optim as optim
import os

from dalle2.models.decoder import Decoder
from dalle2.sampling.noise_scheduler import NoiseScheduler
from dalle2.data.dataset_utils_2 import MidJourneyDecoderDataset

# Device
device = (
    'cuda' if torch.cuda.is_available()
    else 'cpu'
)
print(f'[INFO] Device: {device}')

# Constants
TARGET_IMG_SIZE = 128
BATCH_SIZE = 4  # small batch
REPEATS = 1

# Paths
current_file = os.path.abspath(__file__)
base_dir = os.path.normpath(os.path.join(current_file, '..', '..'))

metadata_path = os.path.join(base_dir, 'data', 'local_datasets', 'midjourney_v6', 'metadata.csv')
images_dir = os.path.join(base_dir, 'data', 'local_datasets', 'midjourney_v6', 'images')

# Components
noise_scheduler = NoiseScheduler(T=200, schedule_type='cosine')
decoder_model = Decoder(device=device, debug=True).to(device)

dataset = MidJourneyDecoderDataset(
    metadata_path=metadata_path,
    images_dir=images_dir,
    device=device,
    resize_size=TARGET_IMG_SIZE,
    noise_scheduler=noise_scheduler,
    n_repeat=REPEATS
)

# Load 1 batch
x_t, z_img, t, eps_img = next(iter(torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)))
x_t, z_img, t = x_t.to(device), z_img.to(device), t.to(device)

# Forward pass
decoder_model.eval()
with torch.no_grad():
    out = decoder_model(x_t, z_img, t)

print(f'[PASS] Forward output shape: {out.shape}')
