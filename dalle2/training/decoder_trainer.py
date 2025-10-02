"""
decoder_trainer.py: trains the decoder model on a single image (for overfitting test).
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import os
import torch
import torch.optim as optim

from dalle2.models.decoder import Decoder
from dalle2.sampling.noise_scheduler import NoiseScheduler
from dalle2.training.dalle2_training_aws import DecoderTrainer
from dalle2.data.dataset_utils_2 import MidJourneyDecoderDataset

# Device
device = (
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)

# Constants
TARGET_IMG_SIZE = 64
BATCH_SIZE = 32
REPEATS = 1  # keep at 1 for the quick test

# Model, optimizer, scheduler
decoder_model = Decoder(device=device, debug=True)
optimizer = optim.AdamW(decoder_model.parameters(), lr=1e-4)
noise_scheduler = NoiseScheduler(T=200, schedule_type='cosine')

# --- S3 paths (use S3 URI scheme) ---
metadata_path = "s3://dalle2-data/train_img/metadata.csv"
images_dir = "s3://dalle2-data/train_img"

# metadata_path = 'dalle2/data/local_datasets/midjourney_v6/metadata.csv'
# images_dir = 'dalle2/data/local_datasets/midjourney_v6/images'

# Dataset
dataset = MidJourneyDecoderDataset(
    metadata_path=metadata_path,
    images_dir=images_dir,
    device=device,
    resize_size=TARGET_IMG_SIZE,
    noise_scheduler=noise_scheduler,
    n_repeat=REPEATS,
    # Optional: pick a larger persistent cache directory on the instance
    # cache_dir="/home/ec2-user/dalle2_cache",
)

# Trainer
trainer = DecoderTrainer(
    train_module=decoder_model,
    optimizer=optimizer,
    noise_scheduler=noise_scheduler,
    dataset=dataset,
    batch_size=BATCH_SIZE,
    model_save_name='decoder_model_overfit',
    debug=False,
    shuffle=True,
    on_aws=True,
    use_amp=True
)

if __name__ == '__main__':
    trainer.train(
        num_epochs=30,
        save_intermediate_output=50,
        save_intermediate_model=200,
       # resume_checkpoint_name='dalle2/checkpoints/prior/epoch5_batch625.pth'
    )
