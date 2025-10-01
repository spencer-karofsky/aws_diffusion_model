"""
prior_trainer.py: trains the prior model on a single image (for overfitting test).
"""
import os
import torch
import torch.optim as optim

from dalle2.models.prior import Prior
from dalle2.sampling.noise_scheduler import NoiseScheduler
from dalle2.training.dalle2_training import PriorTrainer
from dalle2.data.dataset_utils_2 import MidJourneyPriorDataset

TARGET_IMG_SIZE = 128

device = (
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)

prior_model = Prior(device=device, T=200)
optimizer = optim.AdamW(prior_model.parameters(), lr=2e-4)
noise_scheduler = NoiseScheduler(T=200, schedule_type='cosine')

# Start conservatively; raise once it runs stably.
BATCH_SIZE = 64
REPEATS = 1

# --- S3 paths ---
metadata_path = "s3://dalle2-data/train_img/metadata.csv"
images_dir = "s3://dalle2-data/train_img"

dataset = MidJourneyPriorDataset(
    metadata_path=metadata_path,
    images_dir=images_dir,
    batch_size=BATCH_SIZE,
    resize_size=TARGET_IMG_SIZE,
    device=device,
    noise_scheduler=noise_scheduler,
    n_repeat=REPEATS,
    # cache_dir="/home/ec2-user/dalle2_cache",  # if your dataset util supports it
)

trainer = PriorTrainer(
    train_module=prior_model,
    optimizer=optimizer,
    noise_scheduler=noise_scheduler,
    dataset=dataset,
    batch_size=BATCH_SIZE,
    model_save_name='prior_model_overfit',
    debug=False,
    shuffle=False,
    on_aws=False
)

if __name__ == '__main__':
    trainer.train(
        num_epochs=10,
        save_intermediate_output=5,
        save_intermediate_model=100,
        resume_checkpoint_name='epoch3_batch100_ema.pth'
    )
