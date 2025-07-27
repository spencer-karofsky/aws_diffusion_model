"""
prior_trainer.py: trains the prior model on a single image (for overfitting test).
"""
import torch
import torch.optim as optim

import os

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
optimizer = optim.AdamW(prior_model.parameters(), lr=1e-4)
noise_scheduler = NoiseScheduler(T=200)

BATCH_SIZE = 128

REPEATS = 1 # Keep at 1

current_file = os.path.abspath(__file__)
metadata_path = os.path.normpath(os.path.join(
    os.path.dirname(current_file), '..', 'data', 'local_datasets', 'midjourney_v6', 'metadata.csv'
))
images_dir = os.path.normpath(os.path.join(
    os.path.dirname(current_file), '..', 'data', 'local_datasets', 'midjourney_v6', 'images'
))

dataset = MidJourneyPriorDataset(
    metadata_path=metadata_path,
    images_dir=images_dir,
    batch_size=BATCH_SIZE,
    resize_size=TARGET_IMG_SIZE,
    device=device,
    noise_scheduler=noise_scheduler,
    n_repeat=REPEATS
)

trainer = PriorTrainer(
    train_module=prior_model,
    optimizer=optimizer,
    noise_scheduler=noise_scheduler,
    dataset=dataset,
    batch_size=BATCH_SIZE,
    model_save_name='prior_model_overfit',
    debug=False,
    shuffle=False
)

if __name__ == '__main__':
    trainer.train(
        num_epochs=10,
        save_intermediate_output=5,
        save_intermediate_model=200,
    )
