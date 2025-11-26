# """
# prior_trainer.py: trains the prior model on a single image (for overfitting test).
# """
# import os
# import torch
# import torch.optim as optim

# from dalle2.models.prior import Prior
# from dalle2.sampling.noise_scheduler import NoiseScheduler
# from dalle2.training.dalle2_training import PriorTrainer
# from dalle2.data.boston_dataset_utils import BostonPriorDataset

# TARGET_IMG_SIZE = 64

# device = (
#     'cuda' if torch.cuda.is_available()
#     else 'mps' if torch.backends.mps.is_available()
#     else 'cpu'
# )

# prior_model = Prior(device=device, T=1000)
# optimizer = optim.AdamW(prior_model.parameters(), lr=1e-4)
# noise_scheduler = NoiseScheduler(T=1000, schedule_type='cosine')

# BATCH_SIZE = 128
# REPEATS = 1

# # --- S3 paths ---
# # metadata_path = "s3://dalle2-data/train_img/metadata.csv"
# # images_dir = "s3://dalle2-data/train_img"

# # Local Paths
# metadata_path = 'dalle2/data/local_datasets/midjourney_v6/metadata.csv'
# images_dir = 'dalle2/data/local_datasets/midjourney_v6/images'

# dataset = BostonPriorDataset(
#     metadata_path=metadata_path,
#     images_dir=images_dir,
#     batch_size=BATCH_SIZE,
#     resize_size=TARGET_IMG_SIZE,
#     device=device,
#     noise_scheduler=noise_scheduler,
#     n_repeat=REPEATS,
#     precomputed_embeddings_path='dalle2/data/local_datasets/midjourney_v6/precomputed_embeddings_full.pt',
# )

# trainer = PriorTrainer(
#     train_module=prior_model,
#     optimizer=optimizer,
#     noise_scheduler=noise_scheduler,
#     dataset=dataset,
#     batch_size=BATCH_SIZE,
#     model_save_name='prior_model',
#     debug=False,
#     shuffle=True,
#     on_aws=False
# )

# if __name__ == '__main__':
#     trainer.train(
#         num_epochs=150,
#         save_intermediate_output=100,
#         save_intermediate_model=350,
#      #   resume_checkpoint_name='epoch106_batch313_ema.pth'
#     )
"""
prior_trainer.py: trains the prior model on the Boston (Unsplash) dataset.
"""

import os
import torch
import torch.optim as optim

from dalle2.models.prior import Prior
from dalle2.sampling.noise_scheduler import NoiseScheduler
from dalle2.training.dalle2_training import PriorTrainer

# Boston/Unsplash dataset
from dalle2.data.boston_dataset_utils import BostonPriorDataset


TARGET_EMB_SIZE = 512
TARGET_IMG_SIZE = 128        # matches your precompute size if 128 OR 64
BATCH_SIZE = 128
REPEATS = 1

device = (
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)

# ------------------------------------------------------------
# MODEL + OPTIMIZER + SCHEDULER
# ------------------------------------------------------------
prior_model = Prior(device=device, T=1000)
optimizer = optim.AdamW(prior_model.parameters(), lr=1e-4)
noise_scheduler = NoiseScheduler(T=1000, schedule_type='cosine')

# ------------------------------------------------------------
# UNSPLASH DATASET PATHS (Boston)
# ------------------------------------------------------------
metadata_path = "dalle2/data/local_datasets/unsplash/metadata.csv"
images_dir = "dalle2/data/local_datasets/unsplash/images_cropped"
precomputed_embeddings = "dalle2/data/local_datasets/unsplash/precomputed_embeddings_full.pt"

# ------------------------------------------------------------
# DATASET
# ------------------------------------------------------------
dataset = BostonPriorDataset(
    metadata_csv=metadata_path,
    images_dir=images_dir,
    device=device,
    noise_scheduler=noise_scheduler,
    precomputed_path=precomputed_embeddings,
    resize=TARGET_IMG_SIZE,
    n_repeat=REPEATS,
)

# ------------------------------------------------------------
# TRAINER
# ------------------------------------------------------------
trainer = PriorTrainer(
    train_module=prior_model,
    optimizer=optimizer,
    noise_scheduler=noise_scheduler,
    dataset=dataset,
    batch_size=BATCH_SIZE,
    model_save_name="prior_model_unsplash",
    debug=False,
    shuffle=True,
    on_aws=False,
)

# ------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    trainer.train(
        num_epochs=150,
        save_intermediate_output=7,
        save_intermediate_model=350,
        resume_checkpoint_name="final_trained_model.pth"
    )
