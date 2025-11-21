"""
decoder_trainer.py: trains the decoder model on a single image (for overfitting test).
"""
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import os
import torch
import torch.optim as optim

from dalle2.models.decoder import Decoder
from dalle2.sampling.noise_scheduler import NoiseScheduler
from dalle2.training.dalle2_training_aws import DecoderTrainer
#from dalle2.data.dataset_utils_2 import MidJourneyDecoderDataset, SingleImageOverfitDataset
from dalle2.data.boston_dataset_utils import BostonDecoder32Dataset


# Device
device = (
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)

# Constants
TARGET_IMG_SIZE = 64
BATCH_SIZE = 128
REPEATS = 1

# Model, optimizer, scheduler
decoder_model = Decoder(device=device, debug=True, T=1000)
optimizer = optim.AdamW(decoder_model.parameters(), lr=1e-4)
noise_scheduler = NoiseScheduler(T=1000, schedule_type='cosine')

# --- S3 paths (use S3 URI scheme) ---
# metadata_path = "s3://dalle2-data/train_img/metadata.csv"
# images_dir = "s3://dalle2-data/train_img"
metadata_path = "/home/ec2-user/data/train_img/metadata.csv"
images_dir = "/home/ec2-user/data/train_img"
# metadata_path = 'dalle2/data/local_datasets/midjourney_v6/metadata.csv'
# images_dir = 'dalle2/data/local_datasets/midjourney_v6/images'

# Dataset
from dalle2.data.boston_dataset_utils import BostonDecoder32Dataset

dataset = BostonDecoder32Dataset(
    metadata_csv="/home/ec2-user/data/train_img/metadata.csv",
    images_dir="/home/ec2-user/data/train_img",
    precomputed_embeddings="/home/ec2-user/data/precomputed_embeddings.pt",
    device=device,
    noise_scheduler=noise_scheduler,
    lowres=32,
    n_repeat=1,
)

embedding_data = torch.load('/home/ec2-user/data/precomputed_embeddings.pt')

# 512-dim CLIP embedding for image 0
z_img = embedding_data['image_embeddings'][0]

# metadata controls image filenames — precomputed file does not contain them
df = pd.read_csv("/home/ec2-user/data/train_img/metadata.csv")
first_filename = os.path.basename(df.iloc[0]["image_path"])
image_path = f"/home/ec2-user/data/train_img/{first_filename}"


# dataset = SingleImageOverfitDataset(
#     image_path=image_path,
#     z_img=z_img,
#     device=device,
#     noise_scheduler=noise_scheduler,
#     resize_size=TARGET_IMG_SIZE
# )

# Trainer
trainer = DecoderTrainer(
    train_module=decoder_model,
    optimizer=optimizer,
    noise_scheduler=noise_scheduler,
    dataset=dataset,
    batch_size=BATCH_SIZE,
    model_save_name='decoder_model',
    debug=False,
    shuffle=True, # Change back later
    on_aws=True,
    use_amp=True
)

if __name__ == '__main__':
    trainer.train(
        num_epochs=300,
        save_intermediate_output=1,
        save_intermediate_model=5,
        #resume_checkpoint_name='epoch159_batch100.pth'
    )
