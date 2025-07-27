"""
decoder_trainer.py: trains the decoder model on a single image (for overfitting test).
"""
import torch
import torch.optim as optim
import os

from dalle2.models.decoder import Decoder
from dalle2.sampling.noise_scheduler import NoiseScheduler
from dalle2.training.dalle2_training import DecoderTrainer
from dalle2.data.dataset_utils_2 import COCODecoderDataset

# Device Setup
device = (
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)

# Constants
TARGET_IMG_SIZE = 128
BATCH_SIZE = 16
REPEATS = 256

# Model, optimizer, scheduler
decoder_model = Decoder(device=device, debug=True)
optimizer = optim.AdamW(decoder_model.parameters(), lr=1e-4)
noise_scheduler = NoiseScheduler(
    T=200,
    schedule_type='cosine'
)

# Paths
current_file = os.path.abspath(__file__)
metadata_path = os.path.normpath(os.path.join(
    os.path.dirname(current_file), '..', 'data', 'local_datasets', 'coco', 'metadata.csv'
))
images_dir = os.path.normpath(os.path.join(
    os.path.dirname(current_file), '..', 'data', 'local_datasets', 'coco', 'val2017'
))

# Dataset
dataset = COCODecoderDataset(
    metadata_path=metadata_path,
    images_dir=images_dir,
    device=device,
    resize_size=TARGET_IMG_SIZE,
    noise_scheduler=noise_scheduler,
    n_repeat=REPEATS
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
    shuffle=True
)

if __name__ == '__main__':
    # Train the decoder
    trainer.train(
        num_epochs=10,
        save_intermediate_output=100,
        save_intermediate_model=1000,
        resume_checkpoint_name='epoch4_batch2301.pth'
    )
