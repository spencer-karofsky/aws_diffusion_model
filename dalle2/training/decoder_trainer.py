# """
# decoder_trainer.py: trains the decoder model.
# """
# import torch
# import torch.optim as optim

# from dalle2.models.decoder import Decoder
# from dalle2.sampling.noise_scheduler import NoiseScheduler
# from dalle2.training.dalle2_training import DecoderTrainer

# device = (
#     'cuda' if torch.cuda.is_available()
#     else 'mps' if torch.backends.mps.is_available()
#     else 'cpu'
# )

# # Initialize model, optimizer, and noise scheduler
# decoder_model = Decoder(device=device, debug=True)
# optimizer = optim.AdamW(decoder_model.parameters(), lr=1e-4)
# noise_scheduler = NoiseScheduler(T=1000)

# # Dataset imports
# from dalle2.data.dataset_utils import DecoderCOCODataset
# import os

# current_file = os.path.abspath(__file__)
# metadata_path = os.path.join(
#     os.path.dirname(__file__), '..', 'data', 'local_datasets', 'coco', 'metadata.csv'
# )
# images_dir = os.path.join(
#     os.path.dirname(__file__), '..', 'data', 'local_datasets', 'coco', 'val2017'
# )

# metadata_path = os.path.normpath(metadata_path)
# images_dir = os.path.normpath(images_dir)

# # Initialize dataset
# dataset = DecoderCOCODataset(
#     metadata_path=metadata_path,
#     images_dir=images_dir
# )

# # Initialize trainer
# trainer = DecoderTrainer(
#     train_module=decoder_model,
#     optimizer=optimizer,
#     noise_scheduler=noise_scheduler,
#     dataset=dataset,
#     batch_size=32,
#     model_save_name='decoder_model',
#     debug=False
# )

# if __name__ == "__main__":
#     trainer.train(num_epochs=20, save_every=10)

"""
decoder_trainer.py: trains the decoder model on a single image (for overfitting test).
"""
import torch
import torch.optim as optim
import os

from dalle2.models.decoder import Decoder
from dalle2.sampling.noise_scheduler import NoiseScheduler
from dalle2.training.dalle2_training import DecoderTrainer
from dalle2.data.dataset_utils import SingleImageDecoderDataset  # ✅ updated import

device = (
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)

# === Model, optimizer, scheduler ===
decoder_model = Decoder(device=device, debug=True)
optimizer = optim.AdamW(decoder_model.parameters(), lr=1e-4)
noise_scheduler = NoiseScheduler(T=1000)

# === Use single image dataset ===
image_path = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'local_datasets', 'coco', 'val2017', '000000000139.jpg'
)
caption = "A dog wearing sunglasses."  # Use matching caption if possible

dataset = SingleImageDecoderDataset(
    image_path=image_path,
    caption=caption,
    resize_size=128,
    device=device
)

# === Trainer ===
trainer = DecoderTrainer(
    train_module=decoder_model,
    optimizer=optimizer,
    noise_scheduler=noise_scheduler,
    dataset=dataset,
    batch_size=1,  # ✅ overfitting, small batch
    model_save_name='decoder_model_overfit',
    debug=False
)

if __name__ == "__main__":
    trainer.train(num_epochs=20, save_every=10)
