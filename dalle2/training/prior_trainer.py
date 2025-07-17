"""
prior_trainer.py: trains the prior model.
"""
from dalle2.models.prior import Prior
from dalle2.sampling.noise_scheduler import NoiseScheduler
from dalle2_training import PriorTrainer

import torch.optim as optim

prior_model = Prior(dim=512, transformer_blocks=6, attention_heads=8)

optimizer = optim.AdamW(prior_model.parameters(), lr=1e-4)
noise_scheduler = NoiseScheduler(beta_start=1e-4, beta_end=0.02, timesteps=1000)

from data.dataset_utils import COCODataset
dataset = COCODataset()

trainer = PriorTrainer(
    train_module=prior_model,
    optimizer=optimizer,
    noise_scheduler=noise_scheduler,
    dataset=dataset,
    batch_size=32,
    model_save_name='prior_model',
    debug=True
)

trainer.train(num_epochs=5, save_every=10)

