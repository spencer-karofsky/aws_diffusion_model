"""
prior_trainer.py: trains the prior model.
"""
import torch
import torch.optim as optim

from dalle2.models.prior import Prior
from dalle2.sampling.noise_scheduler import NoiseScheduler
from dalle2.training.dalle2_training import PriorTrainer

device = (
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)

prior_model = Prior(device=device)

optimizer = optim.AdamW(prior_model.parameters(), lr=1e-4)
noise_scheduler = NoiseScheduler(T=1000)

from dalle2.data.dataset_utils import COCODataset
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

if __name__ == "__main__":
    trainer.train(num_epochs=5, save_every=10)
