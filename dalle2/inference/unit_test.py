import torch
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
from torch import optim

# ─── 1) Setup your scheduler & model (use same weights & device as training) ─────
from dalle2.sampling.noise_scheduler import NoiseScheduler
from dalle2.models.decoder import Decoder

device = 'mps' if torch.mps.is_available() else 'cpu'
scheduler = NoiseScheduler(T=1000, schedule_type='cosine')
decoder   = Decoder(device=device, T=1000).to(device)
decoder.load_state_dict(torch.load('dalle2/checkpoints/decoder/epoch40_batch1.pth',
                                   map_location=device))
decoder.eval()

# ─── 2) Grab one batch of *clean* images + CLIP embeddings from your dataloader ─────
# (modify your COCODecoderDataset.__getitem__ to also return the clean x0 if needed)
from dalle2.data.dataset_utils_2 import COCODecoderDataset

# === Constants ===
TARGET_IMG_SIZE = 128
BATCH_SIZE = 16
REPEATS = 256

# === Model, optimizer, scheduler ===
decoder_model = Decoder(device=device, debug=True)
optimizer = optim.AdamW(decoder_model.parameters(), lr=1e-4)
noise_scheduler = NoiseScheduler(
    T=200,
    schedule_type='cosine'
)

# === Paths ===
current_file = os.path.abspath(__file__)
metadata_path = os.path.normpath(os.path.join(
    os.path.dirname(current_file), '..', 'data', 'local_datasets', 'coco', 'metadata.csv'
))
images_dir = os.path.normpath(os.path.join(
    os.path.dirname(current_file), '..', 'data', 'local_datasets', 'coco', 'val2017'
))

# === Dataset ===
dataset = COCODecoderDataset(
    metadata_path=metadata_path,
    images_dir=images_dir,
    device=device,
    resize_size=TARGET_IMG_SIZE,
    noise_scheduler=noise_scheduler,
    n_repeat=REPEATS
)
loader  = DataLoader(dataset, batch_size=4, shuffle=True)
batch   = next(iter(loader))

# If your dataset doesn’t return x0, you can reconstruct it here:
#   x0 = dataset.image_tensor.unsqueeze(0).repeat(batch_size,1,1,1)
x0      = batch['x0'].to(device)             # shape (B,3,H,W)
z_img    = batch['z_img'].to(device)          # shape (B,512)
t       = batch['t'].to(device)               # shape (B,)
eps_true = batch['eps_img'].to(device)        # shape (B,3,H,W)

# ─── 3) Recompute x_t exactly as in training ───────────────────────────────────────
#    x_t = √ᾱ_t · x0 + √(1-ᾱ_t) · eps_true
alpha_bar = scheduler.get_alpha_bar(t).view(-1,1,1,1)
x_t = alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * eps_true

# ─── 4) One‑step through your trained decoder ───────────────────────────────────────
with torch.no_grad():
    eps_hat = decoder(x_t, z_img, t)   # predicted noise
    # reconstruct x0_hat from that prediction:
    x0_hat = (x_t - (1 - alpha_bar).sqrt() * eps_hat) / alpha_bar.sqrt()

# ─── 5) Measure MSE against true x0 ────────────────────────────────────────────────
mse = F.mse_loss(x0_hat, x0).item()
print(f"One‑step MSE on x0: {mse:.6f}")
