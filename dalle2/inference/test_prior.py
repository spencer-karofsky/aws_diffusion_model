"""
Quick one‑off prior tester (no argparse).

• Loads an EMA prior checkpoint
• Picks one random (z_txt, z_img_true) pair from the MidJourneyPriorDataset
• Runs DDIM sampling to get z_img_hat
• Prints cosine similarity and saves the raw tensors (optional)

Run from project root:

    python test_prior_single.py

Edit the CONSTANTS block to match your paths.
"""

import os, random, torch, torch.nn.functional as F
from pathlib import Path

# -----------------------------------------------------------------------------
# CONSTANTS — tweak inline
# -----------------------------------------------------------------------------
PROJECT_DIR       = Path("/Users/spencerkarofsky/Desktop/projects/aws_diffusion_model")
PRIOR_EMA         = PROJECT_DIR/"dalle2/checkpoints/prior/epoch5_batch625_ema.pth"
DATASET_ROOT      = "s3://dalle2-data/train_img"  # folder where dataset lives
METADATA_CSV      = "s3://dalle2-data/train_img/metadata.csv"
T_TOTAL           = 200
START_T           = 60
DDIM_STEPS        = 25
CFG_SCALE         = 4.0
DEVICE            = (
    "cuda" if torch.cuda.is_available() else
    "mps"   if torch.backends.mps.is_available() else
    "cpu"
)

# -----------------------------------------------------------------------------
# Imports that depend on project PYTHONPATH
# -----------------------------------------------------------------------------
os.chdir(PROJECT_DIR)

from dalle2.models.prior import Prior
from dalle2.sampling.noise_scheduler import NoiseScheduler
from dalle2.sampling.ddim_sampling import PriorDDIMSampler
from dalle2.data.dataset_utils_2 import MidJourneyPriorDataset

# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------
def cosine(a, b):
    return F.cosine_similarity(a, b, dim=-1)

# -----------------------------------------------------------------------------
# Load model + scheduler + sampler
# -----------------------------------------------------------------------------
print("[INFO] loading EMA prior →", PRIOR_EMA)
prior = Prior(T=200, device='mps')
prior.load_state_dict(torch.load(PRIOR_EMA, map_location="cpu"))  # Always safe to load to CPU
prior.eval()

# Force entire model (and all submodules, buffers, etc.) to correct device
for name, param in prior.named_parameters():
    param.data = param.data.to(DEVICE)
for buffer_name, buffer in prior.named_buffers():
    buffer.data = buffer.data.to(DEVICE)


noise_sched = NoiseScheduler(T=T_TOTAL, schedule_type="cosine")
sampler = PriorDDIMSampler(
    noise_sched,
    num_inference_steps=DDIM_STEPS,
    device='mps',
    eta=0.0,
  #  guidance_scale=CFG_SCALE,
#    start_t=START_T,
)

# -----------------------------------------------------------------------------
# One random sample from dataset
# -----------------------------------------------------------------------------
print("[INFO] sampling dataset …")
ds = MidJourneyPriorDataset(
    metadata_path=METADATA_CSV,
    images_dir=DATASET_ROOT,
    device='mps',
    batch_size=64,
    noise_scheduler=noise_sched
)

z_txt, z_img_true = ds.get_random_text_and_embedding()
z_txt      = z_txt.unsqueeze(0).to('mps')      # (1,512)
z_img_true = z_img_true.unsqueeze(0).to('mps') # (1,512)
print("z_txt norm:", z_txt.norm().item())
print("z_img_true norm:", z_img_true.norm().item())


# -----------------------------------------------------------------------------
# DDIM sampling
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# DDIM sampling
# -----------------------------------------------------------------------------
print("DEVICE:", DEVICE)
print("z_txt device:", z_txt.device)
print("z_img_true device:", z_img_true.device)

for name, param in prior.named_parameters():
    print(f"{name} → {param.device}")
    break  # just show one param

with torch.no_grad():
    null_z_txt = torch.zeros_like(z_txt).to(DEVICE)
    z_img_hat = sampler.sample(
        model=prior,
        z_txt=z_txt,
       # null_z_txt=null_z_txt,
        # Optional: add below if not already set in constructor
        # start_t=START_T,
        # num_inference_steps=DDIM_STEPS,
        # cfg_scale=CFG_SCALE,
    )
    print("z_img_hat norm:", z_img_hat.norm().item())

dot = torch.matmul(z_txt, z_img_true.T).item()
print(f"dot(z_txt, z_img_true): {dot:.4f}")

cos = cosine(z_img_hat, z_img_true).item()
print(f"cosine_similarity(z_hat, z_true) = {cos:.4f}")

# Optional: save to check quickly later
# torch.save({
#     "z_txt"     : z_txt.cpu(),
#     "z_img_true": z_img_true.cpu(),
#     "z_img_hat" : z_img_hat.cpu(),
# }, PROJECT_DIR/"dalle2/checkpoints/prior_single_test.pt")
print("[✓] Saved raw tensors → prior_single_test.pt")



import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("dalle2/checkpoints/logs/prior_batch_losses.csv")
plt.plot(df["mse_loss"])
plt.yscale("log")
plt.title("Prior Loss")
plt.grid(True)
plt.show()


