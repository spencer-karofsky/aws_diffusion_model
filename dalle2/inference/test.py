import torch, pprint

RAW = torch.load("dalle2/checkpoints/prior/epoch2_batch400_ema.pth", map_location="cpu")

# ➊ Lightning‑style   → {'state_dict': …}
# ➋ HF Trainer style  → {'model': …}
# ➌ Plain PyTorch     → {tensor_name: tensor, …}
for k in ("state_dict", "model"):
    if k in RAW and isinstance(RAW[k], dict):
        RAW = RAW[k]
        break       # now RAW is the real state‑dict

print("top‑level keys:", list(RAW.keys())[:10])

from dalle2.models.prior import Prior

# instantiate with the (suspicious) value you used in test_prior_single.py
prior = Prior(T=1000, device='mps')          # <- intentionally wrong
missing, unexpected = prior.load_state_dict(RAW, strict=False)

print("MISSING parameters:\n", pprint.pformat(missing))
print("UNEXPECTED parameters:\n", pprint.pformat(unexpected))
