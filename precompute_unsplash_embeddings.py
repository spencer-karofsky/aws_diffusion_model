import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent  # adjust based on your structure
sys.path.insert(0, str(project_root))

import torch
from dalle2.models.clip_encoding import CLIPEncoder
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import os

# Path to your cropped images
DATA_DIR = project_root / "dalle2" / "data" / "local_datasets" / "unsplash" / "images_cropped"

# Output file
SAVE_PATH = project_root / "dalle2" / "data" / "local_datasets" / "unsplash" / "unsplash_embeddings.pt"

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Load CLIP Encoder
clip = CLIPEncoder().cuda().eval()

# Gather image files
image_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]

embeddings = []
image_paths = []

for fname in tqdm(image_files, desc="Embedding images"):
    img_path = DATA_DIR / fname

    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).cuda()

    with torch.no_grad():
        z_img = clip.encode_image(img_tensor)
        z_img = z_img / z_img.norm(dim=-1, keepdim=True)   # normalize
    embeddings.append(z_img.squeeze(0).cpu())
    image_paths.append(fname)

# Save embeddings + image filenames
torch.save({
    "embeddings": embeddings,
    "image_paths": image_paths
}, SAVE_PATH)

print(f"Saved {len(embeddings)} embeddings to {SAVE_PATH}")
