import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import torch
from dalle2.models.clip_encoding import CLIPEncoder
import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import os

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

clip = CLIPEncoder().cuda().eval()
df = pd.read_csv('/home/ec2-user/data/train_img/metadata.csv')

embeddings = []
for idx, row in tqdm(df.iterrows(), total=len(df)):
    # Just use the filename, not the full path from metadata
    img_path = f"/home/ec2-user/data/train_img/{os.path.basename(row['image_path'])}"
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).cuda()
    
    with torch.no_grad():
        z_img = clip.encode_image(img_tensor)
        z_img = z_img / z_img.norm(dim=-1, keepdim=True)
    embeddings.append(z_img.squeeze(0).cpu())

torch.save({
    'embeddings': embeddings,
    'image_paths': [os.path.basename(p) for p in df['image_path'].tolist()]
}, '/home/ec2-user/data/precomputed_embeddings.pt')

print(f"Saved {len(embeddings)} embeddings")
