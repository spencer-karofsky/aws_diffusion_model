import os
import torch
import pandas as pd
from PIL import Image
import io
import boto3
from torchvision import transforms
from dalle2.models.clip_encoding import CLIPEncoder

BUCKET = "dalle2-data"
IMAGES_PREFIX = "train_img/"
resize = 128

s3 = boto3.client("s3")

clip = CLIPEncoder().to("cuda").eval()

transform = transforms.Compose([
    transforms.Resize((resize, resize), antialias=True),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1),
])

# Load metadata CSV from S3
csv_key = "train_img/metadata.csv"
obj = s3.get_object(Bucket=BUCKET, Key=csv_key)
df = pd.read_csv(obj["Body"])

image_embeddings = []
text_embeddings = []

for _, row in df.iterrows():
    fname = os.path.basename(row["image_path"])
    key = f"{IMAGES_PREFIX}{fname}"

    obj = s3.get_object(Bucket=BUCKET, Key=key)
    img = Image.open(io.BytesIO(obj["Body"].read())).convert("RGB")

    x = transform(img).unsqueeze(0).cuda()

    with torch.no_grad():
        z_img = clip.encode_image(x).squeeze(0)
        z_txt = clip.encode_text([row["caption"]]).squeeze(0)

    image_embeddings.append(z_img.cpu())
    text_embeddings.append(z_txt.cpu())

output_path = "precomputed_unsplash_embeddings.pth"
torch.save(
    {
        "image_embeddings": torch.stack(image_embeddings),
        "text_embeddings": torch.stack(text_embeddings),
    },
    output_path
)

print("Saved:", output_path)
