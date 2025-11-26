# """
# precompute_embeddings.py: Precomputes CLIP text and image embeddings for faster training.

# Description:
#     Encodes all images and captions using CLIP and saves them to disk.
#     This avoids re-encoding during training, significantly speeding up the process.
    
# Author:
#     Spencer Karofsky
# """
# import torch
# import pandas as pd
# from pathlib import Path
# from PIL import Image
# from torchvision import transforms
# from tqdm import tqdm
# import torch.nn.functional as F

# from dalle2.models.clip_encoding import CLIPEncoder

# def precompute_embeddings(
#     metadata_path: str,
#     images_dir: str,
#     output_path: str,
#     resize_size: int = 64,
#     device: str = None
# ):
#     """
#     Precompute CLIP text and image embeddings for all images in the dataset.
    
#     Args:
#         metadata_path: Path to metadata.csv containing image_path and caption columns
#         images_dir: Directory containing the images
#         output_path: Where to save the precomputed embeddings (.pt file)
#         resize_size: Size to resize images to before encoding
#         device: Device to use for encoding (auto-detected if None)
#     """
#     if device is None:
#         device = (
#             'cuda' if torch.cuda.is_available()
#             else 'mps' if torch.backends.mps.is_available()
#             else 'cpu'
#         )
    
#     print(f"Using device: {device}")
    
#     # Load metadata
#     df = pd.read_csv(metadata_path)
#     print(f"Found {len(df)} images in metadata")
    
#     # Initialize CLIP encoder
#     clip_encoder = CLIPEncoder().to(device).eval()
    
#     # Image preprocessing
#     transform = transforms.Compose([
#         transforms.Resize((resize_size, resize_size), antialias=True),
#         transforms.ToTensor(),
#         transforms.Lambda(lambda x: x * 2 - 1)  # [-1, 1]
#     ])
    
#     # Storage for embeddings
#     image_embeddings = []
#     text_embeddings = []
#     image_paths = []
#     captions = []
    
#     print("Encoding images and captions...")
#     for idx, row in tqdm(df.iterrows(), total=len(df)):
#         caption = row['caption']
#         img_path = row['image_path']
        
#         # Resolve full image path
#         if not Path(img_path).is_absolute():
#             img_path = Path(images_dir) / Path(img_path).name
        
#         try:
#             # Load and encode image
#             image = Image.open(img_path).convert('RGB')
#             image_tensor = transform(image).unsqueeze(0).to(device)
            
#             with torch.no_grad():
#                 # Encode image
#                 z_img = clip_encoder.encode_image(image_tensor).squeeze(0)
#                 z_img = F.normalize(z_img, dim=-1)  # Unit normalize
                
#                 # Encode text
#                 z_txt = clip_encoder.encode_text([caption]).squeeze(0)
#                 z_txt = F.normalize(z_txt, dim=-1)  # Unit normalize
            
#             # Store on CPU to save GPU memory
#             image_embeddings.append(z_img.cpu())
#             text_embeddings.append(z_txt.cpu())
#             image_paths.append(str(img_path))
#             captions.append(caption)
            
#         except Exception as e:
#             print(f"Warning: Failed to process {img_path}: {e}")
#             continue
    
#     # Stack into tensors
#     image_embeddings = torch.stack(image_embeddings)
#     text_embeddings = torch.stack(text_embeddings)

#     print(f"Saving {len(image_embeddings)} embeddings to {output_path}")
#     torch.save({
#         'image_embeddings': image_embeddings,
#         'text_embeddings': text_embeddings,
#         'image_paths': image_paths,
#         'captions': captions,
#         'resize_size': resize_size
#     }, output_path)
    
#     print(f"Done! Saved embeddings:")
#     print(f"  - Image embeddings: {image_embeddings.shape}")
#     print(f"  - Text embeddings: {text_embeddings.shape}")


# if __name__ == '__main__':
#     precompute_embeddings(
#         metadata_path='dalle2/data/local_datasets/midjourney_v6/metadata.csv',
#         images_dir='dalle2/data/local_datasets/midjourney_v6/images',
#         output_path='dalle2/data/local_datasets/midjourney_v6/precomputed_embeddings_full.pt',
#         resize_size=64
#     )

"""
precompute_embeddings.py: Precomputes CLIP text and image embeddings for faster training.

Description:
    Encodes all images and captions using CLIP and saves them to disk.
    This avoids re-encoding during training, significantly speeding up the process.
    
Author:
    Spencer Karofsky
"""
import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F

from dalle2.models.clip_encoding import CLIPEncoder


def precompute_embeddings(
    metadata_path: str,
    images_dir: str,
    output_path: str,
    resize_size: int = 64,
    device: str = None
):
    """
    Precompute CLIP text and image embeddings for all images in the dataset.
    
    Args:
        metadata_path: Path to metadata.csv containing image_path and caption columns
        images_dir: Directory containing the images
        output_path: Where to save the precomputed embeddings (.pt file)
        resize_size: Size to resize images to before encoding
        device: Device to use for encoding (auto-detected if None)
    """
    # Auto device select
    if device is None:
        device = (
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )
    device = "cpu"

    
    print(f"Using device: {device}")
    
    # ----------------------------------------------------
    # 1. Load metadata
    # ----------------------------------------------------
    df = pd.read_csv(metadata_path)
    print(f"Found {len(df)} entries in metadata")

    # Validate required columns
    if "image_path" not in df.columns:
        raise ValueError("metadata.csv must include 'image_path' column")
    if "caption" not in df.columns:
        raise ValueError("metadata.csv must include 'caption' column")

    # ----------------------------------------------------
    # 2. Initialize CLIP encoder
    # ----------------------------------------------------
    clip_encoder = CLIPEncoder().to(device).eval()

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((resize_size, resize_size), antialias=True),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1)
    ])
    
    # Storage for embeddings
    image_embeddings = []
    text_embeddings = []
    final_paths = []
    final_captions = []
    
    print("Encoding images and captions...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        caption = row["caption"]
        rel_path = row["image_path"]

        # Full local path
        img_path = Path(images_dir) / Path(rel_path).name

        if not img_path.exists():
            print(f"WARNING: Missing file {img_path}, skipping.")
            continue
        
        try:
            # Load & preprocess image
            image = Image.open(img_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                # Encode image
                z_img = clip_encoder.encode_image(image_tensor).squeeze(0)
                z_img = F.normalize(z_img, dim=-1)

                # Encode text
                z_txt = clip_encoder.encode_text([caption]).squeeze(0)
                z_txt = F.normalize(z_txt, dim=-1)

            # Store CPU version
            image_embeddings.append(z_img.cpu())
            text_embeddings.append(z_txt.cpu())
            final_paths.append(str(img_path))
            final_captions.append(caption)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    # Stack into tensors
    image_embeddings = torch.stack(image_embeddings)
    text_embeddings = torch.stack(text_embeddings)

    print(f"Saving {len(image_embeddings)} embeddings to {output_path}")

    torch.save(
        {
            "image_embeddings": image_embeddings,
            "text_embeddings": text_embeddings,
            "image_paths": final_paths,
            "captions": final_captions,
            "resize_size": resize_size
        },
        output_path
    )

    print(f"Done! Saved:")
    print(f"  - Image embeddings: {image_embeddings.shape}")
    print(f"  - Text embeddings:  {text_embeddings.shape}")


if __name__ == "__main__":
    precompute_embeddings(
        metadata_path="dalle2/data/local_datasets/unsplash/metadata.csv",
        images_dir="dalle2/data/local_datasets/unsplash/images_cropped",
        output_path="dalle2/data/local_datasets/unsplash/precomputed_embeddings_full.pt",
        resize_size=64
    )
