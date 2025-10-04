"""
precompute_upsampler_embeddings.py: Precomputes CLIP embeddings for 128x128 images on AWS.

Description:
    Encodes all images at 128x128 resolution using CLIP and saves embeddings.
    Optimized for AWS EC2/SageMaker with local images and S3 output.
    
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
import boto3
import tempfile
import os
from urllib.parse import urlparse

from dalle2.models.clip_encoding import CLIPEncoder


def is_s3_uri(p: str) -> bool:
    return isinstance(p, str) and p.startswith("s3://")


def split_s3_uri(uri: str):
    pr = urlparse(uri)
    return pr.netloc, pr.path.lstrip("/")


def precompute_upsampler_embeddings(
    metadata_path: str,
    images_dir: str,
    output_path: str,
    resize_size: int = 128,
    device: str = None,
    on_aws: bool = True
):
    """
    Precompute CLIP embeddings for upsampler training.
    
    Args:
        metadata_path: S3 URI or local path to metadata.csv
        images_dir: Local path to images directory (e.g., /home/ec2-user/midjourney_v6/images)
        output_path: S3 URI or local path to save embeddings (.pt file)
        resize_size: Size to resize images to before encoding (128 for upsampler)
        device: Device to use for encoding (auto-detected if None)
        on_aws: Whether running on AWS (enables S3 output)
    """
    if device is None:
        device = (
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )
    
    print(f"[PrecomputeUpsampler] Using device: {device}")
    print(f"[PrecomputeUpsampler] Encoding images at {resize_size}x{resize_size} resolution")
    print(f"[PrecomputeUpsampler] AWS mode: {on_aws}")
    
    # Setup S3 if needed (for metadata and output only)
    s3_client = None
    if on_aws or is_s3_uri(metadata_path) or is_s3_uri(output_path):
        s3_client = boto3.client('s3')
    
    # Load metadata
    if is_s3_uri(metadata_path):
        bucket, key = split_s3_uri(metadata_path)
        print(f"[PrecomputeUpsampler] Downloading metadata from s3://{bucket}/{key}")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
            s3_client.download_file(bucket, key, tmp.name)
            df = pd.read_csv(tmp.name)
            os.remove(tmp.name)
        print(f"[PrecomputeUpsampler] Loaded metadata from S3")
    else:
        df = pd.read_csv(metadata_path)
        print(f"[PrecomputeUpsampler] Loaded metadata from {metadata_path}")
    
    print(f"[PrecomputeUpsampler] Found {len(df)} images in metadata")
    print(f"[PrecomputeUpsampler] Reading images from local directory: {images_dir}")
    
    # Initialize CLIP encoder
    print(f"[PrecomputeUpsampler] Initializing CLIP encoder...")
    clip_encoder = CLIPEncoder().to(device).eval()
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((resize_size, resize_size), antialias=True),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1)  # [-1, 1]
    ])
    
    # Storage for embeddings
    embeddings = []
    image_paths = []
    failed_indices = []
    
    print("[PrecomputeUpsampler] Encoding images...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_path = row['image_path']
        
        try:
            # Resolve local image path
            if not Path(img_path).is_absolute():
                local_img_path = Path(images_dir) / Path(img_path).name
            else:
                local_img_path = Path(img_path)
            
            if not local_img_path.exists():
                raise FileNotFoundError(f"Image not found: {local_img_path}")
            
            # Load and encode image at target resolution
            image = Image.open(local_img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                z_img = clip_encoder.encode_image(image_tensor).squeeze(0)
                z_img = F.normalize(z_img, dim=-1)  # Unit normalize
            
            # Store on CPU to save GPU memory
            embeddings.append(z_img.cpu())
            image_paths.append(str(img_path))
            
        except Exception as e:
            print(f"[PrecomputeUpsampler] Warning: Failed to process {img_path}: {e}")
            failed_indices.append(idx)
            # Append zero embedding as placeholder to maintain indexing
            embeddings.append(torch.zeros(512))
            image_paths.append(str(img_path))
            continue
    
    # Stack into tensor
    embeddings = torch.stack(embeddings)
    
    # Save to disk or S3
    embedding_data = {
        'embeddings': embeddings,
        'image_paths': image_paths,
        'resize_size': resize_size,
        'failed_indices': failed_indices
    }
    
    # Save locally first, then upload to S3 if needed
    if is_s3_uri(output_path):
        # Save to temp file, then upload to S3
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
            torch.save(embedding_data, tmp.name)
            tmp_path = tmp.name
        
        bucket, key = split_s3_uri(output_path)
        print(f"[PrecomputeUpsampler] Uploading to s3://{bucket}/{key}")
        s3_client.upload_file(tmp_path, bucket, key)
        os.remove(tmp_path)
        print(f"[PrecomputeUpsampler] Saved to s3://{bucket}/{key}")
    else:
        # Save to local path
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[PrecomputeUpsampler] Saving to {output_path}")
        torch.save(embedding_data, output_path)
        print(f"[PrecomputeUpsampler] Saved to {output_path}")
    
    print(f"[PrecomputeUpsampler] Done!")
    print(f"  - Embeddings shape: {embeddings.shape}")
    print(f"  - Successfully encoded: {len(embeddings) - len(failed_indices)}/{len(embeddings)}")
    if failed_indices:
        print(f"  - Failed indices: {failed_indices}")


def precompute_decoder_embeddings(
    metadata_path: str,
    images_dir: str,
    output_path: str,
    resize_size: int = 64,
    device: str = None,
    on_aws: bool = True
):
    """
    Precompute CLIP embeddings for decoder training (64x64).
    Same as precompute_upsampler_embeddings but for 64x64 images.
    """
    if device is None:
        device = (
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )
    
    print(f"[PrecomputeDecoder] Using device: {device}")
    print(f"[PrecomputeDecoder] Encoding images at {resize_size}x{resize_size} resolution")
    print(f"[PrecomputeDecoder] AWS mode: {on_aws}")
    
    s3_client = None
    if on_aws or is_s3_uri(metadata_path) or is_s3_uri(output_path):
        s3_client = boto3.client('s3')
    
    # Load metadata
    if is_s3_uri(metadata_path):
        bucket, key = split_s3_uri(metadata_path)
        print(f"[PrecomputeDecoder] Downloading metadata from s3://{bucket}/{key}")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
            s3_client.download_file(bucket, key, tmp.name)
            df = pd.read_csv(tmp.name)
            os.remove(tmp.name)
    else:
        df = pd.read_csv(metadata_path)
        print(f"[PrecomputeDecoder] Loaded metadata from {metadata_path}")
    
    print(f"[PrecomputeDecoder] Found {len(df)} images in metadata")
    print(f"[PrecomputeDecoder] Reading images from local directory: {images_dir}")
    
    print(f"[PrecomputeDecoder] Initializing CLIP encoder...")
    clip_encoder = CLIPEncoder().to(device).eval()
    
    transform = transforms.Compose([
        transforms.Resize((resize_size, resize_size), antialias=True),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1)
    ])
    
    embeddings = []
    image_paths = []
    failed_indices = []
    
    print("[PrecomputeDecoder] Encoding images...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_path = row['image_path']
        
        try:
            if not Path(img_path).is_absolute():
                local_img_path = Path(images_dir) / Path(img_path).name
            else:
                local_img_path = Path(img_path)
            
            if not local_img_path.exists():
                raise FileNotFoundError(f"Image not found: {local_img_path}")
            
            image = Image.open(local_img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                z_img = clip_encoder.encode_image(image_tensor).squeeze(0)
                z_img = F.normalize(z_img, dim=-1)
            
            embeddings.append(z_img.cpu())
            image_paths.append(str(img_path))
            
        except Exception as e:
            print(f"[PrecomputeDecoder] Warning: Failed to process {img_path}: {e}")
            failed_indices.append(idx)
            embeddings.append(torch.zeros(512))
            image_paths.append(str(img_path))
            continue
    
    embeddings = torch.stack(embeddings)
    
    embedding_data = {
        'embeddings': embeddings,
        'image_paths': image_paths,
        'resize_size': resize_size,
        'failed_indices': failed_indices
    }
    
    if is_s3_uri(output_path):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
            torch.save(embedding_data, tmp.name)
            tmp_path = tmp.name
        
        bucket, key = split_s3_uri(output_path)
        print(f"[PrecomputeDecoder] Uploading to s3://{bucket}/{key}")
        s3_client.upload_file(tmp_path, bucket, key)
        os.remove(tmp_path)
        print(f"[PrecomputeDecoder] Saved to s3://{bucket}/{key}")
    else:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[PrecomputeDecoder] Saving to {output_path}")
        torch.save(embedding_data, output_path)
        print(f"[PrecomputeDecoder] Saved to {output_path}")
    
    print(f"[PrecomputeDecoder] Done!")
    print(f"  - Embeddings shape: {embeddings.shape}")
    print(f"  - Successfully encoded: {len(embeddings) - len(failed_indices)}/{len(embeddings)}")


def precompute_both_resolutions(
    metadata_path: str,
    images_dir: str,
    output_dir: str,
    device: str = None,
    on_aws: bool = True
):
    """
    Convenience function to precompute embeddings for both 64x64 (decoder) 
    and 128x128 (upsampler) resolutions.
    
    Args:
        metadata_path: S3 URI or local path to metadata.csv
        images_dir: Local path to images directory on EC2/SageMaker
        output_dir: S3 URI or local path for output directory
        device: Device to use (auto-detected if None)
        on_aws: Whether running on AWS
    """
    # Construct output paths
    if is_s3_uri(output_dir):
        bucket, prefix = split_s3_uri(output_dir)
        prefix = prefix.rstrip('/')
        output_64 = f"s3://{bucket}/{prefix}/precomputed_embeddings_64x64.pt"
        output_128 = f"s3://{bucket}/{prefix}/precomputed_embeddings_128x128.pt"
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_64 = str(output_dir / "precomputed_embeddings_64x64.pt")
        output_128 = str(output_dir / "precomputed_embeddings_128x128.pt")
    
    print("="*70)
    print("PRECOMPUTING EMBEDDINGS FOR DECODER (64x64)")
    print("="*70)
    precompute_decoder_embeddings(
        metadata_path=metadata_path,
        images_dir=images_dir,
        output_path=output_64,
        resize_size=64,
        device=device,
        on_aws=on_aws
    )
    
    print("\n" + "="*70)
    print("PRECOMPUTING EMBEDDINGS FOR UPSAMPLER (128x128)")
    print("="*70)
    precompute_upsampler_embeddings(
        metadata_path=metadata_path,
        images_dir=images_dir,
        output_path=output_128,
        resize_size=128,
        device=device,
        on_aws=on_aws
    )
    
    print("\n" + "="*70)
    print("COMPLETED!")
    print("="*70)


if __name__ == '__main__':
    # SageMaker setup
    # Run from: /home/ec2-user/SageMaker/aws_diffusion_model
    
    import sys
    
    # Add the repo to Python path
    sys.path.insert(0, '/home/ec2-user/SageMaker/aws_diffusion_model')
    
    # Precompute 128x128 embeddings for upsampler
    # (64x64 embeddings already exist at /home/ec2-user/data/precomputed_embeddings.pt)
    
    precompute_upsampler_embeddings(
        metadata_path='/home/ec2-user/data/metadata.csv',
        images_dir='/home/ec2-user/data/train_img',
        output_path='/home/ec2-user/data/precomputed_embeddings_128x128.pt',
        resize_size=128,
        on_aws=False  # Save locally, not to S3
    )
    
    print("\n" + "="*70)
    print("Embeddings saved locally at:")
    print("  - 64x64:  /home/ec2-user/data/precomputed_embeddings.pt (existing)")
    print("  - 128x128: /home/ec2-user/data/precomputed_embeddings_128x128.pt (new)")
    print("="*70)