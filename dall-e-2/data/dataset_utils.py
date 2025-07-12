"""
dataset.py: Contains classes to download different datasets

Description:
    - All datasets sourced from Hugging Face
    - Contains a base Dataset class and then several (Hugging Face) datasets, combined by one large dataset class:
        1. CortexLM/midjourney-v6:
            - Contains (text, image)-pairs from generated images from Midjourney's V6 text-to-image model, conditioned on real prompts.
                - Consideration: Images are synthetic (they are outputs of the V6 model and are not real images).
                    - However, the images look really good and are certainly useful in many contexts.
                    - Solution: Combine with other datasets, combining Midjourney's stunning images with other datasets' realism.
        2. apple/DataCompDR-1B: 
            - Raw scraped image-text pairs from Common Crawl (~12.8B pairs).
            - Contains (text, image)-pairs.
            - Images are web-scrapped.
            - Text is the accompanying HTML text.
            - Text and images are filtered by cosine similarity from CLIP embeddings.
        3. kdexd/red_caps:
            - Large-scale dataset of 12M image-text pairs collected from Reddit.
            - Data is collected from a manually curated set of subreddits (350 total).

Usage:
    from datasets import MergedDataset
    dataset = MergedDataset(...)

    TODO: Update!
    # Download entire dataset
    dataset.download_all(...)

    # Download partial dataset
    dataset.download_partial(...)

    # Download one dataset of all options
    dataset.download_[dataset](...)

Classes:
    - BaseDataset: Parent abstract class that also inherits from torch.utils.data.Dataset.
        - MidjourneyDataset: Child of BaseDataset that assembles dataset from CortexLM/midjourney-v6.
        - DataCompDataset: Child of BaseDataset that assembles dataset from apple/DataCompDR-1B.
        - RedditDataset: Child of BaseDataset that assembles dataset from kdexd/red_caps.
    - MergedDataset: Combines all of the datasets that inherit from BaseDataset and provides control over the weight of each dataset.

References:
    - https://huggingface.co/datasets/CortexLM/midjourney-v6
    - https://huggingface.co/datasets/apple/DataCompDR-1B
    - https://huggingface.co/datasets/kdexd/red_caps

Author:
    - Spencer Karofsky (https://github.com/spencer-karofsky)
"""
import torch
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
import pandas as pd
from typing import Tuple, List
import numpy as np
from datasets import load_dataset
import os
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Parent (abstract) class that also inherits from torch's Dataset class
class BaseDataset(ABC, Dataset):
    @abstractmethod
    def __init__(self):
        """Initializes base dataset.
        """
        super().__init__()
        self.save_dir = '/datasets'

    @abstractmethod
    def download(self, n_images: int) -> bool:
        """Downloads images from the dataset.
        Args:
            n_images: number of images to download
        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def load_metadata(self) -> pd.DataFrame:
        """Loads and returns dataset metadata.
        Returns:
            Metadata table with at least 'url' and 'caption' columns.
        """
        pass
        
    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        """Returns the (image, caption) pair for a given index.
        Args:
            index: the index of the iterable
        Returns:
            The (caption, image) pair
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Compute the number of available samples in the dataset.
        Returns:
            Number of available samples in the dataset
        """
        pass

# Sub-classes of BaseDataset
class MidjourneyDataset(BaseDataset):
    def __init__(self, save_dir='../data/data_scripts/datasets/midjourney', transform=None):
        super().__init__()
        self.save_dir = os.path.abspath(save_dir)
        self.transform = transform
        self.metadata_path = os.path.join(self.save_dir, "metadata.csv")

        if os.path.isfile(self.metadata_path):
            # Load existing metadata with captions
            self.df = pd.read_csv(self.metadata_path)

            # Fix image paths to be absolute if they aren't already
            self.df["image_path"] = self.df["image_path"].apply(
                lambda p: p if os.path.isabs(p) else os.path.join(self.save_dir, os.path.basename(p))
            )
        else:
            # Fallback: no metadata.csv, build from image files (blank captions)
            print("metadata.csv not found — rebuilding metadata from image files...")
            image_files = sorted([
                f for f in os.listdir(self.save_dir)
                if f.endswith(".jpg") and os.path.isfile(os.path.join(self.save_dir, f))
            ])
            if not image_files:
                raise RuntimeError(f"No images found in {self.save_dir}")

            self.df = pd.DataFrame({
                "image_path": [os.path.join(self.save_dir, f) for f in image_files],
                "caption": ["" for _ in image_files]
            })
    
    def _split_into_quadrants(self, img: Image.Image) -> list:
        w, h = img.size
        w2, h2 = w // 2, h // 2
        return [
            img.crop((0,     0,    w2,  h2)),  # top-left
            img.crop((w2,    0,    w,   h2)),  # top-right
            img.crop((0,     h2,   w2,  h)),   # bottom-left
            img.crop((w2,    h2,   w,   h)),   # bottom-right
        ]

    def download(self, n_images: int = 1000) -> bool:
        try:
            print("Loading metadata from Hugging Face...")
            df = pd.read_parquet("hf://datasets/CortexLM/midjourney-v6/data/train-00000-of-00001.parquet")
            df = df.dropna(subset=["image_url", "prompt"]).head(n_images).reset_index(drop=True)

            print("Starting parallel download...")
            saved = []

            def fetch_and_save(idx_row):
                idx, row = idx_row
                url, caption = row["image_url"], row["prompt"]
                try:
                    resp = requests.get(url, timeout=10)
                    resp.raise_for_status()
                    image = Image.open(BytesIO(resp.content)).convert("RGB")
                    quadrants = self._split_into_quadrants(image)
                    results = []
                    for i, quad in enumerate(quadrants):
                        fname = os.path.join(self.save_dir, f"{idx:06d}_{i}.jpg")
                        quad.save(fname)
                        results.append((fname, caption))
                    return results  # list of (fname, caption)
                except Exception as e:
                    return None

            with ThreadPoolExecutor(max_workers=16) as executor:
                futures = [executor.submit(fetch_and_save, item) for item in df.iterrows()]
                for f in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
                    result = f.result()
                    if result is not None:
                        saved.extend(result)  # add all four cropped images

            if saved:
                df = pd.DataFrame(saved, columns=["image_path", "caption"])
                df.to_csv(self.metadata_path, index=False)
                self.df = df
                print(f"Downloaded and cropped {len(df)} image quadrants.")
                return True
            else:
                print("No images downloaded.")
                return False

        except Exception as e:
            print(f"Download failed: {e}")
            return False


    def load_metadata(self) -> pd.DataFrame:
        if self.df is not None:
            return self.df
        elif os.path.exists(self.metadata_path):
            self.df = pd.read_csv(self.metadata_path)
            return self.df
        else:
            raise RuntimeError("Metadata not found. Please run `download()` first.")

    def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], str]:
        row = self.df.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        caption = row["caption"]

        if self.transform:
            image = self.transform(image)

        return image, caption


    def __len__(self) -> int:
        if self.df is None:
            self.load_metadata()
        return len(self.df)
