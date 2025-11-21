"""
training_data_transfer.py: Multithreaded upload of a local training dataset to S3.

- Overwrites existing keys.
- Uses a thread pool for speed.
- Shows a tqdm progress bar.
- Prints intermediate stats every 1000 uploads.
"""

import os
import sys
import time
from pathlib import Path
from typing import Iterable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from tqdm import tqdm

# If you need project root on path for anything else:
sys.path.append(str(Path(__file__).resolve().parent.parent))


class TrainingDataUploader:
    def __init__(
        self,
        local_dir: str,
        bucket_name: str,
        s3_prefix: str = "",
        metadata_path: str | None = None,
        max_workers: int | None = None,
        print_every: int = 1000,
        retries: int = 3,
        retry_backoff: float = 1.0,
    ):
        self.local_dir = Path(local_dir).resolve()
        self.bucket_name = bucket_name
        self.s3_prefix = s3_prefix.strip("/")
        self.metadata_path = Path(metadata_path).resolve() if metadata_path else None
        self.print_every = int(print_every)
        self.retries = int(retries)
        self.retry_backoff = float(retry_backoff)

        if not self.local_dir.exists() or not self.local_dir.is_dir():
            raise ValueError(f"[ERROR] Directory not found: {self.local_dir}")
        if self.metadata_path and not self.metadata_path.exists():
            raise ValueError(f"[ERROR] Metadata file not found: {self.metadata_path}")

        # Reasonable default: plenty of I/O concurrency without going wild
        if max_workers is None:
            cpu = os.cpu_count() or 4
            max_workers = min(64, cpu * 8)
        self.max_workers = int(max_workers)

        # boto3 clients are generally thread-safe to use concurrently
        self.s3 = boto3.client("s3")

        # stats
        self._lock = threading.Lock()
        self._uploaded = 0
        self._failed = 0

    def _iter_files(self) -> Iterable[Path]:
        for root, _, names in os.walk(self.local_dir):
            for name in names:
                yield Path(root) / name

    def _key_for(self, path: Path) -> str:
        rel = path.relative_to(self.local_dir).as_posix()
        return f"{self.s3_prefix}/{rel}" if self.s3_prefix else rel

    def _upload_with_retries(self, local_path: Path, key: str) -> bool:
        for attempt in range(1, self.retries + 1):
            try:
                self.s3.upload_file(str(local_path), self.bucket_name, key)
                return True
            except (BotoCoreError, ClientError) as e:
                if attempt == self.retries:
                    return False
                time.sleep(self.retry_backoff * attempt)
        return False

    def _task(self, local_path: Path, key: str) -> bool:
        ok = self._upload_with_retries(local_path, key)
        with self._lock:
            if ok:
                self._uploaded += 1
                # print intermediate stats every N, as requested
                if self.print_every and (self._uploaded % self.print_every == 0):
                    print(
                        f"\n[INTERMEDIATE] Uploaded: {self._uploaded} | Failed: {self._failed}"
                    )
            else:
                self._failed += 1
        return ok

    def upload_all(self) -> None:
        files = list(self._iter_files())
        total = len(files) + (1 if self.metadata_path else 0)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool, tqdm(
            total=total, desc="Uploading", unit="file"
        ) as pbar:
            futures: list[Tuple[Path, str]] = []
            for f in files:
                key = self._key_for(f)
                futures.append(pool.submit(self._task, f, key))

            # metadata last
            if self.metadata_path:
                meta_key = (
                    f"{self.s3_prefix}/metadata.csv" if self.s3_prefix else "metadata.csv"
                )
                futures.append(pool.submit(self._task, self.metadata_path, meta_key))

            for fut in as_completed(futures):
                _ = fut.result()
                pbar.update(1)

        print("\n[SUMMARY]")
        print(f"Total files processed: {total}")
        print(f"✓ Uploaded: {self._uploaded}")
        print(f"✗ Failed:   {self._failed}")


if __name__ == "__main__":
    uploader = TrainingDataUploader(
        local_dir="/Users/spencerkarofsky/Desktop/projects/aws_diffusion_model/dalle2/data/local_datasets/unsplash/images_cropped",
        bucket_name="dalle2-data",
        s3_prefix="train_img",
        metadata_path="dalle2/data/local_datasets/midjourney_v6/metadata.csv",
        max_workers=None,
        print_every=1000,
        retries=3,
        retry_backoff=1.0,
    )
    uploader.upload_all()
