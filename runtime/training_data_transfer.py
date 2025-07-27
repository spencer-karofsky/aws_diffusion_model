"""
training_data_transfer.py: Uploads a local training dataset to S3.

Description:
    * Recursively traverses a local directory and uploads all files to a specified S3 bucket and prefix.
    * Uses S3ObjectManager to avoid duplicate uploads.
    * Also uploads a metadata CSV file if provided.

Classes:
    * TrainingDataUploader: Handles directory traversal and S3 upload logic.

Author:
    * Spencer Karofsky (https://github.com/spencer-karofsky)
"""
import os
from pathlib import Path
from aws.aws_setup.managers.s3_manager import S3ObjectManager

class TrainingDataUploader:
    def __init__(
        self,
        local_dir: str,
        bucket_name: str,
        s3_prefix: str = "",
        metadata_path: str = None
    ):
        """
        Args:
            local_dir: Path to the local directory containing training data.
            bucket_name: Name of the target S3 bucket.
            s3_prefix: Optional prefix (folder path) within the bucket.
            metadata_path: Optional path to a metadata CSV file to upload alongside.
        """
        self.local_dir = Path(local_dir).resolve()
        self.bucket_name = bucket_name
        self.s3_prefix = s3_prefix.strip('/')
        self.metadata_path = Path(metadata_path).resolve() if metadata_path else None
        self.s3 = S3ObjectManager()

        if not self.local_dir.exists() or not self.local_dir.is_dir():
            raise ValueError(f'[ERROR] Directory not found: {self.local_dir}')
        if self.metadata_path and not self.metadata_path.exists():
            raise ValueError(f'[ERROR] Metadata file not found: {self.metadata_path}')

    def upload_all(self) -> None:
        """
        Upload all files under the local directory to S3 under the specified prefix.
        Also uploads metadata CSV if provided.
        """
        print(f'[INFO] Uploading training images from: {self.local_dir}')
        for root, _, files in os.walk(self.local_dir):
            for file in files:
                local_path = Path(root) / file
                relative_path = local_path.relative_to(self.local_dir)
                s3_key = f'{self.s3_prefix}/{relative_path.as_posix()}' if self.s3_prefix else relative_path.as_posix()

                success = self.s3.upload_object(self.bucket_name, s3_key, str(local_path))
                if success:
                    print(f'[UPLOADED] {local_path} → s3://{self.bucket_name}/{s3_key}')
                else:
                    print(f'[SKIPPED] {local_path} already exists or failed')

        if self.metadata_path:
            s3_key = f'{self.s3_prefix}/metadata.csv' if self.s3_prefix else 'metadata.csv'
            print(f'[INFO] Uploading metadata: {self.metadata_path}')
            success = self.s3.upload_object(self.bucket_name, s3_key, str(self.metadata_path))
            if success:
                print(f'[UPLOADED] metadata.csv → s3://{self.bucket_name}/{s3_key}')
            else:
                print(f'[SKIPPED] metadata.csv already exists or failed')


if __name__ == '__main__':
    uploader = TrainingDataUploader(
        local_dir='dalle2/data/local_datasets/midjourney_v6/images',
        bucket_name='dalle2-data',
        s3_prefix='training_img',
        metadata_path='dalle2/data/local_datasets/midjourney_v6/metadata.csv'
    )
    uploader.upload_all()
