"""
Handles S3 bucket creation and validation for storing datasets, models, and outputs.
"""
import logging
import boto3
from botocore.exceptions import ClientError
from typing import Protocol

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class S3ClientInterface(Protocol):
    """Class to serve as the interface (for testing)"""
    def create_bucket(self, bucket_name: str) -> bool:
        raise NotImplementedError

class BotoS3Client(S3ClientInterface):
    def __init__(self):
        """Define S3 Client"""
        self._client = boto3.client('s3')

    def create_bucket(self, bucket_name: str) -> bool:
        """Create S3 Bucket
        
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3/client/create_bucket.html
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3/client/head_bucket.html
        Args:
            bucket_name: the name of the bucket
        Return:
            True/False to indicate success/failure 
        """
        # Check if bucket exists already
        try:
            self._client.head_bucket(Bucket=bucket_name)
            logger.info(f'[INFO] bucket "{bucket_name}" already exists and is accessible')
            return False
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code in ('404', 'NoSuchBucket'):
                pass # Proceed to bucket creation
            else:
                logger.warning(f'[WARNING] Cannot access bucket "{bucket_name}": {error_code}')
                return False

        # Create bucket if it doesn't exist
        try:
            self._client.create_bucket(Bucket=bucket_name)
            logger.info(f'[SUCCESS] created bucket "{bucket_name}"')
            return True
        except ClientError as e:
            logger.error(f'[FAIL] cannot create bucket "{bucket_name}" ({e})')
            return False