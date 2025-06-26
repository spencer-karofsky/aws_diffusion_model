"""
Implements all S3 functionalities
"""
from core import *

class S3BucketManager(S3BucketInterface):
    def __init__(self):
        """Define S3 Client
        """
        self._client = boto3.client('s3')
    

    def _check_bucket_exists(self, bucket_name: str) -> bool:
        """Check if bucket already exists
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3/client/head_bucket.html
        Args:
            bucket_name: the bucket name
        Return:
            True/False if bucket exists/doesn't exist
        """
        try:
            self._client.head_bucket(Bucket=bucket_name)
            logger.info(f'[INFO] bucket "{bucket_name}" already exists and is accessible')
            return True
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code in ('404', 'NoSuchBucket'):
                return False
            else:
                logger.warning(f'[WARNING] Cannot access bucket "{bucket_name}": {error_code}')
                return False


    def create_bucket(self, bucket_name: str) -> bool:
        """Create S3 Bucket
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3/client/create_bucket.html
        Args:
            bucket_name: the name of the bucket
        Return:
            True/False to indicate success/failure 
        """
        # Check if bucket exists already (can't create a bucket that already exists)
        if self._check_bucket_exists(bucket_name):
            logger.info(f'[SKIP] bucket "{bucket_name}" already exists')
            return False # bucket already exists, so we can't create it

        # Create bucket if it doesn't exist
        try:
            self._client.create_bucket(Bucket=bucket_name)
            logger.info(f'[SUCCESS] created bucket "{bucket_name}"')
            return True
        except ClientError as e:
            logger.error(f'[FAIL] cannot create bucket "{bucket_name}" ({e})')
            return False
        
    def delete_bucket(self, bucket_name: str) -> bool:
        """Delete S3 Bucket
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3/client/delete_bucket.html
        Args:
            bucket_name: the name of the bucket
        Return:
            True/False to indicate success/failure
        """
        # Check if bucket exists already (can't delete a bucket that doesn't exist)
        if not self._check_bucket_exists(bucket_name):
            logger.info(f'[SKIP] bucket "{bucket_name}" does not exist')
            return False # bucket doesn't exist, so we can't delete it
        
        # Delete the bucket, assuming it exists
        try:
            self._client.delete_bucket(Bucket=bucket_name)
            logger.info(f'[SUCCESS] deleted bucket "{bucket_name}"')
            return True
        except ClientError as e:
            logger.error(f'[FAIL] cannot delete bucket "{bucket_name}" ({e})')
            return False