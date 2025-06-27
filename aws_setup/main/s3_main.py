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
        
class S3ObjectManager(S3ObjectInterface):
    def __init__(self):
        """Define S3 Client
        """
        self._client = boto3.client('s3')
    
    def upload_object(self,
                      bucket_name: str,
                      object_key: str,
                      file_path: str) -> bool:
        """Uploads file to S3 bucket
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-uploading-files.html
        Args:
            bucket_name: the S3 Bucket name
            object_key: the key/identifier of the bucket (use '/' to create AWS-recognized virtual folders inside bucket)
            file_path: the local file path to upload
        Return:
            True/False to indicate success/failure of uploading the object
        """
        try:
            self._client.upload_file(file_path, bucket_name, object_key)
        except ClientError as e:
            logger.error(f'[FAIL] cannot upload object in "{file_path}" ({e})')
            return False
        logger.info(f'[SUCCESS] uploaded object in "{file_path}"')
        return True
    
    def download_object(self,
                        bucket_name: str,
                        object_key: str,
                        destination_path: str) -> bool:
        """Download fole from S3 Bucket
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-example-download-file.html
        Args:
            bucket_name: the S3 Bucket name
            object_key: the key/identifier of the bucket (use '/' to create AWS-recognized virtual folders inside bucket)
            destination_path: the local file path to download
        Return:
            True/False to indicate success/failure
        """
        try:
            self._client.download_file(bucket_name, object_key, destination_path)
        except ClientError as e:
            logger.error(f'[FAIL] cannot download object to "{destination_path}" ({e})')
            return False
        logger.info(f'[SUCCESS] downloaded object to "{destination_path}"')
        return True
    
    def list_objects(self, bucket_name: str) -> bool:
        """List objects in a S3 Bucket
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3/client/list_objects.html
        Args:
            bucket_name: the S3 Bucket name

        """
        try:
            response = self._client.list_objects_v2(Bucket=bucket_name)
        except ClientError as e:
            logger.error(f'[FAIL] cannot list objects in bucket "{bucket_name}" ({e})')
            return False

        contents = response.get('Contents', [])
        if not contents:
            logger.info(f'[INFO] bucket "{bucket_name}" is empty')
            return True

        for i, obj in enumerate(contents):
            try:
                key = obj['Key']
                last_modified = obj['LastModified']
                size_mb = round(obj['Size'] / (1024 * 1024), 2)
                logger.info(f'{i}: key={key}, last_modified={last_modified}, size={size_mb} MB')
            except KeyError as e:
                logger.warning(f'[WARN] Malformed object metadata in bucket "{bucket_name}": missing {e}')
                continue
            
        return True