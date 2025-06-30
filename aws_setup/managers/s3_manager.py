"""
Implements all S3 functionalities.
"""
import boto3
from aws_setup.utils.logger import logger
from aws_setup.interfaces.s3_interface import S3BucketInterface, S3ObjectInterface
from botocore.exceptions import ClientError
from typing import List, Tuple

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
            logger.warning(f'[WARNING] bucket "{bucket_name}" does not exist')
            return False # bucket doesn't exist, so we can't delete it
        
        # Delete the bucket, assuming it exists
        try:
            self._client.delete_bucket(Bucket=bucket_name)
            logger.info(f'[SUCCESS] deleted bucket "{bucket_name}"')
            return True
        except ClientError as e:
            logger.error(f'[FAIL] cannot delete bucket "{bucket_name}" ({e})')
            return False
    
    def list_buckets(self) -> List[str]:
        """List S3 Directory Buckets
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3/client/list_directory_buckets.html
        Returns:
            list of S3 directory bucket names
        """
        try:
            response = self._client.list_directory_buckets()
        except ClientError as e:
            logger.error(f'[FAIL] cannot retrieve bucket names ({e})')
            return []
        logger.info(f'[SUCCESS] retrieved S3 directory buckets')
        return response['Buckets']

class S3ObjectManager(S3ObjectInterface):
    def __init__(self):
        """Define S3 Client
        """
        self._client = boto3.client('s3')
    
    def _check_object_exists(self,
                             bucket_name: str,
                             object_key: str) -> bool:
        """Check if object exists
        Docs:
            https://stackoverflow.com/questions/33842944/check-if-a-key-exists-in-a-bucket-in-s3-using-boto3
        Args:
            bucket_name: the S3 bucket name
            object_key: the key/identifier of the bucket (use '/' to create AWS-recognized virtual folders inside bucket)
        Return:
            True/False if object exists/does not exist
        """
        try:
            self._client.head_object(Bucket=bucket_name, Key=object_key)
            return True
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code == "404":
                logger.info(f'[INFO] Object "{object_key}" does not exist in bucket "{bucket_name}".')
                return False
            elif code == "403":
                logger.warning(f'[WARN] Access denied when checking object "{object_key}" in bucket "{bucket_name}".')
                return False
            else:
                logger.error(f'[FAIL] Unexpected error when checking object "{object_key}": {e}')
                return False
    
    def upload_object(self,
                      bucket_name: str,
                      object_key: str,
                      file_path: str) -> bool:
        """Uploads file to S3 bucket and prevents overwrites
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
            if not self._check_object_exists(bucket_name, object_key):
                self._client.upload_file(file_path, bucket_name, object_key)
            else:
                return False
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
    
    def delete_object(self,
                      bucket_name: str,
                      object_key: str) -> bool:
        """Delete object from the bucket
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3/client/delete_object.html
        Args:
            bucket_name: the S3 Bucket name
            object_key: the key/identifier of the bucket
        Return:
            True/False to indicate success/failure of operation
        """
        try:
            self._client.delete_object(Bucket=bucket_name,
                                       Key=object_key)
        except ClientError as e:
            logger.error(f'[FAIL] cannot delete object "{object_key}" in bucket "{bucket_name}" ({e})')
            return False
        logger.info(f'[SUCCESS] deleted object "{object_key}" in bucket "{bucket_name}"')
        return True
    
    def list_objects(self, bucket_name: str) -> List[Tuple[str]]:
        """List objects in a S3 Bucket

        Limited to 1000 objects maximum
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3/client/list_objects.html
        Args:
            bucket_name: the S3 Bucket name
        """
        try:
            response = self._client.list_objects_v2(Bucket=bucket_name)
        except ClientError as e:
            logger.error(f'[FAIL] cannot list objects in bucket "{bucket_name}" ({e})')
            return []

        contents = response.get('Contents', [])
        if not contents:
            logger.info(f'[INFO] bucket "{bucket_name}" is empty')
            return []

        objects_list = []
        for obj in contents:
            try:
                key = obj['Key']
                last_modified = obj['LastModified']
                size_mb = round(obj['Size'] / (1024 * 1024), 2)
                objects_list.append((key, last_modified, size_mb))
            except KeyError as e:
                logger.warning(f'[WARN] Malformed object metadata in bucket "{bucket_name}": missing {e}')
                continue
        
        if len(objects_list) == 1000:
            logger.warning(f'[WARNING] Only first 1000 objects returned')

        return objects_list