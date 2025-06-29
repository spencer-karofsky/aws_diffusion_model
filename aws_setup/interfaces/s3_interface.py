"""
Defines interfaces for all Amazon Simple Storage Service (S3) Functionalities.

Bucket Interface:
1. Create Bucket
2. Delete Bucket
3. List Buckets

Object Interface:
1. Upload Object
2. Download Object
3. Delete Object
4. List Objects
"""
from typing import Protocol
from typing import List, Tuple

class S3BucketInterface(Protocol):
    def _check_bucket_exists(self, bucket_name: str) -> bool:
        raise NotImplementedError
    
    def create_bucket(self, bucket_name: str) -> bool:
        raise NotImplementedError
    
    def delete_bucket(self, bucket_name: str) -> bool:
        raise NotImplementedError
    
    def list_buckets(self) -> List[str]:
        raise NotImplementedError

class S3ObjectInterface(Protocol):
    def _check_object_exists(self,
                             bucket_name: str,
                             object_key: str) -> bool:
        raise NotImplementedError
    
    def upload_object(self,
                      bucket_name: str,
                      object_key: str,
                      file_path: str) -> bool:
        raise NotImplementedError
    
    def download_object(self,
                        bucket_name: str,
                        object_key: str,
                        destination_path: str) -> bool:
        raise NotImplementedError
    
    def delete_object(self,
                      bucket_name: str,
                      object_key: str) -> bool:
        raise NotImplementedError
    
    def list_objects(self, bucket_name: str) -> List[Tuple[str]]:
        raise NotImplementedError