"""
Defines interfaces for all S3 Functionalities
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