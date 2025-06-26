"""
Handles S3 bucket creation and validation for storing datasets, models, and outputs.
"""
from core import *

class S3ClientInterface(Protocol):
    """Class to serve as the interface (for testing)"""
    def create_bucket(self, bucket_name: str) -> bool:
        raise NotImplementedError