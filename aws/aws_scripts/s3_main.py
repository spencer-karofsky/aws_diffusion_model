"""
Create S3 infrastructure after validating in s3_validation.py.
"""
import sys
import os

# Looks at project root directory; used for finding aws_setup/tests
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# We can now import from different directory
from aws_setup.managers.s3_manager import S3BucketManager, S3ObjectManager

if __name__ == "__main__":
    # Instantiate S3BucketManager
    bucket_manager = S3BucketManager()

    # Create S3 buckets
    s3_buckets = ['ddpm-project-data',
                  'ddpm-project-models',
                  'ddpm-project-outputs']
    
    for name in s3_buckets:
        bucket_manager.create_bucket(name)
    
    print('End Process')


