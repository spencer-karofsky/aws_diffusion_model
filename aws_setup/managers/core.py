import boto3
from aws_setup.utils.logger import logger
from aws_setup.interfaces.s3_client import S3BucketInterface, S3ObjectInterface
from botocore.exceptions import ClientError