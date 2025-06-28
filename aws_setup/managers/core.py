import boto3
from aws_setup.utils.logger import logger
from aws_setup.interfaces.s3_interface import S3BucketInterface, S3ObjectInterface
from botocore.exceptions import ClientError

__all__ = [
    "boto3",
    "logger",
    "S3BucketInterface",
    "S3ObjectInterface",
    "ClientError"
]