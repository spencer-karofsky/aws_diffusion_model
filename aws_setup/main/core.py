import boto3
from aws_setup.utils.logger import logger
from aws_setup.clients.s3_client import S3ClientInterface
from botocore.exceptions import ClientError