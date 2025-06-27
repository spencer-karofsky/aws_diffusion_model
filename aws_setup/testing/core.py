import boto3
from aws_setup.utils.logger import logger
from aws_setup.main.s3_main import S3BucketManager, S3ObjectManager
from botocore.exceptions import ClientError
import unittest
from moto import mock_aws
import os