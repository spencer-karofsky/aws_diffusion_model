"""
Implements all VPC functionalities
"""
import boto3
from aws_setup.utils.logger import logger
from aws_setup.interfaces.vpc_interface import VPCSetupInterface, VPCNetworkInterface, VPCSecurityInterface
from botocore.exceptions import ClientError
from botocore.client import BaseClient
from typing import Optional


class VPCSetupManager(VPCSetupInterface):
    def __init__(self, ec2_client: BaseClient):
        """Define VPC client
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ec2/vpc/index.html
        Args:
            ec2_client: the injected EC2 client
        """
        self.client = ec2_client.Vpc('id')
        self.cidr_block = '10.0.0.0/16'
        self.vpc_id = None
        self.dns = False # disabled
    
    def get_cidr_block(self) -> str:
        """Get CIDR block address
        """
        return self.cidr_block
    
    def get_vpc_id(self) -> str:
        """Get VPC id
        """
        return self.vpc_id
    
    def get_dns_status(self) -> bool:
        """Get status of DNS support (True=enabled, False=disabled)
        """
        return self.dns
    
    def _enable_dns(self) -> bool:
        """Enables DNS in the VPC
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ec2/client/modify_vpc_attribute.html
        Return:
            True/False to indicate success/failure
        """
        try:
            self.client.modify_vpc_attribute(EnableDnsSupport=True)
        except ClientError as e:
            logger.error(f'[FAIL] cannot enable DNS support ({e})')
            return False
        logger.info(f'[SUCCESS] enabled DNS support')
        self.dns = True
        return True

    def create_vpc(self, vpc_name: Optional[str] = None) -> bool:
        """Creates a VPC
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ec2/client/create_vpc.html
        Args:
            vpc_name: the name of the vpc (not required, but useful for debugging and console view)
        Return:
            True/False to indicate success/failure
        """
        try:
            response = self.client.create_vpc(CidrBlock=self.cidr_block,
                                              InstanceTenancy='default', # default for free tier
                                              TagSpecifications=[{'ResourceType': 'vpc', 'Tags': [{'Key': 'Name', 'Value': vpc_name}]}])
        except ClientError as e:
            logger.error(f'[FAIL] cannot create VPC ({e})')
            return False
        
        logger.info(f'[SUCCESS] created VPC')
        self.vpc_id = response['Vpc']['VpcId']

        # Enable DNS
        self._enable_dns()

        return True

class VPCNetworkManager(VPCNetworkInterface):
    def __init__(self, ec2_client: BaseClient, vpc_id: Optional[str]):
        """Define network instance
        Args:
            ec2_client: the injected EC2 client
            vpc_id: the id of the VPC (returned by VPCSetupManager.get_vpc_id)
        """
        self.client = ec2_client
        self.vpc_id = vpc_id

    def create_subnet(self, cidr_block: str) -> bool:
        """Create public VPC subnet and enable auto_assigned IP
        Args:
            cidr_block: the CIDR block subset
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ec2/client/create_subnet.html
        Return:
            True/False to indicate success/failure
        """
        self.availability_zone = 'us-east-1a' # Only building small system, so no need to change
        try:
            response = self.client.create_subnet(VpcId=self.vpc_id,
                                                 CidrBlock=cidr_block,
                                                 AvailabilityZone=self.availability_zone,
                                                 TagSpecifications=[{'ResourceType': 'subnet',
                                                                     'Tags': [{'Key': 'Name', 'Value': 'MyPublicSubnet'}]}])
        except ClientError as e:
            logger.error(f'[FAIL] cannot create subnet ({e})')
            return False
        logger.info(f'[SUCCESS] created VPC ({e})')

        # Save subnet id as class attribute for future use
        self.subnet_id = response['Subnet']['SubnetId']

        return True
    