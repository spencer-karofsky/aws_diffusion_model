"""
Implements all VPC functionalities
"""
import boto3
from aws_setup.utils.logger import logger
from aws_setup.interfaces.vpc_interface import VPCSetupInterface, VPCNetworkInterface, VPCSecurityInterface
from botocore.exceptions import ClientError
from botocore.client import BaseClient
from typing import Optional, List


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

    def _enable_auto_assigned_ip(self) -> bool:
        """Allow auto-assigned IP
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ec2/client/modify_subnet_attribute.html
        Return:
            True/False to indicate success/failure
        """
        try:
            self.client.modify_subnet_attribute(MapPublicIpOnLaunch=True,
                                                SubnetId=self.subnet_id)
        except ClientError as e:
            logger.error(f'[FAIL] cannot auto-assign IP ({e})')
            return False
        logger.info(f'[SUCCESS] auto-assigned IP')
        return True

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
        logger.info(f'[SUCCESS] created VPC')

        # Save subnet id as class attribute for future use
        self.subnet_id = response['Subnet']['SubnetId']

        # Enable auto-assigned IP
        status = self._enable_auto_assigned_ip()
        if status:
            return True
        return False
    
    def _attach_internet_gateway(self) -> bool:
        """Attach internet gateway to subnet
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ec2/client/attach_internet_gateway.html
        Return:
            True/False to indicate success/failure
        """
        try:
            self.client.attach_internet_gateway(DryRun=True,
                                                InternetGatewayId=self.igw_id,
                                                VpcId=self.vpc_id)
        except ClientError as e:
            logger.error(f'[FAIL] cannot attach internet gateway ({e})')
            return False
        logger.info(f'[SUCCESS] attached internet gateway')
        return True
    
    def create_internet_gateway(self) -> bool:
        """Create and attach internet gateway
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ec2/client/create_internet_gateway.html
        Return:
            True/False to indicate success/failure
        """
        try:
            response = self.client.create_internet_gateway(TagSpecifications=[{'ResourceType': 'internet-gateway'}],
                                                           DryRun=True)
        except ClientError as e:
            logger.error(f'[FAIL] cannot create internet gateway ({e})')
            return False
        logger.info(f'[SUCCESS] created internet gateway')

        # Save Internet Gateway ID
        self.igw_id = response['InternetGateway']['InternetGatewayId']

        # Attach Internet Gateway
        status = self._attach_internet_gateway()
        if status:
            return True
        return False
    
    def create_route_table(self) -> bool:
        """Attach route table and attach internet gateway
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ec2/client/create_route_table.html
        Return:
            True/False to indicate success/failure
        """
        try:
            response = self.client.create_route_table(TagSpecifications=[{'ResourceType': 'route-table'}],
                                                      DryRun=True,
                                                      VpcId=self.vpc_id)
        except ClientError as e:
            logger.error(f'[FAIL] cannot create route table ({e})')
            return False
        logger.info(f'[SUCCESS] created route table')

        # Save route table id
        self.route_table_id = response['RouteTable']['RouteTableId']
        
        return True
    
    def add_route(self, destination_cidr: str) -> bool:
        """Add public route to route table
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ec2/client/create_route.html
        Args:
            destination_cidr: specifies the range of IP addresses that route applies to
        Return:
            True/False to indicate success/failure
        """
        try:
            self.client.create_route(RouteTableId=self.route_table_id,
                                     DestinationCidrBlock=destination_cidr,
                                     GatewayId=self.igw_id,
                                     DryRun=True)
        except ClientError as e:
            logger.error(f'[FAIL] cannot create route and add to the route table ({e})')
            return False
        logger.info(f'[SUCCESS] created route and added to the route table')
        return True
    
    def list_routes(self) -> List[str]:
        """Retrieve/return all routes in the route table
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ec2/client/describe_route_tables.html
        Return:
            list of destination CIDRs
        """
        try:
            response = self.client.describe_route_tables(DryRun=True)
        except ClientError as e:
            logger.error(f'[FAIL] cannot retrieve routes in the route table ({e})')
            return []
        logger.info(f'[SUCCESS] retrieved routes in the route table')
        return [r['DestinationCidrBlock'] for r in response['RouteTables']['Routes']]
    
    def delete_route(self, destination_cidr: str) -> bool:
        """Delete route in route table
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ec2/client/delete_route.html
        Args:
            destination_cidr: identifies the route to delete
        Return:
            True/False to indicate success/failure
        """
        try:
            self.client.delete_route(RouteTableId=self.route_table_id,
                                     DestinationCidrBlock=destination_cidr,
                                     DryRun=True)
        except ClientError as e:
            logger.error(f'[FAIL] cannot delete route in the route table ({e})')
            return False
        logger.info(f'[SUCCESS] deleted route in the route table')
        return True
    
    def associate_route_table(self, subnet_id: str) -> bool:
        """Associate subnet with route table in the VPC
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ec2/client/associate_route_table.html
        Args:
            subnet_id: the ID of the subnet
        Return:
            True/False to indicate success/failure
        """
        try:
            self.client.associate_route_table(DryRun=True,
                                              SubnetId=subnet_id,
                                              RouteTableId=self.route_table_id)
        except ClientError as e:
            logger.error(f'[FAIL] cannot associate subnet with the route table ({e})')
            return False
        logger.info(f'[SUCCESS] associated subnet with the route table')
        return True

class VPCSecurityManager(VPCSecurityInterface):
    pass