"""
Implements all VPC functionalities.
"""
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
        self.client = ec2_client
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
            self.client.modify_vpc_attribute(VpcId=self.vpc_id,
                                             EnableDnsSupport={'Value': True})
        except ClientError as e:
            logger.error(f'[FAIL] cannot enable DNS support ({e})')
            return False
        logger.info(f'[SUCCESS] enabled DNS support')
        self.dns = True
        return True

    def create_vpc(self, vpc_name: str = None) -> bool:
        """Creates a VPC
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ec2/client/create_vpc.html
        Args:
            vpc_name: the name of the vpc (not required, but useful for debugging and console view)
        Return:
            True/False to indicate success/failure
        """
        if not vpc_name:
            vpc_name = 'MyVPC'
        try:
            response = self.client.create_vpc(CidrBlock=self.cidr_block,
                                              InstanceTenancy='default', # default for free tier
                                              TagSpecifications=[{'ResourceType': 'vpc', 'Tags': [{'Key': 'Name', 'Value': vpc_name}]}])
        except ClientError as e:
            logger.error(f'[FAIL] cannot create VPC ({e})')
            return False
        
        self.vpc_id = response['Vpc']['VpcId']
        logger.info(f'[SUCCESS] created VPC with id "{self.vpc_id}"')

        # Enable DNS
        self._enable_dns()

        return True
    
    def _get_vpc_id_list(self):
        """Get list of VPC IDs
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ec2/client/describe_vpcs.html
        Return:
            List of VPC IDs
        """
        try:
            response = self.client.describe_vpcs()
        except ClientError as e:
            logger.error(f'[FAIL] cannot describe VPCs ({e})')
            return []
        vpc_ids = [vpc['VpcId'] for vpc in response.get('Vpcs', [])]
        return vpc_ids
    
    def delete_vpc(self, vpc_id: str = None) -> bool:
        """Deletes the VPC. If no VPC ID is provided, it deletes all VPCs
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ec2/client/delete_vpc.html
        Args:
            vpc_id: the ID of the VPC. Pass in as parameter in case we just want to delete an existing VPC even if this object did not create it.
        Return:
            True/False to indicate success/failure
        """
        if not vpc_id:
            vpc_ids = self._get_vpc_id_list()
            # Delete each VPC
            if len(vpc_ids) > 0:
                for vpc_id in vpc_ids:
                    try:
                        self.client.delete_vpc(VpcId=vpc_id)
                        logger.info(f'[SUCCESS] deleted VPC "{vpc_id}"')
                    except ClientError as e:
                        logger.error(f'[FAIL] cannot delete VPC "{vpc_id}" ({e})')
                        return False
                return True
            return False
        else:
            try:
                self.client.delete_vpc(VpcId=vpc_id)
            except ClientError as e:
                logger.error(f'[FAIL] cannot delete VPC "{vpc_id}" ({e})')
                return False
            
            logger.info(f'[SUCCESS] deleted VPC "{vpc_id}"')
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
            self.client.modify_subnet_attribute(MapPublicIpOnLaunch={'Value': True},
                                                SubnetId=self.subnet_id)
        except ClientError as e:
            logger.error(f'[FAIL] cannot auto-assign IP ({e})')
            return False
        logger.info(f'[SUCCESS] auto-assigned IP')
        return True

    def create_subnet(self, cidr_block: str, subnet_name: str = 'MyPublicSubnet') -> bool:
        """Create public VPC subnet and enable auto_assigned IP
        Args:
            cidr_block: the CIDR block subset
            subnet_name: the name for the subnet
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
                                                                     'Tags': [{'Key': 'Name', 'Value': subnet_name}]}])
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
            self.client.attach_internet_gateway(InternetGatewayId=self.igw_id,
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
            response = self.client.create_internet_gateway(TagSpecifications=[{
                                                                'ResourceType': 'internet-gateway',
                                                                'Tags': [{'Key': 'Name', 'Value': 'MyInternetGateway'}]
                                                            }])
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
            response = self.client.create_route_table(TagSpecifications=[{
                                                            'ResourceType': 'route-table',
                                                            'Tags': [{'Key': 'Name', 'Value': 'MyInternetGateway'}]
                                                        }],
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
                                     GatewayId=self.igw_id)
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
            response = self.client.describe_route_tables(RouteTableIds=[self.route_table_id])
        except ClientError as e:
            logger.error(f'[FAIL] cannot retrieve routes in the route table ({e})')
            return []
        logger.info(f'[SUCCESS] retrieved routes in the route table')
        # Assume only one route table
        return [r['DestinationCidrBlock'] for r in response['RouteTables'][0]['Routes']]
    
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
                                     DestinationCidrBlock=destination_cidr)
        except ClientError as e:
            logger.error(f'[FAIL] cannot delete route in the route table ({e})')
            return False
        logger.info(f'[SUCCESS] deleted route in the route table')
        return True
    
    def associate_route_table(self) -> bool:
        """Associate subnet with route table in the VPC
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ec2/client/associate_route_table.html
        Return:
            True/False to indicate success/failure
        """
        try:
            self.client.associate_route_table(SubnetId=self.subnet_id,
                                              RouteTableId=self.route_table_id)
        except ClientError as e:
            logger.error(f'[FAIL] cannot associate subnet with the route table ({e})')
            return False
        logger.info(f'[SUCCESS] associated subnet with the route table')
        return True

class VPCSecurityManager(VPCSecurityInterface):
    def __init__(self,
                 ec2_client: BaseClient,
                 vpc_id: str,
                 description: str,
                 group_name: str):
        """Define security instance
        Args:
            ec2_client: the EC2 client
            vpc_id: the ID of the vpc to apply the policy to
            description: description of security group
            group_name: the name of the security group
        """
        self.client = ec2_client
        self.vpc_id = vpc_id
        self.description = description
        self.group_name = group_name

    def _authorize_all_egress(self) -> bool:
        """Authorize outbound traffic to anywhere
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ec2/client/authorize_security_group_egress.html
        Return:
            True/False to indicate success/failure
        """
        try:
            self.client.authorize_security_group_egress(
                GroupId=self.security_group_id,
                IpPermissions=[{
                    'IpProtocol': '-1',
                    'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                }]
            )
        except ClientError as e:
            if 'InvalidPermission.Duplicate' in str(e):
                logger.warning('[WARN] egress rule already exists')
                return True
            logger.error(f'[FAIL] cannot authorize egress ({e})')
            return False
        logger.info(f'[SUCCESS] egress authorized')
        return True

    def create_security_group(self, egress: bool = True) -> bool:
        """Creates the security group
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ec2/client/create_security_group.html
        Args:
            egress: authorizes outbound traffic when enabled
        Return:
            True/False to indicate success/failure
        """
        try:
            response = self.client.create_security_group(Description=self.description,
                                                         GroupName=self.group_name,
                                                         VpcId=self.vpc_id)
        except ClientError as e:
            logger.error(f'[FAIL] cannot create security group ({e})')
            return False
        logger.info(f'[SUCCESS] created security group')
        self.security_group_id = response['GroupId']
        if egress:
            status = self._authorize_all_egress()
            if status:
                return True
            return False
        return True
    