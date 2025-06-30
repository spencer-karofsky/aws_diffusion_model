"""
Create VPC infrastructure.
"""
import sys
import os
import boto3

# Looks at project root directory; used for finding aws_setup/tests
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# We can now import from different directory
from aws_setup.managers.vpc_manager import VPCSetupManager, VPCNetworkManager, VPCSecurityManager

if __name__ == "__main__":
    # Initialize shared EC2 client
    ec2_client = boto3.client('ec2')

    # Set up VPC public subnet
    setup_manager = VPCSetupManager(ec2_client=ec2_client)
    setup_manager.create_vpc('ddpm-vpc')
    cidr_block, vpc_id = setup_manager.get_cidr_block(), setup_manager.get_vpc_id()

    # Setup Network
    network_manager = VPCNetworkManager(ec2_client=ec2_client,
                                        vpc_id=vpc_id)
    network_manager.create_subnet(cidr_block)
    network_manager.create_internet_gateway()
    network_manager.create_route_table()
    network_manager.associate_route_table()

    # Setup security group
    security = VPCSecurityManager(ec2_client=ec2_client,
                                  vpc_id=vpc_id,
                                  description='security group for ddpm project',
                                  group_name='ddpm-sg')
