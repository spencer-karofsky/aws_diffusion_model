"""
Test VPCSetupManager, VPCNetworkManager, and VPCSecurityManager functionalities with moto (fake AWS calls that mimics boto3) and Python's unittest
"""
import boto3
from aws_setup.managers.vpc_manager import VPCSetupManager, VPCNetworkManager, VPCSecurityManager
import unittest
from moto import mock_ec2

class VPCSetupManagerTest(unittest.TestCase):
    def setUp(self):
        """Set up mocked EC2 client and other test resources
        """
        # Mock EC2
        self.mock = mock_ec2()
        self.mock.start()

        self.client = boto3.client('ec2')
        self.setup_manager = VPCSetupManager(self.client)
    
    def tearDown(self):
        """Stop mocking resources
        """
        self.mock.stop()
    
    def test_create_vpc(self):
        """Test VPCSetupManager.create_vpc
        """
        result = self.setup_manager.create_vpc(vpc_name='test-vpc')

        self.assertTrue(result)
        self.assertIsNotNone(self.setup_manager.get_vpc_id())
        self.assertEqual(self.setup_manager.get_cidr_block(), '10.0.0.0/16')
        self.assertTrue(self.setup_manager.get_dns_status())

    def test_delete_vpc(self):
        """Test VPCSetupManager.delete_vpc
        """
        # Create VPC (to later delete)
        self.setup_manager.create_vpc()
        vpc_id = self.setup_manager.get_vpc_id()

        # Delete VPC
        result = self.setup_manager.delete_vpc(vpc_id)
        self.assertTrue(result)

        # Delete VPC again and ensure it fails (since we already deleted it)
        result = self.setup_manager.delete_vpc(vpc_id)
        self.assertFalse(result)

        # Delete VPC with no VPC ID provided
        result = self.setup_manager.delete_vpc()
        self.assertTrue(result)

class VPCNetworkManagerTest(unittest.TestCase):
    def setUp(self):
        """Set up mocked EC2 client and other test resources
        """
        # Mock EC2
        self.mock = mock_ec2()
        self.mock.start()

        self.client = boto3.client('ec2')

        self.setup_manager = VPCSetupManager(self.client)

        # Create VPC and get CIDR block
        self.setup_manager.create_vpc()
        vpc_id = self.setup_manager.get_vpc_id()
        self.cidr_block = self.setup_manager.get_cidr_block()

        self.network_manager = VPCNetworkManager(self.client, vpc_id)

        # Define Dummy Destination CIDR routes to add to the route table
        self.destination_cidrs = ['0.0.0.0/0',
                                  '192.168.1.0/24']
    
    def tearDown(self):
        """Stop mocking resources
        """
        self.mock.stop()

    def test_create_subnet(self):
        """Test VPCNetworkManager.create_subnet
        """
        # Create Subnet and validate success
        result = self.network_manager.create_subnet(self.cidr_block)
        self.assertTrue(result)
    
    def test_create_internet_gateway(self):
        """Test VPCNetworkManager.create_internet_gateway
        """
        result = self.network_manager.create_internet_gateway()
        self.assertTrue(result)
    
    def test_create_route_table(self):
        """Test VPCNetworkManager.create_route_table
        """
        result = self.network_manager.create_route_table()
        self.assertTrue(result)
    
    def test_add_route(self):
        """Test VPCNetworkManager.add_route
        """
        self.assertTrue(self.network_manager.create_internet_gateway())
        self.assertTrue(self.network_manager.create_route_table())
        for route in self.destination_cidrs:
            self.assertTrue(self.network_manager.add_route(route))
        
    def test_list_routes(self):
        """Test VPCNetworkManager.list_routes
        """
        # Create required resources
        self.assertTrue(self.network_manager.create_subnet(self.cidr_block))
        self.assertTrue(self.network_manager.create_internet_gateway())
        self.assertTrue(self.network_manager.create_route_table())

        # Add dummy routes
        for cidr in self.destination_cidrs:
            self.assertTrue(self.network_manager.add_route(cidr))

        # Retrieve routes and verify
        routes = self.network_manager.list_routes()
        for cidr in self.destination_cidrs:
            self.assertIn(cidr, routes)
    
    def test_delete_route(self):
        """Test VPCNetworkManager.delete_route
        """
        # Create required resources
        self.assertTrue(self.network_manager.create_subnet(self.cidr_block))
        self.assertTrue(self.network_manager.create_internet_gateway())
        self.assertTrue(self.network_manager.create_route_table())

        # Add dummy routes
        for cidr in self.destination_cidrs:
            self.assertTrue(self.network_manager.add_route(cidr))
        
        # Delete routes
        for cidr in self.destination_cidrs:
            self.assertTrue(self.network_manager.delete_route(cidr))
    
    def test_associate_route_table(self):
        """Test VPCNetworkManager.associate_route_table
        """
        self.assertTrue(self.network_manager.create_subnet(self.cidr_block))
        self.assertTrue(self.network_manager.create_route_table())
        self.assertTrue(self.network_manager.associate_route_table())

class VPCSecurityManagerTest(unittest.TestCase):
    def setUp(self):
        """Set up mocked EC2 client and other test resources
        """
        # Mock EC2
        self.mock = mock_ec2()
        self.mock.start()

        self.client = boto3.client('ec2')

        self.setup_manager = VPCSetupManager(self.client)
        # Create VPC and get CIDR block
        self.setup_manager.create_vpc()
        vpc_id = self.setup_manager.get_vpc_id()

        self.network_manager = VPCNetworkManager(self.client, vpc_id)
        self.security_manager = VPCSecurityManager(self.client,
                                                   vpc_id,
                                                   description='dummy security group',
                                                   group_name='dummy-sg')
    
    def tearDown(self):
        """Stop mocking resources
        """
        self.mock.stop()

    def test_create_security_group(self):
        """Test VPCSecurityManager.create_security_group
        """
        result = self.security_manager.create_security_group()
        self.assertTrue(result)