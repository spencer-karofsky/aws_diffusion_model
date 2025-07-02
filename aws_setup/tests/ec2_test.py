"""
Test EC2InstancesManager, EC2KeyPairManager, EC2SecurityManager, and EC2VolumeManager functionalities with moto
(fake AWS calls that mimics boto3) and Python's unittest.
"""
import boto3
from aws_setup.managers.ec2_manager import EC2InstancesManager, EC2KeyPairManager, EC2VolumeManager
from aws_setup.managers.vpc_manager import VPCSetupManager, VPCNetworkManager, VPCSecurityManager
import unittest
from moto import mock_ec2

class EC2InstancesManagerTest(unittest.TestCase):
    def _setup_network_sg(self):
        """Set up network and security group
        """
        vpc_setup_manager = VPCSetupManager(self.client)
        vpc_setup_manager.create_vpc('my-fake-vpc')
        vpc_id = vpc_setup_manager.get_vpc_id()
        cidr_block = vpc_setup_manager.get_cidr_block()

        vpc_network_manager = VPCNetworkManager(self.client, vpc_id)
        vpc_network_manager.create_subnet(cidr_block, 'my-subnet-1')
        self.subnet_id = vpc_network_manager.subnet_id

        vpc_security_manager = VPCSecurityManager(self.client, vpc_id, 'new security group', 'sg-1')
        vpc_security_manager.create_security_group()
        self.security_id = vpc_security_manager.security_group_id

        return True

    def _launch_instance(self, instance_name):
        """Launch EC2 Instance
        """
        self.ec2_instances_manager.launch_instance(instance_name=instance_name,
                                                   image_id=self.image_id,
                                                   instance_type=self.instance_type,
                                                   subnet_id=self.subnet_id,
                                                   key_name='my-key-01',
                                                   security_group_ids=[self.security_id])
        return True

    def setUp(self):
        """Set up mocked EC2 Instances Manager
        """
        # Mock EC2
        self.mock = mock_ec2()
        self.mock.start()

        self.client = boto3.client('ec2')
        self.ec2_instances_manager = EC2InstancesManager(self.client)

        self.instances_list = [f'my-ec2-instance-{i}' for i in range(3)]
        self.image_id = 'ami-0a7d80731ae1b2435' # Free tier-eligible
        self.instance_type = 't2.micro' # Free tier-eligble, so optimal for testing, but use more powerful paid instance type for production

        # Set up subnet and security group
        self._setup_network_sg()

    def tearDown(self):
        """Stop mocking resources
        """
        self.mock.stop()

    def test_launch_instance(self):
        """Test EC2InstancesManager.launch_instance
        """
        for instance_name in self.instances_list:
            self.assertTrue(self._launch_instance(instance_name))

    def test_terminate_instances(self):
        """Test EC2InstancesManager.terminate_instances
        """
        # Launch instances
        for instance_name in self.instances_list:
            self._launch_instance(instance_name)
        
        # Get instance IDs
        instance_ids = [instance['InstanceId'] for instance in self.ec2_instances_manager.instances]

        # Terminate instances
        self.assertTrue(self.ec2_instances_manager.terminate_instances(instance_ids))

    def test_stop_instances(self):
        """Test EC2InstancesManager.stop_instances
        """
        # Launch instances
        for instance_name in self.instances_list:
            self._launch_instance(instance_name)

        # Get instance IDs
        instance_ids = [instance['InstanceId'] for instance in self.ec2_instances_manager.instances]

        # Stop instances
        self.assertTrue(self.ec2_instances_manager.stop_instances(instance_ids))

    def test_start_instances(self):
        """Test EC2InstancesManager.start_instances
        """
        # Launch instances
        for instance_name in self.instances_list:
            self._launch_instance(instance_name)
        
        # Get instance IDs
        instance_ids = [instance['InstanceId'] for instance in self.ec2_instances_manager.instances]

        # Stop instances
        self.ec2_instances_manager.stop_instances(instance_ids)

        # Start instances
        self.assertTrue(self.ec2_instances_manager.start_instances(instance_ids))

    def test_reboot_instances(self):
        """Test EC2InstancesManager.reboot_instances
        """
        # Launch instances
        for instance_name in self.instances_list:
            self._launch_instance(instance_name)

        # Get instance IDs
        instance_ids = [instance['InstanceId'] for instance in self.ec2_instances_manager.instances]

        # Reboot instances
        self.assertTrue(self.ec2_instances_manager.reboot_instances(instance_ids))

    def test_list_instance_statuses(self):
        """Test EC2InstancesManager.list_instance_statuses
        """
        # Launch instances
        for instance_name in self.instances_list:
            self._launch_instance(instance_name)

        # Get instance IDs
        instance_ids = [instance['InstanceId'] for instance in self.ec2_instances_manager.instances]

        # Get statuses
        statuses = self.ec2_instances_manager.list_instance_statuses(instance_ids)

        statuses_ids = sorted(list(statuses.keys()))
        
        self.assertListEqual(statuses_ids, sorted(instance_ids))

    def test_list_instance_public_ips(self):
        """Test EC2InstancesManager.list_public_ips
        """
        # Launch instances
        for instance_name in self.instances_list:
            self._launch_instance(instance_name)

        # Get instance IDs
        instance_ids = [instance['InstanceId'] for instance in self.ec2_instances_manager.instances]

        # Get public IPs
        public_ips = self.ec2_instances_manager.list_instance_public_ips(instance_ids)

        public_ips_ids = sorted(list(public_ips.keys()))
        
        self.assertListEqual(public_ips_ids, sorted(instance_ids))

class EC2KeyPairManagerTest(unittest.TestCase):
    def setUp(self):
        """Set up mocked EC2 Key Pair Manager
        """
        # Mock EC2
        self.mock = mock_ec2()
        self.mock.start()

        self.client = boto3.client('ec2')

        self.ec2_key_pair_manager = EC2KeyPairManager(self.client)

    def tearDown(self):
        """Stop mocking resources
        """
        self.mock.stop()

    def test_create_key_pair(self):
        """Test EC2KeyPairManager.create_key_pair
        """
        self.assertTrue(self.ec2_key_pair_manager.create_key_pair('my-key-pair'))

    def test_delete_key_pair(self):
        """Test EC2KeyPairManager.delete_key_pair
        """
        self.assertTrue(self.ec2_key_pair_manager.delete_key_pair('my-key-pair'))

class EC2VolumeManagerTest(unittest.TestCase):
    def _setup_network_sg(self):
        """Set up network and security group
        """
        vpc_setup_manager = VPCSetupManager(self.client)
        vpc_setup_manager.create_vpc('my-fake-vpc')
        vpc_id = vpc_setup_manager.get_vpc_id()
        cidr_block = vpc_setup_manager.get_cidr_block()

        vpc_network_manager = VPCNetworkManager(self.client, vpc_id)
        vpc_network_manager.create_subnet(cidr_block, 'my-subnet-1')
        self.subnet_id = vpc_network_manager.subnet_id

        vpc_security_manager = VPCSecurityManager(self.client, vpc_id, 'new security group', 'sg-1')
        vpc_security_manager.create_security_group()
        self.security_id = vpc_security_manager.security_group_id

        return True
    
    def _launch_instance(self, instance_name):
        """Launch EC2 Instance
        """
        self.ec2_instances_manager.launch_instance(instance_name=instance_name,
                                                   image_id=self.image_id,
                                                   instance_type=self.instance_type,
                                                   subnet_id=self.subnet_id,
                                                   key_name='my-key-01',
                                                   security_group_ids=[self.security_id])
        return True
    
    def setUp(self):
        """Set up mocked EC2 Volume Manager
        """
        # Mock EC2
        self.mock = mock_ec2()
        self.mock.start()

        self.client = boto3.client('ec2')
        self.ec2_instances_manager = EC2InstancesManager(self.client)
        self.ec2_volume_manager = EC2VolumeManager(self.client)

        self.image_id = 'ami-0a7d80731ae1b2435' # Free tier-eligible
        self.instance_type = 't2.micro' # Free tier-eligble, so optimal for testing, but use more powerful paid instance type for production

    def tearDown(self):
        """Stop mocking resources
        """
        self.mock.stop()
    
    def test_attach_volume(self):
        """Test EC2VolumeManager.attach_volume
        """
        ec2_instance = self.ec2_instances_manager.launch_instance()
        instance_id = self.ec2_instances_manager.instances[0]['InstanceId'] 
        volume_id = '' # TODO
        device_name = '' # TODO
        self.assertTrue(self.ec2_volume_manager.attach_volume(instance_id=instance_id,
                                                              volume_id=volume_id,
                                                              device_name=''))
    
    def test_detach_volume(self):
        """Test EC2VolumeManager.detach_volume
        """
        volume_id = '' # TODO
        self.assertTrue(self.ec2_volume_manager.detach_volume(volume_id=volume_id))