"""
Test EBSVolumeManager functionalities with moto
(fake AWS calls that mimics boto3) and Python's unittest.
"""
import boto3
from aws_setup.managers.ebs_manager import EBSVolumeManager
import unittest
from moto import mock_ec2

class EBSVolumeManagerTest(unittest.TestCase):
    def setUp(self):
        """Set up mocked EC2 Instances Manager
        """
        # Mock EC2
        self.mock = mock_ec2()
        self.mock.start()

        self.client = boto3.client('ec2')

        self.ebs_volume_manager = EBSVolumeManager(self.client)

        # Setup volume info
        self.availability_zones = ['us-east-1a',
                                   'us-east-1b',
                                   'us-east-1c']
        self.volume_size = 5 # GiB
        self.volume_type = 'gp2'
        self.volume_names = [f'my-ebs-volume-{i}' for i in range(1, 4)]

    def tearDown(self):
        """Stop mocking resources
        """
        self.mock.stop()
    
    def test_create_volume(self):
        """Test EBSVolumeManager.create_volume
        """
        for avail_zone, vol_name in zip(self.availability_zones, self.volume_names):
            self.assertTrue(self.ebs_volume_manager.create_volume(availability_zone=avail_zone,
                                                                  size=self.volume_size,
                                                                  volume_type=self.volume_type,
                                                                  volume_name=vol_name))

    def test_delete_volume(self):
        """Test EBSVolumeManager.delete_volume
        """
        # Create volumes
        for avail_zone, vol_name in zip(self.availability_zones, self.volume_names):
            self.ebs_volume_manager.create_volume(availability_zone=avail_zone,
                                                  size=self.volume_size,
                                                  volume_type=self.volume_type,
                                                  volume_name=vol_name)
            
        # Delete created volumes
        for vol_id in self.ebs_volume_manager.volume_ids:
            self.assertTrue(self.ebs_volume_manager.delete_volume(vol_id))
    
    def test_list_volumes(self):
        """Test EBSVolumeManager.list_volumes
        """
        # Create volumes
        for az, name in zip(self.availability_zones, self.volume_names):
            self.ebs_volume_manager.create_volume(availability_zone=az,
                                                size=self.volume_size,
                                                volume_type=self.volume_type,
                                                volume_name=name)

        # Call list_volumes
        volumes = self.ebs_volume_manager.list_volumes()

        # Build expected list
        expected = []
        for vol_id, az, name in zip(self.ebs_volume_manager.volume_ids, self.availability_zones, self.volume_names):
            expected.append({
                'name': name,
                'id': vol_id,
                'state': 'available',
                'size': self.volume_size,
                'availability zone': az,
                'type': self.volume_type,
                'attachment status': 'unattached'
            })

        # Assert lists match (order-insensitive)
        self.assertCountEqual(volumes, expected)