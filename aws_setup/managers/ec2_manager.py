"""
Implements all EC2 functionalities.
"""
from aws_setup.utils.logger import logger
from aws_setup.interfaces.ec2_interface import EC2InstancesInterface, EC2KeyPairInterface, EC2VolumeInterface
from botocore.exceptions import ClientError
from botocore.client import BaseClient
from typing import List, Dict
import os

class EC2InstancesManager(EC2InstancesInterface):
    def __init__(self, ec2_client: BaseClient):
        """Initialize EC2 shared resources
        Args:
            ec2_client: the EC2 client
        """
        self.client = ec2_client
        self.instances = []

    def launch_instance(self,
                        instance_name: str,
                        image_id: str,
                        instance_type: str,
                        subnet_id: str,
                        key_name: str,
                        security_group_ids: List[str]) -> bool:
        """Launch an EC2 instance
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ec2/client/run_instances.html
        Args:
            instance_name: the display name for the instance (separate from the generated instance ID)
        Return:
            True/False to indicate success/failure
        """
        try:
            response = self.client.run_instances(ImageId=image_id,
                                                 InstanceType=instance_type,
                                                 MinCount=1,
                                                 MaxCount=1,
                                                 SubnetId=subnet_id,
                                                 KeyName=key_name,
                                                 SecurityGroupIds=security_group_ids,
                                                 TagSpecifications=[
                                                                {
                                                                    'ResourceType': 'instance',
                                                                    'Tags': [
                                                                        {'Key': 'Name', 'Value': instance_name}
                                                                    ]
                                                                }
                                                            ]
                                                        )
        except ClientError as e:
            logger.error(f'[FAIL] cannot launch EC2 instance ({e})')
            return False
        
        # Save instance info
        for instance in response['Instances']:
            info = {
                'InstanceId': instance['InstanceId'],
                'InstanceType': instance['InstanceType'],
                'ImageId': instance['ImageId'],
                'PrivateIpAddress': instance.get('PrivateIpAddress'),
                'PublicIpAddress': instance.get('PublicIpAddress'),
                'SubnetId': instance['SubnetId'],
                'VpcId': instance['VpcId'],
                'KeyName': instance['KeyName'],
                'State': instance['State']['Name'],
                'Tags': instance.get('Tags', []),
                'LaunchTime': str(instance['LaunchTime'])
            }
            self.instances.append(info)

        logger.info(f'[SUCCESS] launched EC2 instance')
        return True
    
    def terminate_instances(self, instance_ids: List[str]) -> bool:
        """Terminate/delete EC2 instances
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ec2/client/terminate_instances.html
        Args:
            instance_ids: list of EC2 instance IDs
        Return:
            True/False to indicate success/failure
        """
        try:
            self.client.terminate_instances(InstanceIds=instance_ids)
        except ClientError as e:
            logger.error(f'[FAIL] cannot terminate EC2 instances ({e})')
            return False
        logger.info(f'[SUCCESS] launched EC2 instances')
        return True
    
    def stop_instances(self, instance_ids: str) -> bool:
        """Stop/pause EC2 instances
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ec2/client/stop_instances.html
        Args:
            instance_ids: the EC2 instance IDs
        Return:
            True/False to indicate success/failure
        """
        try:
            self.client.stop_instances(InstanceIds=instance_ids)
        except ClientError as e:
            logger.error(f'[FAIL] cannot stop EC2 instances ({e})')
            return False
        logger.info(f'[SUCCESS] stopped EC2 instances')
        return True
    
    def start_instances(self, instance_ids: str) -> bool:
        """Start previously stopped EC2 instances
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ec2/client/start_instances.html
        Args:
            instance_ids: the EC2 instance IDs
        Return:
            True/False to indicate success/failure
        """
        try:
            self.client.start_instances(InstanceIds=instance_ids)
        except ClientError as e:
            logger.error(f'[FAIL] cannot start EC2 instances ({e})')
            return False
        logger.info(f'[SUCCESS] started EC2 instances')
        return True
    
    def reboot_instances(self, instance_ids: str) -> bool:
        """Reboot previously stopped EC2 instances
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ec2/client/reboot_instances.html
        Args:
            instance_ids: the EC2 instance IDs
        Return:
            True/False to indicate success/failure
        """
        try:
            self.client.reboot_instances(InstanceIds=instance_ids)
        except ClientError as e:
            logger.error(f'[FAIL] cannot reboot EC2 instances ({e})')
            return False
        logger.info(f'[SUCCESS] rebooted EC2 instances')
        return True
    
    def list_instance_statuses(self, instance_ids: str) -> Dict[str, str]:
        """Retrieve EC2 instance statuses
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ec2/client/describe_instance_status.html
        Args:
            instance_ids: the EC2 instance IDs
        Return:
            Dictionary of the form {'instance_id': 'status', ...}
        """
        try:
            response = self.client.describe_instance_status(InstanceIds=instance_ids)
        except ClientError as e:
            logger.error(f'[FAIL] cannot describe EC2 instances ({e})')
            return []

        statuses_dict = {
            instance_status['InstanceId']: instance_status['InstanceStatus']['Status']
            for instance_status in response.get('InstanceStatuses', [])
        }
        logger.info(f'[SUCCESS] returned EC2 instance statuses')
        return statuses_dict
    
    def list_instance_public_ips(self, instance_ids: str) -> Dict[str, str]:
        """Retrieve public IPs for EC2 instances
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ec2/client/describe_addresses.html
        Args:
            instance_ids: the EC2 instance IDs
        Return:
            Dictionary of the form {'instance_id': 'public_ip', ...}
        """
        try:
            reservations = self.client.describe_instances(InstanceIds=instance_ids)['Reservations']
        except ClientError as e:
            logger.error(f'[FAIL] cannot describe instances ({e})')
            return {}

        result = {}
        for reservation in reservations:
            for instance in reservation['Instances']:
                instance_id = instance['InstanceId']
                public_ip = instance.get('PublicIpAddress')
                if public_ip:
                    result[instance_id] = public_ip

        return result

class EC2KeyPairManager(EC2KeyPairInterface):
    def __init__(self, ec2_client: BaseClient):
        """Initialize EC2 shared resources
        Args:
            ec2_client: the EC2 client
        """
        self.client = ec2_client

    def create_key_pair(self, key_name: str) -> bool:
        """Create a key for access into an EC2 instance
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ec2/client/create_key_pair.html
        Args:
            key_name: the key name
        Return:
            True/False to indicate success/failure
        """
        try:
            response = self.client.create_key_pair(KeyName=key_name)
        except ClientError as e:
            logger.error(f'[FAIL] cannot generate private key ({e})')
            return False
        private_key = response['KeyMaterial']

        # Save key
        key_path = os.path.expanduser(f'~/.ssh/{key_name}.pem')
        os.makedirs(os.path.dirname(key_path), exist_ok=True)
        with open(key_path, 'w') as key_file:
            key_file.write(private_key)

        os.chmod(key_path, 0o400) # Make read only for SSH
        
        logger.info(f'[SUCCESS] generated private key')
        return True
    
    def delete_key_pair(self, key_name: str) -> bool:
        """Delete key pair
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ec2/client/delete_key_pair.html
        Args:
            key_name: the key name
        Return:
            True/False to indicate success/failure
        """
        try:
            response = self.client.delete_key_pair(KeyName=key_name)
        except ClientError as e:
            logger.error(f'[FAIL] cannot delete private key ({e})')
            return False
        # Verify the key was deleted
        if response['Return']:
            logger.info(f'[SUCCESS] deleted private key')
            return True
        logger.error(f'[FAIL] cannot delete private key ({e})')
        return False

class EC2VolumeManager(EC2VolumeInterface):
    def __init__(self, ec2_client: BaseClient):
        """Initialize EC2 shared resources
        Args:
            ec2_client: the EC2 client
        Return:
            True/False to indicate success/failure
        """
        self.client = ec2_client

    def attach_volume(self, instance_id: str, volume_id: str, device_name: str) -> bool:
        """Attach EBS volume to EC2 instance
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ec2/client/attach_volume.html
        Args:
            instance_id: the EC2 instance ID
            volume_id: the EBS volume ID
            device_name: the device name of the volume
        """
        try:
            self.client.attach_volume(Device=device_name,
                                      InstanceId=instance_id,
                                      VolumeId=volume_id)
        except ClientError as e:
            logger.error(f'[FAIL] cannot attach EBS volume ({e})')
            return False
        logger.info(f'[SUCCESS] attached EBS volume')
        return True
    
    def detach_volume(self, volume_id: str) -> bool:
        """Detach an EBS volume
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ec2/client/delete_volume.html
        Args:
            volume_id: the EBS volume ID
        Return:
            True/False to indicate success/failure
        """
        try:
            self.client.delete_volume(VolumeId=volume_id)
        except ClientError as e:
            logger.error(f'[FAIL] cannot detach EBS volume ({e})')
            return False
        logger.info(f'[SUCCESS] detached EBS volume')
        return True