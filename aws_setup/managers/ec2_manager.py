"""
Implements all EC2 functionalities.
"""
import boto3
from aws_setup.utils.logger import logger
from aws_setup.interfaces.ec2_interface import EC2InstancesInterface, EC2KeyPairInterface, EC2SecurityInterface, EC2VolumeInterface
from botocore.exceptions import ClientError
from botocore.client import BaseClient
from typing import Optional, List, Tuple

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
    
    def reboot_instance(self, instance_id: str) -> bool:
        raise NotImplementedError
    
    def get_instance_status(self, instance_id: str) -> str:
        raise NotImplementedError
    
    def list_instances(self) -> List[Tuple[str]]:
        raise NotImplementedError
    
    def get_instance_public_ip(self, instance_id: str) -> str:
        raise NotImplementedError

class EC2KeyPairManager(EC2KeyPairInterface):
    def __init__(self, ec2_client: BaseClient):
        pass

    def create_key_pair(self, destination_path: str) -> bool:
        raise NotImplementedError
    
    def delete_key_pair(self, key_name: str) -> bool:
        raise NotImplementedError

class EC2SecurityManager(EC2SecurityInterface):
    def __init__(self, ec2_client: BaseClient):
        pass

    def assign_security_group(self, sg: str) -> bool:
        raise NotImplementedError
    
    def describe_security_group(self, sg: str) -> bool:
        raise NotImplementedError

class EC2VolumeManager(EC2VolumeInterface):
    def __init__(self, ec2_client: BaseClient):
        pass

    def attach_volume(self, volume_id: str, device_name: str) -> bool:
        raise NotImplementedError
    
    def detach_volume(self, volume_id: str) -> bool:
        raise NotImplementedError