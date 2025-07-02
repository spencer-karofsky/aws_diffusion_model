"""
Implements all EBS functionalities.
"""
from aws_setup.utils.logger import logger
from aws_setup.interfaces.ebs_interface import EBSVolumeInterface
from botocore.exceptions import ClientError
from botocore.client import BaseClient
from typing import List, Dict, Union

class EBSVolumeManager(EBSVolumeInterface):
    def __init__(self, ec2_client: BaseClient):
        """Initialize EBS shared resources
        Args:
            ec2_client: the EC2 client
        """
        self.client = ec2_client
        self.volume_ids = []
    
    def create_volume(self,
                      availability_zone: str,
                      size: str,
                      volume_type: str,
                      volume_name: str = 'my-ebs-volume') -> bool:
        """Creates an EBS volume to give EC2 instances storage.
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ec2/client/create_volume.html
        Args:
            availability_zone: the availability zone is different from the region (e.g., 'us-east-1a', 'us-east-1b')
            size: the amount of desired storage for the EC2 instance, in GiB (1GB is roughly 1.073 GiB)
            volume_type: the type of volume regarding performance and cost. Use either gp2 or gp3 (use gp2 for unit testing and gp3 for deployment)
            volume_name: the name of the volume for easy identification when there's multiple volumes
        Return:
            True/False to indicate success/failure
        """
        try:
            response = self.client.create_volume(
                AvailabilityZone=availability_zone,
                Size=size,
                VolumeType=volume_type,
                TagSpecifications=[{
                    'ResourceType': 'volume',
                    'Tags': [{'Key': 'Name', 'Value': volume_name}]
                }]
            )
        except ClientError as e:
            logger.error(f'[FAIL] cannot create EBS volume ({e})')
            return False

        # Save volume_id
        self.volume_ids.append(response['VolumeId'])

        logger.info(f'[SUCCESS] created EBS volume')
        return True
    
    def delete_volume(self, volume_id: str) -> bool:
        """Deletes an existing EBS volume
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ec2/client/delete_volume.html
        Args:
            volume_id: the EBS volume id
        Return:
            True/False to indicate success/failure
        """
        try:
            self.client.delete_volume(VolumeId=volume_id)
        except ClientError as e:
            logger.error(f'[FAIL] cannot delete EBS volume ({e})')
            return False
        logger.info(f'[SUCCESS] deleted EBS volume')
        return True
    
    def list_volumes(self) -> List[Dict[str, Union[str, int]]]:
        """Lists existing EBS volumes
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ec2/client/describe_volumes.html
        Return:
            List of dictionaries of the form, [{'name': str,
                                                'id': str,
                                                'state': str,
                                                'size': int,
                                                'availability zone': str,
                                                'type': str,
                                                'attachment status': str}]
        """
        try:
            response = self.client.describe_volumes()
        except ClientError as e:
            logger.error(f'[FAIL] cannot retrieve EBS volumes ({e})')
            return []

        volumes = []
        for volume in response['Volumes']:
            # Attempt to get the 'Name' tag
            name = None
            if 'Tags' in volume:
                for tag in volume['Tags']:
                    if tag['Key'] == 'Name':
                        name = tag['Value']
                        break

            attachment_state = (
                volume['Attachments'][0]['State']
                if volume.get('Attachments')
                else 'unattached'
            )

            volumes.append({
                'name': name or 'N/A',
                'id': volume['VolumeId'],
                'state': volume['State'],
                'size': volume['Size'],
                'availability zone': volume['AvailabilityZone'],
                'type': volume['VolumeType'],
                'attachment status': attachment_state
            })

        logger.info(f'[SUCCESS] retrieved and listed {len(volumes)} EBS volumes')
        return volumes