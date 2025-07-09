"""
Defines interfaces for all Amazon Elastic Block Storage (EBS) Functionalities.

Volume Manager:
1. Create Volume. Creates an EBS volume as EC2 instances do not have storage by default (EC2 is just compute).
2. Delete Volume. Deletes an existing EBS volume.
3. List Volumes. Retrieves/returns information about all existing EBS volumes.
"""
from typing import Protocol, List, Dict, Union

class EBSVolumeInterface(Protocol):
    def create_volume(self,
                      availability_zone: str,
                      size: str,
                      volume_type: str,
                      volume_name: str = 'my-ebs-volume') -> bool:
        raise NotImplementedError
    
    def delete_volume(self, volume_id: str) -> bool:
        raise NotImplementedError
    
    def list_volumes(self) -> List[Dict[str, Union[str, int]]]:
        raise NotImplementedError