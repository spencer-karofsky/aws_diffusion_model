"""
Defines interfaces for all Amazon Virtual Private Cloud (VPC) Functionalities.

Instances Manager:
1. Launch instance. Creates a new EC2 instance.
2. Terminate instance. Permanately deletes an EC2 instance.
3. Stop instance. Pauses the instance (and billing of that instance).
4. Start instance. Starts a previously-stopped instance.
5. Reboot instance. Soft reboot without changing instance state.
6. Get Instance Status. Returns pending, running, stopped, etc..
7. List instances. Returns a list of EC2 instances.
8. Get instance public IP. Returns the public IP of a running instance.

Key Pair Manager:
1. Create key pair. Creates a key pair (for SSH access) and saves key pair to a local file.
2. Delete key pair. For clean up.

Security Manager:
1. Assign security group. The security group has already been created with VPC.
2. Describe instance security group. Outputs relevant security information and permissions.

Volume Manager (for integration with EBS):
2. Attach Volume. Attaches a created EBS volume.
3. Detach Volume. Detaches an EBS volume from an EC2 instance.
"""
from typing import Protocol, List, Tuple

class EC2InstancesInterface(Protocol):
    def launch_instance(self,
                        instance_name: str,
                        image_id: str,
                        instance_type: str,
                        subnet_id: str,
                        key_name: str,
                        security_group_ids: List[str]) -> bool:
        raise NotImplementedError
    
    def terminate_instances(self, instance_ids: List[str]) -> bool:
        raise NotImplementedError
    
    def stop_instances(self, instance_ids: str) -> bool:
        raise NotImplementedError
    
    def start_instance(self, instance_id: str) -> bool:
        raise NotImplementedError
    
    def reboot_instance(self, instance_id: str) -> bool:
        raise NotImplementedError
    
    def get_instance_status(self, instance_id: str) -> str:
        raise NotImplementedError
    
    def list_instances(self) -> List[Tuple[str]]:
        raise NotImplementedError
    
    def get_instance_public_ip(self, instance_id: str) -> str:
        raise NotImplementedError

class EC2KeyPairInterface(Protocol):
    def create_key_pair(self, destination_path: str) -> bool:
        raise NotImplementedError
    
    def delete_key_pair(self, key_name: str) -> bool:
        raise NotImplementedError

class EC2SecurityInterface(Protocol):
    def assign_security_group(self, sg: str) -> bool:
        raise NotImplementedError
    
    def describe_security_group(self, sg: str) -> bool:
        raise NotImplementedError

class EC2VolumeInterface(Protocol):
    def attach_volume(self, volume_id: str, device_name: str) -> bool:
        raise NotImplementedError
    
    def detach_volume(self, volume_id: str) -> bool:
        raise NotImplementedError