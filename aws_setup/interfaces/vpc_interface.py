"""
Defines interfaces for all Amazon Virtual Private Cloud (VPC) Functionalities.

Setup Interface:
1. Create VPC
    - Define CIDR block. Represents a group of IP addresses sharing the same prefix to represent the entire network.
    - Enable DNS support. Allows the EC2 Instance to resolve outbound domain names.

Network Interface:
1. Create Public Subnet
    - Define new CIDR block. Represents a subset of IP addresses of the VPC for a specific avaliability zone.
    - Define avaliability zone. A subset of a region for improved redundancy.
    - Auto-Assign Public IP. Boolean setting useful when the IP changes that should be enabled.
2. Attach Internet Gateway
    - Attach an Internet Gateway. Allows instances on the network to access the internet and adds this functionality to the VPC.
3. Define Route Table
    - Create a Route Table. Defines communication routes within the network.

Security Interface:
1. Create Security Group
    - Define inbound rules. Defines who can reach the instance.
    - Define outbound rules. Defines who the instance can reach.
"""
from typing import Protocol, List

class VPCSetupInterface(Protocol):
    # Getter methods
    def get_cidr_block(self) -> str:
        raise NotImplementedError
    
    def get_vpc_id(self) -> str:
        raise NotImplementedError
    
    def get_dns_status(self) -> bool:
        raise NotImplementedError
    
    def _enable_dns(self) -> bool:
        raise NotImplementedError
    
    def create_vpc(self) -> bool:
        raise NotImplementedError

class VPCNetworkInterface(Protocol):
    def _enable_auto_assigned_ip(self) -> bool:
        raise NotImplementedError
    
    def create_subnet(self) -> bool:
        raise NotImplementedError
    
    def _attach_internet_gateway(self) -> bool:
        raise NotImplementedError
    
    def create_internet_gateway(self) -> bool:
        raise NotImplementedError
    
    def create_route_table(self) -> bool:
        raise NotImplementedError
    
    def add_route(self, destination_cidr: str) -> bool:
        raise NotImplementedError
    
    def list_routes(self) -> List[str]:
        raise NotImplementedError
    
    def delete_route(self, destination_cidr: str) -> bool:
        raise NotImplementedError
    
    def associate_route_table(self) -> bool:
        raise NotImplementedError

class VPCSecurityInterface(Protocol):
    def _authorize_all_egress(self) -> bool:
        raise NotImplementedError
    
    def create_security_group(self, egress: bool) -> bool:
        raise NotImplementedError
    
    # def authorize_ssh(self) -> bool: # Implement if needed
    #     raise NotImplementedError
        
    # def authorize_http_https(self) -> bool:
    #     raise NotImplementedError