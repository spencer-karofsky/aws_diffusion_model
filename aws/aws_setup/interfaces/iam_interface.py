"""
Defines interface for Retrieving Identity Access Manager (IAM) ARN role.

Unlike the other AWS services, which are fully defined and managed in this repository, I limit almost all of my IAM usage to the console.
Because I do not create an access key from the root account, I grant all IAM permissions from the AWS Console.
I need to access the role ARN when creating a SageMaker notebook, so I define this behavior in this repository.
"""
from typing import Protocol, Optional

class IAMRoleInterface(Protocol):
    def create_role(self,
                    role_name: str,
                    policy_doc: Optional[str] = None) -> bool:
        raise NotImplementedError
    
    def get_role_arn(self, role_name: str) -> str:
        raise NotImplementedError