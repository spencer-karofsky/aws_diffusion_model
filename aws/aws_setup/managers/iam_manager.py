"""
Implements IAM functionalities.
"""
from aws_setup.utils.logger import logger
from botocore.exceptions import ClientError
from aws_setup.interfaces.iam_interface import IAMRoleInterface
from botocore.client import BaseClient
from typing import Optional

class IAMRoleManager(IAMRoleInterface):
    def __init__(self, iam_client: BaseClient):
        """Initialize IAM resources
        Args:
            iam_client: the IAM client, injectable for unit testing
        """
        self.client = iam_client

    def create_role(self,
                    role_name: str,
                    policy_doc: Optional[str] = None) -> bool:
        """Creates a role with permissions
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/iam/client/create_role.html
        Args:
            role_name: the name of the role
            policy_doc: the permissions, in JSON format
        Return:
            True/False to indicate success/failure
        """
        try:
            self.client.create_role(RoleName=role_name,
                                    AssumeRolePolicyDocument=policy_doc)
        except ClientError as e:
            logger.error(f'[FAIL] cannot create IAM Role ({e})')
            return False
        logger.info(f'[SUCCESS] created IAM Role')
        return True

    def get_role_arn(self, role_name: str) -> str:
        """Gets the role ARN
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/iam/client/get_role.html
        Args:
            role_name: The name of the ARN role in IAM
        Return:
            The role ARN if retrieval is successful, otherwise the function returns an empty string.
        """
        try:
            response = self.client.get_role(RoleName=role_name)
        except ClientError as e:
            logger.error(f'[FAIL] cannot retrieve the ARN for the IAM Role ({e})')
            return ''
        
        # Extract Role ID
        arn = response['Role']['Arn']

        logger.info(f'[SUCCESS] retrieved and returned the ARN for the IAM Role')
        return arn

