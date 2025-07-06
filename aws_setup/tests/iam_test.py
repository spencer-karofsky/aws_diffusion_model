"""
Test IAMRoleManager with moto (fake AWS calls that mimics boto3) and Python's unittest
"""
import boto3
from aws_setup.managers.iam_manager import IAMRoleManager
import unittest
from moto import mock_iam
import os
import tempfile

class IAMRoleManagerTest(unittest.TestCase):
    def _create_role(self) -> bool:
        """Create IAM Role
        Return:
            True/False to indicate success/failure
        """
        status = self.role_manager.create_role(role_name=self.role_name,
                                               policy_doc='{}')
        return status

    def setUp(self):
        """Set up mocked IAM Role Manager
        """
        self.mock = mock_iam()
        self.mock.start()

        self.client = boto3.client('iam', region_name='us-east-1')
        self.role_manager = IAMRoleManager(self.client)

        # Create mock IAM role that SageMaker would use
        self.role_name = 'SageMakerExecutionRole'
        self.expected_arn_prefix = 'arn:aws:iam::123456789012:role/'
    
    def tearDown(self):
        """Clean up test resources
        """
        self.mock.stop()
    
    def test_create_role(self):
        """Test IAMRoleManager.create_role
        """
        self.assertTrue(self._create_role())

    def test_get_role_arn(self):
        """Test IAMRoleManager.get_role_arn
        """
        self._create_role()
        arn = self.role_manager.get_role_arn(self.role_name)
        self.assertTrue(arn.startswith(self.expected_arn_prefix))
        self.assertIn(self.role_name, arn)