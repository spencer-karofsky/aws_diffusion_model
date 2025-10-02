"""
provision_infrastructure.py: Provisions the AWS Infrastructure.

Description:
    * Refer to /aws/planning/AWS Project Plan.pdf for design choice details

Classes:
    * TODO

Author:
    * Spencer Karofsky (https://github.com/spencer-karofsky)
"""
import os
import sys
import boto3
import json
from pathlib import Path

# Add the project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from aws.aws_setup.managers.s3_manager import S3BucketManager
from aws.aws_setup.managers.vpc_manager import (
    VPCSetupManager, VPCNetworkManager, VPCSecurityManager
)
from aws.aws_setup.managers.iam_manager import IAMRoleManager
from aws.aws_setup.managers.sagemaker_manager import SageMakerNotebookManager

class Dalle2InfrastructureProvisioner:
    def __init__(self):
        # AWS Clients
        self.ec2_client = boto3.client('ec2')
        self.iam_client = boto3.client('iam')

        # AWS Resource Managers
        self.vpc_setup = VPCSetupManager(self.ec2_client)
        self.s3_manager = S3BucketManager()
        self.iam_manager = IAMRoleManager(self.iam_client)

        # These will be set during provisioning
        self.network_manager = None
        self.security_manager = None
        self.sagemaker_manager = None

        # Output config
        self.config = {}

    def provision_vpc(self, vpc_name='dalle2-vpc'):
        success = self.vpc_setup.create_vpc(vpc_name)
        if not success:
            raise RuntimeError("Failed to create VPC")

        vpc_id = self.vpc_setup.get_vpc_id()
        if not vpc_id:
            raise RuntimeError("VPC ID is None — likely creation failed silently")

        cidr_block = self.vpc_setup.get_cidr_block()

        self.network_manager = VPCNetworkManager(self.ec2_client, vpc_id)
        self.network_manager.create_subnet(cidr_block)
        self.network_manager.create_internet_gateway()
        self.network_manager.create_route_table()
        self.network_manager.add_route(destination_cidr='0.0.0.0/0')
        self.network_manager.associate_route_table()

        self.security_manager = VPCSecurityManager(
            ec2_client=self.ec2_client,
            vpc_id=vpc_id,
            description='Security group for DALL-E 2',
            group_name='dalle2-sg'
        )
        self.security_manager.create_security_group()

        self.config['vpc'] = {
            'vpc_id': vpc_id,
            'subnet_id': self.network_manager.subnet_id,
            'security_group_id': getattr(self.security_manager, 'security_group_id', 'N/A')
        }

    def provision_s3(self):
        bucket_names = [
            'dalle2-data',
            'dalle2-models',
            'dalle2-outputs'
        ]
        for name in bucket_names:
            self.s3_manager.create_bucket(name)
        self.config['s3_buckets'] = bucket_names

    def provision_iam_role(self, role_name='dalle2-sagemaker-role'):
        trust_policy_path = Path(__file__).parent / '../config/sagemaker_trust_policy.json'
        with open(trust_policy_path) as f:
            policy_doc = f.read()

        self.iam_manager.create_role(role_name, policy_doc)
        role_arn = self.iam_manager.get_role_arn(role_name)
        self.config['iam_role_arn'] = role_arn
        return role_arn

    def provision_sagemaker_notebook(self, role_arn):
        self.sagemaker_manager = SageMakerNotebookManager(
            notebook_name='dalle2-training-notebook',
            role_arn=role_arn,
            instance_type='ml.g4dn.xlarge'
        )
        self.sagemaker_manager.create_notebook()
        # self.sagemaker_manager.start_notebook() DANGER!!!! Incurs charges once started

    def save_config(self, output_path='infra_config.json'):
        config_path = Path(__file__).parent / output_path
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=4)
        print(f'[INFO] Saved config to {config_path.resolve()}')

    def run(self):
        print('[STEP] Provisioning VPC and networking...')
        self.provision_vpc()

        print('[STEP] Creating S3 buckets...')
        self.provision_s3()

        print('[STEP] Creating IAM role...')
        role_arn = self.provision_iam_role()

        print('[STEP] Launching SageMaker notebook...')
        self.provision_sagemaker_notebook(role_arn)

        print('[STEP] Saving configuration...')
        self.save_config()

        print('[SUCCESS] Infrastructure provisioned.')


if __name__ == '__main__':
    #raise Exception('Process Completed') # Uncomment when complete to prevent re-running AWS code
    provisioner = Dalle2InfrastructureProvisioner()
    provisioner.run()
