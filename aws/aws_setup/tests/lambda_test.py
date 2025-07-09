"""
Test Lambda functionalities with moto (fake AWS calls that mimics boto3) and Python's unittest.
"""
import unittest
import boto3
import tempfile
import os
import json
import uuid
from moto import mock_lambda, mock_s3, mock_iam
from aws_setup.managers.lambda_manager import (
    LambdaFunctionManager,
    LambdaDeploymentManager,
    LambdaInvocationManager,
)
from aws_setup.managers.iam_manager import IAMRoleManager

# Prevents accidental real AWS calls
os.environ["AWS_ACCESS_KEY_ID"] = "testing"
os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
os.environ["AWS_SESSION_TOKEN"] = "testing"
os.environ["MOTO_TEST"] = "true"

@mock_lambda
@mock_iam
@mock_s3
class LambdaLambdaFunctionManagerTest(unittest.TestCase):
    def setUp(self):
        """Set up test resources
        """
        self.lambda_client = boto3.client("lambda", region_name="us-east-1")
        self.iam_client = boto3.client("iam", region_name="us-east-1")
        self.s3_client = boto3.client("s3", region_name="us-east-1")
        self.manager = LambdaFunctionManager(self.lambda_client)

        self.temp_dir = tempfile.TemporaryDirectory()
        file_path = os.path.join(self.temp_dir.name, "lambda_function.py")
        with open(file_path, "w") as f:
            f.write(
                "import json\n"
                "def lambda_handler(event, context):\n"
                "    return {'statusCode': 200, 'body': json.dumps(event.get('msg', 'default'))}\n"
            )

        self.deployment = LambdaDeploymentManager(self.lambda_client, self.s3_client)
        self.code = self.deployment.package_from_local_path(self.temp_dir.name)

        self.iam_manager = IAMRoleManager(self.iam_client)
        trust_policy = json.dumps({
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "lambda.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }]
        })
        self.iam_manager.create_role("lambda-role", trust_policy)
        self.role_arn = self.iam_manager.get_role_arn("lambda-role")

    def tearDown(self):
        """Clean up test resources
        """
        self.temp_dir.cleanup()

    def test_create_function(self):
        """Test LambdaFunctionManager.create_function
        """
        result = self.manager.create_function(
            func_name="TestFunction",
            role_arn=self.role_arn,
            code=self.code,
            handler="lambda_function.lambda_handler",
            timeout=10
        )
        self.assertTrue(result)

    def test_list_functions(self):
        """Test LambdaFunctionManager.list_functions
        """
        self.manager.create_function(
            func_name="TestFunction",
            role_arn=self.role_arn,
            code=self.code,
            handler="lambda_function.lambda_handler"
        )
        functions = self.manager.list_functions()
        self.assertEqual(len(functions), 1)
        self.assertGreaterEqual(len(functions), 1)

    def test_delete_function(self):
        """Test LambdaFunctionManager.delete_function
        """
        self.manager.create_function(
            func_name="TestFunction",
            role_arn=self.role_arn,
            code=self.code,
            handler="lambda_function.lambda_handler"
        )
        result = self.manager.delete_function("TestFunction")
        self.assertTrue(result)


@mock_lambda
@mock_iam
@mock_s3
class LambdaDeploymentManagerTest(unittest.TestCase):
    def setUp(self):
        """Set up test resources
        """
        self.lambda_client = boto3.client("lambda", region_name="us-east-1")
        self.s3_client = boto3.client("s3", region_name="us-east-1")
        self.iam_client = boto3.client("iam", region_name="us-east-1")

        self.manager = LambdaDeploymentManager(self.lambda_client, self.s3_client)

        self.temp_dir = tempfile.TemporaryDirectory()
        file_path = os.path.join(self.temp_dir.name, "lambda_function.py")
        with open(file_path, "w") as f:
            f.write("def lambda_handler(event, context): return {'statusCode': 200}")

        self.bucket = f"test-bucket-{uuid.uuid4()}"
        self.s3_client.create_bucket(Bucket=self.bucket)

        self.iam_manager = IAMRoleManager(self.iam_client)
        trust_policy = json.dumps({
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "lambda.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }]
        })
        self.iam_manager.create_role("lambda-role", trust_policy)
        self.role_arn = self.iam_manager.get_role_arn("lambda-role")

        self.func_name = "DeployTestFunc"
        self.function_manager = LambdaFunctionManager(self.lambda_client)
        self.function_manager.create_function(
            func_name=self.func_name,
            role_arn=self.role_arn,
            code=self.manager.package_from_local_path(self.temp_dir.name),
            handler="lambda_function.lambda_handler"
        )

    def tearDown(self):
        """Clean up test resources
        """
        self.temp_dir.cleanup()

    def test_package_from_local_path(self):
        """Test LambdaDeploymentManager.package_from_local_path
        """
        result = self.manager.package_from_local_path(self.temp_dir.name)
        self.assertIn('ZipFile', result)
        self.assertIsInstance(result['ZipFile'], bytes)

    def test_upload_to_s3(self):
        """Test LambdaDeploymentManager.upload_to_s3
        """
        zip_bytes = self.manager._generate_deployment_package(self.temp_dir.name)
        success = self.manager.upload_to_s3(zip_bytes, self.bucket, 'lambda-code.zip')
        self.assertTrue(success)

    def test_publish_new_version(self):
        """Test LambdaDeploymentManager.publish_new_version
        """
        version = self.manager.publish_new_version(self.func_name, description='new version test')
        self.assertIsNotNone(version)
        self.assertEqual(version, '1')


@mock_lambda
@mock_iam
@mock_s3
class LambdaInvocationManagerTest(unittest.TestCase):
    def setUp(self):
        """Set up test resources
        """
        self.lambda_client = boto3.client('lambda', region_name="us-east-1")
        self.s3_client = boto3.client('s3', region_name="us-east-1")
        self.iam_client = boto3.client("iam", region_name="us-east-1")

        self.function_manager = LambdaFunctionManager(self.lambda_client)
        self.invocation_manager = LambdaInvocationManager(self.lambda_client)
        self.deployment_manager = LambdaDeploymentManager(self.lambda_client, self.s3_client)

        self.temp_dir = tempfile.TemporaryDirectory()
        file_path = os.path.join(self.temp_dir.name, 'lambda_function.py')

        with open(file_path, 'w') as f:
            f.write(
                "import json\n"
                "def lambda_handler(event, context):\n"
                "    return {'statusCode': 200, 'body': json.dumps(event.get('msg', 'default'))}\n"
            )

        self.code = self.deployment_manager.package_from_local_path(self.temp_dir.name)

        self.iam_manager = IAMRoleManager(self.iam_client)
        trust_policy = json.dumps({
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "lambda.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }]
        })
        self.iam_manager.create_role("invoke-role", trust_policy)
        self.role_arn = self.iam_manager.get_role_arn("invoke-role")

        self.function_name = 'InvokeTestFunction'
        self.function_manager.create_function(
            func_name=self.function_name,
            role_arn=self.role_arn,
            code=self.code,
            handler='lambda_function.lambda_handler'
        )

    def tearDown(self):
        """Clean up test resources
        """
        self.temp_dir.cleanup()

    def test_invoke_function(self):
        """Test LambdaInvocationManager.invoke_function
        """
        payload = {'msg': 'HelloTest'}
        result = self.invocation_manager.invoke_function(self.function_name, payload)
        self.assertEqual(result, {"statusCode": 200, "body": None})