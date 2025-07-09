"""
Test EBSVolumeManager functionalities with moto
(fake AWS calls that mimics boto3) and Python's unittest.
"""
import boto3
from aws_setup.managers.iam_manager import IAMRoleManager
from aws_setup.managers.s3_manager import S3BucketManager, S3ObjectManager
from aws_setup.managers.sagemaker_manager import SageMakerNotebookManager, SageMakerDataManager, SageMakerTrainingManager
import unittest
from moto import mock_sagemaker, mock_iam, mock_s3
from PIL import Image
import os
import tempfile
from typing import List
import json
from unittest.mock import patch, MagicMock, call
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

class SageMakerNotebookManagerTest(unittest.TestCase):
    def setUp(self):
        """Set up mocked SageMaker Notebook Manager
        """
        # Mock SageMaker and IAM
        self.mock_sagemaker = mock_sagemaker()
        self.mock_iam = mock_iam()

        self.mock_sagemaker.start()
        self.mock_iam.start()

        self.iam_client = boto3.client('iam')

        # Create IAM role and get ARN
        iam_role_manager = IAMRoleManager(self.iam_client)
        self.role_name = 'my-sagemaker-role'
        assume_role_policy = json.dumps({
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "sagemaker.amazonaws.com"
                    },
                    "Action": "sts:AssumeRole"
                }]
            })

        iam_role_manager.create_role(role_name=self.role_name,
                                     policy_doc=assume_role_policy)
        self.arn = iam_role_manager.get_role_arn(self.role_name)

        # Create SageMaker Notebook Manager Instance
        self.notebook_name = 'my-sagemaker-notebook'
        self.instance_type = 'ml.t2.medium' # Cheapest instance type (sufficient for testing)
        self.sagemaker_notebook_manager = SageMakerNotebookManager(notebook_name=self.notebook_name,
                                                                   role_arn=self.arn,
                                                                   instance_type=self.instance_type)

    def tearDown(self):
        """Clean up test resources
        """
        self.mock_sagemaker.stop()
        self.mock_iam.stop()
    
    def test_create_notebook(self):
        """Test SageMakerNotebookManager.create_notebook
        """
        self.assertTrue(
            self.sagemaker_notebook_manager.create_notebook()
        )
    
    def test_start_notebook(self):
        """Test SageMakerNotebookManager.start_notebook
        """
        # Create notebook
        self.sagemaker_notebook_manager.create_notebook()

        # Verify notebook starts
        self.assertTrue(
            self.sagemaker_notebook_manager.start_notebook()
        )
    
    def test_stop_notebook(self):
        """Test SageMakerNotebookManager.stop_notebook
        """
        # Create and start notebook
        self.sagemaker_notebook_manager.create_notebook()
        self.sagemaker_notebook_manager.start_notebook()

        # Verify notebook stops
        self.assertTrue(
            self.sagemaker_notebook_manager.stop_notebook()
        )
    
    @patch.object(SageMakerNotebookManager, "get_notebook_status", return_value="InService")
    def test_get_notebook_status(self, mock_status):
        self.sagemaker_notebook_manager.create_notebook()
        self.sagemaker_notebook_manager.start_notebook()
        self.assertEqual(self.sagemaker_notebook_manager.get_notebook_status(), "InService")

    @patch.object(SageMakerNotebookManager, "get_notebook_status", return_value="Stopped")
    def test_delete_notebook(self, mock_status):
        self.sagemaker_notebook_manager.create_notebook()
        self.sagemaker_notebook_manager.stop_notebook()
        self.assertTrue(self.sagemaker_notebook_manager.delete_notebook())


class SageMakerDataManagerTest(unittest.TestCase):
    def _create_dummy_images(self,
                             local_dir: str,
                             n: int = 9,
                             size=(32, 32)) -> List[Image.Image]:
        """Generate dummy images for upload and return their paths
        Args:
            local_dir: the local directory
            n: the number of images to generate
            size: the image size, in pixels
        Return:
            List of generated images
        """
        os.makedirs(local_dir, exist_ok=True)
        paths = []
        for i in range(n):
            img = Image.new('RGB', size, color=(i * 20 % 255, i * 40 % 255, i * 60 % 255))
            path = os.path.join(local_dir, f'dummy_image_{i}.png')
            img.save(path)
            paths.append(path)
        return paths
    
    def _upload_dummy_images_to_s3(self,
                                   bucket: str,
                                   prefix: str,
                                   local_paths: List[str]):
        """Upload dummy images to a specific S3 prefix
        Args:
            bucket: the S3 bucket name
            prefix: the S3 bucket prefix
            local_paths: the paths of images
        """
        for path in local_paths:
            key = f"{prefix.rstrip('/')}/{os.path.basename(path)}"
            self.s3_object_manager.upload_object(bucket_name=bucket, object_key=key, file_path=path)

    def setUp(self):
        """Set up mocked SageMaker, IAM, and S3 resources
        """
        self.mock_sagemaker = mock_sagemaker()
        self.mock_iam = mock_iam()
        self.mock_s3 = mock_s3()

        self.mock_sagemaker.start()
        self.mock_iam.start()
        self.mock_s3.start()

        # IAM setup
        self.iam_client = boto3.client('iam')
        iam_role_manager = IAMRoleManager(self.iam_client)
        self.role_name = 'my-sagemaker-role'
        assume_role_policy = json.dumps({
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "sagemaker.amazonaws.com"
                    },
                    "Action": "sts:AssumeRole"
                }]
            })
        iam_role_manager.create_role(role_name=self.role_name,
                                     policy_doc=assume_role_policy)
        self.arn = iam_role_manager.get_role_arn(self.role_name)

        # S3 setup
        self.s3_bucket_manager = S3BucketManager()
        self.s3_object_manager = S3ObjectManager()
        self.bucket_name = 'test-bucket'
        self.prefix = 'test-images/'

        self.s3_bucket_manager.create_bucket(self.bucket_name)

        # SageMaker setup
        self.notebook_name = 'test-notebook'
        self.instance_type = 'ml.t2.medium'
        self.sagemaker_notebook_manager = SageMakerNotebookManager(
            notebook_name=self.notebook_name,
            role_arn=self.arn,
            instance_type=self.instance_type
        )
        self.sagemaker_notebook_manager.create_notebook()

        self.sagemaker_data_manager = SageMakerDataManager(
            s3_object_manager=self.s3_object_manager,
            notebook_instance=self.sagemaker_notebook_manager
        )

        # Temp directory for image storage
        self.temp_dir = tempfile.TemporaryDirectory()
        self.local_dir = self.temp_dir.name

        self.image_paths = self._create_dummy_images(self.local_dir)
        self._upload_dummy_images_to_s3(bucket=self.bucket_name, prefix=self.prefix, local_paths=self.image_paths)

    def tearDown(self):
        """Clean up test resources
        """
        self.temp_dir.cleanup()
        self.mock_sagemaker.stop()
        self.mock_iam.stop()
        self.mock_s3.stop()

    def test_download_data(self):
        """Test SageMakerDataManager.download_data
        """
        download_dir = tempfile.mkdtemp()
        result = self.sagemaker_data_manager.download_data(
            local_dir=download_dir,
            bucket=self.bucket_name,
            prefix=self.prefix,
            n_samples=9,
            skip_notebook_check=True
        )
        self.assertTrue(result)
        downloaded_files = os.listdir(download_dir)
        self.assertEqual(len(downloaded_files), 9)
    
    def test_visualize_samples(self):
        """Test SageMakerDataManager.visualize_samples
        """
        download_dir = tempfile.mkdtemp()
        self.sagemaker_data_manager.download_data(
            local_dir=download_dir,
            bucket=self.bucket_name,
            prefix=self.prefix,
            n_samples=9,
            skip_notebook_check=True
        )
        self.assertTrue(self.sagemaker_data_manager.visualize_samples(local_dir=download_dir))

    @patch.object(SageMakerNotebookManager, "get_notebook_status", return_value="InService")
    def test_validate_data(self, mock_status):
        """Test SageMakerDataManager.validate_data
        """
        # Make temporary directory
        download_dir = tempfile.mkdtemp()

        # Download data
        self.sagemaker_data_manager.download_data(
            local_dir=download_dir,
            bucket=self.bucket_name,
            prefix=self.prefix,
            n_samples=9
        )

        self.sagemaker_data_manager.all_files = [
            f for f in os.listdir(download_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.sagemaker_data_manager.local_dir = download_dir

        self.assertTrue(self.sagemaker_data_manager.validate_data())

class SageMakerTrainingManagerTest(unittest.TestCase):
    def setUp(self):
        """Set up testing resources
        """
        X = torch.randn(20, 10)
        y = torch.randn(20, 1)
        dataset = TensorDataset(X, y)
        self.dataloader = DataLoader(dataset, batch_size=5)

        self.model = DummyModel()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()

        # Mock S3 manager
        self.s3_object_manager = MagicMock()
        self.s3_object_manager.upload_file = MagicMock(return_value=True)
        self.s3_object_manager.download_file = MagicMock(return_value=True)

        # Notebook mock
        self.notebook = MagicMock()

        self.manager = SageMakerTrainingManager(
            model=self.model,
            dataloader=self.dataloader,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn,
            s3_object_manager=self.s3_object_manager,
            notebook_instance=self.notebook
        )

    @patch("torch.save")
    def test_save_checkpoint(self, mock_save):
        """Test SageMakerTrainingInterface._save_checkpoint
        """
        with tempfile.NamedTemporaryFile() as tmp:
            result = self.manager._save_checkpoint("my-bucket", tmp.name)
            mock_save.assert_called_once()
            self.s3_object_manager.upload_file.assert_called_once()
            self.assertTrue(result)

    @patch("torch.load")
    def test_load_checkpoint(self, mock_load):
        """Test SageMakerTrainingInterface._load_checkpoint
        """
        mock_load.return_value = self.model.state_dict()
        self.s3_object_manager.download_file.return_value = True

        result = self.manager._load_checkpoint("my-bucket", "checkpoint.pt")
        self.s3_object_manager.download_file.assert_called_once()
        self.assertTrue(result)

    def test_train(self):
        """Test SageMakerTrainingInterface.train
        """
        self.assertTrue(
            self.manager.train(epochs=1, model_bucket="unused")
        )

    @patch.object(SageMakerTrainingManager, "_load_checkpoint")
    def test_train_resume_from(self, mock_load):
        result = self.manager.train(epochs=1, model_bucket="bucket", resume_from="checkpoint.pt")
        mock_load.assert_called_once_with(bucket="bucket", save_name="checkpoint.pt")
        self.assertTrue(result)
    
class DummyModel(nn.Module):
    def __init__(self):
        """Dummy Neural Network to train
        """
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)