"""
Implements all SageMaker functionalities.
"""
from aws_setup.utils.logger import logger
from aws_setup.managers.s3_manager import S3BucketManager, S3ObjectManager
from aws_setup.interfaces.sagemaker_interface import SageMakerNotebookInterface, SageMakerDataInterface, SageMakerTrainingInterface
from botocore.exceptions import ClientError
from botocore.client import BaseClient
from typing import List, Dict, Optional
import os
import tempfile
import boto3
import matplotlib.pyplot as plt
from PIL import Image

class SageMakerNotebookManager(SageMakerNotebookInterface):
    def __init__(self,
                 notebook_name: str,
                 role_arn: str,
                 instance_type: str):
        """Initialize SageMaker Notebook resources
        Args:
            notebook_name: must be a unique name without spaces or special characters
            role_arn: the Amazon Resource Name of the IAM role that the notebook will assume
            instance_type: the performance/compute resources of the notebook (use ml.g5.xlarge for training & preprocessing notebook and ml.t3.medium for unit testing)
        """
        self.client = boto3.client('sagemaker')
        self.notebook_name = notebook_name
        self.role_arn = role_arn
        self.instance_type = instance_type
    
    def create_notebook(self) -> bool:
        """Creates a SageMaker notebook instance
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker/client/create_notebook_instance.html
        Return:
            True/False to indicate success/failure
        """
        try:
            self.client.create_notebook_instance(NotebookInstanceName=self.notebook_name,
                                                 InstanceType=self.instance_type,
                                                 RoleArn=self.role_arn)
        except ClientError as e:
            logger.error(f'[FAIL] cannot create SageMaker Notebook instance ({e})')
            return False
        logger.info(f'[SUCCESS] created SageMaker Notebook instance')
        return True

    def start_notebook(self) -> bool:
        """Starts the created SageMaker notebook instance
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/1.28.1/reference/services/sagemaker/client/start_notebook_instance.html
        Return:
            True/False to indicate success/failure
        """
        try:
            self.client.start_notebook_instance(NotebookInstanceName=self.notebook_name)
        except ClientError as e:
            logger.error(f'[FAIL] cannot start SageMaker Notebook instance ({e})')
            return False
        logger.info(f'[SUCCESS] started SageMaker Notebook instance')
        return True

    def stop_notebook(self) -> bool:
        """Stops the running SageMaker Notebook instance
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker/client/stop_notebook_instance.html
        Return:
            True/False to indicate success/failure
        """
        try:
            self.client.stop_notebook_instance(NotebookInstanceName=self.notebook_name)
        except ClientError as e:
            logger.error(f'[FAIL] cannot stop SageMaker Notebook instance ({e})')
            return False
        logger.info(f'[SUCCESS] stopped SageMaker Notebook instance')
        return True

    def get_notebook_status(self) -> str:
        """Gets the status of the SageMaker notebook
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker/client/list_notebook_instances.html
        Return:
            the status of the Notebook or an empty string if there's a failure in the API call
        """
        try:
            response = self.client.list_notebook_instances()
        except ClientError as e:
            logger.error(f'[FAIL] cannot retrieve SageMaker Notebook instance status ({e})')
            return ''
        
        # Get notebook status
        notebook_status = response['NotebookInstances'][0]['NotebookInstanceStatus']

        logger.info(f'[SUCCESS] retrieved and returned SageMaker Notebook instance status: {notebook_status}')
        
        return notebook_status

    def delete_notebook(self) -> bool:
        """Deletes a notebook AND ALL OF ITS DATA
        Docs:
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker/client/delete_notebook_instance.html
        Return:
            True/False to indicate success/failure
        """
        try:
            self.client.delete_notebook_instance(NotebookInstanceName=self.notebook_name)
        except ClientError as e:
            logger.error(f'[FAIL] cannot delete SageMaker Notebook instance ({e})')
            return False
        logger.info(f'[SUCCESS] deleted SageMaker Notebook instance')
        return True

class SageMakerDataInspectorManager(SageMakerDataInterface):
    def __init__(self,
                 s3_object_manager: S3ObjectManager,
                 notebook_instance: SageMakerNotebookInterface):
        """Initialize SageMaker resources
        Args:
            s3_object_manager: the S3 object manager
            notebook_instance: the SageMaker Notebook
        """
        self.s3_object_manager = s3_object_manager
        self.notebook = notebook_instance
    
    def download_data(self,
                  local_dir: str,
                  bucket: str,
                  prefix: str,
                  n_samples: Optional[int] = 100) -> bool:
        """
        Download up to n_samples from a S3 bucket prefix to a local SageMaker (EC2) directory.

        Args:
            local_dir: Local directory (in SageMaker EC2) to store downloaded data
            bucket: The S3 bucket name
            prefix: S3 prefix (like a folder path) to filter objects (e.g. 'raw/' or 'preprocessed/')
            n_samples: Max number of samples to download (downloads all if None)

        Returns:
            True if all requested files are downloaded successfully, False otherwise
        """
        try:
            # Check notebook status
            notebook_status = self.notebook.get_notebook_status()
            if notebook_status != 'InService':
                logger.warning(f'[WARNING] SageMaker Notebook is not in service!\n(status={notebook_status})')
                return False

            # Create the local directory in SageMaker (EC2)
            os.makedirs(local_dir, exist_ok=True)

            # Get filtered list of objects in S3 with prefix
            objects_list = self.s3_object_manager.list_objects(bucket, prefix=prefix)

        except ClientError as e:
            logger.error(f'[FAIL] cannot get notebook status and/or retrieve S3 objects ({e})')
            return False

        if not objects_list:
            logger.error(f'[FAIL] no objects found at s3://{bucket}/{prefix}')
            return False

        # Isolate object keys
        object_keys = [obj[0] for obj in objects_list]

        # Limit to n_samples if specified
        if n_samples is not None:
            object_keys = object_keys[:n_samples]

        # Download the objects
        download_fails = []
        for obj_key in object_keys:
            # Keep folder structure flat if needed
            filename = os.path.basename(obj_key)
            local_path = os.path.join(local_dir, filename)

            download_status = self.s3_object_manager.download_object(
                bucket_name=bucket,
                object_key=obj_key,
                destination_path=local_path
            )

            if not download_status:
                download_fails.append(obj_key)

        if len(download_fails) == 0:
            logger.info(f'[SUCCESS] downloaded {len(object_keys)}/{len(object_keys)} to local dir "{local_dir}"')
            return True
        else:
            download_successes = len(object_keys) - len(download_fails)
            logger.warning(f'[WARNING] {len(download_fails)} failed downloads. Successfully downloaded {download_successes}/{len(object_keys)} objects to \"{local_dir}\"')
            return False

    def visualize_samples(self):
        """Visualize the data samples graphically (with Matplotlib)

        """
        pass
            
        

            