"""
Implements all SageMaker functionalities.
"""
from aws_setup.utils.logger import logger
from aws_setup.managers.s3_manager import S3ObjectManager
from aws_setup.interfaces.sagemaker_interface import SageMakerNotebookInterface, SageMakerDataInterface, SageMakerTrainingInterface
from botocore.exceptions import ClientError
from typing import Optional
import os
import boto3
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import time

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

class SageMakerDataManager(SageMakerDataInterface):
    def __init__(self,
                 s3_object_manager: S3ObjectManager,
                 notebook_instance: SageMakerNotebookInterface):
        """Initialize SageMaker data resources
        Args:
            s3_object_manager: the S3 object manager
            notebook_instance: the SageMaker Notebook
        """
        self.s3_object_manager = s3_object_manager
        self.notebook = notebook_instance
        self.local_dir = ''
        self.all_files = []
    
    def download_data(self,
                  local_dir: str,
                  bucket: str,
                  prefix: str,
                  n_samples: Optional[int] = 100,
                  skip_notebook_check=False) -> bool:
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
        self.bucket_name = bucket
        self.local_dir = local_dir
        try:
            # Check notebook status
            if not skip_notebook_check:
                status = self.notebook.get_notebook_status()
                if status != "InService":
                    logger.error("[FAIL] Notebook is not active")
                    return False

            # Create the local directory in SageMaker (EC2)
            os.makedirs(local_dir, exist_ok=True)

            # Get list of objects
            objects_list = self.s3_object_manager.list_objects(bucket)

        except ClientError as e:
            logger.error(f'[FAIL] cannot get notebook status and/or retrieve S3 objects ({e})')
            return False

        if not objects_list:
            logger.error(f'[FAIL] no objects found at s3://{bucket}/{prefix}')
            return False

        # Isolate object keys and filter by prefix
        object_keys = []
        for obj in objects_list:
            obj_key = obj[0]
            if obj_key.startswith(prefix):
                object_keys.append(obj_key)

        # Limit to n_samples if specified
        if n_samples is not None:
            object_keys = object_keys[:n_samples]

        # Download the objects
        download_fails = []
        for obj_key in object_keys:
            # Keep folder structure flat if needed
            filename = os.path.basename(obj_key)
            local_path = os.path.join(local_dir, filename)

            download_status = self.s3_object_manager.download_object(bucket_name=bucket,
                                                                     object_key=obj_key,
                                                                     destination_path=local_path)

            if not download_status:
                download_fails.append(obj_key)

        if len(download_fails) == 0:
            logger.info(f'[SUCCESS] downloaded {len(object_keys)}/{len(object_keys)} to local dir "{local_dir}"')
            return True
        else:
            download_successes = len(object_keys) - len(download_fails)
            logger.warning(f'[WARNING] {len(download_fails)} failed downloads. Successfully downloaded {download_successes}/{len(object_keys)} objects to \"{local_dir}\"')
            return False

    def visualize_samples(self, local_dir: str) -> bool:
        """
        Visualize up to 9 data samples graphically (with Matplotlib)
        Args:
            local_dir: Local directory containing downloaded images
        Returns:
            True if visualization is successful, False otherwise
        """
        MAX_SAMPLES = 9
        try:
            self.all_files = [f for f in os.listdir(local_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            if not self.all_files:
                print(f'[INFO] No image files found in {local_dir}')
                return False

            sample_files = self.all_files[:MAX_SAMPLES]
            num_samples = len(sample_files)

            cols = 3
            rows = int(np.ceil(num_samples / cols))

            _, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
            axs = axs.flatten()

            for i, ax in enumerate(axs):
                if i < num_samples:
                    img_path = os.path.join(local_dir, sample_files[i])
                    with Image.open(img_path) as img:
                        ax.imshow(img)
                        ax.set_title(sample_files[i])
                        ax.axis('off')
                else:
                    ax.axis('off')

            plt.tight_layout()
            plt.show()
            return True

        except Exception as e:
            print(f'[ERROR] Failed to visualize samples: {e}')
            return False
        
    def validate_data(self) -> bool:
        """Ensure all images are the same size
        Return:
            True/False to indicate success/failure
        """
        try:
            if not self.all_files:
                print(f'[INFO] No files in self.all_files')
                return False

            expected_size = None
            for fname in self.all_files:
                path = os.path.join(self.local_dir, fname)
                with Image.open(path) as img:
                    size = img.size  # (width, height)

                    if expected_size is None:
                        expected_size = size
                    elif size != expected_size:
                        print(f'[FAIL] Image size mismatch: {fname} is {size}, expected {expected_size}')
                        return False

            print(f'[SUCCESS] All {len(self.all_files)} images are size {expected_size}')
            return True
        except Exception as e:
            print(f'[ERROR] Failed to validate data: {e}')
            return False
        
class SageMakerTrainingManager(SageMakerTrainingInterface):
    def __init__(self,
                 model: torch.nn.Module,
                 dataloader: torch.utils.data.DataLoader,
                 optimizer: torch.optim.Optimizer,
                 loss_fn,
                 s3_object_manager: S3ObjectManager,
                 notebook_instance: SageMakerNotebookInterface):
        """Initialize SageMaker training resources
        Args:
            s3_object_manager: the S3 object manager
            notebook_instance: the SageMaker Notebook
        """
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.s3_object_manager = s3_object_manager
        self.notebook = notebook_instance

    def _save_checkpoint(self, bucket: str, save_name: str) -> bool:
        """Save model to bucket
        Args:
            bucket: the bucketname to save the model to
            save_name: the file name for saving the bucket
        Return:
            True/False to indicate success/failure
        """
        try:
            torch.save(self.model.state_dict(), save_name)
            self.s3_object_manager.upload_file(bucket, save_name, save_name)
            print(f'[SUCCESS] Saved checkpoint {save_name} to s3://{bucket}/{save_name}')
            return True
        except Exception as e:
            print(f'[ERROR] Failed to save checkpoint: {e}')
            return False
    
    def _load_checkpoint(self, bucket: str, save_name: str) -> bool:
        """Load a checkpoint from the given S3 bucket and apply it to the model
        Args:
            bucket: S3 bucket name
            save_name: Filename of the checkpoint in S3
        Return:
            True/False to indicate success/failure
        """
        try:
            # Download checkpoint from S3 to local file
            
            local_path = f'/tmp/{save_name}'  # SageMaker safe temp dir
            self.s3_object_manager.download_file(bucket_name=bucket,
                                                object_key=save_name,
                                                destination_path=local_path)

            # Load weights into model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.load_state_dict(torch.load(local_path, map_location=device))
            self.model.to(device)

            print(f'[SUCCESS] Loaded checkpoint to CUDA from s3://{bucket}/{save_name}')
            return True

        except Exception as e:
            print(f'[ERROR] Failed to load checkpoint: {e}')
            return False


    def train(self,
              epochs: int,
              model_bucket: str,
              resume_from: Optional[str] = None) -> bool:
        """Train the model (optionally from a checkpoint)
        Args:
            epochs: the number of epochs
            model_bucket: the S3 model bucket
            resume_from: the path of the model it resumes training from, if enabled
        Return:
            True/False to indicate success/failure 
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        if resume_from:
            self._load_checkpoint(bucket=model_bucket, save_name=resume_from)

        try:
            for epoch in range(epochs):
                self.model.train()
                epoch_loss = 0.0
                num_batches = 0
                start_time = time.time()

                for inputs, targets in self.dataloader:
                    inputs, targets = inputs.to(device), targets.to(device)

                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    num_batches += 1

                # Compute average loss and time
                avg_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
                epoch_time = time.time() - start_time

                print(f'[INFO] Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s')

            return True
        except Exception as e:
            print(f'[ERROR] Training failed: {e}')
            return False