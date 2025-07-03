"""
Defines interfaces for all Amazon SageMaker Functionalities.


Notebook Manager:
1. Create Notebook
2. Start Notebook
3. Stop Notebook
4. Get Notebook Status
5. Delete Notebook

Data Manager:
1. Download Data from S3
2. Visualize Samples
3. Validate Data Integrity

Training Manager:
1. Train the model.
    1.1. Save Checkpoint Data to S3
    1.2. Resume from Checkpoint
2. Save Final Model
"""
from typing import Protocol, Optional

class SageMakerNotebookInterface(Protocol):
    def create_notebook(self) -> bool:
        raise NotImplementedError
    
    def start_notebook(self) -> bool:
        raise NotImplementedError
    
    def stop_notebook(self) -> bool:
        raise NotImplementedError
    
    def get_notebook_status(self) -> str:
        raise NotImplementedError
    
    def delete_notebook(self) -> bool:
        raise NotImplementedError

class SageMakerDataInterface(Protocol):
    def download_data(self,
                      local_dir: str,
                      bucket: str,
                      prefix: str,
                      n_samples: Optional[int] = 100) -> bool:
        raise NotImplementedError
    
    def visualize_samples(self) -> bool:
        raise NotImplementedError
    
    def validate_data(self) -> bool:
        raise NotImplementedError

class SageMakerTrainingInterface(Protocol):
    def _save_checkpoint(self,
                         bucket: str,
                         save_name: str) -> bool:
        raise NotImplementedError
    
    def _load_checkpoint(self,
                         bucket: str,
                         save_name: str) -> bool:
        raise NotImplementedError
    
    def train(self,
              epochs: int,
              resume_from: Optional[str] = None) -> bool:
        raise NotImplementedError
