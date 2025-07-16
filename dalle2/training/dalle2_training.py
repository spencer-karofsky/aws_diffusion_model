"""
dalle2_training.py: Provides functionality to train the prior or decoder.

Description:
    * Provides classes that train the prior and the decoders.

Classes:
    * BaseTrainer(ABC): A base trainer containing functionalities
    * PriorTrainer(BaseTrainer): Trains the prior.
    * DecoderTrainer(BaseTrainer): Trains the decoder.

References:
    * DALL·E 2 Paper: https://cdn.openai.com/papers/dall-e-2.pdf
    * My DALL·E 2 Notes: https://github.com/spencer-karofsky/aws_diffusion_model/blob/main/dalle2/research_notes/DALL-E-2%202022.pdf or /dalle2/research_notes/DALL-E-2 2022.pdf

Author:
    * Spencer Karofsky (https://github.com/spencer-karofsky)
"""
# PyTorch imports
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Module imports
from models.prior import Prior
from models.decoder import Decoder
from models.dalle2 import DALLe2

from sampling.noise_scheduler import NoiseScheduler

# Other imports
from abc import ABC, abstractmethod
from typing import Union, Tuple
import os
from tqdm import tqdm

class BaseTrainer(ABC):
    def __init__(
            self,
            train_module: Union[Prior, Decoder],
            optimizer: torch.optim,
            noise_scheduler: NoiseScheduler,
            dataset: Dataset,
            batch_size: int,
            model_save_name: str,
            shuffle: bool = True,
            on_aws: bool = False,
            debug: bool = False,
    ):
        """
        Base trainer for all classes.

        Args:
            train_module: either the prior or decoder module
            optimizer: the optimizer, used to update the model's parameters
            noise_scheduler: the noise scheduler, which retrieves beta_t, alpha_t, and alpha-bar_t
            dataset: a PyTorch dataset, which will be passed into the PyTorch dataloader
            batch_size: the batch size
            model_save_name: the name of the model to save
            on_aws: configures the training for AWS (TODO)
            debug: provides useful debugging information (mostly for testing than deployment and heavy inference)
        """
        # Save params for training
        self.train_module = train_module
        self.optimizer = optimizer
        self.noise_scheduler = noise_scheduler
        self.model_save_name = model_save_name
        self.on_aws = on_aws
        self.debug = debug
        
        # Create PyTorch DataLoader
        num_workers = min(4, os.cpu_count() or 1)

        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )

        # Assign device, ideally using a GPU-accelerated framework
        self.device = (
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )
        self.train_module.to(self.device)
        
        # If training on AWS, configure training resources for AWS as opposed to on a local machine
        if self.on_aws:
            self._configure_for_aws()

        # Print out useful information for debugging
        if self.debug:
            print(f'Training module: {self.train_module.__class__.__name__}')
            print(f'Device: {self.device}')
            print(f'Number of training batches: {len(self.dataloader)}')
    
    def _configure_for_aws(self) -> None:
        """
        Configure resources for AWS (TODO)
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def module_type(self) -> str:
        """
        Since most of the training logic is defined in the abstract class, this property tells the object if it's the prior or the decoder.

        Returns:
            either 'prior' or 'decoder', which informs self._save_current_model on which directory to save the model to
        """
        pass

    @abstractmethod
    def _run_batch(
            self,
            batch_input: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    ) -> torch.Tensor:
        """
        Defines how a batch is passed through the model.

        Args:
            batch_input: the input to our batch, X
        
        Returns:
            the predicted target
        """
        pass

    @abstractmethod
    def _compute_batch_loss(
            self,
            target: torch.Tensor,
            predicted: torch.Tensor
    
    ) -> torch.Tensor:
        """
        Computes the per-batch loss, given predicted and target.

        Both the prior and decoder use the same loss function (MSE), but since they are semantically different, we define separate methods.

        Args:
            target: target Tensor
            predicted: predicted Tensor
        
        Returns:
            the batch loss as a 0-dimension Tensor ([batch loss scalar])
        """
        pass
    
    def _save_current_model(
            self,
            epoch: int,
            batch_i: int
    ) -> None:
        """
        Saves the partially-trained model, either locally or to S3

        Args:
            epoch: the current epoch
            batch: the current batch index
        """
        if self.on_aws:
            # Save to S3 bucket
            raise NotImplementedError
        else:
            # Save to local directory
            save_path = f'dalle2/checkpoints/{self.module_type}/epoch{epoch + 1}_batch{batch_i + 1}.pth'
            torch.save(self.train_module.state_dict(), save_path)

    def train(
            self,
            num_epochs: int,
            save_every: int = 10,
            resume_checkpoint_name: str = None
    ) -> None:
        """
        Trains the DALL·E 2 module.

        Args:
            num_epochs: number of epochs
            save_every: how often to save the model state (in batches, not epochs)
            resume_checkpoint_name: if resuming training, the path to load from
        """
        # Internally switches PyTorch to training mode
        self.train_module.train()

        # Optionally load model from checkpoint
        if resume_checkpoint_name is not None:
            checkpoint_path = os.path.join(
                'dalle2', 'checkpoints', self.module_type, resume_checkpoint_name
            )
            if os.path.isfile(checkpoint_path):
                print(f'Resuming from checkpoint: {checkpoint_path}')
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.train_module.load_state_dict(checkpoint)
            else:
                raise FileNotFoundError(f'No checkpoint found at: {checkpoint_path}')

        # Training loop
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            epoch_loss = 0.0
            
            # Wrap the dataloader with tqdm for a batch-level progress bar
            for batch_i, batch in enumerate(tqdm(self.dataloader, desc='Training', leave=True)): # leave=True keeps each progress bar after training
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward Pass
                target_pred = self._run_batch(
                    batch_input=inputs
                )

                # Compute Batch Loss
                loss = self._compute_batch_loss(
                    target=targets,
                    predicted=target_pred
                )

                epoch_loss += loss.item()

                # Clear previous batch gradients, compute current batch gradients, and update weights
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Periodically save the current model state in case of training interruptions
                if (batch_i + 1) % save_every == 0:
                    self._save_current_model(epoch, batch_i)

            print(f'Average Epoch loss: {epoch_loss / len(self.dataloader):.4f}')

        
class PriorTrainer(BaseTrainer):
    @property
    def module_type(self) -> str:
        return 'prior'
    
    def _run_batch(
            self,
            batch_input: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    ) -> torch.Tensor:
        """
        Defines how a batch is passed through the prior.

        Args:
            batch_input: the input of the batch, (z_T (B, 512), z_txt (B, 512), t_emb (B, 512))
        
        Returns:
            the predicted target
        """
        # Extract the text embedding, the timestep embedding, and the fully-noised imaage (DDPM-defined)
        z_T, z_txt, t_emb = batch_input

        # Forward pass
        return self.train_module.forward(
            z_txt=z_txt,
            t_emb=t_emb,
            z_T=z_T
        )
    
    def _compute_batch_loss(
            self,
            target: torch.Tensor,
            predicted: torch.Tensor
    
    ) -> torch.Tensor:
        """
        Computes the prior's per-batch loss, given predicted and target

        Args:
            target: target Tensor
            predicted: predicted Tensor
        
        Returns:
            the prior's batch loss as a 0-dimension Tensor ([batch loss scalar])
        """
        # The prior's loss is the MSE between the predicted noise and true noise added to the CLIP image embeddings
        return F.mse_loss(predicted, target)

class DecoderTrainer(BaseTrainer):
    @property
    def module_type(self) -> str:
        return 'decoder'

    def _run_batch(
            self,
            batch_input: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    ) -> torch.Tensor:
        """
        Defines how a batch is passed through the decoder.

        Args:
            batch_input: the input of the batch, (z_t (B, 512), z_img (B, 512), t_emb (B, 512))
        
        Returns:
            the predicted target
        """
        # Extract the partially-noisy images, the image embeddings, and the timestep embeddings from the batch input
        x_t, z_img, t_emb = batch_input

        # Forward pass
        return self.train_module.forward(
            x_t=x_t,
            z_img=z_img,
            t_emb=t_emb,
        )
    
    def _compute_batch_loss(
            self,
            target: torch.Tensor,
            predicted: torch.Tensor
    
    ) -> torch.Tensor:
        """
        Computes the decoder's per-batch loss, given predicted and target.

        Args:
            target: target Tensor
            predicted: predicted Tensor
        
        Returns:
            the decoder's batch loss as a 0-dimension Tensor ([batch loss scalar])
        """
        # The decoder's loss is the MSE between the predicted noise and true noise added to the final generated images
        return F.mse_loss(predicted, target)


    

        