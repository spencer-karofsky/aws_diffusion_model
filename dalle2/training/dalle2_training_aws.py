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
from dalle2.models.prior import Prior
from dalle2.models.decoder import Decoder

from dalle2.sampling.noise_scheduler import NoiseScheduler
from dalle2.sampling.ddim_sampling import DecoderDDIMSampler
from dalle2.models.clip_encoding import CLIPEncoder

# Other imports
from abc import ABC, abstractmethod
from typing import Union, Tuple
import os
from tqdm import tqdm
import csv
import boto3
import tempfile
import math
from torch.amp import autocast, GradScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def cosine_lr(step: int, total_steps: int, lr_max: float, lr_min: float, warmup_steps: int = 0) -> float:
    if step < warmup_steps:
        return lr_max * (step + 1) / max(1, warmup_steps)
    t = step - warmup_steps
    T = max(1, total_steps - warmup_steps)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * t / T))


def maybe_ablate_cond(z_img: torch.Tensor,
                      t_emb: torch.Tensor,
                      ablate_zero: bool = False,
                      ablate_shuffle: bool = False):
    """
    Quickly probe whether conditioning matters.
    ablate_zero    : z_img, t_emb -> 0
    ablate_shuffle : z_img, t_emb are shuffled across batch
    """
    if ablate_zero:
        z_img = torch.zeros_like(z_img)
        t_emb = torch.zeros_like(t_emb)
        return z_img, t_emb

    if ablate_shuffle:
        perm = torch.randperm(z_img.size(0), device=z_img.device)
        z_img = z_img[perm]
        t_emb = t_emb[perm]
        return z_img, t_emb

    return z_img, t_emb

@torch.no_grad()
def debug_direct_sampling(denoiser, sched, z_img, steps=200, H=64, W=64, device=None):
    if device is None:
        device = next(denoiser.parameters()).device
    denoiser.eval()

    sampler = DecoderDDIMSampler(sched, num_inference_steps=steps, eta=0.0)
    ts = sampler.timesteps  # strictly descending, ends at 0

    B = z_img.size(0)
    x_t = torch.randn(B, 3, H, W, device=device)

    for i, t_i in enumerate(ts):
        t   = torch.full((B,), t_i, dtype=torch.long, device=device)
        eps = denoiser(x_t=x_t, z_img=z_img, t=t)  # ε̂θ

        abar = sched.get_alpha_bar(t).view(B,1,1,1)
        x0   = (x_t - torch.sqrt(1 - abar) * eps) / torch.sqrt(abar)

        if i == len(ts) - 1:
            return x0.clamp(-1, 1)

        t_next    = torch.full((B,), ts[i+1], dtype=torch.long, device=device)
        abar_next = sched.get_alpha_bar(t_next).view(B,1,1,1)
        x_t = torch.sqrt(abar_next) * x0 + torch.sqrt(1 - abar_next) * eps


@torch.no_grad()
def debug_inference(model, scheduler, batch, device, n_steps=50,
                    epoch: int = 0, batch_i: int = 0, on_aws: bool = False, s3_client=None):
    """
    Runs DDIM denoising for one batch and saves rollout to intermediate outputs.
    Also mirrors to S3 if on_aws=True.
    """
    model.eval()

    z_img = batch["z_img"].to(device)
    B, H, W = batch["x_t"].shape[0], batch["x_t"].shape[-2], batch["x_t"].shape[-1]

    sampler = DecoderDDIMSampler(
        noise_scheduler=scheduler,
        num_inference_steps=n_steps,
        eta=0.0
    )

    imgs = sampler.sample(model, z_img, image_size=(H, W))

    # map [-1,1] → [0,1] for saving
    imgs = (imgs + 1) / 2

    # Local save
    save_dir = "dalle2/checkpoints/decoder_intermediate_outputs"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"epoch_{epoch}_batch_{batch_i}_rollout.png")

    from torchvision.utils import save_image
    save_image(imgs, save_path)

    print(f"[DEBUG] Saved rollout → {save_path}")

    # Optional S3 upload
    if on_aws and s3_client is not None:
        key = f"decoder/intermediate_outputs/epoch_{epoch}_batch_{batch_i}_rollout.png"
        try:
            s3_client.upload_file(save_path, "dalle2-outputs", key)
            print(f"[DEBUG] Uploaded rollout to s3://dalle2-outputs/{key}")
        except Exception as e:
            print(f"[WARN] S3 upload failed: {e}")

    model.train()
    return imgs





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
            use_amp: bool = False
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
        self.use_amp = use_amp
        self.scaler = GradScaler('cuda') if use_amp else None

        self.use_cosine_lr = True
        self.lr_max = 1e-4
        self.lr_min = 2e-5
        self.warmup_frac = 0.05
#        self.steps_per_epoch = len(self.dataloader)
        self.total_steps = 0  # to be initialized later

        
        # Create PyTorch DataLoader
        num_workers = min(4, os.cpu_count() or 1)

        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            # num_workers=4
            num_workers=0
        )

        self.steps_per_epoch = len(self.dataloader)
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

        from copy import deepcopy

        self.ema_model = deepcopy(self.train_module)
        self.ema_decay = 0.999  # You can tune this
        for p in self.ema_model.parameters():
            p.requires_grad = False


    
    def _configure_for_aws(self) -> None:
        """
        Prepare S3 for artifact uploads.
        """
        self.outputs_bucket = 'dalle2-outputs'
        self.models_bucket = 'dalle2-models'
        self.s3 = boto3.client('s3')


    
    def _s3_put(self, local_path: str, key: str):
        """
        Upload file to S3. 
        - Keys under {prior|decoder}/checkpoints/* go to the models bucket.
        - Everything else goes to the outputs bucket.
        """
        try:
            if not self.on_aws:
                return

            if not hasattr(self, "s3"):
                self.s3 = boto3.client("s3")

            # Route by key
            is_checkpoint = (
                key.startswith("prior/checkpoints/")
                or key.startswith("decoder/checkpoints/")
                or key.endswith(".pth")
            )
            bucket = self.models_bucket if is_checkpoint else self.outputs_bucket

            self.s3.upload_file(local_path, bucket, key)
        except Exception as e:
            print(f"[WARN] S3 upload failed for key '{key}': {e}")



    
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
    
    def _save_current_model(self, epoch: int, batch_i: int) -> None:
        fname = f'epoch{epoch}_batch{batch_i}.pth'
        if self.on_aws:
            # save to a temp file then upload
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                local_path = os.path.join(td, fname)
                torch.save(self.train_module.state_dict(), local_path)
                key = f'{self.module_type}/checkpoints/{fname}'
                self._s3_put(local_path, key)
        else:
            os.makedirs(f'dalle2/checkpoints/{self.module_type}', exist_ok=True)
            save_path = f'dalle2/checkpoints/{self.module_type}/{fname}'
            torch.save(self.train_module.state_dict(), save_path)

    def _save_final_model(self) -> None:
        fname = 'final_trained_model.pth'
        if self.on_aws:
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                local_path = os.path.join(td, fname)
                torch.save(self.train_module.state_dict(), local_path)
                key = f'{self.module_type}/checkpoints/{fname}'
                self._s3_put(local_path, key)
        else:
            os.makedirs(f'dalle2/checkpoints/{self.module_type}', exist_ok=True)
            save_path = f'dalle2/checkpoints/{self.module_type}/{fname}'
            torch.save(self.train_module.state_dict(), save_path)


    def _upload_file(self, local_path: str, key: str):
        """
        Utility: upload file to the appropriate S3 bucket and remove local file.
        """
        if not hasattr(self, "s3"):
            self.s3 = boto3.client("s3")

        is_checkpoint = (
            key.startswith("prior/checkpoints/")
            or key.startswith("decoder/checkpoints/")
            or key.endswith(".pth")
        )
        bucket = getattr(self, "models_bucket", "dalle2-models") if is_checkpoint \
                else getattr(self, "outputs_bucket", "dalle2-outputs")

        self.s3.upload_file(local_path, bucket, key)
        os.remove(local_path)

    def train(
            self,
            num_epochs: int,
            save_intermediate_output: int = 100,
            save_intermediate_model: int = 1000,
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
        
        self.total_steps = num_epochs * self.steps_per_epoch
        warmup_steps = int(self.warmup_frac * self.total_steps)

        # Optionally load model from checkpoint
    #    if resume_checkpoint_name is not None:
     #       checkpoint_path = os.path.join(
      #          'dalle2', 'checkpoints', self.module_type, resume_checkpoint_name
       #     )
        #    if os.path.isfile(checkpoint_path):
         #       print(f'Resuming from checkpoint: {checkpoint_path}')
          #      checkpoint = torch.load(checkpoint_path, map_location=self.device)
           #     self.train_module.load_state_dict(checkpoint)
          #  else:
           #     raise FileNotFoundError(f'No checkpoint found at: {checkpoint_path}')
        if resume_checkpoint_name is not None:
            if self.on_aws:
                checkpoint_path = self._download_checkpoint_from_s3(resume_checkpoint_name)
            else:
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
            print(f'Epoch {epoch + 1}/{num_epochs}')
            epoch_loss = 0.0
            
            for batch_i, batch in enumerate(tqdm(self.dataloader, desc='Training', leave=True)): # leave=True keeps each progress bar after training                
                if batch_i % 200 == 0 and self.module_type == "decoder":
                    debug_inference(
                        self.train_module,
                        self.noise_scheduler,
                        batch,
                        self.device,
                        n_steps=50,
                        epoch=epoch+1,
                        batch_i=batch_i+1,
                        on_aws=self.on_aws,
                        s3_client=getattr(self, "s3", None)
                    )

                
                # Get batch data (different depending on the prior vs decoder) and pass to correct PyTorch device
                if self.module_type == 'prior':
                    # Inputs
                    z_txt = batch['z_txt'].to(self.device)

                    if not hasattr(self, "_null_txt"):
                        clip = CLIPEncoder().to(self.device).eval()
                        with torch.no_grad():
                            self._null_txt = clip.encode_text([""]).to(self.device) # [1,512]

                    p_uncond = 0.2
                    if self.module_type == 'prior':
                        B = z_txt.shape[0]
                        null_batch = self._null_txt.expand(B, -1) # [B,512]
                        mask = (torch.rand(B, device=self.device) < p_uncond).view(B, 1) # True => use null
                        z_txt_train = torch.where(mask, null_batch, z_txt)
                    else:
                        z_txt_train = z_txt # unused for decoder

                    t = batch['t'].to(self.device)
                    z_img_noisy = batch['z_img_noisy'].to(self.device)

                    # Target (for computing loss)
                    eps_img = batch['eps_img'].to(self.device)

                    if self.debug:
                        print('[BaseTrainer] (Batch Data Passing into Prior:)')
                        print(f'[BaseTrainer] z_txt: {z_txt.shape}')
                        print(f'[BaseTrainer] t: {t.shape}')
                        print(f'[BaseTrainer] z_img_noisy: {z_img_noisy.shape}')
                        print(f'[BaseTrainer] eps_img (target noise): {eps_img.shape}')
                    if self.module_type == 'decoder':
                        batch_input = (
                            z_txt,
                            t,
                            z_img_noisy
                        )
                    elif self.module_type == 'prior':
                        batch_input = (
                            z_txt_train,
                            t,
                            z_img_noisy
                        )

                elif self.module_type == 'decoder':
                    # Inputs
                    x_t = batch['x_t'].to(self.device)
                    z_img = batch['z_img'].to(self.device)
                    t = batch['t'].to(self.device)

                    # Target (for computing loss)
                    eps_img = batch['eps_img'].to(self.device)

                    if self.debug:
                        print('[BaseTrainer] (Batch Data Passing into Decoder:)')
                        print(f'[BaseTrainer] x_t: {x_t.shape}')
                        print(f'[BaseTrainer] z_img: {z_img.shape}')
                        print(f'[BaseTrainer] t: {t.shape}')
                        print(f'[BaseTrainer] eps_img (target noise): {eps_img.shape}')
                    
                    batch_input = (
                        x_t,
                        z_img,
                        t
                    )
                z_img, t = self._abl(z_img, t, self.ABLATE_ZERO, self.ABLATE_SHUFFLE)
               # t_emb = self.get_timestep_embedding(t)
                batch_input = (x_t, z_img, t)
                # eps_hat = self._run_batch(batch_input=batch_input)
                if self.use_amp:
                    with autocast('cuda'):
                        eps_hat = self._run_batch(batch_input=batch_input)
                        # Compute loss inside autocast too
                        loss = self._compute_batch_loss(
                            target=eps_img,
                            predicted=eps_hat
                        )
                else:
                    eps_hat = self._run_batch(batch_input=batch_input)
                    loss = self._compute_batch_loss(
                        target=eps_img,
                        predicted=eps_hat
                    )

                cos_eps = F.cosine_similarity(
                    eps_hat.flatten(1), eps_img.flatten(1), dim=1
                ).mean().item()

                if batch_i % save_intermediate_output == 0:
                    print(f"[cos_eps] {cos_eps:.4f}")

                # Debug: log stats every 100 batches
                if (batch_i + 1) % save_intermediate_output == 0:
                    print(f"[Epoch {epoch + 1}, Batch {batch_i + 1}] Loss: {loss.item():.6f}")
                    print(f"  x_t:     mean={x_t.mean().item():.4f}, std={x_t.std().item():.4f}")
                    print(f"  eps_img: mean={eps_img.mean().item():.4f}, std={eps_img.std().item():.4f}")
                    print(f"  pred:    mean={eps_hat.mean().item():.4f}, std={eps_hat.std().item():.4f}")

                epoch_loss += loss.item()

                # Clear previous batch gradients, compute current batch gradients, and update weights
                self.optimizer.zero_grad()

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
        
                global_step = epoch * self.steps_per_epoch + batch_i
                if self.use_cosine_lr:
                    lr = cosine_lr(
                        step=global_step,
                        total_steps=self.total_steps,
                        lr_max=self.lr_max,
                        lr_min=self.lr_min,
                        warmup_steps=warmup_steps
                    )
                    for pg in self.optimizer.param_groups:
                        pg['lr'] = lr


                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
           
                with torch.no_grad():
                    for ema_p, p in zip(self.ema_model.parameters(), self.train_module.parameters()):
                        ema_p.data.mul_(self.ema_decay).add_(p.data, alpha=1.0 - self.ema_decay)


                if loss.item() < 1e-3:
                    fname = f"epoch{epoch+1}_batch{batch_i+1}_goodloss.pth"
                    print(f"[INFO] 🎉 Loss {loss.item():.6f} < 1e32 — saving checkpoint: {fname}")
                    self._run_intermediate_decoder_preview(epoch + 1, batch_i + 1, loss, steps=200, n_img=3)
                    if self.on_aws:
                        with tempfile.TemporaryDirectory() as td:
                            local_path = os.path.join(td, fname)
                            torch.save(self.train_module.state_dict(), local_path)
                            key = f"{self.module_type}/checkpoints/{fname}"
                            self._s3_put(local_path, key)
                    else:
                        os.makedirs(f'dalle2/checkpoints/{self.module_type}', exist_ok=True)
                        save_path = f'dalle2/checkpoints/{self.module_type}/{fname}'
                        torch.save(self.train_module.state_dict(), save_path)
                # Save intermediate outputs
                self._log_loss(epoch + 1, batch_i + 1, loss, plot_interval=save_intermediate_output)

                if (batch_i + 1) % save_intermediate_output == 0 and self.module_type == 'decoder':
                    self._run_intermediate_decoder_preview(epoch + 1, batch_i + 1, loss, steps=200, n_img=3)
                
                if (batch_i + 1) % save_intermediate_output == 0 and self.module_type == 'prior':
                    self.save_intermediate_prior_cosine(epoch + 1, batch_i + 1, loss, n_embs=1)

                if (batch_i + 1) % save_intermediate_model == 0:
                    self._save_current_model(epoch + 1, batch_i + 1)

            print(f'Average Epoch Loss: {epoch_loss / len(self.dataloader):.4f}')
            self._save_current_model(epoch + 1, batch_i + 1)

        self._save_final_model()


    def _download_checkpoint_from_s3(self, checkpoint_name: str) -> str:
        """
        Downloads a checkpoint from S3 to a temp local path and returns the local path.
        """
        if not hasattr(self, "s3"):
            self.s3 = boto3.client("s3")

        key = f"{self.module_type}/checkpoints/{checkpoint_name}"
        local_dir = tempfile.mkdtemp()
        local_path = os.path.join(local_dir, checkpoint_name)

        try:
            self.s3.download_file(self.models_bucket, key, local_path)
            return local_path
        except Exception as e:
            raise FileNotFoundError(f"[ERROR] Failed to download checkpoint {key} from S3: {e}")


    def _run_intermediate_decoder_preview(
            self,
            epoch: int,
            batch: int,
            loss: float,
            steps: int,
            n_img: int = 3
    ) -> None:
        """
        Generate and save an image from the current decoder checkpoint using DDIM.

        Args:
            epoch: used for naming the file
            batch: the current batch
            loss: the current batch loss
            steps: number of DDIM steps to use
            n_img: number of sample images to generate
        """
        plt.close()
        _, ax = plt.subplots(2, n_img, sharex=True, sharey=True)

        for i in range(n_img):
            img_true, z_img = self.dataloader.dataset.get_random_clean_image_and_embedding()
            z_img = z_img.to(self.device)
            
            # print(f"Target image - mean: {img_true.mean().item():.4f}, std: {img_true.std().item():.4f}")
            # print(f"Target image - min: {img_true.min().item():.4f}, max: {img_true.max().item():.4f}")

            # sampler = DecoderDDIMSampler(self.noise_scheduler, num_inference_steps=steps, guidance_scale=1.0)

            img_gen = debug_direct_sampling(
                denoiser=self.train_module,           # << the SAME module you train
                sched=self.noise_scheduler,
                z_img=z_img,
                steps=200,
                H=img_true.shape[-2],
                W=img_true.shape[-1],
                device=self.device
            )

            # Clamp and convert to [0, 1]
            img_gen = img_gen.clamp(-1, 1)
            img_gen = (img_gen + 1) / 2
            img_gen = img_gen.squeeze(0).permute(1, 2, 0).cpu().numpy()

            # Process ground truth image
            img_true_np = img_true.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
            img_true_np = (img_true_np + 1) / 2  # optional: only if your images are [-1,1]

            ax[0, i].imshow(img_true_np)
            ax[0, i].axis('off')
            ax[0, i].set_title('target')

            ax[1, i].imshow(img_gen)
            ax[1, i].axis('off')
            ax[1, i].set_title('generated')

        # Create directory if needed
        output_dir = 'dalle2/checkpoints/decoder_intermediate_outputs'
        os.makedirs(output_dir, exist_ok=True)

        # Save figure
        plt.suptitle(f'Epoch: {epoch}, Batch: {batch}, Loss: {loss:.6f}')
        save_path = os.path.join(output_dir, f'epoch_{epoch}_batch_{batch}_output.png')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # NEW: upload to S3
        if self.on_aws:
            key = f"decoder/intermediate_outputs/epoch_{epoch}_batch_{batch}_output.png"
            self._s3_put(save_path, key)

        self.train_module.train()

    
    def save_intermediate_prior_cosine(
            self,
            epoch: int,
            batch: int,
            loss: float,
            n_embs: int = 3
    ) -> None:
        """
        Computes cosine similarity between predicted and true CLIP image embeddings
        during Prior training and writes them to a CSV file.

        Args:
            epoch: current training epoch
            batch: current training batch index
            loss: current batch loss
            n_embs: number of random embeddings to log (default = 3)
        """
        csv_path = 'dalle2/checkpoints/prior_intermediate_outputs/cosine_similarity_log.csv'
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        records = []
        for _ in range(n_embs):
            z_txt, z_img_true = self.dataloader.dataset.get_random_text_and_embedding()

            z_txt = z_txt.to(self.device).unsqueeze(0) # (1, 512)
            z_img_true = z_img_true.to(self.device).unsqueeze(0) # (1, 512)


            T = self.noise_scheduler.alpha_bar_t.shape[0]
            t_int = torch.randint(low=0, high=T, size=(1,))
            t = t_int.to(self.device).long()

            # Add noise to z_img_true
            noise = torch.randn_like(z_img_true)
            alpha_bar = self.noise_scheduler.get_alpha_bar(t)
            z_img_noisy = (alpha_bar.sqrt() * z_img_true) + ((1 - alpha_bar).sqrt() * noise)

            # Predict noise using current prior
            self.train_module.eval()
            with torch.no_grad():
                eps_hat = self.train_module(z_txt=z_txt, t=t, z_T=z_img_noisy)

            self.train_module.train()

            # Cosine similarity between predicted and true noise
            cos_sim = F.cosine_similarity(eps_hat, noise, dim=-1).item()
            records.append([epoch, batch, t_int.item(), cos_sim, loss.item()])

        # Write data to csv
        write_header = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['epoch', 'batch', 'timestep', 'cosine_similarity', 'mse_loss'])
            writer.writerows(records)

        # NEW: upload to S3
        if self.on_aws:
            key = 'prior/intermediate_outputs/cosine_similarity_log.csv'
            self._s3_put(csv_path, key)

    
    def _log_loss(
            self,
            epoch: int,
            batch: int,
            mse_loss: float,
            plot_interval: int = 500
    ) -> None:
        """
        Logs the MSE loss to CSV each batch and periodically writes a loss plot.
        When self.on_aws is True, mirrors artifacts to s3://dalle2-outputs/.
        """
        # Ensure output directories exist
        log_dir = 'dalle2/checkpoints/logs'
        plot_dir = 'dalle2/checkpoints/plots'
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)

        csv_path = os.path.join(log_dir, f'{self.module_type}_batch_losses.csv')
        plot_path = os.path.join(plot_dir, f'{self.module_type}_loss_curve.png')

        # Append the current loss to the CSV
        write_header = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['epoch', 'batch', 'mse_loss'])
            writer.writerow([epoch, batch, float(mse_loss.item() if hasattr(mse_loss, "item") else mse_loss)])

        # Mirror CSV to S3 every batch
        if getattr(self, "on_aws", False):
            try:
                # Prefer helper if present
                if hasattr(self, "_s3_put"):
                    self._s3_put(csv_path, f"{self.module_type}/logs/{self.module_type}_batch_losses.csv")
                else:
                    # Fallback direct upload
                    if not hasattr(self, "s3"):
                        self.s3 = boto3.client("s3")
                        self.s3_bucket = getattr(self, "s3_bucket", "dalle2-outputs")
                    self.s3.upload_file(csv_path, self.s3_bucket,
                                        f"{self.module_type}/logs/{self.module_type}_batch_losses.csv")
            except Exception as e:
                print(f"[WARN] S3 upload (CSV) failed: {e}")

        # Periodically regenerate loss plots
        if (batch + 1) % plot_interval == 0:
            # Load the csv to build the curve
            epochs, batches, losses = [], [], []
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    epochs.append(int(row['epoch']))
                    batches.append(int(row['batch']))
                    losses.append(float(row['mse_loss']))

            plt.figure(figsize=(14, 6))

            # Linear scale
            plt.subplot(1, 2, 1)
            plt.plot(range(len(losses)), losses, label='Batch MSE Loss', linewidth=1.5)
            plt.xlabel('Batch Index')
            plt.ylabel('MSE Loss')
            plt.title('Training Loss (Linear Scale)')
            plt.grid(True)
            plt.legend()

            # Log scale
            plt.subplot(1, 2, 2)
            plt.plot(range(len(losses)), losses, label='Batch MSE Loss (Log)', linewidth=1.5)
            plt.xlabel('Batch Index')
            plt.ylabel('MSE Loss (log)')
            plt.title('Training Loss (Log Scale)')
            plt.yscale('log')
            plt.grid(True, which='both')
            plt.legend()

            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()

            # Mirror plot to S3
            if getattr(self, "on_aws", False):
                try:
                    if hasattr(self, "_s3_put"):
                        self._s3_put(plot_path, f"{self.module_type}/plots/{self.module_type}_loss_curve.png")
                    else:
                        if not hasattr(self, "s3"):
                            self.s3 = boto3.client("s3")
                            self.s3_bucket = getattr(self, "s3_bucket", "dalle2-outputs")
                        self.s3.upload_file(plot_path, self.s3_bucket,
                                            f"{self.module_type}/plots/{self.module_type}_loss_curve.png")
                except Exception as e:
                    print(f"[WARN] S3 upload (plot) failed: {e}")

        
class PriorTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
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
            batch_input: the input of the batch, (z_T (B, 512), z_txt (B, 512), t (B, 512))
        
        Returns:
            the predicted target
        """
        # Extract the text embedding, the timestep embedding, and the fully-noised imaage (DDPM-defined)
        if self.debug:
            print(f'Batch structure: {type(batch_input)}, len: {len(batch_input)}')
            print(f'Batch content: {[type(b) for b in batch_input]}')

        z_txt, t, z_T = batch_input

        # Forward pass
        return self.train_module.forward(
            z_txt=z_txt,
            t=t,
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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    #    ABLATE_ZERO=False; ABLATE_SHUFFLE=False
    @property
    def module_type(self) -> str:
        return 'decoder'

    ABLATE_ZERO=False; ABLATE_SHUFFLE=False
    @staticmethod
    def _abl(z,t,zero,shuf):
        if zero:  return torch.zeros_like(z), torch.zeros_like(t)
        if shuf:  p=torch.randperm(z.size(0),device=z.device); return z[p],t[p]
        return z,t


    def _run_batch(
            self,
            batch_input: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    ) -> Tuple[torch.Tensor, dict]:
        """
        Defines how a batch is passed through the decoder.

        Args:
            batch_input: the input of the batch, (z_t (B, 512), z_img (B, 512), t (B, 512))
        
        Returns:
            the predicted target
        """
        # Extract the partially-noisy images, the image embeddings, and the timestep embeddings from the batch input
        x_t, z_img, t = batch_input

        # Forward pass
        return self.train_module.forward(
            x_t=x_t,
            z_img=z_img,
            t=t
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
