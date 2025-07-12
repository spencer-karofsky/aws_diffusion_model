"""
dalle_trainer: trains DALL·E 2

Usage:
    From the CLI, run:
        $ cd aws_diffusion_model  # Root project directory
        $ python -m dalle_core.dalle_model.dalle_trainer

Classes:
    - DALLETrainer: trains and runs inference on the DALL·E 2 architecture

References:
    - DALL·E 2 Paper: https://cdn.openai.com/papers/dall-e-2.pdf
    - My DALL·E 2 Notes: https://github.com/spencer-karofsky/aws_diffusion_model/blob/main/dall-e-2/research_notes/DALL-E-2%202022.pdf or /dall-e-2/research_notes/DALL-E-2 2022.pdf

Author:
    - Spencer Karofsky (https://github.com/spencer-karofsky)
"""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from dalle_core.dalle_model.dalle_2 import DALLE2

class DALLETrainer:
    def __init__(
        self,
        dalle_model: DALLE2,
        clip_embedder,
        noise_scheduler,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        num_epochs: int = 10,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.dalle = dalle_model
        self.clip = clip_embedder
        self.scheduler = noise_scheduler
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.device = device

    def train(self, on_aws: bool = False) -> bool:
        if on_aws:
            raise NotImplementedError('AWS training pipeline not yet implemented.')

        self.dalle.train()
        self.clip.eval()  # CLIP is frozen
        total_steps = 0

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            pbar = tqdm(self.dataloader, desc=f'Epoch {epoch + 1}/{self.num_epochs}')

            for images, captions in pbar:
                images = images.to(self.device)
                B = images.size(0)

                with torch.no_grad():
                    # Get CLIP text embeddings: [B, 512]
                    text_embed = self.clip.encode_text(captions).to(self.device)

                    # Get CLIP image embeddings: [B, 512]
                    clean_img_embed = self.clip.encode_image(images).to(self.device)

                    # Sample time steps: [B]
                    timesteps = torch.randint(0, self.scheduler.num_timesteps, (B,), device=self.device).long()

                    # Add noise to image embeddings → x_t, ε
                    noised_embed, true_noise = self.scheduler.add_noise(clean_img_embed, timesteps)

                    # Get timestep embedding: [B, 512]
                    timestep_embed = self.scheduler.get_timestep_embedding(timesteps).to(self.device)

                # Predict noise using prior
                predicted_noise = self.dalle.prior(text_embed, timestep_embed, noised_embed)

                # Compute loss
                loss = F.mse_loss(predicted_noise, true_noise)

                # Backprop and update weights
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                total_steps += 1
                pbar.set_postfix(loss=loss.item())

            print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(self.dataloader):.4f}")

        return True

if __name__ == "__main__":
    dalle = DALLE2
    print('Begin training...')
    pass
