"""
decoder_inference.py: To test the decoder, run the decoder only using the true CLIP image embedding.

NOTE: This file is for debugging only. The file "dalle2.py" runs the actual inference pipeline.
"""
import torch
import torch.nn.functional as F
from torchvision import transforms

import os
from PIL import Image
import matplotlib.pyplot as plt

from dalle2.models.clip_encoding import CLIPEncoder
from dalle2.models.decoder import Decoder
from dalle2.sampling.ddim_sampling import DecoderDDIMSampler

from dalle2.sampling.noise_scheduler import NoiseScheduler

class DecoderOnlyInference:
    def __init__(
            self,
            decoder_path: str,
            decoder_sampler: DecoderDDIMSampler,
            H: int,
            W: int,
            T: int,
            num_inference_steps: int,
            debug: bool
    ):
        """
        Initialize decoder-only inference using the true CLIP image embedding in replacement of the prior.

        Args:
            decoder_path: the path to the trained decoder model
            decoder_sampler: the DDIM-powered sampler that given a CLIP image embedding, generates the actual clean image
            H: image height (pixels)
            W: image width (pixels)
            T: total number of timesteps sampled from during training
            num_inference_steps: number of uniformly-distributed inference steps
            debug: prints debugging information
        """
        # Ideally use a GPU-accelerated framework for faster inference
        self.device = (
                'cuda' if torch.cuda.is_available()
                else 'mps' if torch.backends.mps.is_available()
                else 'cpu'
        )

        # Load the decoder
        try:
            if not os.path.exists(decoder_path):
                raise FileNotFoundError(f'Decoder checkpoint not found at: {decoder_path}')

            self.decoder = Decoder(device=self.device, debug=debug)
            self.decoder.load_state_dict(torch.load(decoder_path, map_location=self.device), strict=False)
            self.decoder.to(self.device)
            self.decoder.eval()
            if debug:
                print('Successfully loaded the decoder and switched to mode "eval".')

        except FileNotFoundError as e:
            print('[ERROR] failed to load either the trained prior or decoder weights.')
            raise e

        except RuntimeError as e:
            raise RuntimeError(f'Model loading failed — checkpoint might not match model architecture:\n{e}')

        # Save params
        self.clip_encoder = CLIPEncoder()
        self.decoder_sampler = decoder_sampler
        self.H = H
        self.W = W
        self.T = T
        self.debug = debug
        self.num_inference_steps = num_inference_steps
    
    def generate_img(
            self,
            caption: str,
            image: Image.Image
    ) -> torch.Tensor:
        """
        Generate the true image

        Args:
            caption: the caption of the image
            image: the image we seek to reconstruct, of shape (1, 3, H, W)
        
        Returns:
            generated_img: the generated image, of shape (3, H, W)
        """
        # add once, top-level
        self.img_transform = transforms.Compose([
            transforms.Resize((self.H, self.W), antialias=True),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1)
        ])

        image_tensor = self.img_transform(image).unsqueeze(0).to(self.device)

        z_img_true = self.clip_encoder.encode_image(image_tensor).to(self.device)
        z_img_true = z_img_true / z_img_true.norm(dim=-1, keepdim=True)

        # Encode the caption (not needed for inference, but useful to compare cos similarity with z_img_true)
        z_txt_true = self.clip_encoder.encode_text([caption])
        assert z_txt_true.size(0) == 1 and z_txt_true.size(1) == 512, f'[DecoderOnlyInference] Incorrect text embedding shape ({z_img_true.shape}), expected (1, 512)'

        print(z_img_true.norm(dim=-1).mean())
        print(z_txt_true.norm(dim=-1).mean())
        
        self.decoder.eval()

        generated_images = self.decoder.sample(
            z_img=z_img_true,
            sampler=self.decoder_sampler
        )
        
        if self.debug:
            print(f'[DecoderOnlyInference.generate] generated_images shape: {generated_images.shape}')

        return generated_images
    
if __name__ == '__main__':
    INFERENCE_STEPS = 50

    # MUST match training
    TRAIN_T = 200
    SCHEDULE = 'cosine'

    noise_scheduler = NoiseScheduler(T=TRAIN_T, schedule_type=SCHEDULE)
    decoder_sampler = DecoderDDIMSampler(noise_scheduler, num_inference_steps=INFERENCE_STEPS)

    img_generator = DecoderOnlyInference(
        decoder_path='dalle2/checkpoints/decoder/epoch550_batch1.pth',
        decoder_sampler=decoder_sampler,
        H=128, W=128, T=1000,
        num_inference_steps=INFERENCE_STEPS,
        debug=False
    )

    img_generator.decoder.sampler = decoder_sampler

    image = Image.open('dalle2/data/local_datasets/coco/val2017/000000219578.jpg').convert('RGB')
    to_tensor = transforms.ToTensor()
    image_tensor = to_tensor(image)

    print(f'Input Image Min: {image_tensor.min().item()}')
    print(f'Input Image Max: {image_tensor.max().item()}')

    caption = 'a dog and cat lying together on an orange couch.'

    generated_image = img_generator.generate_img(
        caption=caption,
        image=image
    )

    with torch.no_grad():
        generated_images = img_generator.generate_img(caption, image)

    # Debug ranges
    print("eps_hat range during sampling?", "-> add in sampler loop if needed")

    # Map to [0,1] for display
    gen = generated_images.clamp(-1, 1)
    vis = (gen + 1) / 2
    plt.imshow(vis.squeeze(0).permute(1,2,0).cpu().numpy())

    gen_np = vis.squeeze(0).permute(1, 2, 0).cpu().numpy()
    print('Output min:', gen_np.min(), 'max:', gen_np.max())

    plt.imshow(gen_np)
    plt.axis('off')
    plt.show()