from dalle2.inference.inference import DALLe2Text2Image
from dalle2.data.dataset_utils_2 import MidJourneyDecoderDataset
from dalle2.sampling.noise_scheduler import NoiseScheduler
import torch, torchvision, os

os.chdir('/Users/spencerkarofsky/Desktop/projects/aws_diffusion_model')

pipe = DALLe2Text2Image(
    prior_path         = 'dalle2/checkpoints/prior/epoch10_batch625_ema.pth',
    decoder_path       = 'dalle2/checkpoints/decoder/epoch18_batch1250.pth',
    prior_T            = 200,
    start_T            = 20,
    steps_prior        = 10,
    prior_cfg_scale    = 3.0,
    decoder_T          = 200,
    steps_decoder      = 50,
    decoder_cfg_scale  = 5.75
)
metadata_path = "s3://dalle2-data/train_img/metadata.csv"
images_dir = "s3://dalle2-data/train_img"
noise_scheduler = NoiseScheduler(T=200, schedule_type='cosine')
ds = MidJourneyDecoderDataset(
    metadata_path=metadata_path,
    images_dir=images_dir,
    resize_size=128,
    device='mps',
    noise_scheduler=noise_scheduler,
    n_repeat=1,
    # cache_dir="/home/ec2-user/dalle2_cache",  # if your dataset util supports it
)
img_true, z_img_true = ds.get_random_clean_image_and_embedding()
with torch.no_grad():
    img_pred = pipe.decoder.sample(z_img_true.to(pipe.dev))

torchvision.utils.save_image(img_pred, 'decoder_only.png')
print('saved → decoder_only.png')
