import torch
import matplotlib.pyplot as plt
from dalle2.inference.inference import DALLe2Text2Image

def main():
    # Initialize pipeline (prior loads but won't be used)
    pipeline = DALLe2Text2Image(
        prior_path='dalle2/checkpoints/prior/epoch116_batch313.pth',
        decoder_path='dalle2/checkpoints/decoder/epoch149_batch100.pth',
        steps_decoder=50,
        decoder_cfg_scale=1.0
    )

    # Load precomputed embeddings
    embeds = torch.load('dalle2/data/local_datasets/midjourney_v6/precomputed_embeddings_full.pt')

    # Extract only image embeddings
    clip_embeds = embeds['image_embeddings']
    print(f'Loaded {clip_embeds.shape[0]} image embeddings of dimension {clip_embeds.shape[1]}')

    # Pick random embedding
    rand_idx = torch.randint(0, clip_embeds.size(0), (1,))
    z_img_true = clip_embeds[rand_idx].to(pipeline.dev)

    # Run decoder-only inference
    img = pipeline.embedding_to_image(z_img_true)

    # Visualize
    vis = (img.clamp(-1, 1) + 1) / 2
    plt.imshow(vis.squeeze(0).permute(1, 2, 0).cpu().numpy())
    plt.title('Decoder-only from true CLIP embedding')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()
