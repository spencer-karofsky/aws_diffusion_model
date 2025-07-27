"""
run_inference.py: Runs the text-to-image generation pipeline on DALL·E 2

Instructions (run from CLI):
    cd [path to directory]
    python -m dalle2.inference.run_inference

Author:
    * Spencer Karofsky (https://github.com/spencer-karofsky)
"""

from dalle2.inference.inference import DALLe2Text2Image
import matplotlib.pyplot as plt

def visualize(img_tensor):
    gen = img_tensor.clamp(-1, 1)
    vis = (gen + 1) / 2
    img_np = vis.squeeze(0).permute(1, 2, 0).cpu().numpy()
    plt.imshow(img_np)
    plt.axis('off')
    plt.show()

def main():
    prompt = ''
    prompt = input('Enter a Text Prompt to Generate: ')

    pipeline = DALLe2Text2Image(
        prior_path='dalle2/checkpoints/prior/epoch150_batch1_single_img.pth',
        decoder_path='dalle2/checkpoints/decoder/epoch550_batch1_single_img.pth',
        prior_T=200,
        start_T=20,
        steps_prior=10,
        cfg_scale=3.0,
        decoder_t=200,
        steps_decoder=50
    )

    img, _ = pipeline.text_to_image(prompt)
    visualize(img)

if __name__ == '__main__':
    main()
