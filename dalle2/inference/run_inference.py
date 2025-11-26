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
import time
import math

import matplotlib.pyplot as plt
import math
import textwrap
import numpy as np

def visualize(img_tensor, prompt, generation_time):
    gen = img_tensor.clamp(-1, 1)
    vis = (gen + 1) / 2
    img_np = vis.squeeze(0).permute(1, 2, 0).cpu().numpy()
    plt.imshow(img_np)
    plt.title(f'128x128 Image Generated in {generation_time:.3f}s:\n"{prompt}"')
    plt.axis('off')
    plt.show()

def visualize_grid(
    img_tuples_list,
    prompts,
    generation_time,
    n_cols: int = 2,
    wrap_width: int = 40,
):
    flat_entries = []
    for prompt, img_tuples in zip(prompts, img_tuples_list):
        for cfg, img_tensor in img_tuples:
            flat_entries.append((prompt, cfg, img_tensor))

    n_images = len(flat_entries)
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes = np.atleast_2d(axes).reshape(n_rows, n_cols)

    for idx, (prompt, cfg, img_tensor) in enumerate(flat_entries):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        img = img_tensor.clamp(-1, 1)
        img = (img + 1) / 2
        img_np = img.squeeze(0).permute(1, 2, 0).cpu().numpy()

        wrapped_prompt = textwrap.fill(prompt, width=wrap_width)
        ax.imshow(img_np)
        ax.set_title(f'"{wrapped_prompt}"', fontsize=9)
        ax.axis('off')

    # Hide unused subplots
    for i in range(n_images, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row, col].axis('off')

    fig.suptitle(f'Generated {n_images} images in {generation_time:.2f}s', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def main():
    prompts = ['a city sunset' for _ in range(4)]
    # prompts = [
    #     'city scape during nighttime',
    #     'a street light in the middle of a snowy park'
    # ]

    pipeline = DALLe2Text2Image(
        prior_path='dalle2/checkpoints/prior/boston_prior_ema.pth',
        decoder_path='dalle2/checkpoints/decoder/boston_decoder_final.pth',
        upsampler_path='dalle2/checkpoints/upsampler/boston_upsampler_final.pth',
        prior_T=1000,
        start_T=999,
        steps_prior=400,
        prior_cfg_scale=2.0,
        decoder_T=1000,
        steps_decoder=400,
        decoder_cfg_scale=4.2,
        upsampler_T=400,
        steps_upsampler=220
    )

    t0 = time.time()
    img_tuples_list = []

    for prompt in prompts:
        img, _ = pipeline.text_to_image(prompt)
        img_tuples_list.append([(pipeline.prior_cfg, img)])

    gen_time = time.time() - t0
    visualize_grid(img_tuples_list, prompts, gen_time)

if __name__ == '__main__':
    main()
