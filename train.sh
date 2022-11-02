#!/bin/bash

# SD text-conditioned model
# python main.py --base configs/stable-diffusion/txt2img-finetune.yaml -t --gpus 0,

# LDM text-conditioned model
# python main.py --base configs/latent-diffusion/txt2img-finetune.yaml -t --gpus 0,

# LDM CLIP image embedding-conditioned model
# python main.py --base configs/latent-diffusion/clip_img_cond-finetune.yaml -t --gpus 0, -r logs/2022-10-28T04-10-19_clip_img_cond-finetune

# LDM inpainting CLIP image embedding-conditioned model
python main.py --base configs/latent-diffusion/inpainting-clip_img_cond-finetune.yaml -t --gpus 0,