#!/bin/bash

# SD text-conditioned model
# python main.py --base configs/stable-diffusion/txt2img-finetune.yaml -t --gpus 0,

# LDM text-conditioned model
# python main.py --base configs/latent-diffusion/txt2img-finetune.yaml -t --gpus 0,

# LDM CLIP image embedding-conditioned model
# python main.py --base configs/latent-diffusion/clip_img_cond-finetune.yaml -t --gpus 0, -n "123rf_vectors-clip_img_cond"
# python main.py --base configs/stable-diffusion/clip_img_cond-finetune.yaml -t --gpus 0, -n "123rf_vectors-clip_img_cond"
# python main.py --base configs/stable-diffusion/clip_img_cond-finetune.yaml -t --gpus 0, -r logs/2022-11-04T00-54-43_123rf

# LDM inpainting CLIP image embedding-conditioned model
# python main.py --base configs/latent-diffusion/inpainting-clip_img_cond-finetune.yaml -t --gpus 0, -r logs/2022-11-07T09-51-06_inpainting-clip_img_cond-finetune
python main.py -t --gpus 0, -r logs/2022-11-07T09-51-06_inpainting-clip_img_cond-finetune
