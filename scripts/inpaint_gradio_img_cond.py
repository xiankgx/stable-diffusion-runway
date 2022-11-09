import sys
sys.path.append(".")

from PIL import Image
from omegaconf import OmegaConf
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from einops import repeat
import torch
import numpy as np
import gradio as gr
import cv2
import torch.nn.functional as F


MAX_SIZE = 640


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def initialize_model(config, ckpt):
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)

    model.load_state_dict(torch.load(ckpt)["state_dict"], strict=False)

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    return sampler


def make_batch_sd(
        image,
        mask,
        # txt,
        cond_image,
        device,
        num_samples=1):
    image = np.array(image.convert("RGB").resize((256, 256)))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32)

    cond_image = np.array(cond_image.convert("RGB").resize((256, 256)))
    cond_image = cond_image[None].transpose(0, 3, 1, 2)
    cond_image = torch.from_numpy(cond_image).to(dtype=torch.float32)/127.5-1.0

    mask = np.array(mask.convert("L").resize((256, 256)))
    mask = mask.astype(np.float32)/255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    # masked_image = image * (mask < 0.5)
    masked_image = image * mask

    image = image/127.5-1.0
    masked_image = masked_image/127.5 - 1.0

    batch = {
        "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
        # "txt": num_samples * [txt],
        "cond_image": repeat(cond_image.to(device=device), "1 ... -> n ...", n=num_samples),
        "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
        "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
    }
    return batch


def inpaint(sampler, image, mask, cond_image, seed, scale, ddim_steps, num_samples=1, w=256, h=256):
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = sampler.model

    prng = np.random.RandomState(seed)
    start_code = prng.randn(num_samples, 4, h//8, w//8)
    start_code = torch.from_numpy(start_code).to(device=device, dtype=torch.float32)
    # start_code = torch.randn(num_samples, 4, h//8, w//8, device=device)
    print(f"start_code.shape: {start_code.shape}")

    with torch.no_grad():
        with torch.autocast("cuda"):
            batch = make_batch_sd(
                image, mask, cond_image=cond_image, device=device, num_samples=num_samples)

            c = model.cond_stage_model(batch["cond_image"])

            x0 = model.get_first_stage_encoding(model.encode_first_stage(batch["image"]))
            print(f"x0.shape: {x0.shape}")

            c_cat = list()
            for ck in model.concat_keys:
                print(f"ck: {ck}")
                cc = batch[ck].float()
                if ck != model.masked_image_key:
                    bchw = [num_samples, 4, h//8, w//8]
                    cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
                else:
                    cc = model.get_first_stage_encoding(
                        model.encode_first_stage(cc))
                c_cat.append(cc)
            c_cat = torch.cat(c_cat, dim=1)

            # cond
            cond = {"c_concat": [c_cat], "c_crossattn": [c]}

            # uncond cond
            # uc_cross = model.get_unconditional_conditioning(num_samples, "")
            uc_cross = torch.zeros_like(c)

            uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

            shape = [model.channels, h//8, w//8]
            print(f"shape: {shape}")

            print(f"batch_mask.shape: {batch['mask'].shape}")

            samples_cfg, intermediates = sampler.sample(
                ddim_steps,
                num_samples,
                shape,
                cond,
                verbose=False,
                eta=1.0,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=uc_full,
                x_T=start_code,

                # XXX ADDED THIS
                x0=x0,
                mask=F.interpolate(batch["mask"], size=shape[-2:], mode="nearest")
            )

            x_samples_ddim = model.decode_first_stage(samples_cfg)

            result = torch.clamp((x_samples_ddim+1.0)/2.0,
                                 min=0.0, max=1.0)

            result = result.cpu().numpy().transpose(0, 2, 3, 1)
            result = result*255

    result = [img.astype(np.uint8) for img in result]
    return result[0]


def run():
    sampler = initialize_model(sys.argv[1], sys.argv[2])

    def predict(input_img_data, cond_img, cond_scale, full_zero_mask):
        input_img = cv2.resize(input_img_data["image"], (256, 256))
        input_mask = cv2.resize(input_img_data["mask"][..., 0], (256, 256))
        print(f"input_img.shape : {input_img.shape}")
        print(f"input_mask.shape: {input_mask.shape}")

        input_mask = 255 - input_mask
        if full_zero_mask:
            input_mask = np.zeros_like(input_mask)

        # print(f"mask unique values: {np.unique(input_mask)}")

        h, w = input_img.shape[:2]

        result = inpaint(
            sampler=sampler,
            image=Image.fromarray(input_img),
            mask=Image.fromarray(input_mask),
            cond_image=Image.fromarray(cond_img),
            # prompt=prompt,
            seed=np.random.randint(100),
            scale=cond_scale,
            ddim_steps=50,
            num_samples=1,
            h=h, w=w
        )

        alpha = (input_mask/np.float32(255))[..., None]
        blended = (alpha * input_img.astype(np.float32) + (1 - alpha)
                   * result.astype(np.float32)).astype(np.uint8)
        # blended = (input_img.astype(np.float32) * alpha).astype(np.uint8)

        # return result, blended, input_img, input_mask
        return np.concatenate([
            input_img,
            result,
            # input_mask
        ], axis=1)

    gr.close_all()
    demo = gr.Interface(
        predict,
        inputs=[
            gr.Image(tool="sketch"),
            gr.Image(),
            gr.Number(value=2, precision=1, label="cond_scale"),
            gr.Checkbox(False, label="full zero mask")
        ],
        outputs=[
            gr.Image(),
            # gr.Image(),
            # gr.Image(),
            # gr.Image()
        ]
    )
    demo.launch(server_name="0.0.0.0", server_port=5000)


if __name__ == "__main__":
    run()
