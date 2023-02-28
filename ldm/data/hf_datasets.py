import os
from functools import partial

import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision import transforms

from ldm.data.salient_crop import SaliencyRandomCrop


def preprocess_train(
    examples,
    train_transforms,
    image_column,
    text_column,
    image_output_key,
    text_output_key,
):
    images = [image.convert("RGB") for image in examples[image_column]]
    examples[image_output_key] = [train_transforms(image) for image in images]
    examples[text_output_key] = [text for text in examples[text_column]]
    return examples


def identity(x):
    return x


def normalize(img):
    return (np.asarray(img) / np.float32(255) - 0.5) * 2


def get_dataset(
    dataset_name: str = None,
    dataset_config_name=None,
    train_data_dir: str = None,
    cache_dir=None,
    resolution: int = 512,
    center_crop: bool = False,
    random_flip: bool = False,
    image_column: str = "image",
    text_column: str = "text",
    image_output_key: str = "image",
    text_output_key: str = "text",
) -> Dataset:
    if dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            dataset_name,
            dataset_config_name,
            cache_dir=cache_dir,
        )
    else:
        data_files = {}
        if train_data_dir is not None:
            data_files["train"] = os.path.join(train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=cache_dir,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    train_transforms = transforms.Compose(
        [
            transforms.Resize(
                resolution, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(resolution)
            if center_crop
            else SaliencyRandomCrop(height=resolution, width=resolution, device="cpu"),
            transforms.RandomHorizontalFlip()
            if random_flip
            else transforms.Lambda(identity),
            # transforms.ToTensor(),
            # transforms.Normalize([0.5], [0.5]),
            transforms.Lambda(normalize),
        ]
    )

    train_dataset = dataset["train"].with_transform(
        partial(
            preprocess_train,
            train_transforms=train_transforms,
            image_column=image_column,
            text_column=text_column,
            image_output_key=image_output_key,
            text_output_key=text_output_key,
        )
    )

    return train_dataset
