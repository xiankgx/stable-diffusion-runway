import os

import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision import transforms


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
            else transforms.RandomCrop(resolution),
            transforms.RandomHorizontalFlip()
            if random_flip
            else transforms.Lambda(lambda x: x),
            # transforms.ToTensor(),
            # transforms.Normalize([0.5], [0.5]),
            transforms.Lambda(
                lambda img: (np.asarray(img) / np.float32(255) - 0.5) * 2
            ),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples[image_output_key] = [train_transforms(image) for image in images]
        examples[text_output_key] = [text for text in examples[text_column]]
        return examples

    train_dataset = dataset["train"].with_transform(preprocess_train)

    return train_dataset
