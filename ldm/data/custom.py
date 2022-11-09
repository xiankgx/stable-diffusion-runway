import glob
import os

import numpy as np
import pandas as pd
import webdataset as wds
from ldm.data.base import Txt2ImgIterableBaseDataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchvision import transforms

import braceexpand
import cv2
import random

Image.MAX_IMAGE_PIXELS = None


class ImageCaptioningDataset(Dataset):

    def __init__(self, annotations_csv_file=None,
                 image_col="image",
                 text_col="caption",
                 image_root=None,
                 size=256,
                 hflip=False,
                 random_crop=True,
                 random_crop_scale=(0.8, 1.0),
                 repeats=1,
                 inpaint_task=False):
        super().__init__()

        self.inpaint_task = inpaint_task

        if annotations_csv_file is not None:
            # Assuming image paths and captions providied in annotations csv file

            df = pd.read_csv(annotations_csv_file)

            if image_root is not None:
                df[image_col] = df[image_col].map(
                    lambda p: os.path.join(image_root, p)
                )

            images = df[image_col].tolist()
            captions = df[text_col].tolist()
        else:
            # No annotations csv file provided, assuming image filenames as captions

            assert image_root is not None, "Must provide at least one of 'annotations_csv_file' and 'image_root'."
            images = glob.glob(image_root + "/**/*.jpg", recursive=True) + \
                glob.glob(image_root + "/**/*.png", recursive=True)
            captions = list(map(lambda p: os.path.splitext(os.path.basename(p))[0],
                                images))

        # Repeat lists for some number of times
        if repeats > 1:
            images = images * repeats
            captions = captions * repeats

        print(f"num files: {len(images)}")
        # print(f"top 5 files: {images[:5]}")
        self.images = images
        self.captions = captions

        # Image transforms
        size = (size, size) if not isinstance(size, (list, tuple)) else size
        flip = [transforms.RandomHorizontalFlip(p=0.5), ] if hflip else []
        crop = [
            transforms.RandomResizedCrop(
                size=size,
                scale=random_crop_scale,
                ratio=(1, 1),
                interpolation=transforms.InterpolationMode.BICUBIC
            )
        ] if random_crop else [
            transforms.Resize(
                size, interpolation=transforms.InterpolationMode.BICUBIC),
        ]
        self.transform = transforms.Compose(
            flip + crop
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")

        caption = self.captions[idx]

        img = self.transform(img)

        # Output image in np.ndarray format
        img = np.asarray(img)

        data = {}

        if self.inpaint_task:
            # Full 0 mask by default to make model predict everything
            mask = np.zeros(list(img.shape[:2]) + [1, ], dtype=np.float32)

            # With some probability, generate some random mask
            if np.random.rand() < 0.8:
                if np.random.rand() < 0.35:
                    mask = generate_rectangular_mask(
                        (256, 256), min_height=0.25, max_height=0.9, min_width=0.25, max_width=0.9, x=0.5, y=0.5, center=True)
                elif np.random.rand() < 0.7:
                    mask = generate_circular_mask(
                        (256, 256), min_radius=0.125, max_radius=0.5, x=0.5, y=0.5)
                else:
                    mask = generate_stroke_mask((256, 256))

            masked_image = mask * img

            # data["mask"] = (mask - 0.5) * 2
            data["mask"] = mask
            data["masked_image"] = (masked_image/np.float32(255) - 0.5) * 2

        # Normalize: [0, 255] -> [-1, 1]
        data["image"] = (img/np.float32(255) - 0.5) * 2
        data["caption"] = caption

        return data


class WebdatasetImageCaptionDataset(IterableDataset):

    def __init__(
        self,

        urls,
        shuffle=1000,

        num_records=None,

        size=256,

        # Image augmentations
        hflip=True,
        random_crop=True,
        random_crop_scale=(0.5, 1.0)
    ):
        urls = list(map(lambda p: braceexpand.braceexpand(p), urls))
        urls = [el for l in urls for el in l]

        self.urls = urls
        self.shuffle = shuffle

        self.num_records = num_records
        # self.valid_ids = valid_ids
        # self.sample_ids = valid_ids
        # print(f'{self.__class__.__name__} dataset contains {self.__len__()} examples.')

        ds = wds.WebDataset(
            self.urls,
            # nodesplitter=wds.split_by_node,
            shardshuffle=True,
            handler=wds.handlers.warn_and_continue,
            verbose=True
        )
        if self.shuffle:
            ds = ds.shuffle(self.shuffle)
        ds = ds.decode("pil")
        ds = ds.map(self.transform)
        self.ds = ds

        self.size = size

        self.hflip = hflip
        self.random_crop = random_crop
        self.random_crop_scale = random_crop_scale

    def __len__(self):
        if self.num_records is not None:
            return self.num_records
        return len(self.urls) * 10_000

    def transform(self, d):
        # Get data
        img = d["jpg"]
        caption = d["json"]["caption"]
        key = d["__key__"]

        size = self.size
        size = (size, size) if not isinstance(size, (list, tuple)) else size

        flip = [transforms.RandomHorizontalFlip(p=0.5), ] if self.hflip else []
        crop = [
            transforms.RandomResizedCrop(
                size=size,
                scale=self.random_crop_scale,
                interpolation=transforms.InterpolationMode.BICUBIC
            )
        ] if self.random_crop else [transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC), ]

        transform = transforms.Compose(
            flip + crop
        )

        # Apply transforms
        img = transform(img)
        # PIL to np array
        img = np.array(img)
        # Normalize [0, 255] -> [-1, 1]
        img = ((img / np.float32(255)) - 0.5) * 2

        return {
            "key": key,
            "image": img,
            "caption": caption
        }

    def __iter__(self):
        return iter(self.ds)


def generate_rectangular_mask(im_size, min_height=0.25, max_height=0.5, min_width=0.25, max_width=0.5, channels=1, invert_mask=True, y=None, x=None, center=False):
    """Getnerate rectangular mask."""
    mask = np.zeros((im_size[0], im_size[1]), dtype=np.float32)

    if isinstance(min_height, float):
        min_height = int(min_height * im_size[0])
    if isinstance(max_height, float):
        max_height = int(max_height * im_size[0])
    if isinstance(min_width, float):
        min_width = int(min_width * im_size[1])
    if isinstance(max_width, float):
        max_width = int(max_width * im_size[1])

    h = np.random.randint(min_height, max_height)
    w = np.random.randint(min_width, max_width)
    if y is None:
        y = np.random.randint(0, im_size[0] - h)
    else:
        y = int(y * im_size[0])
    if x is None:
        x = np.random.randint(0, im_size[1] - w)
    else:
        x = int(x * im_size[1])
    # print(f"x: {x}, y: {y}")
    if center:
        mask[y - h//2:y + (h - h//2), x - w//2: x + (w - w//2)] = 1.0
    else:
        mask[y:y+h, x:x+w] = 1.0
    mask = np.stack([mask, ] * channels, axis=-1)
    if invert_mask:
        return 1 - mask
    return mask


def generate_circular_mask(im_size, min_radius=0.125, max_radius=0.25, channels=1, invert_mask=True, y=None, x=None):
    """Generate circular mask."""
    mask = np.zeros((im_size[0], im_size[1]), dtype=np.float32)

    # Get the length of the shorter side
    min_size = np.min(mask.shape[:2])

    # Compute radius in terms of ratio to the length of the shorter side
    if isinstance(min_radius, float):
        min_radius = int(min_radius * min_size)
    if isinstance(max_radius, float):
        max_radius = int(max_radius * min_size)

    # Generate circle center coordinate and radius
    if y is None:
        cy = np.random.randint(0, im_size[0])
    else:
        cy = int(y * im_size[0])
    if x is None:
        cx = np.random.randint(0, im_size[1])
    else:
        cx = int(x * im_size[1])
    r = np.random.randint(min_radius, max_radius)

    # Compute selection mask based on Euclidean distance to circle center
    y = np.linspace(0, im_size[0] - 1, im_size[0])
    x = np.linspace(0, im_size[1] - 1, im_size[1])
    xv, yv = np.meshgrid(x, y)
    dist = ((xv - cx) ** 2 + (yv - cy) ** 2) ** 0.5

    mask[(dist < r)] = 1.0
    mask = np.stack([mask, ] * channels, axis=-1)
    if invert_mask:
        return 1 - mask
    return mask


def generate_stroke_mask(im_size, max_parts=15, maxVertex=25, maxLength=100, maxBrushWidth=24 * 2, maxAngle=360, channels=1):
    mask = np.zeros((im_size[0], im_size[1], 1), dtype=np.float32)
    parts = random.randint(1, max_parts)
    for i in range(parts):
        mask = mask + np_free_form_mask(maxVertex, maxLength,
                                        maxBrushWidth, maxAngle, im_size[0], im_size[1])
    mask = np.minimum(mask, 1.0)
    mask = np.concatenate([mask, ] * channels, axis=2)
    return mask


def np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):
    mask = np.zeros((h, w, 1), np.float32)
    numVertex = np.random.randint(maxVertex + 1)
    startY = np.random.randint(h)
    startX = np.random.randint(w)
    brushWidth = 0
    for i in range(numVertex):
        angle = np.random.randint(maxAngle + 1)
        angle = angle / 360.0 * 2 * np.pi
        if i % 2 == 0:
            angle = 2 * np.pi - angle
        length = np.random.randint(maxLength + 1)
        brushWidth = np.random.randint(10, maxBrushWidth + 1) // 2 * 2
        nextY = startY + length * np.cos(angle)
        nextX = startX + length * np.sin(angle)
        nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(np.int32)
        nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(np.int32)
        cv2.line(mask, (startX, startY), (nextX, nextY), 1, brushWidth)
        cv2.circle(mask, (startX, startY), brushWidth // 2, 2)
        startY, startX = nextY, nextX
    cv2.circle(mask, (startX, startY), brushWidth // 2, 2)
    return mask
