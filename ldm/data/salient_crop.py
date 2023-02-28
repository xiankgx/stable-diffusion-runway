import copy
import math

import albumentations as albu
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from torch.nn import Dropout, LayerNorm, Linear, Softmax
from torchvision import transforms
from torchvision.models import resnet

cfg1 = {
    "hidden_size": 768,
    "mlp_dim": 768 * 4,
    "num_heads": 12,
    "num_layers": 2,
    "attention_dropout_rate": 0,
    "dropout_rate": 0.0,
}

cfg2 = {
    "hidden_size": 768,
    "mlp_dim": 768 * 4,
    "num_heads": 12,
    "num_layers": 2,
    "attention_dropout_rate": 0,
    "dropout_rate": 0.0,
}

cfg3 = {
    "hidden_size": 512,
    "mlp_dim": 512 * 4,
    "num_heads": 8,
    "num_layers": 2,
    "attention_dropout_rate": 0,
    "dropout_rate": 0.0,
}

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu}


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_attention_heads = config["num_heads"]  # 12
        self.attention_head_size = int(
            config["hidden_size"] / self.num_attention_heads
        )  # 42
        self.all_head_size = (
            self.num_attention_heads * self.attention_head_size
        )  # 12*42=504

        self.query = Linear(config["hidden_size"], self.all_head_size)  # (512, 504)
        self.key = Linear(config["hidden_size"], self.all_head_size)
        self.value = Linear(config["hidden_size"], self.all_head_size)

        # self.out = Linear(config['hidden_size'], config['hidden_size'])
        self.out = Linear(self.all_head_size, config["hidden_size"])
        self.attn_dropout = Dropout(config["attention_dropout_rate"])
        self.proj_dropout = Dropout(config["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config["hidden_size"], config["mlp_dim"])
        self.fc2 = Linear(config["mlp_dim"], config["hidden_size"])
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config["dropout_rate"])
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.flag = config["num_heads"]
        self.hidden_size = config["hidden_size"]
        self.ffn_norm = LayerNorm(config["hidden_size"], eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)
        self.attention_norm = LayerNorm(config["hidden_size"], eps=1e-6)

    def forward(self, x):
        h = x

        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config["hidden_size"], eps=1e-6)
        for _ in range(config["num_layers"]):
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states)
        encoded = self.encoder_norm(hidden_states)

        return encoded


class TranSalNet(nn.Module):
    def __init__(self):
        super(TranSalNet, self).__init__()
        self.encoder = _Encoder()
        self.decoder = _Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class _Encoder(nn.Module):
    def __init__(self):
        super(_Encoder, self).__init__()
        base_model = resnet.resnet50(pretrained=False)
        base_layers = list(base_model.children())[:8]
        self.encoder = nn.ModuleList(base_layers).eval()

    def forward(self, x):
        outputs = []
        for ii, layer in enumerate(self.encoder):
            x = layer(x)
            if ii in {5, 6, 7}:
                outputs.append(x)
        return outputs


class _Decoder(nn.Module):
    def __init__(self):
        super(_Decoder, self).__init__()
        self.conv1 = nn.Conv2d(
            768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv2 = nn.Conv2d(
            768, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv3 = nn.Conv2d(
            512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv4 = nn.Conv2d(
            256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv5 = nn.Conv2d(
            128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv6 = nn.Conv2d(
            64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv7 = nn.Conv2d(32, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.batchnorm1 = nn.BatchNorm2d(
            768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.batchnorm2 = nn.BatchNorm2d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.batchnorm3 = nn.BatchNorm2d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.batchnorm4 = nn.BatchNorm2d(
            128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.batchnorm5 = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.batchnorm6 = nn.BatchNorm2d(
            32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )

        self.TransEncoder1 = TransEncoder(
            in_channels=2048, spatial_size=9 * 12, cfg=cfg1
        )
        self.TransEncoder2 = TransEncoder(
            in_channels=1024, spatial_size=18 * 24, cfg=cfg2
        )
        self.TransEncoder3 = TransEncoder(
            in_channels=512, spatial_size=36 * 48, cfg=cfg3
        )

        self.add = torch.add
        self.relu = nn.ReLU(True)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x3, x4, x5 = x

        x5 = self.TransEncoder1(x5)
        x5 = self.conv1(x5)
        x5 = self.batchnorm1(x5)
        x5 = self.relu(x5)
        x5 = self.upsample(x5)

        x4_a = self.TransEncoder2(x4)
        x4 = x5 * x4_a
        x4 = self.relu(x4)
        x4 = self.conv2(x4)
        x4 = self.batchnorm2(x4)
        x4 = self.relu(x4)
        x4 = self.upsample(x4)

        x3_a = self.TransEncoder3(x3)
        x3 = x4 * x3_a
        x3 = self.relu(x3)
        x3 = self.conv3(x3)
        x3 = self.batchnorm3(x3)
        x3 = self.relu(x3)
        x3 = self.upsample(x3)

        x2 = self.conv4(x3)
        x2 = self.batchnorm4(x2)
        x2 = self.relu(x2)
        x2 = self.upsample(x2)
        x2 = self.conv5(x2)
        x2 = self.batchnorm5(x2)
        x2 = self.relu(x2)

        x1 = self.upsample(x2)
        x1 = self.conv6(x1)
        x1 = self.batchnorm6(x1)
        x1 = self.relu(x1)
        x1 = self.conv7(x1)
        x = self.sigmoid(x1)

        return x


class TransEncoder(nn.Module):
    def __init__(self, in_channels, spatial_size, cfg):
        super(TransEncoder, self).__init__()

        self.patch_embeddings = nn.Conv2d(
            in_channels=in_channels,
            out_channels=cfg["hidden_size"],
            kernel_size=1,
            stride=1,
        )
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, spatial_size, cfg["hidden_size"])
        )

        self.transformer_encoder = Encoder(cfg)

    def forward(self, x):
        a, b = x.shape[2], x.shape[3]
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)

        embeddings = x + self.position_embeddings
        x = self.transformer_encoder(embeddings)
        B, n_patch, hidden = x.shape
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, a, b)

        return x


def preprocess_img(img, channels: int = 3):
    # if channels == 1:
    #     img = cv2.imread(img_dir, 0)
    # elif channels == 3:
    #     img = cv2.imread(img_dir)

    # img = np.asarray(img)[..., ::-1]

    shape_r = 288
    shape_c = 384

    img_padded = np.zeros((shape_r, shape_c, channels), dtype=np.uint8)
    if channels == 1:
        img_padded = np.zeros((shape_r, shape_c), dtype=np.uint8)

    original_shape = img.shape
    rows_rate = original_shape[0] / shape_r
    cols_rate = original_shape[1] / shape_c

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = cv2.resize(img, (new_cols, shape_r))
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[
            :,
            ((img_padded.shape[1] - new_cols) // 2) : (
                (img_padded.shape[1] - new_cols) // 2 + new_cols
            ),
        ] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = cv2.resize(img, (shape_c, new_rows))

        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[
            ((img_padded.shape[0] - new_rows) // 2) : (
                (img_padded.shape[0] - new_rows) // 2 + new_rows
            ),
            :,
        ] = img

    return img_padded


def postprocess_img(pred, ori_size):
    pred = np.array(pred)
    # org = cv2.imread(org_dir, 0)

    # shape_r = org.shape[0]
    # shape_c = org.shape[1]
    shape_c, shape_r = ori_size

    predictions_shape = pred.shape

    rows_rate = shape_r / predictions_shape[0]
    cols_rate = shape_c / predictions_shape[1]

    if rows_rate > cols_rate:
        new_cols = (predictions_shape[1] * shape_r) // predictions_shape[0]
        pred = cv2.resize(pred, (new_cols, shape_r))
        img = pred[
            :,
            ((pred.shape[1] - shape_c) // 2) : (
                (pred.shape[1] - shape_c) // 2 + shape_c
            ),
        ]
    else:
        new_rows = (predictions_shape[0] * shape_c) // predictions_shape[1]
        pred = cv2.resize(pred, (shape_c, new_rows))
        img = pred[
            ((pred.shape[0] - shape_r) // 2) : (
                (pred.shape[0] - shape_r) // 2 + shape_r
            ),
            :,
        ]

    return img


class SaliencyRandomCrop:
    def __init__(
        self,
        height: int = 512,
        width: int = 512,
        sal_threshold: int = 127,
        p_sal_crop: float = 0.5,
        checkpoint_path="TranSalNet_Res.pth",  # https://drive.google.com/file/d/14czAAQQcRLGeiddPOM6AaTJTieu6QiHy/view?usp=sharing or s3://gx-stuffs/stable-diffusion/checkpoints/TranSalNet_Res.pth
        device="cpu",
    ) -> None:
        self.crop = albu.CropNonEmptyMaskIfExists(height=height, width=width)
        self.rand_crop = albu.RandomCrop(height=height, width=width)
        self.height = height
        self.width = width
        self.sal_threshold = sal_threshold
        self.p_sal_crop = p_sal_crop
        self.device = device

        # Instantiate saliency model
        model = TranSalNet()
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        model.eval()
        model.to(device)
        self.model = model

        self.test_transform = transforms.Compose(
            [
                transforms.Lambda(preprocess_img),
                transforms.ToTensor(),
            ]
        )

    @torch.no_grad()
    def __call__(self, image: Image.Image) -> Image.Image:
        assert not self.model.training

        image = np.array(image)[..., ::-1]  # BGR
        # print(f"image.shape: {image.shape}")
        # print(f"image.dtype: {image.dtype}")

        input_tensor = self.test_transform(image).unsqueeze(0).to(self.device)

        pred_saliency = self.model(input_tensor)

        sal_map = TF.to_pil_image(pred_saliency.squeeze())
        sal_map = postprocess_img(sal_map, image.shape[:2][::-1])
        # print(f"sal_map.shape: {sal_map.shape}")
        # print(f"sal_map.dtype: {sal_map.dtype}")

        assert sal_map.shape[:2] == image.shape[:2]
        # print(f"min: {sal_map.min()}, max: {sal_map.max()}")

        if np.random.rand() < self.p_sal_crop:
            mask = ((sal_map > self.sal_threshold).astype(np.float32) * 255).astype(
                np.uint8
            )
            cropped = self.crop(image=image, mask=mask)["image"]
        else:
            # print("rand crop")
            mask = np.zeros_like(sal_map)
            cropped = self.rand_crop(image=image)["image"]
        cropped = cropped[..., ::-1]
        # print(f"mask.dtype: {mask.dtype}")
        # print(f"mask.unique: {np.unique(mask)}")
        assert cropped.shape[:2] == (self.height, self.width)

        # return (
        #     Image.fromarray(cropped),
        #     Image.fromarray(image[..., ::-1]),
        #     Image.fromarray(sal_map),
        #     Image.fromarray(mask),
        # )
        return Image.fromarray(cropped)
