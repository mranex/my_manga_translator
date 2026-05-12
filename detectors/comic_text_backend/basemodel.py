from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn

from .yolov5_common import C3
from .yolov5_model import Model


TEXTDET_MASK = 0
TEXTDET_DET = 1
TEXTDET_INFERENCE = 2

YOLOV5_TEXT_BLOCK_CFG = {
    "nc": 2,
    "depth_multiple": 0.33,
    "width_multiple": 0.50,
    "anchors": [
        [10, 13, 16, 30, 33, 23],
        [30, 61, 62, 45, 59, 119],
        [116, 90, 156, 198, 373, 326],
    ],
    "backbone": [
        [-1, 1, "Conv", [64, 6, 2, 2]],
        [-1, 1, "Conv", [128, 3, 2]],
        [-1, 3, "C3", [128]],
        [-1, 1, "Conv", [256, 3, 2]],
        [-1, 6, "C3", [256]],
        [-1, 1, "Conv", [512, 3, 2]],
        [-1, 9, "C3", [512]],
        [-1, 1, "Conv", [1024, 3, 2]],
        [-1, 3, "C3", [1024]],
        [-1, 1, "SPPF", [1024, 5]],
    ],
    "head": [
        [-1, 1, "Conv", [512, 1, 1]],
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
        [[-1, 6], 1, "Concat", [1]],
        [-1, 3, "C3", [512, False]],
        [-1, 1, "Conv", [256, 1, 1]],
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
        [[-1, 4], 1, "Concat", [1]],
        [-1, 3, "C3", [256, False]],
        [-1, 1, "Conv", [256, 3, 2]],
        [[-1, 14], 1, "Concat", [1]],
        [-1, 3, "C3", [512, False]],
        [-1, 1, "Conv", [512, 3, 2]],
        [[-1, 10], 1, "Concat", [1]],
        [-1, 3, "C3", [1024, False]],
        [[17, 20, 23], 1, "Detect", [2, 3]],
    ],
}


def _load_safetensors_state_dict(path: str):
    from safetensors.torch import load_file

    return load_file(path, device="cpu")


def _normalize_state_dict_keys(
    module: nn.Module,
    raw_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    target_keys = set(module.state_dict().keys())
    candidates = ["", "module.", "weights.", "state_dict.", "model."]

    best_state_dict = raw_state_dict
    best_score = -1
    for prefix in candidates:
        if prefix:
            candidate_state_dict = {
                (
                    key[len(prefix) :]
                    if key.startswith(prefix)
                    else key
                ): value
                for key, value in raw_state_dict.items()
            }
        else:
            candidate_state_dict = dict(raw_state_dict)

        score = len(target_keys.intersection(candidate_state_dict.keys()))
        if score > best_score:
            best_score = score
            best_state_dict = candidate_state_dict

    return best_state_dict


def _load_state_dict_strict(
    module: nn.Module,
    raw_state_dict: dict[str, torch.Tensor],
    name: str,
) -> None:
    state_dict = _normalize_state_dict_keys(module, raw_state_dict)
    incompatible = module.load_state_dict(state_dict, strict=False)

    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        print(f"[ComicTextDetector] Failed loading {name} weights.")
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:20]}")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:20]}")
        raise RuntimeError(f"Incompatible {name} weights")


class DoubleConvUpC3(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, act=True):
        super().__init__()
        self.conv = nn.Sequential(
            C3(in_ch + mid_ch, mid_ch, act=act),
            nn.ConvTranspose2d(mid_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DoubleConvC3(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, act=True):
        super().__init__()
        self.down = nn.AvgPool2d(2, stride=2) if stride > 1 else None
        self.conv = C3(in_ch, out_ch, act=act)

    def forward(self, x):
        if self.down is not None:
            x = self.down(x)
        return self.conv(x)


class UnetHead(nn.Module):
    def __init__(self, act=True) -> None:
        super().__init__()
        self.down_conv1 = DoubleConvC3(512, 512, 2, act=act)
        self.upconv0 = DoubleConvUpC3(0, 512, 256, act=act)
        self.upconv2 = DoubleConvUpC3(256, 512, 256, act=act)
        self.upconv3 = DoubleConvUpC3(0, 512, 256, act=act)
        self.upconv4 = DoubleConvUpC3(128, 256, 128, act=act)
        self.upconv5 = DoubleConvUpC3(64, 128, 64, act=act)
        self.upconv6 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, f160, f80, f40, f20, f3, forward_mode=TEXTDET_MASK):
        d10 = self.down_conv1(f3)
        u20 = self.upconv0(d10)
        u40 = self.upconv2(torch.cat([f20, u20], dim=1))
        if forward_mode == TEXTDET_DET:
            return f80, f40, u40
        u80 = self.upconv3(torch.cat([f40, u40], dim=1))
        u160 = self.upconv4(torch.cat([f80, u80], dim=1))
        u320 = self.upconv5(torch.cat([f160, u160], dim=1))
        mask = self.upconv6(u320)
        if forward_mode == TEXTDET_MASK:
            return mask
        return mask, [f80, f40, u40]


class DBHead(nn.Module):
    def __init__(self, in_channels, k=50, shrink_with_sigmoid=True, act=True):
        super().__init__()
        self.k = k
        self.shrink_with_sigmoid = shrink_with_sigmoid
        self.upconv3 = DoubleConvUpC3(0, 512, 256, act=act)
        self.upconv4 = DoubleConvUpC3(128, 256, 128, act=act)
        self.conv = nn.Sequential(
            nn.Conv2d(128, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.binarize = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, 2),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, 1, 2, 2),
        )
        self.thresh = self._init_thresh(in_channels)

    def forward(self, f80, f40, u40, step_eval=False):
        u80 = self.upconv3(torch.cat([f40, u40], dim=1))
        x = self.upconv4(torch.cat([f80, u80], dim=1))
        x = self.conv(x)
        threshold_maps = self.thresh(x)
        x = self.binarize(x)
        shrink_maps = torch.sigmoid(x)

        if self.training:
            binary_maps = self.step_function(shrink_maps, threshold_maps)
            if self.shrink_with_sigmoid:
                return torch.cat((shrink_maps, threshold_maps, binary_maps), dim=1)
            return torch.cat((shrink_maps, threshold_maps, binary_maps, x), dim=1)

        if step_eval:
            return self.step_function(shrink_maps, threshold_maps)
        return torch.cat((shrink_maps, threshold_maps), dim=1)

    def _init_thresh(self, inner_channels, serial=False, smooth=False, bias=False):
        in_channels = inner_channels + 1 if serial else inner_channels
        return nn.Sequential(
            nn.Conv2d(in_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, inner_channels // 4, smooth=smooth, bias=bias),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, 1, smooth=smooth, bias=bias),
            nn.Sigmoid(),
        )

    def _init_upsample(self, in_channels, out_channels, smooth=False, bias=False):
        if smooth:
            inter_out_channels = out_channels if out_channels != 1 else in_channels
            layers = [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias),
            ]
            if out_channels == 1:
                layers.append(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=1,
                        bias=True,
                    )
                )
            return nn.Sequential(*layers)
        return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))


def _build_yolov5_backbone(
    raw_state_dict: dict[str, torch.Tensor],
    device: str,
) -> Model:
    backbone = Model(YOLOV5_TEXT_BLOCK_CFG)
    _load_state_dict_strict(backbone, raw_state_dict, "YOLOv5 block detector")
    backbone.out_indices = [1, 3, 5, 7, 9]
    backbone.model = backbone.model[: max(backbone.out_indices) + 1]
    return backbone.eval().to(device)


class TextDetBase(nn.Module):
    def __init__(
        self,
        yolo_weights_path: str,
        unet_weights_path: str,
        dbnet_weights_path: str,
        device: str = "cpu",
        act: str = "leaky",
    ):
        super().__init__()
        yolo_state_dict = _load_safetensors_state_dict(yolo_weights_path)
        unet_state_dict = _load_safetensors_state_dict(unet_weights_path)
        dbnet_state_dict = _load_safetensors_state_dict(dbnet_weights_path)

        self.blk_det = _build_yolov5_backbone(yolo_state_dict, device)
        self.text_seg = UnetHead(act=act)
        _load_state_dict_strict(self.text_seg, unet_state_dict, "UNet text segmentation")
        self.text_seg = self.text_seg.eval().to(device)

        self.text_det = DBHead(64, act=act)
        _load_state_dict_strict(self.text_det, dbnet_state_dict, "DBNet text line head")
        self.text_det = self.text_det.eval().to(device)

    def forward(self, x):
        blks, features = self.blk_det(x, detect=True)
        mask, features = self.text_seg(*features, forward_mode=TEXTDET_INFERENCE)
        lines = self.text_det(*features, step_eval=False)
        return blks[0], mask, lines


__all__ = [
    "DBHead",
    "TEXTDET_DET",
    "TEXTDET_INFERENCE",
    "TEXTDET_MASK",
    "TextDetBase",
    "UnetHead",
]
