from __future__ import annotations

import importlib
from pathlib import Path
from typing import Sequence

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None
    nn = None
    F = None

from .strategy import crop_windows_from_text_regions, run_inpaint_crop, run_inpaint_resize


if nn is None:
    _ModuleBase = object
else:
    _ModuleBase = nn.Module


REQUIRED_RUNTIME_MODULES = [
    ("torch", "torch"),
    ("numpy", "numpy"),
    ("huggingface_hub", "huggingface_hub"),
    ("safetensors.torch", "safetensors"),
]

MODEL_REPO_ID = "mayocream/lama-manga"
MODEL_FILENAME = "lama-manga.safetensors"


class LamaMangaUnavailable(RuntimeError):
    pass


def _require_numpy():
    if np is None:
        raise ModuleNotFoundError("numpy is required for LaMa Manga inpainting")
    return np


def _check_runtime_dependencies() -> None:
    failures = []
    for module_name, package_hint in REQUIRED_RUNTIME_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            failures.append((module_name, package_hint, repr(exc)))
    if failures:
        details = "\n".join(
            f"- import {module_name} failed; install/check {package_hint}: {error}"
            for module_name, package_hint, error in failures
        )
        raise LamaMangaUnavailable(
            "LaMa Manga runtime dependencies are not available:\n"
            f"{details}"
        )


def _import_huggingface_hub():
    try:
        return importlib.import_module("huggingface_hub")
    except Exception as exc:
        raise LamaMangaUnavailable(
            "LaMa Manga weight download requires huggingface_hub: "
            f"{exc}"
        ) from exc


def _default_model_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "model" / "lama_manga"


def ensure_lama_manga_weights(model_dir: str | Path | None = None) -> Path:
    model_directory = Path(model_dir) if model_dir is not None else _default_model_dir()
    model_directory.mkdir(parents=True, exist_ok=True)
    model_path = model_directory / MODEL_FILENAME
    if model_path.exists():
        return model_path

    huggingface_hub = _import_huggingface_hub()
    downloaded_path = huggingface_hub.hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=MODEL_FILENAME,
        local_dir=str(model_directory),
        local_dir_use_symlinks=False,
    )
    return Path(downloaded_path)


class Conv2dPad(_ModuleBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if nn is None or torch is None:
            self.weight = None
            self.bias = None
            self.stride = stride
            self.padding = padding
            self.dilation = dilation
            self.groups = groups
            return

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x):
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode="reflect")
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            groups=self.groups,
        )


class FourierUnit(_ModuleBase):
    def __init__(self, in_channels: int, out_channels: int, groups: int = 1) -> None:
        super().__init__()
        if nn is None:
            return
        self.groups = groups
        self.conv_layer = nn.Conv2d(
            in_channels=in_channels * 2,
            out_channels=out_channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch = x.shape[0]
        ffted = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1) + ffted.size()[3:])

        ffted = self.conv_layer(ffted)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        return torch.fft.irfft2(ffted, s=x.shape[-2:], dim=(-2, -1), norm="ortho")


class SpectralTransform(_ModuleBase):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        if nn is None:
            return
        self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2) if stride == 2 else nn.Identity()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, groups=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups=1)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, groups=1, bias=False)

    def forward(self, x):
        x = self.downsample(x)
        y = self.conv1(x)
        return self.conv2(y + self.fu(y))


class FFC(_ModuleBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        ratio_gin: float,
        ratio_gout: float,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        if nn is None:
            return

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg

        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        self.convl2l = (
            Conv2dPad(in_cl, out_cl, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
            if in_cl > 0 and out_cl > 0
            else nn.Identity()
        )
        self.convl2g = (
            Conv2dPad(in_cl, out_cg, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
            if in_cl > 0 and out_cg > 0
            else nn.Identity()
        )
        self.convg2l = (
            Conv2dPad(in_cg, out_cl, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
            if in_cg > 0 and out_cl > 0
            else nn.Identity()
        )
        self.convg2g = (
            SpectralTransform(in_cg, out_cg, stride=stride)
            if in_cg > 0 and out_cg > 0
            else nn.Identity()
        )

    def forward(self, x):
        x_l, x_g = x if isinstance(x, tuple) else (x, 0)
        out_xl, out_xg = 0, 0

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l)
            if torch.is_tensor(x_g):
                out_xl = out_xl + self.convg2l(x_g)
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l)
            if torch.is_tensor(x_g):
                out_xg = out_xg + self.convg2g(x_g)
        return out_xl, out_xg


class FFCBnAct(_ModuleBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        ratio_gin: float,
        ratio_gout: float,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        if nn is None:
            return
        self.ffc = FFC(
            in_channels,
            out_channels,
            kernel_size,
            ratio_gin=ratio_gin,
            ratio_gout=ratio_gout,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        global_channels = int(out_channels * ratio_gout)
        local_channels = out_channels - global_channels
        self.bn_l = nn.BatchNorm2d(local_channels) if local_channels > 0 else nn.Identity()
        self.bn_g = nn.BatchNorm2d(global_channels) if global_channels > 0 else nn.Identity()
        self.act_l = nn.ReLU(inplace=True) if local_channels > 0 else nn.Identity()
        self.act_g = nn.ReLU(inplace=True) if global_channels > 0 else nn.Identity()

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        if torch.is_tensor(x_g):
            x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class FFCResBlock(_ModuleBase):
    def __init__(self, dim: int, *, ratio_gin: float, ratio_gout: float) -> None:
        super().__init__()
        if nn is None:
            return
        self.conv1 = FFCBnAct(
            dim,
            dim,
            3,
            ratio_gin=ratio_gin,
            ratio_gout=ratio_gout,
            stride=1,
            padding=1,
            dilation=1,
        )
        self.conv2 = FFCBnAct(
            dim,
            dim,
            3,
            ratio_gin=ratio_gin,
            ratio_gout=ratio_gout,
            stride=1,
            padding=1,
            dilation=1,
        )

    def forward(self, x):
        x_l, x_g = x if isinstance(x, tuple) else (x, 0)
        id_l, id_g = x_l, x_g
        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))
        x_l = id_l + x_l
        if torch.is_tensor(id_g) and torch.is_tensor(x_g):
            x_g = id_g + x_g
        return x_l, x_g


class ConcatTupleLayer(_ModuleBase):
    def forward(self, x):
        x_l, x_g = x
        if not torch.is_tensor(x_g):
            return x_l
        return torch.cat((x_l, x_g), dim=1)


class LamaMangaModel(_ModuleBase):
    def __init__(self) -> None:
        super().__init__()
        if nn is None:
            return
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            FFCBnAct(4, 64, 7, ratio_gin=0.0, ratio_gout=0.0, padding=0),
            FFCBnAct(64, 128, 3, ratio_gin=0.0, ratio_gout=0.0, stride=2, padding=1),
            FFCBnAct(128, 256, 3, ratio_gin=0.0, ratio_gout=0.0, stride=2, padding=1),
            FFCBnAct(256, 512, 3, ratio_gin=0.0, ratio_gout=0.75, stride=2, padding=1),
        ]
        for _ in range(18):
            layers.append(FFCResBlock(512, ratio_gin=0.75, ratio_gout=0.75))
        layers.extend(
            [
                ConcatTupleLayer(),
                nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.ReflectionPad2d(3),
                nn.Conv2d(64, 3, kernel_size=7, padding=0),
                nn.Sigmoid(),
            ]
        )
        self.model = nn.Sequential(*layers)

    def forward(self, image, mask):
        mask_inv = 1.0 - mask
        mask3 = mask.expand(-1, 3, -1, -1)
        masked_image = image * mask_inv.expand(-1, 3, -1, -1)
        predicted = self.model(torch.cat((masked_image, mask), dim=1))
        return predicted * mask3 + image * mask_inv.expand(-1, 3, -1, -1)


class LamaMangaInpainter:
    def __init__(
        self,
        model_path: str | None = None,
        device: str | None = None,
        crop_trigger_size: int = 800,
        crop_margin: int = 128,
        resize_limit: int = 1280,
        pad_mod: int = 8,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.crop_trigger_size = int(crop_trigger_size)
        self.crop_margin = int(crop_margin)
        self.resize_limit = int(resize_limit)
        self.pad_mod = int(pad_mod)
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return

        _check_runtime_dependencies()
        if torch is None:
            raise LamaMangaUnavailable("torch is required for LaMa Manga inpainting")

        model_path = Path(self.model_path) if self.model_path is not None else ensure_lama_manga_weights()
        model_path.parent.mkdir(parents=True, exist_ok=True)

        load_file = importlib.import_module("safetensors.torch").load_file
        target_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

        model = LamaMangaModel()
        state_dict = load_file(str(model_path), device="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys or unexpected_keys:
            diagnostics = []
            if missing_keys:
                diagnostics.append("missing keys:\n  " + "\n  ".join(sorted(missing_keys)))
            if unexpected_keys:
                diagnostics.append("unexpected keys:\n  " + "\n  ".join(sorted(unexpected_keys)))
            raise RuntimeError(
                "LaMa Manga weight loading failed due to key mismatch:\n"
                + "\n".join(diagnostics)
            )

        model.to(target_device)
        model.eval()
        self._model = model
        self.device = target_device

    def _run_model_on_patch(self, image_bgr, text_mask, bubble_mask=None):
        np_module = _require_numpy()
        self.load()
        if self._model is None or torch is None:
            raise LamaMangaUnavailable("LaMa Manga model is not loaded")

        rgb_image = image_bgr[:, :, ::-1].astype(np_module.float32) / 255.0
        mask_array = np_module.where(text_mask > 0, 1.0, 0.0).astype(np_module.float32)

        image_tensor = torch.from_numpy(rgb_image.transpose(2, 0, 1)).unsqueeze(0).to(self.device, dtype=torch.float32)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.float32)

        try:
            with torch.no_grad():
                output_tensor = self._model(image_tensor, mask_tensor)
            output_rgb = output_tensor.squeeze(0).detach().cpu().clamp(0.0, 1.0).numpy().transpose(1, 2, 0)
        finally:
            del image_tensor
            del mask_tensor

        output_bgr = np_module.clip(output_rgb[:, :, ::-1] * 255.0, 0, 255).astype(np_module.uint8)
        output = image_bgr.copy()
        output[text_mask > 0] = output_bgr[text_mask > 0]
        return output

    def inpaint(
        self,
        image_bgr,
        text_mask,
        bubble_mask=None,
        text_regions: list | None = None,
        crop_windows: list[tuple[int, int, int, int]] | None = None,
    ):
        np_module = _require_numpy()
        if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
            raise ValueError("image_bgr must contain image data")

        binary_mask = np_module.where(text_mask > 0, 255, 0).astype(np_module.uint8)
        if not np_module.any(binary_mask):
            return image_bgr.copy()

        working_bubble_mask = None
        if bubble_mask is not None:
            working_bubble_mask = np_module.where(bubble_mask > 0, 255, 0).astype(np_module.uint8)

        windows = list(crop_windows or [])
        if not windows:
            windows = crop_windows_from_text_regions(text_regions or [], image_bgr.shape)
        max_side = max(image_bgr.shape[0], image_bgr.shape[1])
        if windows:
            output = run_inpaint_crop(
                self._run_model_on_patch,
                image_bgr,
                binary_mask,
                bubble_mask=working_bubble_mask,
                crop_trigger_size=self.crop_trigger_size,
                crop_margin=self.crop_margin,
                resize_limit=self.resize_limit,
                pad_mod=self.pad_mod,
                text_regions=text_regions,
                crop_windows=windows,
            )
        elif max_side > self.crop_trigger_size:
            output = run_inpaint_crop(
                self._run_model_on_patch,
                image_bgr,
                binary_mask,
                bubble_mask=working_bubble_mask,
                crop_trigger_size=self.crop_trigger_size,
                crop_margin=self.crop_margin,
                resize_limit=self.resize_limit,
                pad_mod=self.pad_mod,
                text_regions=text_regions,
                crop_windows=None,
            )
        elif max_side > self.resize_limit:
            output = run_inpaint_resize(
                self._run_model_on_patch,
                image_bgr,
                binary_mask,
                bubble_mask=working_bubble_mask,
                resize_limit=self.resize_limit,
                pad_mod=self.pad_mod,
            )
        else:
            output = self._run_model_on_patch(image_bgr, binary_mask, working_bubble_mask)
        if torch is not None and torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()
        return output


__all__ = [
    "LamaMangaInpainter",
    "LamaMangaModel",
    "LamaMangaUnavailable",
    "ensure_lama_manga_weights",
]
