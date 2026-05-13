from __future__ import annotations

import importlib
from typing import Sequence

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


def _require_numpy():
    if np is None:
        raise ModuleNotFoundError("numpy is required for text rendering effects")
    return np


def _get_cv2():
    try:
        return importlib.import_module("cv2")
    except ModuleNotFoundError:
        return None


def _normalize_local_mask(mask, width: int, height: int):
    np_module = _require_numpy()
    if mask is None:
        return None
    current = mask
    if hasattr(current, "detach"):
        current = current.detach()
    if hasattr(current, "cpu"):
        current = current.cpu()
    if hasattr(current, "numpy"):
        current = current.numpy()
    array = np_module.asarray(current)
    if array.ndim == 3:
        if array.shape[0] == 1:
            array = array[0]
        elif array.shape[-1] == 1:
            array = array[..., 0]
    if array.shape != (height, width):
        cv2 = _get_cv2()
        if cv2 is not None:
            array = cv2.resize(
                array.astype(np_module.float32),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            y_indices = np_module.clip(
                np_module.floor(np_module.arange(height) * (array.shape[0] / height)).astype(int),
                0,
                array.shape[0] - 1,
            )
            x_indices = np_module.clip(
                np_module.floor(np_module.arange(width) * (array.shape[1] / width)).astype(int),
                0,
                array.shape[1] - 1,
            )
            array = array[y_indices][:, x_indices]
    return np_module.where(array > 0, 255, 0).astype(np_module.uint8)


def choose_text_color_for_region(image_bgr, bbox: tuple[int, int, int, int], mask=None):
    np_module = _require_numpy()
    x1, y1, x2, y2 = [int(value) for value in bbox]
    if x2 <= x1 or y2 <= y1:
        return (0, 0, 0), (255, 255, 255), 1

    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return (0, 0, 0), (255, 255, 255), 1

    local_mask = None
    if mask is not None:
        mask_array = np_module.asarray(mask)
        if mask_array.ndim >= 2 and mask_array.shape[0] == image_bgr.shape[0] and mask_array.shape[1] == image_bgr.shape[1]:
            local_mask = _normalize_local_mask(mask_array[y1:y2, x1:x2], crop.shape[1], crop.shape[0])
        else:
            local_mask = _normalize_local_mask(mask_array, crop.shape[1], crop.shape[0])
    if local_mask is not None and np_module.any(local_mask):
        pixels = crop[local_mask > 0]
    else:
        pixels = crop.reshape(-1, crop.shape[-1])

    if pixels.size == 0:
        pixels = crop.reshape(-1, crop.shape[-1])

    luminance = (
        0.114 * pixels[:, 0].astype(np_module.float32)
        + 0.587 * pixels[:, 1].astype(np_module.float32)
        + 0.299 * pixels[:, 2].astype(np_module.float32)
    )
    median_luma = float(np_module.median(luminance))

    min_side = max(1, min(crop.shape[0], crop.shape[1]))
    stroke_width = max(1, int(round(min_side / 36.0)))
    if median_luma < 128.0:
        return (255, 255, 255), (0, 0, 0), stroke_width
    return (0, 0, 0), (255, 255, 255), stroke_width


def alpha_composite_onto_bgr(image_bgr, overlay_rgba, bbox: tuple[int, int, int, int]) -> None:
    np_module = _require_numpy()
    x1, y1, x2, y2 = [int(value) for value in bbox]
    if x2 <= x1 or y2 <= y1:
        return

    roi = image_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return

    overlay = np_module.asarray(overlay_rgba, dtype=np_module.float32)
    if overlay.shape[0] != roi.shape[0] or overlay.shape[1] != roi.shape[1]:
        raise ValueError("Overlay dimensions must match the target ROI")

    alpha = overlay[..., 3:4] / 255.0
    overlay_bgr = overlay[..., :3][:, :, ::-1]
    blended = overlay_bgr * alpha + roi.astype(np_module.float32) * (1.0 - alpha)
    roi[:] = np_module.clip(blended, 0, 255).astype(np_module.uint8)


__all__ = [
    "alpha_composite_onto_bgr",
    "choose_text_color_for_region",
]
