from __future__ import annotations

import importlib
from typing import Sequence

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

from .blocks import MangaRenderBlock


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


def _extract_region_stats(image_bgr, bbox: tuple[int, int, int, int], mask=None):
    np_module = _require_numpy()
    x1, y1, x2, y2 = [int(value) for value in bbox]
    if x2 <= x1 or y2 <= y1:
        return {
            "median_luma": 255.0,
            "std_luma": 0.0,
            "min_side": 1,
            "pixels": np_module.zeros((0, 3), dtype=np_module.uint8),
        }

    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return {
            "median_luma": 255.0,
            "std_luma": 0.0,
            "min_side": 1,
            "pixels": np_module.zeros((0, 3), dtype=np_module.uint8),
        }

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
    return {
        "median_luma": float(np_module.median(luminance)) if luminance.size else 255.0,
        "std_luma": float(np_module.std(luminance)) if luminance.size else 0.0,
        "min_side": max(1, min(crop.shape[0], crop.shape[1])),
        "pixels": pixels,
    }


def choose_text_color_for_region(
    image_bgr,
    bbox: tuple[int, int, int, int],
    mask=None,
    *,
    prefer_stroke: bool = False,
    is_dark: bool | None = None,
):
    stats = _extract_region_stats(image_bgr, bbox, mask=mask)
    median_luma = stats["median_luma"]
    std_luma = stats["std_luma"]
    min_side = stats["min_side"]

    base_stroke = max(1, int(round(min_side / 40.0)))
    max_stroke = max(1, int(round(min_side * 0.07)))
    is_background_dark = bool(is_dark) if is_dark is not None else median_luma < 132.0
    is_complex = std_luma >= 28.0

    if is_background_dark:
        text_color = (255, 255, 255)
        stroke_color = (0, 0, 0)
        stroke_width = min(max_stroke, max(1, base_stroke + (1 if is_complex or prefer_stroke else 0)))
        return text_color, stroke_color, stroke_width

    text_color = (0, 0, 0)
    if prefer_stroke or is_complex:
        stroke_color = (255, 255, 255)
        stroke_width = min(max_stroke, max(1, base_stroke + (1 if prefer_stroke else 0)))
    else:
        stroke_color = None
        stroke_width = min(max_stroke, max(1, base_stroke))
    return text_color, stroke_color, stroke_width


def choose_render_style_for_item(image_bgr, render_item, bbox, base_font_path, text: str = "") -> MangaRenderBlock:
    kind = render_item.get("kind", "bubble")
    color_mask = None
    if kind == "outside_text" and render_item.get("text_region") is not None:
        color_mask = render_item["text_region"].mask
    elif render_item.get("text_regions"):
        first_mask_region = next(
            (region for region in render_item["text_regions"] if getattr(region, "mask", None) is not None),
            None,
        )
        color_mask = None if first_mask_region is None else first_mask_region.mask
    elif render_item.get("bubble_region") is not None:
        color_mask = render_item["bubble_region"].mask

    bubble_region = render_item.get("bubble_region")
    bubble_is_dark = None if bubble_region is None else bubble_region.is_dark
    text_color, stroke_color, stroke_width = choose_text_color_for_region(
        image_bgr,
        bbox,
        mask=color_mask,
        prefer_stroke=(kind == "outside_text"),
        is_dark=bubble_is_dark,
    )

    if kind == "outside_text":
        stroke_width = max(1, int(round(stroke_width * 1.35)))
    else:
        stroke_width = max(1, int(round(stroke_width * 1.0)))

    return MangaRenderBlock(
        bbox=tuple(int(value) for value in bbox),
        text=text,
        kind=kind,
        source_direction=render_item.get("source_direction"),
        is_dark=bubble_is_dark,
        font_path=base_font_path,
        text_color=text_color,
        stroke_color=stroke_color,
        stroke_width=stroke_width,
        align="center",
    )


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
    "choose_render_style_for_item",
    "choose_text_color_for_region",
]
