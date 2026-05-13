from __future__ import annotations

import importlib
from typing import Sequence

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

from detectors.runtime_utils import (
    clamp_bbox_to_image,
    expand_bbox,
    normalize_binary_mask,
    union_text_regions_bbox,
)

from .strategy import crop_windows_from_bboxes


def _require_numpy():
    if np is None:
        raise ModuleNotFoundError("numpy is required for inpainting mask helpers")
    return np


def _get_cv2():
    try:
        return importlib.import_module("cv2")
    except ModuleNotFoundError:
        return None


def _empty_mask(image_shape: Sequence[int]):
    np_module = _require_numpy()
    return np_module.zeros((int(image_shape[0]), int(image_shape[1])), dtype=np_module.uint8)


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = [int(value) for value in bbox]
    return max(0, x2 - x1) * max(0, y2 - y1)


def _page_area(image_shape: Sequence[int]) -> int:
    return max(1, int(image_shape[0]) * int(image_shape[1]))


def _is_huge_bbox(
    bbox: tuple[int, int, int, int] | None,
    image_shape: Sequence[int],
    max_region_ratio: float = 0.35,
) -> bool:
    if bbox is None:
        return False
    clamped = clamp_bbox_to_image(bbox, image_shape)
    return _bbox_area(clamped) > (_page_area(image_shape) * float(max_region_ratio))


def _apply_bbox_mask(mask, bbox: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return
    mask[y1:y2, x1:x2] = 255


def _apply_region_mask(mask, region, image_shape: Sequence[int]) -> None:
    np_module = _require_numpy()
    if region is None:
        return
    if getattr(region, "mask", None) is not None:
        full_mask = normalize_binary_mask(region.mask, image_shape)
        mask[:] = np_module.maximum(mask, full_mask)
        return
    _apply_bbox_mask(mask, clamp_bbox_to_image(region.bbox, image_shape))


def _collect_text_regions(item) -> list:
    text_regions = list(item.get("text_regions") or [])
    if (
        item.get("kind") == "outside_text"
        and item.get("text_region") is not None
        and not any(region is item.get("text_region") for region in text_regions)
    ):
        text_regions.append(item["text_region"])
    return text_regions


def _dilate_mask(mask, dilation: int):
    np_module = _require_numpy()
    if dilation <= 0:
        return mask

    cv2 = _get_cv2()
    if cv2 is not None:
        kernel = np_module.ones((dilation * 2 + 1, dilation * 2 + 1), dtype=np_module.uint8)
        return cv2.dilate(mask, kernel, iterations=1)

    padded = np_module.pad(mask, dilation, mode="constant", constant_values=0)
    out = np_module.zeros_like(mask)
    for offset_y in range(dilation * 2 + 1):
        for offset_x in range(dilation * 2 + 1):
            out = np_module.maximum(
                out,
                padded[offset_y:offset_y + mask.shape[0], offset_x:offset_x + mask.shape[1]],
            )
    return out.astype(np_module.uint8)


def _candidate_bbox_from_item(
    item,
    image_shape: Sequence[int],
    *,
    block_padding: int,
    min_padding: int,
    prefer_block_bbox: bool,
    max_region_ratio: float,
):
    explicit_inpaint_bbox = item.get("inpaint_bbox")
    if explicit_inpaint_bbox is not None:
        explicit_bbox = clamp_bbox_to_image(explicit_inpaint_bbox, image_shape)
        if not _is_huge_bbox(explicit_bbox, image_shape, max_region_ratio=max_region_ratio):
            return explicit_bbox

    text_regions = _collect_text_regions(item)
    huge_region_skipped = bool(item.get("huge_region_skipped"))
    if prefer_block_bbox and not huge_region_skipped and text_regions:
        text_bbox = union_text_regions_bbox(text_regions, image_shape, padding=0)
        if text_bbox is not None:
            expanded_text_bbox = expand_bbox(text_bbox, image_shape, max(block_padding, min_padding))
            if not _is_huge_bbox(expanded_text_bbox, image_shape, max_region_ratio=max_region_ratio):
                return expanded_text_bbox

    if huge_region_skipped:
        fallback_candidates = [
            item.get("fallback_text_bbox"),
            item.get("ocr_bbox"),
        ]
        for fallback_bbox in fallback_candidates:
            if fallback_bbox is None:
                continue
            clamped_fallback = clamp_bbox_to_image(fallback_bbox, image_shape)
            if not _is_huge_bbox(clamped_fallback, image_shape, max_region_ratio=max_region_ratio):
                return clamped_fallback
        return None

    render_bbox = item.get("render_bbox")
    if render_bbox is not None:
        expanded_render_bbox = expand_bbox(render_bbox, image_shape, min_padding)
        if not _is_huge_bbox(expanded_render_bbox, image_shape, max_region_ratio=max_region_ratio):
            return expanded_render_bbox

    ocr_bbox = item.get("ocr_bbox")
    if ocr_bbox is not None:
        expanded_ocr_bbox = expand_bbox(ocr_bbox, image_shape, min_padding)
        if not _is_huge_bbox(expanded_ocr_bbox, image_shape, max_region_ratio=max_region_ratio):
            return expanded_ocr_bbox

    return None


def build_text_block_removal_mask(
    image_shape,
    render_items,
    *,
    block_padding: int = 8,
    min_padding: int = 4,
    dilation: int = 2,
    prefer_block_bbox: bool = True,
):
    np_module = _require_numpy()
    mask = _empty_mask(image_shape)

    for item in render_items or []:
        item_mask = _empty_mask(image_shape)

        candidate_bbox = _candidate_bbox_from_item(
            item,
            image_shape,
            block_padding=block_padding,
            min_padding=min_padding,
            prefer_block_bbox=prefer_block_bbox,
            max_region_ratio=0.35,
        )
        if candidate_bbox is not None:
            _apply_bbox_mask(item_mask, candidate_bbox)

        for region in _collect_text_regions(item):
            if getattr(region, "mask", None) is not None:
                _apply_region_mask(item_mask, region, image_shape)

        mask[:] = np_module.maximum(mask, item_mask)

    return _dilate_mask(mask, dilation)


def build_text_removal_mask(image_shape: Sequence[int], render_items, dilation: int = 4):
    return build_text_block_removal_mask(
        image_shape,
        render_items,
        block_padding=8,
        min_padding=4,
        dilation=dilation,
        prefer_block_bbox=True,
    )


def build_text_block_crop_windows(
    render_items,
    image_shape,
    ratio: float = 1.7,
    aspect_ratio: float = 1.0,
):
    boxes = []
    for item in render_items or []:
        candidate_bbox = _candidate_bbox_from_item(
            item,
            image_shape,
            block_padding=0,
            min_padding=0,
            prefer_block_bbox=True,
            max_region_ratio=0.35,
        )
        if candidate_bbox is None:
            continue
        boxes.append(candidate_bbox)
    return crop_windows_from_bboxes(
        boxes,
        image_shape,
        ratio=ratio,
        aspect_ratio=aspect_ratio,
    )


def build_bubble_mask(image_shape: Sequence[int], render_items):
    np_module = _require_numpy()
    mask = _empty_mask(image_shape)

    for item in render_items or []:
        if item.get("kind") != "bubble":
            continue
        bubble_region = item.get("bubble_region")
        if bubble_region is None:
            continue
        if bubble_region.mask is not None:
            bubble_mask = normalize_binary_mask(bubble_region.mask, image_shape)
            mask[:] = np_module.maximum(mask, bubble_mask)
            continue
        _apply_bbox_mask(mask, clamp_bbox_to_image(bubble_region.bbox, image_shape))

    return mask


__all__ = [
    "build_bubble_mask",
    "build_text_block_crop_windows",
    "build_text_block_removal_mask",
    "build_text_removal_mask",
]
