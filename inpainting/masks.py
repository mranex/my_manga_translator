from __future__ import annotations

import importlib
from typing import Any, Sequence

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

from detectors.runtime_utils import (
    clamp_bbox_to_image,
    expand_bbox,
    normalize_binary_mask,
)

from .strategy import crop_windows_from_bboxes


DEFAULT_MAX_REGION_RATIO = 0.35
DEFAULT_EMERGENCY_REGION_RATIO = 0.5


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
    max_region_ratio: float = DEFAULT_MAX_REGION_RATIO,
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


def _close_mask(mask, kernel_size: int = 3):
    np_module = _require_numpy()
    if kernel_size <= 1:
        return mask

    cv2 = _get_cv2()
    if cv2 is not None:
        kernel = np_module.ones((kernel_size, kernel_size), dtype=np_module.uint8)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    radius = max(1, kernel_size // 2)
    dilated = _dilate_mask(mask, radius)
    padded = np_module.pad(dilated, radius, mode="constant", constant_values=0)
    out = np_module.full_like(mask, 255)
    for offset_y in range(radius * 2 + 1):
        for offset_x in range(radius * 2 + 1):
            out = np_module.minimum(
                out,
                padded[offset_y:offset_y + mask.shape[0], offset_x:offset_x + mask.shape[1]],
            )
    return out.astype(np_module.uint8)


def _dedupe_bboxes(boxes):
    deduped = []
    seen = set()
    for box in boxes:
        if box is None:
            continue
        normalized = tuple(int(value) for value in box)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _candidate_boxes_from_item(
    item,
    image_shape: Sequence[int],
    *,
    block_padding: int,
    min_padding: int,
):
    candidate_bboxes = []

    explicit_bboxes = list(item.get("text_mask_bboxes") or [])
    explicit_bboxes.extend(item.get("inpaint_bboxes") or [])
    if item.get("inpaint_bbox") is not None:
        explicit_bboxes.append(item.get("inpaint_bbox"))

    for explicit_bbox in explicit_bboxes:
        clamped_bbox = clamp_bbox_to_image(explicit_bbox, image_shape)
        if clamped_bbox is not None:
            candidate_bboxes.append(clamped_bbox)

    for padded_key, padding in (
        ("render_bbox", block_padding),
        ("ocr_bbox", min_padding),
        ("bbox", min_padding),
        ("fallback_text_bbox", max(min_padding, block_padding // 2)),
    ):
        raw_bbox = item.get(padded_key)
        if raw_bbox is None:
            continue
        clamped_bbox = clamp_bbox_to_image(raw_bbox, image_shape)
        if clamped_bbox is None:
            continue
        candidate_bboxes.append(expand_bbox(clamped_bbox, image_shape, padding))

    return _dedupe_bboxes(candidate_bboxes)


def collect_item_inpaint_bboxes(
    item,
    image_shape: Sequence[int],
    *,
    block_padding: int = 14,
    min_padding: int = 8,
    max_region_ratio: float = DEFAULT_MAX_REGION_RATIO,
    emergency_max_region_ratio: float = DEFAULT_EMERGENCY_REGION_RATIO,
):
    resolved_cached = item.get("resolved_inpaint_bboxes")
    if resolved_cached is not None:
        return list(resolved_cached)

    deduped = _candidate_boxes_from_item(
        item,
        image_shape,
        block_padding=block_padding,
        min_padding=min_padding,
    )

    accepted = []
    emergency_candidates = []
    for bbox in deduped:
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        if not _is_huge_bbox(bbox, image_shape, max_region_ratio=max_region_ratio):
            accepted.append(bbox)
            continue
        if item.get("inpaint_fallback_used") and not _is_huge_bbox(
            bbox,
            image_shape,
            max_region_ratio=emergency_max_region_ratio,
        ):
            emergency_candidates.append(bbox)

    if not accepted and emergency_candidates:
        accepted.extend(emergency_candidates)
    return accepted


def build_text_block_removal_mask(
    image_shape,
    render_items,
    *,
    block_padding: int = 14,
    min_padding: int = 8,
    dilation: int = 4,
):
    np_module = _require_numpy()
    mask = _empty_mask(image_shape)

    for item in render_items or []:
        candidate_bboxes = collect_item_inpaint_bboxes(
            item,
            image_shape,
            block_padding=block_padding,
            min_padding=min_padding,
        )
        for candidate_bbox in candidate_bboxes:
            _apply_bbox_mask(mask, candidate_bbox)

    mask = _dilate_mask(mask, dilation)
    mask = _close_mask(mask, 3 if dilation <= 4 else 5)
    return np_module.where(mask > 0, 255, 0).astype(np_module.uint8)


def build_text_removal_mask(image_shape: Sequence[int], render_items, dilation: int = 4):
    return build_text_block_removal_mask(
        image_shape,
        render_items,
        block_padding=14,
        min_padding=8,
        dilation=max(4, dilation),
    )


def build_text_block_crop_windows(
    render_items,
    image_shape,
    ratio: float = 2.0,
    aspect_ratio: float = 1.0,
):
    boxes = []
    for item in render_items or []:
        boxes.extend(
            collect_item_inpaint_bboxes(
                item,
                image_shape,
                block_padding=14,
                min_padding=8,
            )
        )
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

        bubble_mask = item.get("bubble_mask")
        if bubble_mask is not None:
            binary_mask = normalize_binary_mask(bubble_mask, image_shape)
            mask[:] = np_module.maximum(mask, binary_mask)
            continue

        bubble_region = item.get("bubble_region")
        if bubble_region is not None:
            if getattr(bubble_region, "mask", None) is not None:
                binary_mask = normalize_binary_mask(bubble_region.mask, image_shape)
                mask[:] = np_module.maximum(mask, binary_mask)
                continue
            bubble_bbox = getattr(bubble_region, "bbox", None)
        else:
            bubble_bbox = item.get("bbox")

        clamped_bbox = clamp_bbox_to_image(bubble_bbox, image_shape) if bubble_bbox is not None else None
        if clamped_bbox is not None:
            _apply_bbox_mask(mask, clamped_bbox)

    return mask


__all__ = [
    "build_bubble_mask",
    "build_text_block_crop_windows",
    "build_text_block_removal_mask",
    "build_text_removal_mask",
    "collect_item_inpaint_bboxes",
]
