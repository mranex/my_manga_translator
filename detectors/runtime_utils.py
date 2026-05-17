from __future__ import annotations

import importlib
from dataclasses import replace
from typing import Sequence

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

from .base import BubbleRegion, bubble_region_from_legacy_detection
from .matching import (
    bbox_area,
    bbox_center,
    bbox_intersection_area,
    bbox_iou,
    point_in_bbox,
)


def legacy_detection_to_bubble_region(result: Sequence[object]) -> BubbleRegion:
    return bubble_region_from_legacy_detection(result)


def convert_legacy_detections_to_bubble_regions(
    results: Sequence[Sequence[object]],
) -> list[BubbleRegion]:
    return [legacy_detection_to_bubble_region(result) for result in results]


def clamp_bbox_to_image(
    bbox: tuple[int, int, int, int],
    image_shape: Sequence[int],
) -> tuple[int, int, int, int]:
    height = int(image_shape[0])
    width = int(image_shape[1])

    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), width))
    y1 = max(0, min(int(y1), height))
    x2 = max(0, min(int(x2), width))
    y2 = max(0, min(int(y2), height))

    return (
        min(x1, x2),
        min(y1, y2),
        max(x1, x2),
        max(y1, y2),
    )


def expand_bbox(
    bbox: tuple[int, int, int, int],
    image_shape: Sequence[int],
    padding: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return clamp_bbox_to_image(
        (x1 - padding, y1 - padding, x2 + padding, y2 + padding),
        image_shape,
    )


def crop_bbox(image, bbox: tuple[int, int, int, int]):
    x1, y1, x2, y2 = bbox
    try:
        return image[y1:y2, x1:x2]
    except TypeError:
        return [row[x1:x2] for row in image[y1:y2]]


def _get_cv2():
    try:
        return importlib.import_module("cv2")
    except ModuleNotFoundError:
        return None


def _require_numpy():
    if np is None:
        raise ModuleNotFoundError("numpy is required for mask-based bubble processing")
    return np


def _grayscale_values(image: np.ndarray) -> np.ndarray:
    np_module = _require_numpy()
    if image.ndim == 2:
        return image.astype(np_module.float32)
    return (
        (0.114 * image[..., 0])
        + (0.587 * image[..., 1])
        + (0.299 * image[..., 2])
    ).astype(np_module.float32)


def _resize_mask_nearest(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    np_module = _require_numpy()
    source_height, source_width = mask.shape[:2]
    if source_height == height and source_width == width:
        return mask
    if source_height <= 0 or source_width <= 0:
        return np_module.zeros((height, width), dtype=mask.dtype)

    y_indices = np_module.clip(
        np_module.floor(np_module.arange(height) * (source_height / height)).astype(int),
        0,
        source_height - 1,
    )
    x_indices = np_module.clip(
        np_module.floor(np_module.arange(width) * (source_width / width)).astype(int),
        0,
        source_width - 1,
    )
    return mask[y_indices][:, x_indices]


def _to_numpy_mask(mask) -> np.ndarray:
    np_module = _require_numpy()
    if mask is None:
        return np_module.zeros((0, 0), dtype=np_module.uint8)

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
    if array.ndim != 2:
        raise ValueError("Mask must be 2D after normalization")
    return array


def normalize_binary_mask(mask, image_shape: Sequence[int]) -> np.ndarray:
    np_module = _require_numpy()
    height = int(image_shape[0])
    width = int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("Image shape must have positive height and width")

    array = _to_numpy_mask(mask)
    if array.size == 0:
        return np_module.zeros((height, width), dtype=np_module.uint8)

    if array.shape != (height, width):
        cv2 = _get_cv2()
        if cv2 is not None:
            array = cv2.resize(
                array.astype(np_module.float32),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            array = _resize_mask_nearest(array, height, width)

    return np_module.where(array > 0, 255, 0).astype(np_module.uint8)


def detect_dark_bubble_from_mask(
    image: np.ndarray,
    mask,
    threshold: float = 140.0,
) -> bool:
    np_module = _require_numpy()
    if image is None or image.size == 0:
        return False

    binary_mask = normalize_binary_mask(mask, image.shape)
    masked_pixels = binary_mask > 0
    if not np_module.any(masked_pixels):
        return False

    gray = _grayscale_values(image)
    median_intensity = float(np_module.median(gray[masked_pixels]))
    return median_intensity < float(threshold)


def map_bbox_from_roi_to_page(
    local_bbox: tuple[int, int, int, int],
    roi_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    roi_x1, roi_y1, _, _ = roi_bbox
    x1, y1, x2, y2 = local_bbox
    return (
        int(x1) + int(roi_x1),
        int(y1) + int(roi_y1),
        int(x2) + int(roi_x1),
        int(y2) + int(roi_y1),
    )


def map_mask_from_roi_to_page(mask, roi_bbox, image_shape):
    if mask is None:
        return None

    np_module = _require_numpy()
    page_height = int(image_shape[0])
    page_width = int(image_shape[1])
    roi_x1, roi_y1, roi_x2, roi_y2 = clamp_bbox_to_image(roi_bbox, image_shape)
    roi_width = max(0, roi_x2 - roi_x1)
    roi_height = max(0, roi_y2 - roi_y1)
    if roi_width <= 0 or roi_height <= 0:
        return np_module.zeros((page_height, page_width), dtype=np_module.uint8)

    full_mask = np_module.zeros((page_height, page_width), dtype=np_module.uint8)
    local_mask = normalize_binary_mask(mask, (roi_height, roi_width))
    full_mask[roi_y1:roi_y2, roi_x1:roi_x2] = np_module.maximum(
        full_mask[roi_y1:roi_y2, roi_x1:roi_x2],
        local_mask[:roi_height, :roi_width],
    )
    return full_mask


def map_bubble_region_from_roi_to_page(
    bubble_region: BubbleRegion,
    roi_bbox,
    image_shape,
) -> BubbleRegion:
    return replace(
        bubble_region,
        bbox=map_bbox_from_roi_to_page(bubble_region.bbox, roi_bbox),
        mask=map_mask_from_roi_to_page(bubble_region.mask, roi_bbox, image_shape),
    )


def _union_bbox(
    bbox_a: tuple[int, int, int, int],
    bbox_b: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    return (
        min(int(bbox_a[0]), int(bbox_b[0])),
        min(int(bbox_a[1]), int(bbox_b[1])),
        max(int(bbox_a[2]), int(bbox_b[2])),
        max(int(bbox_a[3]), int(bbox_b[3])),
    )


def _merge_masks(mask_a, mask_b, image_shape):
    np_module = _require_numpy()
    if mask_a is None and mask_b is None:
        return None
    if mask_a is None:
        return normalize_binary_mask(mask_b, image_shape)
    if mask_b is None:
        return normalize_binary_mask(mask_a, image_shape)

    normalized_a = normalize_binary_mask(mask_a, image_shape)
    normalized_b = normalize_binary_mask(mask_b, image_shape)
    return np_module.maximum(normalized_a, normalized_b).astype(np_module.uint8)


def _infer_image_shape_from_masks(mask_a, mask_b):
    for candidate in (mask_a, mask_b):
        if candidate is None:
            continue
        shape = getattr(candidate, "shape", None)
        if shape is not None and len(shape) >= 2:
            return (int(shape[0]), int(shape[1]))
        try:
            height = len(candidate)
            width = len(candidate[0]) if height > 0 else 0
            if height > 0 and width > 0:
                return (int(height), int(width))
        except (TypeError, IndexError, KeyError):
            continue
    return None


def _center_in_bbox(point, bbox, *, padding: int = 0) -> bool:
    return point_in_bbox(point, bbox, padding=padding)


def _mask_overlap_metrics(mask_a, mask_b, image_shape):
    if np is None or image_shape is None or mask_a is None or mask_b is None:
        return (0.0, 0.0)

    np_module = _require_numpy()
    normalized_a = normalize_binary_mask(mask_a, image_shape) > 0
    normalized_b = normalize_binary_mask(mask_b, image_shape) > 0
    area_a = int(np_module.count_nonzero(normalized_a))
    area_b = int(np_module.count_nonzero(normalized_b))
    if area_a <= 0 or area_b <= 0:
        return (0.0, 0.0)

    intersection = int(np_module.count_nonzero(normalized_a & normalized_b))
    if intersection <= 0:
        return (0.0, 0.0)

    union = max(area_a + area_b - intersection, 1)
    return (
        float(intersection / union),
        float(intersection / max(min(area_a, area_b), 1)),
    )


def merge_duplicate_bubble_regions(
    bubbles: Sequence[BubbleRegion],
    *,
    iou_threshold: float = 0.45,
    image_shape=None,
) -> list[BubbleRegion]:
    merged: list[BubbleRegion] = []

    for bubble in bubbles:
        match_index = None
        best_score = 0.0
        for index, existing in enumerate(merged):
            intersection = bbox_intersection_area(existing.bbox, bubble.bbox)
            if intersection <= 0.0:
                continue

            bbox_overlap_ratio = float(
                intersection / max(min(bbox_area(existing.bbox), bbox_area(bubble.bbox)), 1.0)
            )
            current_iou = bbox_iou(existing.bbox, bubble.bbox)
            existing_center = bbox_center(existing.bbox)
            bubble_center = bbox_center(bubble.bbox)
            center_overlap = (
                _center_in_bbox(existing_center, bubble.bbox)
                or _center_in_bbox(bubble_center, existing.bbox)
            )
            merge_shape = (
                image_shape
                if image_shape is not None
                else _infer_image_shape_from_masks(existing.mask, bubble.mask)
            )
            mask_iou, mask_overlap_ratio = _mask_overlap_metrics(
                existing.mask,
                bubble.mask,
                merge_shape,
            )
            is_duplicate = (
                current_iou >= float(iou_threshold)
                or mask_iou >= 0.35
                or bbox_overlap_ratio >= 0.80
                or (center_overlap and bbox_overlap_ratio >= 0.65)
                or mask_overlap_ratio >= 0.70
            )
            if not is_duplicate:
                continue

            current_score = max(
                current_iou,
                mask_iou,
                bbox_overlap_ratio,
                mask_overlap_ratio,
            )
            if current_score >= best_score:
                best_score = current_score
                match_index = index

        if match_index is None:
            merged.append(bubble)
            continue

        existing = merged[match_index]
        preferred = bubble if bubble.score > existing.score else existing
        merge_shape = (
            image_shape
            if image_shape is not None
            else _infer_image_shape_from_masks(existing.mask, bubble.mask)
        )
        merged[match_index] = BubbleRegion(
            bbox=_union_bbox(existing.bbox, bubble.bbox),
            score=max(existing.score, bubble.score),
            class_id=preferred.class_id,
            mask=_merge_masks(existing.mask, bubble.mask, merge_shape)
            if (
                merge_shape is not None
                and (existing.mask is not None or bubble.mask is not None)
            )
            else None,
            is_dark=bool(existing.is_dark or bubble.is_dark),
            fill_color=preferred.fill_color,
        )

    return merged


__all__ = [
    "clamp_bbox_to_image",
    "convert_legacy_detections_to_bubble_regions",
    "crop_bbox",
    "detect_dark_bubble_from_mask",
    "expand_bbox",
    "legacy_detection_to_bubble_region",
    "map_bbox_from_roi_to_page",
    "map_bubble_region_from_roi_to_page",
    "map_mask_from_roi_to_page",
    "merge_duplicate_bubble_regions",
    "normalize_binary_mask",
]
