from __future__ import annotations

import importlib
from dataclasses import replace
from typing import Sequence

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

from .base import BubbleRegion, TextRegion, bubble_region_from_legacy_detection
from .matching import bbox_iou


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

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    return (x1, y1, x2, y2)


def union_text_regions_bbox(
    text_regions: Sequence[TextRegion],
    image_shape: Sequence[int],
    padding: int = 12,
) -> tuple[int, int, int, int] | None:
    if not text_regions:
        return None

    x1 = min(region.bbox[0] for region in text_regions) - padding
    y1 = min(region.bbox[1] for region in text_regions) - padding
    x2 = max(region.bbox[2] for region in text_regions) + padding
    y2 = max(region.bbox[3] for region in text_regions) + padding

    return clamp_bbox_to_image((x1, y1, x2, y2), image_shape)


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

    binary_mask = np_module.where(array > 0, 255, 0).astype(np_module.uint8)
    return binary_mask


def _rect_contour(width: int, height: int) -> np.ndarray:
    np_module = _require_numpy()
    max_x = max(int(width) - 1, 0)
    max_y = max(int(height) - 1, 0)
    return np_module.array(
        [
            [[0, 0]],
            [[max_x, 0]],
            [[max_x, max_y]],
            [[0, max_y]],
        ],
        dtype=np_module.int32,
    )


def mask_to_contour(mask) -> np.ndarray:
    np_module = _require_numpy()
    mask_array = _to_numpy_mask(mask)

    if mask_array.size == 0:
        return _rect_contour(1, 1)

    binary_mask = np_module.where(mask_array > 0, 255, 0).astype(np_module.uint8)
    height, width = binary_mask.shape[:2]

    cv2 = _get_cv2()
    if cv2 is not None:
        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if contours:
            return max(contours, key=cv2.contourArea)

    ys, xs = np_module.nonzero(binary_mask > 0)
    if xs.size == 0 or ys.size == 0:
        return _rect_contour(width, height)

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return np_module.array(
        [
            [[x1, y1]],
            [[max(x2 - 1, x1), y1]],
            [[max(x2 - 1, x1), max(y2 - 1, y1)]],
            [[x1, max(y2 - 1, y1)]],
        ],
        dtype=np_module.int32,
    )


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


def _detect_fill_color_from_mask(
    image: np.ndarray,
    binary_mask: np.ndarray,
    *,
    bubble_is_dark: bool,
) -> tuple[int, int, int]:
    np_module = _require_numpy()
    default_color = (0, 0, 0) if bubble_is_dark else (255, 255, 255)
    if image is None or image.size == 0:
        return default_color

    pixels = image[binary_mask > 0]
    if pixels.size == 0:
        return default_color

    gray = _grayscale_values(pixels.reshape((-1, 1, 3))).reshape(-1)

    if bubble_is_dark:
        cutoff = float(np_module.percentile(gray, 60))
        background_pixels = pixels[gray <= cutoff]
    else:
        cutoff = float(np_module.percentile(gray, 40))
        background_pixels = pixels[gray >= cutoff]

    if background_pixels.size == 0:
        background_pixels = pixels

    color = np_module.median(background_pixels, axis=0)
    return tuple(int(np_module.clip(channel, 0, 255)) for channel in color)


def process_bubble_with_mask(
    bubble_crop: np.ndarray,
    mask_crop,
    force_dark: bool = False,
):
    np_module = _require_numpy()
    if bubble_crop is None or getattr(bubble_crop, "size", 0) == 0:
        raise ValueError("Bubble crop must contain image data")

    binary_mask = normalize_binary_mask(mask_crop, bubble_crop.shape)
    if not np_module.any(binary_mask):
        binary_mask = np_module.full(
            bubble_crop.shape[:2],
            255,
            dtype=np_module.uint8,
        )

    contour = mask_to_contour(binary_mask)
    bubble_is_dark = detect_dark_bubble_from_mask(bubble_crop, binary_mask)
    if force_dark:
        bubble_is_dark = True

    detected_color = _detect_fill_color_from_mask(
        bubble_crop,
        binary_mask,
        bubble_is_dark=bubble_is_dark,
    )
    bubble_crop[binary_mask > 0] = detected_color

    return bubble_crop, contour, bubble_is_dark, detected_color


def bubble_region_to_crop_data(
    image: np.ndarray,
    bubble_region: BubbleRegion,
    matched_text_regions: Sequence[TextRegion] | None = None,
    *,
    padding: int = 12,
):
    np_module = _require_numpy()
    bubble_bbox = clamp_bbox_to_image(bubble_region.bbox, image.shape)
    bubble_crop = crop_bbox(image, bubble_bbox)

    if bubble_region.mask is None:
        mask_crop = np_module.full(
            bubble_crop.shape[:2],
            255,
            dtype=np_module.uint8,
        )
    else:
        full_mask = normalize_binary_mask(bubble_region.mask, image.shape)
        mask_crop = crop_bbox(full_mask, bubble_bbox)

    ocr_bbox = union_text_regions_bbox(
        matched_text_regions or [],
        image.shape,
        padding=padding,
    )
    if ocr_bbox is None:
        ocr_bbox = bubble_bbox

    ocr_crop = crop_bbox(image, ocr_bbox)

    return {
        "bubble_bbox": bubble_bbox,
        "bubble_crop": bubble_crop,
        "mask_crop": mask_crop,
        "ocr_bbox": ocr_bbox,
        "ocr_crop": ocr_crop,
    }


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


def map_text_region_from_roi_to_page(
    text_region: TextRegion,
    roi_bbox,
    image_shape,
) -> TextRegion:
    return replace(
        text_region,
        bbox=map_bbox_from_roi_to_page(text_region.bbox, roi_bbox),
        mask=map_mask_from_roi_to_page(text_region.mask, roi_bbox, image_shape),
    )


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


def merge_duplicate_bubble_regions(
    bubbles: Sequence[BubbleRegion],
    *,
    iou_threshold: float = 0.5,
    image_shape=None,
) -> list[BubbleRegion]:
    merged: list[BubbleRegion] = []

    for bubble in bubbles:
        match_index = None
        best_iou = 0.0
        for index, existing in enumerate(merged):
            current_iou = bbox_iou(existing.bbox, bubble.bbox)
            if current_iou >= iou_threshold and current_iou >= best_iou:
                best_iou = current_iou
                match_index = index

        if match_index is None:
            merged.append(bubble)
            continue

        existing = merged[match_index]
        preferred = bubble if bubble.score > existing.score else existing
        merge_shape = image_shape if image_shape is not None else _infer_image_shape_from_masks(existing.mask, bubble.mask)
        merged[match_index] = BubbleRegion(
            bbox=_union_bbox(existing.bbox, bubble.bbox),
            score=max(existing.score, bubble.score),
            class_id=preferred.class_id,
            mask=_merge_masks(
                existing.mask,
                bubble.mask,
                merge_shape,
            )
            if (
                merge_shape is not None
                and (existing.mask is not None or bubble.mask is not None)
            )
            else None,
            is_dark=bool(existing.is_dark or bubble.is_dark),
            fill_color=preferred.fill_color,
        )

    return merged


def merge_duplicate_text_regions(
    text_regions: Sequence[TextRegion],
    *,
    iou_threshold: float = 0.5,
    image_shape=None,
) -> list[TextRegion]:
    merged: list[TextRegion] = []

    for text_region in text_regions:
        match_index = None
        best_iou = 0.0
        for index, existing in enumerate(merged):
            current_iou = bbox_iou(existing.bbox, text_region.bbox)
            if current_iou >= iou_threshold and current_iou >= best_iou:
                best_iou = current_iou
                match_index = index

        if match_index is None:
            merged.append(text_region)
            continue

        existing = merged[match_index]
        preferred = (
            text_region
            if text_region.confidence > existing.confidence
            else existing
        )
        reading_orders = [
            value
            for value in (existing.reading_order, text_region.reading_order)
            if value is not None
        ]
        merge_shape = image_shape if image_shape is not None else _infer_image_shape_from_masks(existing.mask, text_region.mask)
        merged[match_index] = TextRegion(
            bbox=_union_bbox(existing.bbox, text_region.bbox),
            score=max(existing.score, text_region.score),
            class_id=preferred.class_id,
            mask=_merge_masks(
                existing.mask,
                text_region.mask,
                merge_shape,
            )
            if (
                merge_shape is not None
                and (existing.mask is not None or text_region.mask is not None)
            )
            else None,
            text=preferred.text if preferred.text else existing.text or text_region.text,
            confidence=max(existing.confidence, text_region.confidence),
            bubble_id=existing.bubble_id if existing.bubble_id is not None else text_region.bubble_id,
            reading_order=min(reading_orders) if reading_orders else None,
        )

    return merged


__all__ = [
    "bubble_region_to_crop_data",
    "clamp_bbox_to_image",
    "convert_legacy_detections_to_bubble_regions",
    "crop_bbox",
    "detect_dark_bubble_from_mask",
    "legacy_detection_to_bubble_region",
    "map_bbox_from_roi_to_page",
    "map_bubble_region_from_roi_to_page",
    "map_mask_from_roi_to_page",
    "map_text_region_from_roi_to_page",
    "mask_to_contour",
    "merge_duplicate_bubble_regions",
    "merge_duplicate_text_regions",
    "normalize_binary_mask",
    "process_bubble_with_mask",
    "union_text_regions_bbox",
]
