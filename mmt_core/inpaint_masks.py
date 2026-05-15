"""Conservative text-removal mask generation from cached OCR and detection JSON."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Sequence

from .image_io import ensure_path, load_image_grayscale


def clamp_bbox(bbox: Any, image_shape: Sequence[int]) -> tuple[int, int, int, int] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None

    height = int(image_shape[0])
    width = int(image_shape[1])
    try:
        x1, y1, x2, y2 = [int(value) for value in bbox[:4]]
    except Exception:
        return None

    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def expand_bbox(
    bbox: tuple[int, int, int, int] | None,
    image_shape: Sequence[int],
    padding: int,
) -> tuple[int, int, int, int] | None:
    if bbox is None:
        return None
    return clamp_bbox(
        (
            int(bbox[0]) - int(padding),
            int(bbox[1]) - int(padding),
            int(bbox[2]) + int(padding),
            int(bbox[3]) + int(padding),
        ),
        image_shape,
    )


def bbox_area(bbox: tuple[int, int, int, int] | None) -> int:
    if bbox is None:
        return 0
    return max(0, int(bbox[2]) - int(bbox[0])) * max(0, int(bbox[3]) - int(bbox[1]))


def build_text_mask_from_ocr(
    image_shape: Sequence[int],
    ocr_items: Sequence[dict[str, Any]],
    *,
    padding: int = 0,
    include_skipped_items: bool = True,
    min_box_area: int = 16,
    strict_mask: bool = True,
) -> tuple[Any, list[tuple[int, int, int, int]], int]:
    """Build a binary text-removal mask from OCR cache items."""

    np_module = _require_numpy()
    mask = np_module.zeros((int(image_shape[0]), int(image_shape[1])), dtype=np_module.uint8)
    valid_boxes: list[tuple[int, int, int, int]] = []

    for item in ocr_items or []:
        if not isinstance(item, dict):
            continue
        if bool(item.get("excluded", False)):
            continue

        status = str(item.get("status", "") or "").strip().lower()
        if status == "skipped" and not include_skipped_items:
            continue

        bbox = item.get("ocr_bbox") or item.get("bbox")
        resolved_bbox = clamp_bbox(bbox, image_shape)
        if resolved_bbox is not None and int(padding) > 0:
            resolved_bbox = expand_bbox(resolved_bbox, image_shape, padding)
        if resolved_bbox is None:
            continue
        if bbox_area(resolved_bbox) < int(min_box_area):
            continue

        x1, y1, x2, y2 = resolved_bbox
        mask[y1:y2, x1:x2] = 255
        valid_boxes.append(resolved_bbox)

    if not strict_mask or int(padding) > 0:
        mask = _close_mask(mask, 3 if padding <= 4 else 5)
    mask = np_module.where(mask > 0, 255, 0).astype(np_module.uint8)
    return mask, valid_boxes, int(np_module.count_nonzero(mask))


def build_text_mask_from_canon_items(
    image_shape: Sequence[int],
    canon_items: Sequence[dict[str, Any]],
    *,
    padding: int = 0,
    min_box_area: int = 16,
    strict_mask: bool = True,
    return_stats: bool = False,
) -> tuple[Any, list[tuple[int, int, int, int]], int] | tuple[Any, list[tuple[int, int, int, int]], int, dict[str, int]]:
    """Build a binary text-removal mask from active canon items.

    Inpaint text removal follows the current OCR target geometry:
    - primary: canon_state.items[].ocr_bbox
    - fallback: canon_state.items[].bbox only if ocr_bbox is missing/invalid

    text_mask_bboxes remains legacy/debug geometry and does not override ocr_bbox.
    """

    return build_text_mask_from_canon_ocr_bboxes(
        image_shape,
        canon_items,
        padding=padding,
        min_box_area=min_box_area,
        strict_mask=strict_mask,
        return_stats=return_stats,
    )


def build_text_mask_from_canon_ocr_bboxes(
    image_shape: Sequence[int],
    canon_items: Sequence[dict[str, Any]],
    *,
    padding: int = 0,
    min_box_area: int = 16,
    strict_mask: bool = True,
    return_stats: bool = False,
) -> tuple[Any, list[tuple[int, int, int, int]], int] | tuple[Any, list[tuple[int, int, int, int]], int, dict[str, int]]:
    """Build a binary text-removal mask from canon OCR targets."""

    np_module = _require_numpy()
    mask = np_module.zeros((int(image_shape[0]), int(image_shape[1])), dtype=np_module.uint8)
    valid_boxes: list[tuple[int, int, int, int]] = []
    active_item_count = 0
    used_ocr_bboxes = 0
    fallback_to_bbox_count = 0
    skipped_invalid_bbox_count = 0

    for item in canon_items or []:
        if not isinstance(item, dict):
            continue
        if not bool(item.get("enabled", True)):
            continue
        active_item_count += 1

        used_fallback = False
        raw_bbox = item.get("ocr_bbox")
        resolved_bbox = clamp_bbox(raw_bbox, image_shape)
        if resolved_bbox is None:
            raw_bbox = item.get("bbox")
            resolved_bbox = clamp_bbox(raw_bbox, image_shape)
            used_fallback = resolved_bbox is not None

        if resolved_bbox is not None and int(padding) > 0:
            resolved_bbox = expand_bbox(resolved_bbox, image_shape, padding)
        if resolved_bbox is None or bbox_area(resolved_bbox) < int(min_box_area):
            skipped_invalid_bbox_count += 1
            continue

        x1, y1, x2, y2 = resolved_bbox
        mask[y1:y2, x1:x2] = 255
        valid_boxes.append(resolved_bbox)
        used_ocr_bboxes += 1
        if used_fallback:
            fallback_to_bbox_count += 1

    if not strict_mask or int(padding) > 0:
        mask = _close_mask(mask, 3 if padding <= 4 else 5)
    mask = np_module.where(mask > 0, 255, 0).astype(np_module.uint8)
    masked_pixel_count = int(np_module.count_nonzero(mask))
    if not return_stats:
        return mask, valid_boxes, masked_pixel_count
    return mask, valid_boxes, masked_pixel_count, {
        "active_item_count": active_item_count,
        "used_ocr_bboxes": used_ocr_bboxes,
        "fallback_to_bbox_count": fallback_to_bbox_count,
        "skipped_invalid_bbox_count": skipped_invalid_bbox_count,
        "used_text_mask_bboxes": used_ocr_bboxes,
        "skipped_items_without_text_mask": skipped_invalid_bbox_count,
    }


def build_bubble_guidance_mask(
    image_shape: Sequence[int],
    detection_data: dict[str, Any] | None,
    *,
    project_root: Path | str | None = None,
) -> Any | None:
    """Build a combined bubble guidance mask from cached detection bubble masks or bboxes."""

    if not isinstance(detection_data, dict):
        return None

    np_module = _require_numpy()
    mask = np_module.zeros((int(image_shape[0]), int(image_shape[1])), dtype=np_module.uint8)
    root_path = ensure_path(project_root) if project_root is not None else None
    has_any_content = False
    allowed_bubble_ids: set[int] | None = None
    canon_state = detection_data.get("canon_state")
    if isinstance(canon_state, dict):
        allowed_bubble_ids = {
            int(detector_refs.get("bubble_id"))
            for item in canon_state.get("items", [])
            if isinstance(item, dict)
            and bool(item.get("enabled", True))
            and str(item.get("kind", "") or "") == "bubble"
            for detector_refs in [item.get("detector_refs", {})]
            if isinstance(detector_refs, dict)
            and detector_refs.get("bubble_id") not in (None, "")
        }

    for bubble in detection_data.get("bubbles", []):
        if not isinstance(bubble, dict):
            continue
        bubble_id = bubble.get("id")
        if allowed_bubble_ids is not None:
            try:
                normalized_bubble_id = int(bubble_id)
            except Exception:
                continue
            if normalized_bubble_id not in allowed_bubble_ids:
                continue

        bubble_mask_relative = str(bubble.get("mask_path", "") or "").strip()
        if bubble_mask_relative and root_path is not None:
            bubble_mask_file = root_path / bubble_mask_relative
            if bubble_mask_file.exists():
                try:
                    bubble_mask = load_image_grayscale(bubble_mask_file)
                except Exception:
                    bubble_mask = None
                if bubble_mask is not None:
                    binary_mask = np_module.where(bubble_mask > 0, 255, 0).astype(np_module.uint8)
                    if binary_mask.shape == mask.shape:
                        mask[:] = np_module.maximum(mask, binary_mask)
                        has_any_content = True
                        continue

        bubble_bbox = clamp_bbox(bubble.get("bbox"), image_shape)
        if bubble_bbox is None:
            continue
        x1, y1, x2, y2 = bubble_bbox
        mask[y1:y2, x1:x2] = 255
        has_any_content = True

    if not has_any_content:
        return None
    return np_module.where(mask > 0, 255, 0).astype(np_module.uint8)


def build_preview_mask(text_mask: Any, bubble_mask: Any | None = None) -> Any:
    np_module = _require_numpy()
    preview = np_module.where(text_mask > 0, 255, 0).astype(np_module.uint8)
    if bubble_mask is not None:
        preview = np_module.maximum(preview, np_module.where(bubble_mask > 0, 96, 0).astype(np_module.uint8))
    return preview


def build_crop_windows_from_boxes(
    boxes: Sequence[Sequence[int]],
    image_shape: Sequence[int],
) -> list[tuple[int, int, int, int]]:
    if not boxes:
        return []
    try:
        strategy_module = importlib.import_module("inpainting.strategy")
        crop_windows_from_bboxes = strategy_module.crop_windows_from_bboxes
    except Exception:
        return [box for box in boxes if clamp_bbox(box, image_shape) is not None]
    return list(crop_windows_from_bboxes(boxes, image_shape))


def _require_numpy():
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("NumPy is required for inpaint mask generation.") from exc
    return np


def _close_mask(mask, kernel_size: int) -> Any:
    np_module = _require_numpy()
    if kernel_size <= 1:
        return mask

    try:
        cv2 = importlib.import_module("cv2")
    except Exception:
        cv2 = None

    if cv2 is not None:
        kernel = np_module.ones((kernel_size, kernel_size), dtype=np_module.uint8)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    radius = max(1, kernel_size // 2)
    padded = np_module.pad(mask, radius, mode="constant", constant_values=0)
    out = np_module.full_like(mask, 255)
    for offset_y in range(radius * 2 + 1):
        for offset_x in range(radius * 2 + 1):
            out = np_module.minimum(
                out,
                padded[offset_y:offset_y + mask.shape[0], offset_x:offset_x + mask.shape[1]],
            )
    return out.astype(np_module.uint8)


__all__ = [
    "bbox_area",
    "build_bubble_guidance_mask",
    "build_text_mask_from_canon_ocr_bboxes",
    "build_text_mask_from_canon_items",
    "build_crop_windows_from_boxes",
    "build_preview_mask",
    "build_text_mask_from_ocr",
    "clamp_bbox",
    "expand_bbox",
]
