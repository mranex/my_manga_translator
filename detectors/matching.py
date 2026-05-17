from __future__ import annotations

from math import floor
from typing import Sequence


BBox = tuple[int, int, int, int]
Point = tuple[float, float]


def bbox_center(bbox: BBox) -> Point:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def point_in_bbox(point: Point, bbox: BBox, padding: int = 0) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    return (
        (x1 - padding) <= x <= (x2 + padding)
        and (y1 - padding) <= y <= (y2 + padding)
    )


def point_in_mask(point: Point, mask) -> bool:
    if mask is None:
        return False

    x, y = point
    x_idx = floor(x)
    y_idx = floor(y)

    shape = getattr(mask, "shape", None)
    if shape is not None and len(shape) >= 2:
        height, width = int(shape[0]), int(shape[1])
    else:
        try:
            height = len(mask)
            width = len(mask[0]) if height > 0 else 0
        except (TypeError, IndexError, KeyError):
            return False

    if x_idx < 0 or y_idx < 0 or x_idx >= width or y_idx >= height:
        return False

    try:
        value = mask[y_idx, x_idx]
    except (TypeError, IndexError, KeyError):
        try:
            value = mask[y_idx][x_idx]
        except (TypeError, IndexError, KeyError):
            return False

    try:
        positive = value > 0
    except TypeError:
        return False

    if hasattr(positive, "any"):
        return bool(positive.any())
    return bool(positive)


def bbox_area(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    return float(width * height)


def bbox_intersection_area(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    width = max(0, inter_x2 - inter_x1)
    height = max(0, inter_y2 - inter_y1)
    return float(width * height)


def bbox_iou(a: BBox, b: BBox) -> float:
    intersection = bbox_intersection_area(a, b)
    if intersection <= 0:
        return 0.0

    union = bbox_area(a) + bbox_area(b) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


__all__ = [
    "bbox_area",
    "bbox_center",
    "bbox_intersection_area",
    "bbox_iou",
    "point_in_bbox",
    "point_in_mask",
]
