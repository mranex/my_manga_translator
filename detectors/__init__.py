from __future__ import annotations

from typing import Any

from .base import BubbleRegion, LayoutRegion, PageDetectionResult, Region
from .matching import (
    bbox_area,
    bbox_center,
    bbox_intersection_area,
    bbox_iou,
    point_in_bbox,
    point_in_mask,
)
from .pp_doclayout_v3 import PPDocLayoutV3Detector


def detect_segmented_page_regions(
    image: Any,
    *,
    model_path: str | None = None,
    confidence: float = 0.25,
    iou: float = 0.5,
    device: str | None = None,
) -> PageDetectionResult:
    from .yolov8_seg_bubble import detect_page_regions as _detect_page_regions

    return _detect_page_regions(
        image,
        model_path=model_path,
        confidence=confidence,
        iou=iou,
        device=device,
    )


def detect_segmented_bubble_regions(
    image: Any,
    *,
    model_path: str | None = None,
    confidence: float = 0.25,
    iou: float = 0.5,
    device: str | None = None,
) -> list[BubbleRegion]:
    from .yolov8_seg_bubble import (
        detect_segmented_bubble_regions as _detect_segmented_bubble_regions,
    )

    return _detect_segmented_bubble_regions(
        image,
        model_path=model_path,
        confidence=confidence,
        iou=iou,
        device=device,
    )


def get_yolov8_seg_bubble_detector(
    model_path: str | None = None,
    confidence: float = 0.25,
    iou: float = 0.5,
    device: str | None = None,
):
    from .yolov8_seg_bubble import (
        get_yolov8_seg_bubble_detector as _get_yolov8_seg_bubble_detector,
    )

    return _get_yolov8_seg_bubble_detector(
        model_path=model_path,
        confidence=confidence,
        iou=iou,
        device=device,
    )


def get_pp_doclayout_v3_detector(
    model_id: str = "PaddlePaddle/PP-DocLayoutV3_safetensors",
    cache_dir: str = "model/pp_doclayout_v3",
    confidence_threshold: float = 0.25,
    device: str | None = None,
    roi_padding: int = 24,
    min_region_area: int = 64,
    max_full_page_region_ratio: float = 0.92,
):
    from .pp_doclayout_v3 import (
        get_pp_doclayout_v3_detector as _get_pp_doclayout_v3_detector,
    )

    return _get_pp_doclayout_v3_detector(
        model_id=model_id,
        cache_dir=cache_dir,
        confidence_threshold=confidence_threshold,
        device=device,
        roi_padding=roi_padding,
        min_region_area=min_region_area,
        max_full_page_region_ratio=max_full_page_region_ratio,
    )


def detect_page_regions_layout_first(
    image: Any,
    *,
    layout_detector=None,
    bubble_detector=None,
) -> PageDetectionResult:
    from .page_detector import (
        detect_page_regions_layout_first as _detect_page_regions_layout_first,
    )

    return _detect_page_regions_layout_first(
        image,
        layout_detector=layout_detector,
        bubble_detector=bubble_detector,
    )


__all__ = [
    "Region",
    "BubbleRegion",
    "LayoutRegion",
    "PageDetectionResult",
    "detect_segmented_bubble_regions",
    "detect_segmented_page_regions",
    "detect_page_regions_layout_first",
    "bbox_area",
    "bbox_center",
    "bbox_intersection_area",
    "bbox_iou",
    "point_in_bbox",
    "point_in_mask",
    "get_pp_doclayout_v3_detector",
    "get_yolov8_seg_bubble_detector",
    "PPDocLayoutV3Detector",
]
