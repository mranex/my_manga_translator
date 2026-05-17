from __future__ import annotations

from .base import PageDetectionResult
from .pp_doclayout_v3 import get_pp_doclayout_v3_detector
from .runtime_utils import merge_duplicate_bubble_regions
from .yolov8_seg_bubble import get_yolov8_seg_bubble_detector


def detect_page_regions_layout_first(
    image,
    *,
    layout_detector=None,
    bubble_detector=None,
) -> PageDetectionResult:
    active_layout_detector = (
        layout_detector
        if layout_detector is not None
        else get_pp_doclayout_v3_detector()
    )
    active_bubble_detector = (
        bubble_detector
        if bubble_detector is not None
        else get_yolov8_seg_bubble_detector()
    )

    layout_regions = active_layout_detector.detect_layout_regions(image)
    raw_bubbles = active_bubble_detector.detect_segmented_bubble_regions(image)
    bubbles = merge_duplicate_bubble_regions(
        raw_bubbles,
        image_shape=image.shape,
    )

    return PageDetectionResult(
        bubbles=bubbles,
        layout_regions=layout_regions,
        method="pp_doclayout_v3+yolov8_seg_bubble",
        stats={
            "raw_bubbles": getattr(active_bubble_detector, "last_raw_bubble_count", len(raw_bubbles)),
            "merged_bubbles": getattr(active_bubble_detector, "last_merged_bubble_count", len(bubbles)),
            "raw_layout_regions": len(layout_regions),
        },
    )


__all__ = [
    "detect_page_regions_layout_first",
]
