from __future__ import annotations

from .base import PageDetectionResult
from .comic_text_detector import get_comic_text_detector
from .matching import assign_text_regions_to_bubbles
from .pp_doclayout_v3 import build_layout_rois, get_pp_doclayout_v3_detector
from .yolov8_seg_bubble import get_yolov8_seg_bubble_detector


def detect_page_regions_layout_first(
    image,
    *,
    layout_detector=None,
    bubble_detector=None,
    text_detector=None,
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
    active_text_detector = (
        text_detector if text_detector is not None else get_comic_text_detector()
    )

    layout_regions = active_layout_detector.detect_layout_regions(image)
    layout_rois = build_layout_rois(
        layout_regions,
        image.shape,
        padding=getattr(active_layout_detector, "roi_padding", 24),
    )

    bubbles = active_bubble_detector.detect_bubble_regions_in_rois(
        image,
        layout_rois,
    )
    text_regions = active_text_detector.detect_text_regions_in_rois(
        image,
        layout_rois,
    )
    matched_text_regions = assign_text_regions_to_bubbles(
        text_regions,
        bubbles,
    )

    return PageDetectionResult(
        bubbles=bubbles,
        text_regions=matched_text_regions,
        layout_regions=layout_regions,
        method="pp_doclayout_v3+yolov8_seg_bubble+comic_text_detector",
    )


__all__ = ["detect_page_regions_layout_first"]
