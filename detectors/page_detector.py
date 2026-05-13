from __future__ import annotations

from .base import PageDetectionResult, TextRegion
from .comic_text_detector import get_comic_text_detector
from .matching import assign_text_regions_to_bubbles, bbox_iou
from .pp_doclayout_v3 import (
    get_pp_doclayout_v3_detector,
    layout_regions_to_text_regions,
)
from .runtime_utils import merge_duplicate_text_regions
from .yolov8_seg_bubble import get_yolov8_seg_bubble_detector


def merge_text_region_candidates(
    pp_text_regions,
    comic_text_regions,
    *,
    image_shape,
    iou_threshold: float = 0.5,
) -> list[TextRegion]:
    merged_comic = merge_duplicate_text_regions(
        comic_text_regions,
        iou_threshold=iou_threshold,
        image_shape=image_shape,
    )

    merged_regions: list[TextRegion] = list(merged_comic)
    for pp_region in pp_text_regions:
        if any(bbox_iou(pp_region.bbox, comic_region.bbox) >= iou_threshold for comic_region in merged_comic):
            continue
        merged_regions.append(pp_region)

    return [
        region
        for _, region in sorted(
            enumerate(merged_regions),
            key=lambda item: (
                item[1].reading_order if item[1].reading_order is not None else 10**9,
                item[1].bbox[1],
                item[1].bbox[0],
                item[0],
            ),
        )
    ]


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
    pp_text_regions = layout_regions_to_text_regions(
        layout_regions,
        image.shape,
    )
    bubbles = active_bubble_detector.detect_segmented_bubble_regions(image)
    comic_text_regions = active_text_detector.detect_text_regions(image)
    merged_text_regions = merge_text_region_candidates(
        pp_text_regions,
        comic_text_regions,
        image_shape=image.shape,
    )
    matched_text_regions = assign_text_regions_to_bubbles(
        merged_text_regions,
        bubbles,
    )

    return PageDetectionResult(
        bubbles=bubbles,
        text_regions=matched_text_regions,
        layout_regions=layout_regions,
        method="pp_doclayout_v3_text_source+yolov8_seg_bubble+comic_text_detector",
    )


__all__ = [
    "detect_page_regions_layout_first",
    "merge_text_region_candidates",
]
