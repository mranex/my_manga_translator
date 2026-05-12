from __future__ import annotations

from typing import Any

from detect_bubbles import detect_bubbles

from .base import PageDetectionResult, bubble_region_from_legacy_detection


def detect_page_regions(
    model_path: str,
    image: Any,
    enable_black_bubble: bool = True,
) -> PageDetectionResult:
    legacy_detections = detect_bubbles(
        model_path,
        image,
        enable_black_bubble=enable_black_bubble,
    )

    return PageDetectionResult(
        bubbles=[
            bubble_region_from_legacy_detection(detection)
            for detection in legacy_detections
        ],
        text_regions=[],
        method="legacy_yolo",
    )
