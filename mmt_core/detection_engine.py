"""Resident detection engine for preloaded detector ownership."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

from detectors import (
    PageDetectionResult,
    assign_text_regions_to_bubbles,
    get_comic_text_detector,
    get_pp_doclayout_v3_detector,
    get_yolov8_seg_bubble_detector,
)
from detectors.page_detector import merge_text_region_candidates
from detectors.pp_doclayout_v3 import layout_regions_to_text_regions
from detectors.runtime_utils import merge_duplicate_bubble_regions

from .runtime_diagnostics import write_runtime_diagnostic


Logger = Callable[[str], None] | None
StatusCallback = Callable[[str], None] | None


@dataclass(slots=True)
class DetectionEngine:
    """Owns the detector instances used by the desktop studio runtime."""

    bubble_detector: Any | None = None
    layout_detector: Any | None = None
    text_detector: Any | None = None
    disable_pp_layout_for_debug: bool = field(init=False, default=False)
    disable_yolo_for_debug: bool = field(init=False, default=False)
    disable_comic_text_for_debug: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.disable_pp_layout_for_debug = _env_flag("MMT_DISABLE_PP_LAYOUT")
        self.disable_yolo_for_debug = _env_flag("MMT_DISABLE_YOLO_BUBBLE")
        self.disable_comic_text_for_debug = _env_flag("MMT_DISABLE_COMIC_TEXT")

    def preload(self, *, logger: Logger = None, status_callback: StatusCallback = None) -> None:
        if self.disable_yolo_for_debug:
            _status(status_callback, "YOLO bubble detector disabled for debug.")
            _log(logger, "YOLO bubble detector disabled for debug.")
            self.bubble_detector = None
        else:
            _status(status_callback, "Loading YOLO bubble detector...")
            _log(logger, "Loading YOLO bubble detector...")
            self.bubble_detector = get_yolov8_seg_bubble_detector()
            if hasattr(self.bubble_detector, "load"):
                self.bubble_detector.load()

        if self.disable_pp_layout_for_debug:
            _status(status_callback, "PPLayout detector disabled for debug.")
            _log(logger, "PPLayout detector disabled for debug.")
            self.layout_detector = None
        else:
            _status(status_callback, "Loading PPLayout detector...")
            _log(logger, "Loading PPLayout detector...")
            self.layout_detector = get_pp_doclayout_v3_detector()
            if hasattr(self.layout_detector, "load"):
                self.layout_detector.load()

        if self.disable_comic_text_for_debug:
            _status(status_callback, "Comic text detector disabled for debug.")
            _log(logger, "Comic text detector disabled for debug.")
            self.text_detector = None
        else:
            _status(status_callback, "Loading comic/text detector...")
            _log(logger, "Loading comic/text detector...")
            self.text_detector = get_comic_text_detector()
            if hasattr(self.text_detector, "load"):
                self.text_detector.load()

    def is_ready(self) -> bool:
        return (
            (self.disable_yolo_for_debug or self.bubble_detector is not None)
            and (self.disable_pp_layout_for_debug or self.layout_detector is not None)
            and (self.disable_comic_text_for_debug or self.text_detector is not None)
        )

    def clear(self) -> None:
        self.bubble_detector = None
        self.layout_detector = None
        self.text_detector = None

    def detect_image(
        self,
        image: Any,
        *,
        logger: Logger = None,
        diagnostics_path: Path | str | None = None,
        page_name: str = "",
    ) -> PageDetectionResult:
        if not self.is_ready():
            raise RuntimeError("Detection models are not loaded. Restart the Detection service.")
        _log(logger, "Running resident detection inference...")
        _diag(diagnostics_path, page_name, "before_pp_layout", "before PPLayout detect_layout_regions")
        if self.disable_pp_layout_for_debug:
            layout_regions = []
            _log(logger, "PPLayout detector disabled for debug; returning no layout regions.")
            _diag(diagnostics_path, page_name, "after_pp_layout", "PPLayout disabled for debug")
        else:
            layout_regions = self.layout_detector.detect_layout_regions(image)
            _diag(diagnostics_path, page_name, "after_pp_layout", "after PPLayout detect_layout_regions")

        _diag(diagnostics_path, page_name, "before_layout_to_text", "before layout_regions_to_text_regions")
        pp_text_regions = (
            layout_regions_to_text_regions(layout_regions, image.shape) if layout_regions else []
        )
        _diag(diagnostics_path, page_name, "after_layout_to_text", "after layout_regions_to_text_regions")

        _diag(diagnostics_path, page_name, "before_yolo", "before YOLO detect_segmented_bubble_regions")
        if self.disable_yolo_for_debug:
            raw_bubbles = []
            _log(logger, "YOLO bubble detector disabled for debug; returning no bubbles.")
            _diag(diagnostics_path, page_name, "after_yolo", "YOLO disabled for debug")
        else:
            raw_bubbles = self.bubble_detector.detect_segmented_bubble_regions(image)
            _diag(diagnostics_path, page_name, "after_yolo", "after YOLO detect_segmented_bubble_regions")

        _diag(diagnostics_path, page_name, "before_merge_bubbles", "before merge_duplicate_bubble_regions")
        bubbles = merge_duplicate_bubble_regions(raw_bubbles, image_shape=image.shape)
        _diag(diagnostics_path, page_name, "after_merge_bubbles", "after merge_duplicate_bubble_regions")

        _diag(diagnostics_path, page_name, "before_comic_text", "before comic_text_detector.detect_text_regions")
        if self.disable_comic_text_for_debug:
            comic_text_regions = []
            _log(logger, "Comic text detector disabled for debug; returning no text regions.")
            _diag(diagnostics_path, page_name, "after_comic_text", "comic text detector disabled for debug")
        else:
            comic_text_regions = self.text_detector.detect_text_regions(image)
            _diag(diagnostics_path, page_name, "after_comic_text", "after comic_text_detector.detect_text_regions")

        _diag(diagnostics_path, page_name, "before_merge_text_regions", "before merge_text_region_candidates")
        merged_text_regions = merge_text_region_candidates(
            pp_text_regions,
            comic_text_regions,
            image_shape=image.shape,
        )
        _diag(diagnostics_path, page_name, "after_merge_text_regions", "after merge_text_region_candidates")

        _diag(diagnostics_path, page_name, "before_assign_text_regions", "before assign_text_regions_to_bubbles")
        matched_text_regions = assign_text_regions_to_bubbles(merged_text_regions, bubbles)
        _diag(diagnostics_path, page_name, "after_assign_text_regions", "after assign_text_regions_to_bubbles")

        active_method_parts = []
        if not self.disable_pp_layout_for_debug:
            active_method_parts.append("pp_doclayout_v3_text_source")
        if not self.disable_yolo_for_debug:
            active_method_parts.append("yolov8_seg_bubble")
        if not self.disable_comic_text_for_debug:
            active_method_parts.append("comic_text_detector")
        method_name = "+".join(active_method_parts) if active_method_parts else "debug_no_detectors"

        return PageDetectionResult(
            bubbles=bubbles,
            text_regions=matched_text_regions,
            layout_regions=layout_regions,
            method=method_name,
            stats={
                "raw_bubbles": getattr(self.bubble_detector, "last_raw_bubble_count", len(raw_bubbles))
                if self.bubble_detector is not None
                else 0,
                "merged_bubbles": getattr(self.bubble_detector, "last_merged_bubble_count", len(bubbles))
                if self.bubble_detector is not None
                else len(bubbles),
                "raw_pp_text_regions": len(pp_text_regions),
                "raw_comic_text_blocks": getattr(self.text_detector, "last_raw_comic_text_blocks", len(comic_text_regions))
                if self.text_detector is not None
                else 0,
                "raw_comic_line_regions": getattr(self.text_detector, "last_raw_comic_line_regions", 0)
                if self.text_detector is not None
                else 0,
                "comic_grouped_from_lines": getattr(self.text_detector, "last_comic_grouped_from_lines", 0)
                if self.text_detector is not None
                else 0,
                "raw_comic_text_regions": len(comic_text_regions),
                "merged_text_regions": len(merged_text_regions),
                "matched_text_regions": len(matched_text_regions),
            },
        )


def _log(logger: Logger, message: str) -> None:
    if logger is not None:
        logger(str(message or ""))


def _status(callback: StatusCallback, message: str) -> None:
    if callback is not None:
        callback(str(message or ""))


def _env_flag(name: str) -> bool:
    value = str(os.environ.get(name, "") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _diag(log_path: Path | str | None, page_name: str, step: str, message: str) -> None:
    if log_path is None:
        return
    write_runtime_diagnostic(
        message,
        log_path=log_path,
        service="detection",
        page=page_name,
        step=step,
    )


__all__ = ["DetectionEngine"]
