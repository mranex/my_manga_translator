"""Resident detection engine for preloaded detector ownership."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import os
from typing import Any

from detectors import (
    PageDetectionResult,
    get_pp_doclayout_v3_detector,
    get_yolov8_seg_bubble_detector,
)
from detectors.page_detector import detect_page_regions_layout_first
from .crash_logging import write_crash_breadcrumb


Logger = Callable[[str], None] | None
StatusCallback = Callable[[str], None] | None


@dataclass(slots=True)
class DetectionEngine:
    """Owns the detector instances used by the desktop studio runtime."""

    bubble_detector: Any | None = None
    layout_detector: Any | None = None
    disable_pp_layout_for_debug: bool = field(init=False, default=False)
    disable_yolo_for_debug: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.disable_pp_layout_for_debug = _env_flag("MMT_DISABLE_PP_LAYOUT")
        self.disable_yolo_for_debug = _env_flag("MMT_DISABLE_YOLO_BUBBLE")

    def preload(self, *, logger: Logger = None, status_callback: StatusCallback = None) -> None:
        if self.disable_yolo_for_debug:
            _status(status_callback, "MMT_DISABLE_YOLO_BUBBLE is set; YOLO bubble detector will remain unavailable.")
            _log(logger, "MMT_DISABLE_YOLO_BUBBLE is set; resident detection requires this detector to be loaded.")
            self.bubble_detector = None
        else:
            _status(status_callback, "Loading YOLO bubble detector...")
            _log(logger, "Loading YOLO bubble detector...")
            self.bubble_detector = get_yolov8_seg_bubble_detector()
            if hasattr(self.bubble_detector, "load"):
                self.bubble_detector.load()

        if self.disable_pp_layout_for_debug:
            _status(status_callback, "MMT_DISABLE_PP_LAYOUT is set; PPLayout detector will remain unavailable.")
            _log(logger, "MMT_DISABLE_PP_LAYOUT is set; resident detection requires this detector to be loaded.")
            self.layout_detector = None
        else:
            _status(status_callback, "Loading PPLayout detector...")
            _log(logger, "Loading PPLayout detector...")
            self.layout_detector = get_pp_doclayout_v3_detector()
            if hasattr(self.layout_detector, "load"):
                self.layout_detector.load()

    def is_ready(self) -> bool:
        return not self.missing_detectors()

    def missing_detectors(self) -> list[str]:
        missing: list[str] = []
        if self.layout_detector is None:
            missing.append("PPLayout")
        if self.bubble_detector is None:
            missing.append("YOLO bubble")
        return missing

    def clear(self) -> None:
        self.bubble_detector = None
        self.layout_detector = None

    def detect_image(
        self,
        image: Any,
        *,
        logger: Logger = None,
        diagnostics_path: Any | None = None,
        page_name: str = "",
    ) -> PageDetectionResult:
        current_page = page_name
        write_crash_breadcrumb(
            "DetectionEngine.detect_image entered",
            page=current_page,
            has_layout_detector=self.layout_detector is not None,
            has_bubble_detector=self.bubble_detector is not None,
        )
        del diagnostics_path, page_name
        missing = self.missing_detectors()
        if missing:
            missing_text = ", ".join(missing)
            raise RuntimeError(
                "Detection models are not loaded. Restart the Detection service. "
                f"Missing resident detectors: {missing_text}. "
                "Active detection does not support getter fallback or partial debug-disable pipelines."
            )
        _log(logger, "Running resident detection inference...")
        write_crash_breadcrumb("before detect_page_regions_layout_first", page=current_page)
        result = detect_page_regions_layout_first(
            image,
            layout_detector=self.layout_detector,
            bubble_detector=self.bubble_detector,
        )
        write_crash_breadcrumb("after detect_page_regions_layout_first", page=current_page)
        return result


def _log(logger: Logger, message: str) -> None:
    if logger is not None:
        logger(str(message or ""))


def _status(callback: StatusCallback, message: str) -> None:
    if callback is not None:
        callback(str(message or ""))


def _env_flag(name: str) -> bool:
    value = str(os.environ.get(name, "") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


__all__ = ["DetectionEngine"]
