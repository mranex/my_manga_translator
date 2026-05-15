"""Resident detection service."""

from __future__ import annotations

from .base_service import WorkerTaskService
from .resource_scheduler import ResourceScheduler
from mmt_gui.workers import _run_detection_task


class DetectionService(WorkerTaskService):
    def __init__(self, *, scheduler: ResourceScheduler, startup_options: dict | None = None) -> None:
        super().__init__(
            "detection",
            scheduler=scheduler,
            startup_options=startup_options,
            action_callbacks={
                "detect_page": _run_detection_task,
                "detect_pages": _run_detection_task,
            },
            lane_by_action={
                "detect_page": "gpu_model_lane",
                "detect_pages": "gpu_model_lane",
            },
        )

    def on_initialize(self) -> None:
        if not bool(self.startup_options.get("preload_detection", True)):
            self._emit_log("info", "Detection preload disabled; service is ready.")
            return

        from detectors import (
            get_comic_text_detector,
            get_pp_doclayout_v3_detector,
            get_yolov8_seg_bubble_detector,
        )

        self._emit_status("loading", "Preloading detection models...")
        bubble_detector = get_yolov8_seg_bubble_detector()
        if hasattr(bubble_detector, "load"):
            bubble_detector.load()
        layout_detector = get_pp_doclayout_v3_detector()
        if hasattr(layout_detector, "load"):
            layout_detector.load()
        text_detector = get_comic_text_detector()
        if hasattr(text_detector, "load"):
            text_detector.load()
        self._emit_log("info", "Detection models are ready.")


__all__ = ["DetectionService"]
