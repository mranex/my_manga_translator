"""Resident OCR service."""

from __future__ import annotations

from .base_service import WorkerTaskService
from .resource_scheduler import ResourceScheduler
from mmt_gui.workers import _run_ocr_inference_task, _run_ocr_preparation_task


class OCRService(WorkerTaskService):
    def __init__(self, *, scheduler: ResourceScheduler, startup_options: dict | None = None) -> None:
        super().__init__(
            "ocr",
            scheduler=scheduler,
            startup_options=startup_options,
            action_callbacks={
                "prepare_page": _run_ocr_preparation_task,
                "prepare_pages": _run_ocr_preparation_task,
                "run_page": _run_ocr_inference_task,
                "run_pages": _run_ocr_inference_task,
                "run_selected_items": _run_ocr_inference_task,
            },
            lane_by_action={
                "prepare_page": "cpu_image_lane",
                "prepare_pages": "cpu_image_lane",
                "run_page": "network_lane",
                "run_pages": "network_lane",
                "run_selected_items": "network_lane",
            },
        )

    def on_initialize(self) -> None:
        self._emit_log("info", "OCR service is ready. PaddleOCR-VL server remains user-controlled.")


__all__ = ["OCRService"]
