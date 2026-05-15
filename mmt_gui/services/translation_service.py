"""Resident translation service."""

from __future__ import annotations

from .base_service import WorkerTaskService
from .resource_scheduler import ResourceScheduler
from mmt_gui.workers import _run_translation_initialization_task, _run_translation_task


class TranslationService(WorkerTaskService):
    def __init__(self, *, scheduler: ResourceScheduler, startup_options: dict | None = None) -> None:
        super().__init__(
            "translation",
            scheduler=scheduler,
            startup_options=startup_options,
            action_callbacks={
                "initialize_page": _run_translation_initialization_task,
                "initialize_pages": _run_translation_initialization_task,
                "translate_page": _run_translation_task,
                "translate_pages": _run_translation_task,
                "translate_selected_items": _run_translation_task,
            },
            lane_by_action={
                "initialize_page": "cpu_image_lane",
                "initialize_pages": "cpu_image_lane",
                "translate_page": "network_lane",
                "translate_pages": "network_lane",
                "translate_selected_items": "network_lane",
            },
        )

    def on_initialize(self) -> None:
        self._emit_log("info", "Translation service is ready.")


__all__ = ["TranslationService"]
