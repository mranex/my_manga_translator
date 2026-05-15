"""Resident export service."""

from __future__ import annotations

from .base_service import WorkerTaskService
from .resource_scheduler import ResourceScheduler
from mmt_gui.workers import _run_export_task


class ExportService(WorkerTaskService):
    def __init__(self, *, scheduler: ResourceScheduler, startup_options: dict | None = None) -> None:
        super().__init__(
            "export",
            scheduler=scheduler,
            startup_options=startup_options,
            action_callbacks={
                "export_current": _run_export_task,
                "export_selected": _run_export_task,
                "export_all": _run_export_task,
            },
            lane_by_action={
                "export_current": "cpu_image_lane",
                "export_selected": "cpu_image_lane",
                "export_all": "cpu_image_lane",
            },
        )

    def on_initialize(self) -> None:
        self._emit_log("info", "Export service is ready.")


__all__ = ["ExportService"]
