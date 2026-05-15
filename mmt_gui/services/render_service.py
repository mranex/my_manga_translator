"""Resident render service."""

from __future__ import annotations

from .base_service import WorkerTaskService
from .resource_scheduler import ResourceScheduler
from mmt_gui.workers import _run_render_preparation_task, _run_render_task


class RenderService(WorkerTaskService):
    def __init__(self, *, scheduler: ResourceScheduler, startup_options: dict | None = None) -> None:
        super().__init__(
            "render",
            scheduler=scheduler,
            startup_options=startup_options,
            action_callbacks={
                "prepare_page": _run_render_preparation_task,
                "prepare_pages": _run_render_preparation_task,
                "render_page": _run_render_task,
                "render_pages": _run_render_task,
            },
            lane_by_action={
                "prepare_page": "cpu_image_lane",
                "prepare_pages": "cpu_image_lane",
                "render_page": "cpu_image_lane",
                "render_pages": "cpu_image_lane",
            },
        )

    def on_initialize(self) -> None:
        self._emit_log("info", "Render service is ready.")


__all__ = ["RenderService"]
