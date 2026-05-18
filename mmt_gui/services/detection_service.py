"""Qt-facing detection service backed by a resident Python runtime thread."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from mmt_core import DetectionConfig
from mmt_core.crash_logging import write_crash_breadcrumb
from mmt_gui.workers import DetectionTask, MangaDetectorTask

from .base_service import BaseService, ServiceCanceledError, WorkerSignalsBridge
from .detection_runtime import (
    DetectionRuntimeBusyError,
    DetectionRuntimeCallbacks,
    DetectionRuntimeRequest,
    DetectionRuntimeThread,
)
from .models import ServiceCommand


class DetectionService(BaseService):
    def __init__(self, *, scheduler: Any | None = None, startup_options: dict | None = None) -> None:
        super().__init__("detection", scheduler=scheduler, startup_options=startup_options)
        workspace_root_text = str((startup_options or {}).get("workspace_root", "") or "").strip()
        self._workspace_root = Path(workspace_root_text).resolve() if workspace_root_text else Path.cwd()
        self._runtime: DetectionRuntimeThread | None = None

    def on_initialize(self) -> None:
        if not bool(self.startup_options.get("preload_detection", True)):
            raise RuntimeError("Detection preload is disabled. Enable startup preload or reload the service.")
        if str(os.environ.get("QT_QPA_PLATFORM", "") or "").strip().lower() in {"offscreen", "minimal"}:
            raise RuntimeError("Detection preload is not available in headless Qt mode.")

        self._runtime = DetectionRuntimeThread(
            workspace_root=self._workspace_root,
            startup_detection_config=DetectionConfig.from_value(
                self.startup_options.get("detection_config", {})
            ),
            logger=lambda message: self._emit_log("info", message),
            status_callback=lambda message: self._emit_status("loading", message),
        )
        self._runtime.start()
        if not self._runtime.wait_until_ready(timeout=300.0):
            self._runtime.stop()
            self._runtime = None
            raise RuntimeError("Detection runtime did not become ready. Check crash logs for the last breadcrumb.")
        if not self._runtime.is_ready():
            error_message = self._runtime.error_message() or "Detection runtime failed to initialize."
            self._runtime.stop()
            self._runtime = None
            raise RuntimeError(error_message)
        self._emit_log("info", "Detection runtime thread is ready.")

    def on_restart(self) -> None:
        runtime = self._runtime
        self._runtime = None
        if runtime is not None:
            runtime.stop()
        self.on_initialize()

    def on_shutdown(self) -> None:
        runtime = self._runtime
        self._runtime = None
        if runtime is not None:
            runtime.stop()

    def execute_command(self, command: ServiceCommand, bridge: WorkerSignalsBridge) -> Any:
        task = command.task
        if isinstance(task, DetectionTask):
            write_crash_breadcrumb(
                "DetectionService command received",
                command_id=command.command_id,
                action=command.action,
                page_total=len(task.image_paths),
            )
            return self._run_detection_task(command, task, bridge)
        if isinstance(task, MangaDetectorTask):
            return self._run_manga_detector_task(command, task, bridge)
        if command.action == "reload_models":
            self.on_restart()
            return {"reloaded": True}
        if command.action == "status":
            return self._status_payload()
        raise RuntimeError(f"Detection service does not support action '{command.action}'.")

    def _run_detection_task(
        self,
        command: ServiceCommand,
        task: DetectionTask,
        bridge: WorkerSignalsBridge,
    ) -> Any:
        if not task.image_paths:
            raise ValueError("No source images were provided for detection.")

        runtime = self._runtime
        if runtime is None or not runtime.is_ready():
            raise RuntimeError("Detection runtime is not ready. Restart the Detection service.")
        if runtime.is_busy():
            write_crash_breadcrumb(
                "DetectionService runtime busy",
                level="warning",
                command_id=command.command_id,
                action=command.action,
            )
            raise RuntimeError("Detection runtime is busy.")

        request = DetectionRuntimeRequest(
            command_id=command.command_id,
            action=command.action,
            image_paths=[Path(path) for path in task.image_paths],
            detection_cache_dir=Path(task.detection_cache_dir),
            masks_cache_dir=Path(task.masks_cache_dir),
            force=bool(task.force),
            cancel_token=command.cancel_token,
            callbacks=DetectionRuntimeCallbacks(
                progress=bridge.progress.emit,
                message=bridge.message.emit,
                event=bridge.event.emit,
            ),
            workspace_root=self._workspace_root,
            detection_config=DetectionConfig.from_value(task.config),
        )

        write_crash_breadcrumb(
            "DetectionService before submit to DetectionRuntimeThread",
            command_id=command.command_id,
            action=command.action,
            page_total=len(request.image_paths),
        )
        try:
            outcome = runtime.submit_and_wait(
                request,
                accepted_callback=lambda: write_crash_breadcrumb(
                    "DetectionService after submit accepted",
                    command_id=command.command_id,
                    action=command.action,
                ),
            )
        except DetectionRuntimeBusyError as exc:
            write_crash_breadcrumb(
                "DetectionService runtime busy",
                level="warning",
                command_id=command.command_id,
                error=str(exc),
            )
            raise RuntimeError(str(exc)) from exc

        if outcome.canceled:
            write_crash_breadcrumb(
                "DetectionService runtime failed",
                level="warning",
                command_id=command.command_id,
                error=outcome.error or "Detection canceled.",
            )
            raise ServiceCanceledError(outcome.error or "Detection canceled.")
        if outcome.error:
            write_crash_breadcrumb(
                "DetectionService runtime failed",
                level="critical",
                command_id=command.command_id,
                error=outcome.error,
            )
            raise RuntimeError(outcome.error)

        write_crash_breadcrumb(
            "DetectionService runtime done",
            command_id=command.command_id,
            action=command.action,
        )
        return outcome.result

    def _run_manga_detector_task(
        self,
        command: ServiceCommand,
        task: MangaDetectorTask,
        bridge: WorkerSignalsBridge,
    ) -> Any:
        runtime = self._runtime
        if runtime is None or not runtime.is_ready():
            raise RuntimeError("Detection runtime is not ready. Restart the Detection service.")
        if runtime.is_busy():
            raise RuntimeError("Detection runtime is busy.")

        request = DetectionRuntimeRequest(
            command_id=command.command_id,
            action=command.action,
            image_paths=[],
            detection_cache_dir=None,
            masks_cache_dir=None,
            force=False,
            cancel_token=command.cancel_token,
            callbacks=DetectionRuntimeCallbacks(
                progress=bridge.progress.emit,
                message=bridge.message.emit,
                event=bridge.event.emit,
            ),
            workspace_root=self._workspace_root,
            detection_config=DetectionConfig.from_value(task.config),
        )
        try:
            outcome = runtime.submit_and_wait(request)
        except DetectionRuntimeBusyError as exc:
            raise RuntimeError(str(exc)) from exc
        if outcome.canceled:
            raise ServiceCanceledError(outcome.error or "Manga detector command canceled.")
        if outcome.error:
            raise RuntimeError(outcome.error)
        return outcome.result

    def _status_payload(self) -> dict[str, Any]:
        runtime = self._runtime
        return {
            "ready": bool(runtime is not None and runtime.is_ready()),
            "busy": bool(runtime is not None and runtime.is_busy()),
            "error": "" if runtime is None else runtime.error_message(),
        }


__all__ = ["DetectionService"]
