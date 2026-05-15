"""Resident detection service with thread-owned preloaded detector models."""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any

from mmt_core.detection_engine import DetectionEngine
from mmt_core.detection_io import save_detection_result
from mmt_core.image_io import ensure_path, load_image_bgr
from mmt_core.runtime_diagnostics import resolve_runtime_diagnostics_path, write_runtime_diagnostic
from mmt_gui.workers import DetectionPageResult, DetectionTask, DetectionWorkerResult

from .base_service import BaseService, WorkerSignalsBridge
from .models import ServiceCommand


class DetectionService(BaseService):
    def __init__(self, *, scheduler: Any | None = None, startup_options: dict | None = None) -> None:
        super().__init__("detection", scheduler=scheduler, startup_options=startup_options)
        self._engine = DetectionEngine()
        self._workspace_root = Path(str((startup_options or {}).get("workspace_root", "") or "")).resolve() if str((startup_options or {}).get("workspace_root", "") or "").strip() else Path.cwd()

    def on_initialize(self) -> None:
        if not bool(self.startup_options.get("preload_detection", True)):
            raise RuntimeError("Detection preload is disabled. Enable startup preload or reload the service.")
        if str(os.environ.get("QT_QPA_PLATFORM", "") or "").strip().lower() in {"offscreen", "minimal"}:
            raise RuntimeError("Detection preload is not available in headless Qt mode.")
        self._engine.preload(
            logger=lambda message: self._emit_log("info", message),
            status_callback=lambda message: self._emit_status("loading", message),
        )
        if not self._engine.is_ready():
            missing = ", ".join(self._engine.missing_detectors()) or "unknown detectors"
            raise RuntimeError(
                "Detection resident models failed to load. "
                f"Missing resident detectors: {missing}. "
                "Active detection requires PPLayout, YOLO bubble, and comic/text detectors."
            )
        self._emit_log("info", "Detection models are ready.")

    def on_restart(self) -> None:
        self._engine.clear()
        gc.collect()
        self._engine = DetectionEngine()
        self.on_initialize()

    def on_shutdown(self) -> None:
        self._engine.clear()
        try:
            from detectors import comic_text_detector, pp_doclayout_v3, yolov8_seg_bubble

            if hasattr(yolov8_seg_bubble, "_DETECTOR_CACHE"):
                yolov8_seg_bubble._DETECTOR_CACHE["key"] = None
                yolov8_seg_bubble._DETECTOR_CACHE["detector"] = None
            if hasattr(pp_doclayout_v3, "_DETECTOR_CACHE"):
                pp_doclayout_v3._DETECTOR_CACHE["key"] = None
                pp_doclayout_v3._DETECTOR_CACHE["detector"] = None
            if hasattr(comic_text_detector, "_TEXT_DETECTOR_CACHE"):
                comic_text_detector._TEXT_DETECTOR_CACHE["key"] = None
                comic_text_detector._TEXT_DETECTOR_CACHE["detector"] = None
        except Exception:
            pass
        gc.collect()

    def execute_command(self, command: ServiceCommand, bridge: WorkerSignalsBridge) -> Any:
        task = command.task
        if isinstance(task, DetectionTask):
            return self._run_detection_task(command, task, bridge)
        if command.action == "reload_models":
            self.on_restart()
            return {"reloaded": True}
        if command.action == "status":
            return {"ready": self._engine.is_ready()}
        raise RuntimeError(f"Detection service does not support action '{command.action}'.")

    def _run_detection_task(
        self,
        command: ServiceCommand,
        task: DetectionTask,
        bridge: WorkerSignalsBridge,
    ) -> DetectionWorkerResult:
        if not task.image_paths:
            raise ValueError("No source images were provided for detection.")
        if not self._engine.is_ready():
            raise RuntimeError("Detection service is not ready. Restart the Detection service.")

        total_pages = len(task.image_paths)
        page_results: list[DetectionPageResult] = []
        bridge.message.emit(f"Starting detection for {total_pages} page(s).")
        bridge.progress.emit(0)
        self._diag_for_task(task, step="command_start", message=f"detection command received ({total_pages} page(s))")

        for index, image_path in enumerate(task.image_paths, start=1):
            self._check_canceled(command, message="Detection canceled before the next page.")
            page_path = Path(image_path)
            bridge.event.emit(
                {
                    "stage": "detection",
                    "event": "page_start",
                    "image_path": str(page_path),
                    "page_index": index,
                    "page_total": total_pages,
                }
            )
            bridge.message.emit(f"[{index}/{total_pages}] Detecting {page_path.name}")

            try:
                json_path = self._run_detection_for_image(task, page_path, bridge)
            except Exception as exc:
                readable_error = f"{page_path.name}: {exc}"
                bridge.message.emit(f"Detection failed: {readable_error}")
                page_results.append(
                    DetectionPageResult(
                        image_path=page_path,
                        json_path=None,
                        error=str(exc),
                    )
                )
                bridge.event.emit(
                    {
                        "stage": "detection",
                        "event": "page_error",
                        "image_path": str(page_path),
                        "page_index": index,
                        "page_total": total_pages,
                        "message": str(exc),
                        "error": str(exc),
                    }
                )
            else:
                page_results.append(
                    DetectionPageResult(
                        image_path=page_path,
                        json_path=json_path,
                        error=None,
                    )
                )
                bridge.message.emit(f"Detection complete: {json_path}")
                bridge.event.emit(
                    {
                        "stage": "detection",
                        "event": "page_done",
                        "image_path": str(page_path),
                        "output_path": str(json_path),
                        "page_index": index,
                        "page_total": total_pages,
                    }
                )

            bridge.progress.emit(int((index / total_pages) * 100))

        successful_pages = [result for result in page_results if result.json_path is not None]
        if not successful_pages:
            first_error = page_results[0].error if page_results else "Unknown detection failure."
            raise RuntimeError(f"Detection failed for all pages. {first_error}")

        failed_count = len([result for result in page_results if result.error is not None])
        if failed_count:
            bridge.message.emit(
                f"Detection finished with {failed_count} failed page(s) and {len(successful_pages)} successful page(s)."
            )
        else:
            bridge.message.emit(f"Detection finished successfully for {len(successful_pages)} page(s).")
        return DetectionWorkerResult(page_results=page_results)

    def _run_detection_for_image(
        self,
        task: DetectionTask,
        image_path: Path,
        bridge: WorkerSignalsBridge,
    ) -> Path:
        source_image_path = ensure_path(image_path)
        detection_dir = ensure_path(task.detection_cache_dir)
        masks_dir = ensure_path(task.masks_cache_dir)
        detection_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        detection_json_path = detection_dir / f"{source_image_path.stem}.json"
        page_mask_dir = masks_dir / source_image_path.stem
        project_root = detection_dir.parents[1]
        diagnostics_path = resolve_runtime_diagnostics_path(
            project_root=project_root,
            workspace_root=self._workspace_root,
        )

        if not task.force and detection_json_path.exists():
            bridge.message.emit(f"Reusing cached detection for {source_image_path.name}")
            return detection_json_path

        bridge.message.emit(f"Loading image for detection: {source_image_path.name}")
        write_runtime_diagnostic(
            "before load image",
            log_path=diagnostics_path,
            service="detection",
            page=source_image_path.name,
            step="before_load_image",
        )
        image = load_image_bgr(source_image_path)
        write_runtime_diagnostic(
            "after load image",
            log_path=diagnostics_path,
            service="detection",
            page=source_image_path.name,
            step="after_load_image",
        )

        bridge.message.emit(f"Running detection: {source_image_path.name}")
        write_runtime_diagnostic(
            "before detect_image",
            log_path=diagnostics_path,
            service="detection",
            page=source_image_path.name,
            step="before_detect_image",
        )
        result = self._engine.detect_image(
            image,
            logger=bridge.message.emit,
            diagnostics_path=diagnostics_path,
            page_name=source_image_path.name,
        )
        write_runtime_diagnostic(
            "after detect_image",
            log_path=diagnostics_path,
            service="detection",
            page=source_image_path.name,
            step="after_detect_image",
        )

        write_runtime_diagnostic(
            "before save_detection_result",
            log_path=diagnostics_path,
            service="detection",
            page=source_image_path.name,
            step="before_save_detection_result",
        )
        output_path = save_detection_result(
            result,
            image_path=source_image_path,
            image_shape=image.shape,
            detection_json_output_path=detection_json_path,
            mask_output_dir=page_mask_dir,
            project_root=project_root,
            logger=bridge.message.emit,
        )
        write_runtime_diagnostic(
            "after save_detection_result",
            log_path=diagnostics_path,
            service="detection",
            page=source_image_path.name,
            step="after_save_detection_result",
        )
        bridge.message.emit(f"Saved detection cache: {output_path}")
        return output_path

    def _diag_for_task(
        self,
        task: DetectionTask,
        *,
        step: str,
        message: str,
        page_path: Path | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        try:
            project_root = None
            detection_cache_dir = ensure_path(task.detection_cache_dir)
            if len(detection_cache_dir.parents) >= 2:
                project_root = detection_cache_dir.parents[1]
        except Exception:
            project_root = None
        write_runtime_diagnostic(
            message,
            project_root=project_root,
            workspace_root=self._workspace_root,
            service="detection",
            page=page_path.name if page_path is not None else "",
            step=step,
            extra=extra,
        )


__all__ = ["DetectionService"]
