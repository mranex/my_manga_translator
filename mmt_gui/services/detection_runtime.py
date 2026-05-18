"""Resident Python-threaded detection runtime for CUDA/model execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import gc
from pathlib import Path
import threading
from typing import Any

from mmt_core import DetectionConfig, detection_configs_match, load_detection_json
from mmt_core.crash_logging import write_crash_breadcrumb
from mmt_core.detection_engine import DetectionEngine
from mmt_core.detection_io import save_detection_result
from mmt_core.image_io import ensure_path, load_image_bgr
from mmt_core.runtime_diagnostics import resolve_runtime_diagnostics_path, write_runtime_diagnostic
from mmt_gui.workers import DetectionPageResult, DetectionWorkerResult, MangaDetectorTaskResult

from .models import CancelToken


LoggerCallback = Callable[[str], None] | None
StatusCallback = Callable[[str], None] | None
ProgressCallback = Callable[[int], None]
MessageCallback = Callable[[str], None]
EventCallback = Callable[[object], None]


class DetectionRuntimeBusyError(RuntimeError):
    """Raised when the resident detection runtime is already processing a command."""


class DetectionRuntimeCanceledError(RuntimeError):
    """Raised when a resident detection runtime command is canceled cooperatively."""


@dataclass(slots=True)
class DetectionRuntimeCallbacks:
    progress: ProgressCallback
    message: MessageCallback
    event: EventCallback


@dataclass(slots=True)
class DetectionRuntimeRequest:
    command_id: str
    action: str
    image_paths: list[Path]
    detection_cache_dir: Path | None
    masks_cache_dir: Path | None
    force: bool
    cancel_token: CancelToken
    callbacks: DetectionRuntimeCallbacks
    workspace_root: Path
    detection_config: DetectionConfig = field(default_factory=DetectionConfig)


@dataclass(slots=True)
class DetectionRuntimeOutcome:
    result: DetectionWorkerResult | None = None
    error: str = ""
    canceled: bool = False


@dataclass(slots=True)
class _RuntimeJob:
    request: DetectionRuntimeRequest
    done_event: threading.Event = field(default_factory=threading.Event)
    outcome: DetectionRuntimeOutcome = field(default_factory=DetectionRuntimeOutcome)


class DetectionRuntimeThread:
    def __init__(
        self,
        *,
        workspace_root: Path,
        startup_detection_config: DetectionConfig | dict[str, Any] | None = None,
        logger: LoggerCallback = None,
        status_callback: StatusCallback = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self._startup_detection_config = DetectionConfig.from_value(startup_detection_config)
        self._logger = logger
        self._status_callback = status_callback
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._stop_requested = False
        self._ready = False
        self._startup_error = ""
        self._engine: DetectionEngine | None = None
        self._pending_job: _RuntimeJob | None = None
        self._active_job: _RuntimeJob | None = None
        self._ready_event = threading.Event()

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_requested = False
            self._ready = False
            self._startup_error = ""
            self._pending_job = None
            self._active_job = None
            self._ready_event.clear()
            self._thread = threading.Thread(
                target=self._thread_main,
                name="MMTDetectionRuntimeThread",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        thread_to_join: threading.Thread | None = None
        with self._condition:
            self._stop_requested = True
            active_job = self._active_job
            pending_job = self._pending_job
            if active_job is not None:
                active_job.request.cancel_token.request_cancel()
            if pending_job is not None:
                pending_job.request.cancel_token.request_cancel()
            self._condition.notify_all()
            thread_to_join = self._thread
        if thread_to_join is not None:
            thread_to_join.join(timeout=20.0)

    def wait_until_ready(self, timeout: float | None = None) -> bool:
        return self._ready_event.wait(timeout)

    def is_ready(self) -> bool:
        with self._lock:
            return self._ready and not self._startup_error and self._engine is not None

    def is_busy(self) -> bool:
        with self._lock:
            return self._active_job is not None or self._pending_job is not None

    def error_message(self) -> str:
        with self._lock:
            return self._startup_error

    def submit_and_wait(
        self,
        request: DetectionRuntimeRequest,
        *,
        accepted_callback: Callable[[], None] | None = None,
    ) -> DetectionRuntimeOutcome:
        with self._condition:
            if not self.is_ready():
                raise RuntimeError(self.error_message() or "Detection runtime is not ready.")
            if self._active_job is not None or self._pending_job is not None:
                raise DetectionRuntimeBusyError("Detection runtime is busy.")
            thread = self._thread
            if thread is None or not thread.is_alive():
                raise RuntimeError("Detection runtime thread is not running.")
            job = _RuntimeJob(request=request)
            self._pending_job = job
            self._condition.notify_all()
        write_crash_breadcrumb(
            "DetectionRuntimeThread submit accepted",
            command_id=request.command_id,
            action=request.action,
        )
        if accepted_callback is not None:
            try:
                accepted_callback()
            except Exception:
                pass

        while not job.done_event.wait(timeout=0.10):
            thread = self._thread
            if thread is not None and not thread.is_alive():
                job.outcome.error = job.outcome.error or "Detection runtime thread stopped unexpectedly."
                job.done_event.set()
                break
        return job.outcome

    def _thread_main(self) -> None:
        write_crash_breadcrumb("DetectionRuntimeThread thread started")
        self._emit_status("Starting detection runtime...")
        try:
            self._engine = DetectionEngine()
            write_crash_breadcrumb("DetectionRuntimeThread preload started")
            self._emit_status("Preloading detection engine...")
            self._engine.preload(
                self._startup_detection_config,
                logger=self._startup_log,
                status_callback=self._emit_status,
            )
            if self._engine.bubble_detector is not None:
                write_crash_breadcrumb("DetectionRuntimeThread YOLO loaded")
            if self._engine.layout_detector is not None:
                write_crash_breadcrumb("DetectionRuntimeThread PPLayout loaded")
            if self._engine.manga_detector_status().get("loaded", False):
                write_crash_breadcrumb("DetectionRuntimeThread manga detector loaded")
            write_crash_breadcrumb("DetectionRuntimeThread preload done")
            write_crash_breadcrumb("DetectionRuntimeThread runtime ready")
            self._startup_log("Detection runtime is ready.")
            self._emit_status("Detection runtime ready.")
            with self._condition:
                self._ready = True
                self._ready_event.set()
        except Exception as exc:
            with self._condition:
                self._ready = False
                self._startup_error = str(exc)
                self._ready_event.set()
            self._startup_log(f"Detection runtime preload failed: {exc}", level="error")
            write_crash_breadcrumb(
                "DetectionRuntimeThread preload failed",
                level="critical",
                error=str(exc),
            )
            self._cleanup_runtime()
            return

        try:
            while True:
                with self._condition:
                    while not self._stop_requested and self._pending_job is None:
                        self._condition.wait(timeout=0.25)
                    if self._stop_requested:
                        break
                    job = self._pending_job
                    self._pending_job = None
                    self._active_job = job

                if job is None:
                    continue

                try:
                    job.outcome.result = self._execute_request(job.request)
                except DetectionRuntimeCanceledError as exc:
                    job.outcome.canceled = True
                    job.outcome.error = str(exc)
                except Exception as exc:
                    job.outcome.error = str(exc)
                    write_crash_breadcrumb(
                        "DetectionRuntimeThread command failed",
                        level="critical",
                        command_id=job.request.command_id,
                        error=str(exc),
                    )
                finally:
                    with self._condition:
                        self._active_job = None
                    job.done_event.set()
        finally:
            self._cleanup_runtime()
            write_crash_breadcrumb("DetectionRuntimeThread thread stopped")

    def _execute_request(self, request: DetectionRuntimeRequest) -> Any:
        if self._engine is None or not self._engine.is_ready():
            raise RuntimeError("Detection runtime is not ready.")

        if request.action in {
            "load_manga_detector",
            "reload_manga_detector",
            "unload_manga_detector",
            "manga_detector_status",
        }:
            return self._execute_manga_detector_request(request)

        callbacks = request.callbacks
        total_pages = len(request.image_paths)
        page_results: list[DetectionPageResult] = []

        write_crash_breadcrumb(
            "DetectionRuntimeThread command received",
            command_id=request.command_id,
            action=request.action,
            page_total=total_pages,
        )
        self._engine.ensure_detection_ready(request.detection_config, logger=callbacks.message)
        callbacks.message(f"Starting detection for {total_pages} page(s).")
        callbacks.progress(0)
        self._write_task_diag(
            request,
            step="command_start",
            message=f"detection command received ({total_pages} page(s))",
        )

        for index, image_path in enumerate(request.image_paths, start=1):
            self._check_canceled(request.cancel_token, message="Detection canceled before the next page.")
            page_path = Path(image_path)
            write_crash_breadcrumb(
                "DetectionRuntimeThread before page",
                command_id=request.command_id,
                page=str(page_path),
                page_index=index,
                page_total=total_pages,
            )
            callbacks.event(
                {
                    "stage": "detection",
                    "event": "page_start",
                    "image_path": str(page_path),
                    "page_index": index,
                    "page_total": total_pages,
                }
            )
            callbacks.message(f"[{index}/{total_pages}] Detecting {page_path.name}")

            try:
                json_path = self._run_detection_for_image(request, page_path, callbacks)
            except DetectionRuntimeCanceledError:
                raise
            except Exception as exc:
                readable_error = f"{page_path.name}: {exc}"
                callbacks.message(f"Detection failed: {readable_error}")
                page_results.append(
                    DetectionPageResult(
                        image_path=page_path,
                        json_path=None,
                        error=str(exc),
                    )
                )
                callbacks.event(
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
                callbacks.message(f"Detection complete: {json_path}")
                callbacks.event(
                    {
                        "stage": "detection",
                        "event": "page_done",
                        "image_path": str(page_path),
                        "output_path": str(json_path),
                        "page_index": index,
                        "page_total": total_pages,
                    }
                )

            callbacks.progress(int((index / total_pages) * 100))
            self._check_canceled(request.cancel_token, message="Detection canceled after the current page.")

        successful_pages = [result for result in page_results if result.json_path is not None]
        if not successful_pages:
            first_error = page_results[0].error if page_results else "Unknown detection failure."
            raise RuntimeError(f"Detection failed for all pages. {first_error}")

        failed_count = len([result for result in page_results if result.error is not None])
        if failed_count:
            callbacks.message(
                f"Detection finished with {failed_count} failed page(s) and {len(successful_pages)} successful page(s)."
            )
        else:
            callbacks.message(f"Detection finished successfully for {len(successful_pages)} page(s).")

        write_crash_breadcrumb(
            "DetectionRuntimeThread command done",
            command_id=request.command_id,
            page_total=total_pages,
        )
        return DetectionWorkerResult(page_results=page_results)

    def _run_detection_for_image(
        self,
        request: DetectionRuntimeRequest,
        image_path: Path,
        callbacks: DetectionRuntimeCallbacks,
    ) -> Path:
        if self._engine is None:
            raise RuntimeError("Detection runtime is not initialized.")

        source_image_path = ensure_path(image_path)
        detection_dir = ensure_path(request.detection_cache_dir)
        masks_dir = ensure_path(request.masks_cache_dir)
        detection_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        detection_json_path = detection_dir / f"{source_image_path.stem}.json"
        page_mask_dir = masks_dir / source_image_path.stem
        project_root = detection_dir.parents[1]
        diagnostics_path = resolve_runtime_diagnostics_path(
            project_root=project_root,
            workspace_root=request.workspace_root,
        )

        if not request.force and detection_json_path.exists():
            cached_payload: dict[str, Any] | None = None
            try:
                cached_payload = load_detection_json(detection_json_path)
            except Exception:
                cached_payload = None
            if detection_configs_match(
                request.detection_config,
                (cached_payload or {}).get("detection_config"),
            ):
                callbacks.message(f"Reusing cached detection for {source_image_path.name}")
                return detection_json_path
            callbacks.message(
                f"Detection config changed for {source_image_path.name}. Regenerating detection cache."
            )

        callbacks.message(f"Loading image for detection: {source_image_path.name}")
        write_crash_breadcrumb("DetectionRuntimeThread before load image", page=source_image_path.name)
        write_runtime_diagnostic(
            "before load image",
            log_path=diagnostics_path,
            service="detection",
            page=source_image_path.name,
            step="before_load_image",
        )
        image = load_image_bgr(source_image_path)
        write_crash_breadcrumb("DetectionRuntimeThread after load image", page=source_image_path.name)
        write_runtime_diagnostic(
            "after load image",
            log_path=diagnostics_path,
            service="detection",
            page=source_image_path.name,
            step="after_load_image",
        )

        callbacks.message(f"Running detection: {source_image_path.name}")
        write_crash_breadcrumb("DetectionRuntimeThread before engine.detect_image", page=source_image_path.name)
        write_runtime_diagnostic(
            "before detect_image",
            log_path=diagnostics_path,
            service="detection",
            page=source_image_path.name,
            step="before_detect_image",
        )
        result = self._engine.detect_image(
            image,
            detection_config=request.detection_config,
            logger=callbacks.message,
            diagnostics_path=diagnostics_path,
            page_name=source_image_path.name,
        )
        write_crash_breadcrumb("DetectionRuntimeThread after engine.detect_image", page=source_image_path.name)
        write_runtime_diagnostic(
            "after detect_image",
            log_path=diagnostics_path,
            service="detection",
            page=source_image_path.name,
            step="after_detect_image",
        )

        write_crash_breadcrumb("DetectionRuntimeThread before save_detection_result", page=source_image_path.name)
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
            detection_config=request.detection_config,
            logger=callbacks.message,
        )
        write_crash_breadcrumb("DetectionRuntimeThread after save_detection_result", page=source_image_path.name)
        write_runtime_diagnostic(
            "after save_detection_result",
            log_path=diagnostics_path,
            service="detection",
            page=source_image_path.name,
            step="after_save_detection_result",
        )
        callbacks.message(f"Saved detection cache: {output_path}")
        return output_path

    def _execute_manga_detector_request(self, request: DetectionRuntimeRequest) -> MangaDetectorTaskResult:
        if self._engine is None:
            raise RuntimeError("Detection runtime is not initialized.")

        callbacks = request.callbacks
        config = DetectionConfig.from_value(request.detection_config)
        callbacks.progress(0)

        if request.action == "load_manga_detector":
            callbacks.message("Loading Manga RT-DETR detector...")
            payload = self._engine.load_manga_detector(config, logger=callbacks.message)
        elif request.action == "reload_manga_detector":
            callbacks.message("Reloading Manga RT-DETR detector...")
            payload = self._engine.reload_manga_detector(config, logger=callbacks.message)
        elif request.action == "unload_manga_detector":
            callbacks.message("Unloading Manga RT-DETR detector...")
            payload = self._engine.unload_manga_detector()
        else:
            callbacks.message("Checking Manga RT-DETR detector status...")
            payload = self._engine.manga_detector_status()

        callbacks.progress(100)
        return MangaDetectorTaskResult(
            loaded=bool(payload.get("loaded", False)),
            device=str(payload.get("device", "") or ""),
            model_id=str(payload.get("model_id", config.manga_model_id) or config.manga_model_id),
            message=str(payload.get("message", "") or ""),
            detection_config=dict(payload.get("detection_config", {}) or {}),
            error=str(payload.get("error", "") or ""),
        )

    def _write_task_diag(
        self,
        request: DetectionRuntimeRequest,
        *,
        step: str,
        message: str,
        page_path: Path | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        try:
            project_root = None
            detection_cache_dir = ensure_path(request.detection_cache_dir)
            if len(detection_cache_dir.parents) >= 2:
                project_root = detection_cache_dir.parents[1]
        except Exception:
            project_root = None
        write_runtime_diagnostic(
            message,
            project_root=project_root,
            workspace_root=request.workspace_root,
            service="detection",
            page=page_path.name if page_path is not None else "",
            step=step,
            extra=extra,
        )

    def _check_canceled(self, cancel_token: CancelToken, *, message: str) -> None:
        if cancel_token.is_cancel_requested():
            raise DetectionRuntimeCanceledError(message)

    def _startup_log(self, message: str, *, level: str = "info") -> None:
        if self._logger is not None:
            self._logger(str(message or ""))
        write_crash_breadcrumb(
            "DetectionRuntimeThread startup log",
            level=level,
            message_text=str(message or ""),
        )

    def _emit_status(self, message: str) -> None:
        if self._status_callback is not None:
            self._status_callback(str(message or ""))

    def _cleanup_runtime(self) -> None:
        engine = self._engine
        self._engine = None
        with self._lock:
            self._ready = False
        if engine is not None:
            try:
                engine.clear()
            except Exception:
                pass
        try:
            from detectors import pp_doclayout_v3, yolov8_seg_bubble

            if hasattr(yolov8_seg_bubble, "_DETECTOR_CACHE"):
                yolov8_seg_bubble._DETECTOR_CACHE["key"] = None
                yolov8_seg_bubble._DETECTOR_CACHE["detector"] = None
            if hasattr(pp_doclayout_v3, "_DETECTOR_CACHE"):
                pp_doclayout_v3._DETECTOR_CACHE["key"] = None
                pp_doclayout_v3._DETECTOR_CACHE["detector"] = None
        except Exception:
            pass
        gc.collect()


__all__ = [
    "DetectionRuntimeBusyError",
    "DetectionRuntimeCallbacks",
    "DetectionRuntimeCanceledError",
    "DetectionRuntimeOutcome",
    "DetectionRuntimeRequest",
    "DetectionRuntimeThread",
]
