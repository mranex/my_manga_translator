"""Service registry and command router for resident stage services."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
import traceback
from typing import Any

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from mmt_core.crash_logging import write_crash_breadcrumb
from mmt_gui.workers import (
    DetectionTask,
    ExportTask,
    InpaintMaskTask,
    InpaintTask,
    LamaModelTask,
    MangaDetectorTask,
    OCRInferenceTask,
    OCRPreparationTask,
    ProcessTask,
    RenderPreparationTask,
    RenderTask,
    TranslationInitializationTask,
    TranslationTask,
)

from .base_service import ServiceCanceledError
from .detection_service import DetectionService
from .export_service import ExportService
from .inpaint_service import InpaintService
from .models import (
    CancelToken,
    ServiceCommand,
    ServiceCommandHandle,
    ServiceCommandResult,
    ServiceDispatchError,
    ServiceStatusSnapshot,
    cache_dir_from_task,
    command_id,
    image_relative_paths_from_task,
    project_root_from_task,
)
from .ocr_service import OCRService
from .process_service import ProcessService
from .render_service import RenderService
from .translation_service import TranslationService


@dataclass(slots=True)
class _PendingCommand:
    command: ServiceCommand
    handle: ServiceCommandHandle | None = None
    sync_event: threading.Event | None = None
    sync_result: ServiceCommandResult | None = None
    started_emitted: bool = False


@dataclass(slots=True)
class _ServiceEntry:
    thread: QThread
    service: QObject


class ServiceManager(QObject):
    """Owns resident services, routes commands, and exposes diagnostics."""

    service_status_changed = pyqtSignal(object)
    diagnostic_message = pyqtSignal(str, str)

    def __init__(
        self,
        *,
        workspace_root: Path,
        startup_options: dict[str, Any] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.workspace_root = Path(workspace_root)
        self.startup_options = dict(startup_options or {})
        self.scheduler = None
        self._services: dict[str, _ServiceEntry] = {}
        self._pending_commands: dict[str, _PendingCommand] = {}
        self._service_active_commands: dict[str, str] = {}
        self._service_statuses: dict[str, ServiceStatusSnapshot] = {}
        self._lock = threading.RLock()
        self._shutting_down = False
        self._diagnostics_path = self.workspace_root / "logs" / "runtime_diagnostics.log"
        self._diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_services()

    def start_services(self) -> None:
        for entry in self._services.values():
            if entry.thread.isRunning():
                continue
            write_crash_breadcrumb(
                "service thread start requested",
                service=entry.service.objectName() if hasattr(entry.service, "objectName") else "",
                thread_name=entry.thread.objectName(),
            )
            entry.thread.start()

    def shutdown(self) -> None:
        with self._lock:
            self._shutting_down = True
            pending_ids = list(self._pending_commands)
        for command_id_value in pending_ids:
            self.cancel_command(command_id_value)
        for entry in self._services.values():
            try:
                write_crash_breadcrumb(
                    "service shutdown requested",
                    service=entry.service.objectName() if hasattr(entry.service, "objectName") else "",
                    thread_name=entry.thread.objectName(),
                )
                entry.service.shutdown_service()  # type: ignore[attr-defined]
            except Exception:
                pass
        for entry in self._services.values():
            entry.thread.quit()
        for entry in self._services.values():
            entry.thread.wait(4000)

    def dispatch_task(self, task: Any) -> ServiceCommandHandle:
        write_crash_breadcrumb(
            "ServiceManager dispatch_task entered",
            task_type=type(task).__name__,
        )
        service_name, action = self._resolve_service_action(task)
        write_crash_breadcrumb(
            "ServiceManager resolved service/action",
            service=service_name,
            action=action,
            task_type=type(task).__name__,
        )
        self._ensure_service_ready(service_name)
        command = self._build_command(task, service_name=service_name, action=action)
        write_crash_breadcrumb(
            "ServiceManager command built",
            service=service_name,
            action=action,
            command_id=command.command_id,
        )
        handle = ServiceCommandHandle(
            command_id=command.command_id,
            task=task,
            service_name=service_name,
            cancel_callback=self.cancel_command,
        )
        pending = _PendingCommand(command=command, handle=handle)
        with self._lock:
            if self._shutting_down:
                raise ServiceDispatchError("Services are shutting down.")
            self._reserve_service_slot_locked(service_name=service_name, command_id_value=command.command_id)
            self._pending_commands[command.command_id] = pending
        write_crash_breadcrumb(
            "ServiceManager service slot reserved",
            service=service_name,
            action=action,
            command_id=command.command_id,
        )
        try:
            write_crash_breadcrumb(
                "ServiceManager before submit_to_service",
                service=service_name,
                action=action,
                command_id=command.command_id,
            )
            self._submit_to_service(command)
            write_crash_breadcrumb(
                "ServiceManager after submit_to_service",
                service=service_name,
                action=action,
                command_id=command.command_id,
            )
        except Exception:
            with self._lock:
                self._pending_commands.pop(command.command_id, None)
                self._release_service_slot_locked(service_name)
            raise
        return handle

    def run_task_sync(self, task: Any, parent_command: ServiceCommand) -> ServiceCommandResult:
        service_name, action = self._resolve_service_action(task)
        self._ensure_service_ready(service_name)
        command = self._build_command(task, service_name=service_name, action=action)
        sync_event = threading.Event()
        pending = _PendingCommand(command=command, sync_event=sync_event)
        with self._lock:
            if self._shutting_down:
                raise ServiceDispatchError("Services are shutting down.")
            self._reserve_service_slot_locked(service_name=service_name, command_id_value=command.command_id)
            self._pending_commands[command.command_id] = pending
        try:
            self._submit_to_service(command)
        except Exception:
            with self._lock:
                self._pending_commands.pop(command.command_id, None)
                self._release_service_slot_locked(service_name)
            raise

        cancel_forwarded = False
        while not sync_event.wait(timeout=0.10):
            if parent_command.cancel_token.is_cancel_requested() and not cancel_forwarded:
                self.cancel_command(command.command_id)
                cancel_forwarded = True

        result = pending.sync_result
        if result is None:
            raise RuntimeError("Resident service completed without a result payload.")
        if result.state == "busy":
            raise ServiceDispatchError(result.error or f"{service_name} worker is busy.")
        if result.state == "failed":
            raise RuntimeError(result.error or f"{service_name} command failed.")
        if result.state == "canceled":
            raise ServiceCanceledError(result.error or "Command canceled.")
        return result

    def cancel_command(self, command_id_value: str) -> None:
        normalized = str(command_id_value or "").strip()
        if not normalized:
            return
        with self._lock:
            pending = self._pending_commands.get(normalized)
        if pending is None:
            return
        pending.command.cancel_token.request_cancel()
        entry = self._services.get(pending.command.service_name)
        if entry is None:
            return
        try:
            entry.service.cancel_command(normalized)  # type: ignore[attr-defined]
        except Exception as exc:
            self._emit_diagnostic("warning", f"Cancel forwarding failed for {normalized}: {exc}")

    def restart_service(self, service_name: str) -> None:
        normalized = str(service_name or "").strip().lower()
        entry = self._services.get(normalized)
        if entry is None:
            raise ServiceDispatchError(f"Unknown service: {service_name}")
        if self.has_active_commands():
            raise ServiceDispatchError("A task is running. Stop it before restarting a service.")
        entry.service.restart_service()  # type: ignore[attr-defined]

    def has_active_commands(self) -> bool:
        with self._lock:
            return bool(self._service_active_commands)

    def service_statuses(self) -> dict[str, ServiceStatusSnapshot]:
        with self._lock:
            return dict(self._service_statuses)

    def _create_services(self) -> None:
        startup_options = {
            "preload_detection": bool(self.startup_options.get("preload_detection", True)),
            "preload_inpaint": bool(self.startup_options.get("preload_inpaint", True)),
            "preload_render": bool(self.startup_options.get("preload_render", True)),
            "device": str(self.startup_options.get("inpaint_device", "") or ""),
            "detection_config": dict(self.startup_options.get("detection_config", {}) or {}),
            "workspace_root": str(self.workspace_root),
        }

        self._register_service(
            "detection",
            DetectionService(scheduler=self.scheduler, startup_options=startup_options),
        )
        self._register_service(
            "ocr",
            OCRService(scheduler=self.scheduler, startup_options=startup_options),
        )
        self._register_service(
            "translation",
            TranslationService(scheduler=self.scheduler, startup_options=startup_options),
        )
        self._register_service(
            "inpaint",
            InpaintService(scheduler=self.scheduler, startup_options=startup_options),
        )
        self._register_service(
            "render",
            RenderService(scheduler=self.scheduler, startup_options=startup_options),
        )
        self._register_service(
            "export",
            ExportService(scheduler=self.scheduler, startup_options=startup_options),
        )
        self._register_service(
            "process",
            ProcessService(
                scheduler=self.scheduler,
                sync_dispatch=self.run_task_sync,
                startup_options=startup_options,
            ),
        )

    def _register_service(self, service_name: str, service: QObject) -> None:
        thread = QThread(self)
        thread.setObjectName(f"MMT{str(service_name).title()}ServiceThread")
        service.setObjectName(f"{str(service_name).strip().lower()}_service")
        service.moveToThread(thread)
        write_crash_breadcrumb(
            "service thread created",
            service=service_name,
            thread_name=thread.objectName(),
        )
        thread.started.connect(
            lambda service_name=service_name, thread_name=thread.objectName(): write_crash_breadcrumb(
                "service thread started",
                service=service_name,
                thread_name=thread_name,
            )
        )
        thread.started.connect(service.initialize)  # type: ignore[attr-defined]
        service.status_changed.connect(self._on_service_status_changed)  # type: ignore[attr-defined]
        service.command_started.connect(self._on_service_command_started)  # type: ignore[attr-defined]
        service.command_progress.connect(self._on_service_command_progress)  # type: ignore[attr-defined]
        service.command_event.connect(self._on_service_command_event)  # type: ignore[attr-defined]
        service.command_finished.connect(self._on_service_command_finished)  # type: ignore[attr-defined]
        service.command_failed.connect(self._on_service_command_failed)  # type: ignore[attr-defined]
        service.command_canceled.connect(self._on_service_command_canceled)  # type: ignore[attr-defined]
        service.log_message.connect(self._on_service_log_message)  # type: ignore[attr-defined]
        thread.finished.connect(
            lambda service_name=service_name, thread_name=thread.objectName(): write_crash_breadcrumb(
                "service thread finished",
                service=service_name,
                thread_name=thread_name,
            )
        )
        thread.finished.connect(service.deleteLater)
        self._services[str(service_name).strip().lower()] = _ServiceEntry(thread=thread, service=service)
        self._service_statuses[str(service_name).strip().lower()] = ServiceStatusSnapshot(
            service_name=str(service_name).strip().lower(),
            state="starting",
            message="Starting...",
        )

    def _submit_to_service(self, command: ServiceCommand) -> None:
        entry = self._services.get(command.service_name)
        if entry is None:
            raise ServiceDispatchError(f"Unknown service: {command.service_name}")
        try:
            entry.service.submit_command(command)  # type: ignore[attr-defined]
        except Exception as exc:
            with self._lock:
                self._pending_commands.pop(command.command_id, None)
            raise ServiceDispatchError(str(exc)) from exc

    def _ensure_service_ready(self, service_name: str) -> None:
        normalized = str(service_name or "").strip().lower()
        with self._lock:
            status = self._service_statuses.get(normalized)
            active_command_id = self._service_active_commands.get(normalized)
        if status is None:
            raise ServiceDispatchError(f"{normalized.title()} service is still starting. Please wait a moment.")
        if active_command_id:
            raise ServiceDispatchError(f"{normalized.title()} worker is busy.")
        if status.state == "error":
            raise ServiceDispatchError(
                f"{normalized.title()} service is in an error state: {status.message or 'Unknown service error.'}"
            )
        if status.state in {"starting", "loading", "stopping", "stopped"}:
            raise ServiceDispatchError(
                f"{normalized.title()} service is {status.state}. Please wait for it to become ready."
            )

    def _build_command(self, task: Any, *, service_name: str, action: str) -> ServiceCommand:
        image_paths = image_relative_paths_from_task(task)
        current_page = image_paths[0] if image_paths else ""
        force_value = bool(getattr(task, "force", False))
        return ServiceCommand(
            command_id=command_id(service_name),
            service_name=service_name,
            action=action,
            stage=str(getattr(task, "stage", service_name) or service_name),
            task=task,
            project_root=project_root_from_task(task),
            cache_dir=cache_dir_from_task(task),
            image_relative_paths=image_paths,
            current_page=current_page,
            force=force_value,
            config=self._task_config(task),
            metadata={},
        )

    def _task_config(self, task: Any) -> dict[str, Any]:
        config = {}
        for attr_name in ("config", "server_url", "timeout", "scope"):
            if not hasattr(task, attr_name):
                continue
            value = getattr(task, attr_name)
            if value in (None, "", {}, []):
                continue
            config[attr_name] = value
        return config

    def _resolve_service_action(self, task: Any) -> tuple[str, str]:
        if isinstance(task, DetectionTask):
            return "detection", "detect_page" if len(task.image_paths) <= 1 else "detect_pages"
        if isinstance(task, MangaDetectorTask):
            action = str(task.action or "").strip().lower()
            if action == "load":
                return "detection", "load_manga_detector"
            if action == "reload":
                return "detection", "reload_manga_detector"
            if action == "unload":
                return "detection", "unload_manga_detector"
            return "detection", "manga_detector_status"
        if isinstance(task, OCRPreparationTask):
            return "ocr", "prepare_page" if len(task.image_relative_paths) <= 1 else "prepare_pages"
        if isinstance(task, OCRInferenceTask):
            if task.selected_item_ids_by_page:
                return "ocr", "run_selected_items"
            return "ocr", "run_page" if len(task.image_relative_paths) <= 1 else "run_pages"
        if isinstance(task, TranslationInitializationTask):
            return "translation", "initialize_page" if len(task.image_relative_paths) <= 1 else "initialize_pages"
        if isinstance(task, TranslationTask):
            if task.selected_item_ids_by_page:
                return "translation", "translate_selected_items"
            return "translation", "translate_page" if len(task.image_relative_paths) <= 1 else "translate_pages"
        if isinstance(task, InpaintMaskTask):
            return "inpaint", "prepare_mask_page" if len(task.image_relative_paths) <= 1 else "prepare_mask_pages"
        if isinstance(task, InpaintTask):
            return "inpaint", "inpaint_page" if len(task.image_relative_paths) <= 1 else "inpaint_pages"
        if isinstance(task, LamaModelTask):
            action = str(task.action or "").strip().lower()
            if action == "load":
                return "inpaint", "preload_model"
            if action == "reload":
                return "inpaint", "reload_model"
            return "inpaint", "unload_model"
        if isinstance(task, RenderPreparationTask):
            return "render", "prepare_page" if len(task.image_relative_paths) <= 1 else "prepare_pages"
        if isinstance(task, RenderTask):
            return "render", "render_page" if len(task.image_relative_paths) <= 1 else "render_pages"
        if isinstance(task, ExportTask):
            scope = str(getattr(getattr(task, "config", {}), "get", lambda *_args, **_kwargs: "")("page_scope", "") or "")
            normalized_scope = scope.strip().lower()
            if normalized_scope == "current":
                return "export", "export_current"
            if normalized_scope == "selected":
                return "export", "export_selected"
            return "export", "export_all"
        if isinstance(task, ProcessTask):
            normalized_scope = str(task.scope or "").strip().lower()
            if normalized_scope == "current":
                return "process", "reprocess_current_page" if task.force else "process_current_page"
            return "process", "reprocess_chapter" if task.force else "process_chapter"
        raise ServiceDispatchError(f"Unsupported service task type: {type(task).__name__}")

    def _on_service_status_changed(self, payload: object) -> None:
        if not isinstance(payload, ServiceStatusSnapshot):
            return
        with self._lock:
            self._service_statuses[payload.service_name] = payload
        self.service_status_changed.emit(payload)
        self._emit_diagnostic("info", f"[{payload.service_name}] {payload.state}: {payload.message}")

    def _on_service_command_started(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        pending = self._pending_for_payload(payload)
        if pending is None:
            return
        if pending.handle is not None and not pending.started_emitted:
            pending.handle.signals.started.emit(getattr(pending.command.task, "name", pending.command.action))
            pending.started_emitted = True
        self._emit_diagnostic("info", self._command_log_line(payload, "started"))

    def _on_service_command_progress(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        pending = self._pending_for_payload(payload)
        if pending is None:
            return
        if pending.handle is not None:
            try:
                pending.handle.signals.progress.emit(int(payload.get("progress", 0) or 0))
            except Exception:
                pending.handle.signals.progress.emit(0)

    def _on_service_command_event(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        pending = self._pending_for_payload(payload)
        if pending is None:
            return
        if pending.handle is not None:
            pending.handle.signals.event.emit(payload)
        self._emit_diagnostic("info", self._command_log_line(payload, str(payload.get("event", "event") or "event")))

    def _on_service_command_finished(self, payload: object) -> None:
        if not isinstance(payload, ServiceCommandResult):
            return
        pending = self._pop_pending(payload.command_id)
        if pending is None:
            return
        pending.sync_result = payload
        if pending.handle is not None:
            pending.handle.signals.finished.emit(payload.result)
        if pending.sync_event is not None:
            pending.sync_event.set()
        self._emit_diagnostic("info", f"[{payload.service_name}] {payload.action} finished")

    def _on_service_command_failed(self, payload: object) -> None:
        if not isinstance(payload, ServiceCommandResult):
            return
        pending = self._pop_pending(payload.command_id)
        if pending is None:
            return
        pending.sync_result = payload
        if pending.handle is not None:
            pending.handle.signals.failed.emit(payload.error or "Resident service command failed.")
        if pending.sync_event is not None:
            pending.sync_event.set()
        level = "warning" if payload.state == "busy" else "error"
        suffix = "busy" if payload.state == "busy" else "failed"
        self._emit_diagnostic(level, f"[{payload.service_name}] {payload.action} {suffix}: {payload.error}")

    def _on_service_command_canceled(self, payload: object) -> None:
        if not isinstance(payload, ServiceCommandResult):
            return
        pending = self._pop_pending(payload.command_id)
        if pending is None:
            return
        pending.sync_result = payload
        if pending.handle is not None:
            pending.handle.signals.failed.emit(payload.error or "Command canceled.")
        if pending.sync_event is not None:
            pending.sync_event.set()
        self._emit_diagnostic("warning", f"[{payload.service_name}] {payload.action} canceled")

    def _on_service_log_message(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        level = str(payload.get("level", "info") or "info")
        message = str(payload.get("message", "") or "")
        command_id_value = str(payload.get("command_id", "") or "").strip()
        if command_id_value:
            with self._lock:
                pending = self._pending_commands.get(command_id_value)
            if pending is not None and pending.handle is not None and message:
                pending.handle.signals.message.emit(message)
        self._emit_diagnostic(level, self._format_log_payload(payload))

    def _pop_pending(self, command_id_value: str) -> _PendingCommand | None:
        with self._lock:
            pending = self._pending_commands.pop(str(command_id_value or "").strip(), None)
            if pending is not None:
                self._release_service_slot_locked(pending.command.service_name)
            return pending

    def _pending_for_payload(self, payload: dict[str, Any]) -> _PendingCommand | None:
        command_id_value = str(payload.get("command_id", "") or "").strip()
        if not command_id_value:
            return None
        with self._lock:
            return self._pending_commands.get(command_id_value)

    def _emit_diagnostic(self, level: str, message: str) -> None:
        normalized_level = str(level or "info").strip().lower() or "info"
        line = f"[{normalized_level.upper()}] {message}"
        try:
            with self._diagnostics_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{line}\n")
        except Exception:
            pass
        self.diagnostic_message.emit(line, normalized_level)

    def _reserve_service_slot_locked(self, *, service_name: str, command_id_value: str) -> None:
        normalized = str(service_name or "").strip().lower()
        active_command_id = self._service_active_commands.get(normalized)
        if active_command_id:
            raise ServiceDispatchError(f"{normalized.title()} worker is busy.")
        self._service_active_commands[normalized] = str(command_id_value or "").strip()

    def _release_service_slot_locked(self, service_name: str) -> None:
        normalized = str(service_name or "").strip().lower()
        self._service_active_commands.pop(normalized, None)

    def _format_log_payload(self, payload: dict[str, Any]) -> str:
        service_name = str(payload.get("service_name", "") or "").strip() or "service"
        command_id_value = str(payload.get("command_id", "") or "").strip()
        lane = str(payload.get("lane", "") or "").strip()
        thread_name = str(payload.get("thread_name", "") or "").strip()
        parts = [f"[{service_name}]"]
        if command_id_value:
            parts.append(command_id_value)
        if lane:
            parts.append(f"lane={lane}")
        if thread_name:
            parts.append(f"thread={thread_name}")
        message = str(payload.get("message", "") or "").strip()
        if message:
            parts.append(message)
        return " ".join(part for part in parts if part)

    def _command_log_line(self, payload: dict[str, Any], suffix: str) -> str:
        service_name = str(payload.get("service_name", "") or "").strip() or "service"
        action = str(payload.get("action", "") or "").strip() or "command"
        command_id_value = str(payload.get("command_id", "") or "").strip()
        page_name = str(payload.get("image_relative_path", "") or "").strip()
        parts = [f"[{service_name}]", action, suffix]
        if command_id_value:
            parts.append(command_id_value)
        if page_name:
            parts.append(page_name)
        return " ".join(part for part in parts if part)


__all__ = ["ServiceManager"]
