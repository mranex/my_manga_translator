"""Shared command, result, and status models for resident stage services."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import threading
import uuid
from typing import Any, Callable

from mmt_gui.workers import WorkerSignals


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class CancelToken:
    """Thread-safe cooperative cancellation token."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._requested_at = ""

    def request_cancel(self) -> None:
        self._requested_at = self._requested_at or utc_timestamp()
        self._event.set()

    def is_cancel_requested(self) -> bool:
        return self._event.is_set()

    @property
    def requested_at(self) -> str:
        return self._requested_at


@dataclass(slots=True)
class ServiceCommand:
    """One queued command sent to a resident service."""

    command_id: str
    service_name: str
    action: str
    stage: str
    task: Any
    project_root: str = ""
    cache_dir: str = ""
    image_relative_paths: list[str] = field(default_factory=list)
    current_page: str = ""
    force: bool = False
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_timestamp)
    cancel_token: CancelToken = field(default_factory=CancelToken)


@dataclass(slots=True)
class ServiceStatusSnapshot:
    """Current public state for one resident service."""

    service_name: str
    state: str
    message: str = ""
    active_command_id: str = ""
    queued_count: int = 0
    updated_at: str = field(default_factory=utc_timestamp)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ServiceEvent:
    """Structured event emitted by a resident service."""

    command_id: str
    service_name: str
    action: str
    stage: str
    event: str
    image_relative_path: str = ""
    page_index: int = 0
    page_total: int = 0
    progress: int = 0
    message: str = ""
    output_path: str = ""
    summary: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "command_id": self.command_id,
            "service_name": self.service_name,
            "action": self.action,
            "stage": self.stage,
            "event": self.event,
        }
        if self.image_relative_path:
            payload["image_relative_path"] = self.image_relative_path
        if self.page_index:
            payload["page_index"] = int(self.page_index)
        if self.page_total:
            payload["page_total"] = int(self.page_total)
        if self.progress:
            payload["progress"] = int(self.progress)
        if self.message:
            payload["message"] = self.message
        if self.output_path:
            payload["output_path"] = self.output_path
        if self.summary:
            payload["summary"] = dict(self.summary)
        if self.error:
            payload["error"] = self.error
        if self.metadata:
            payload.update(dict(self.metadata))
        return payload


@dataclass(slots=True)
class ServiceCommandResult:
    """Final outcome stored for one service command."""

    command_id: str
    service_name: str
    action: str
    stage: str
    state: str
    result: Any = None
    error: str = ""
    summary: dict[str, Any] = field(default_factory=dict)
    canceled: bool = False
    finished_at: str = field(default_factory=utc_timestamp)


class ServiceDispatchError(RuntimeError):
    """Raised when a command cannot be submitted to a resident service."""


class ServiceCommandHandle:
    """UI-facing handle that mimics the old worker signal surface."""

    def __init__(
        self,
        *,
        command_id: str,
        task: Any,
        service_name: str,
        cancel_callback: Callable[[str], None],
    ) -> None:
        self.command_id = command_id
        self.task = task
        self.service_name = service_name
        self.signals = WorkerSignals()
        self._cancel_callback = cancel_callback

    def request_cancel(self) -> None:
        self._cancel_callback(self.command_id)

    def cancel_requested(self) -> bool:
        return False


def command_id(prefix: str) -> str:
    normalized = str(prefix or "cmd").strip().lower().replace(" ", "_")
    return f"{normalized}_{uuid.uuid4().hex}"


def project_root_from_task(task: Any) -> str:
    project = getattr(task, "project", None)
    root_dir = getattr(project, "root_dir", None)
    if root_dir is not None:
        return str(Path(root_dir))

    detection_cache_dir = getattr(task, "detection_cache_dir", None)
    if detection_cache_dir is not None:
        try:
            return str(Path(detection_cache_dir).resolve().parents[1])
        except Exception:
            return str(Path(detection_cache_dir))
    return ""


def cache_dir_from_task(task: Any) -> str:
    project = getattr(task, "project", None)
    cache_dir = getattr(project, "cache_dir", None)
    if cache_dir is not None:
        return str(Path(cache_dir))
    detection_cache_dir = getattr(task, "detection_cache_dir", None)
    if detection_cache_dir is not None:
        return str(Path(detection_cache_dir))
    return ""


def image_relative_paths_from_task(task: Any) -> list[str]:
    if hasattr(task, "image_relative_paths"):
        return [str(path) for path in list(getattr(task, "image_relative_paths") or []) if str(path).strip()]
    if hasattr(task, "image_paths"):
        return [str(Path(path)) for path in list(getattr(task, "image_paths") or [])]
    current_page = getattr(task, "current_page", None)
    if current_page:
        return [str(current_page)]
    return []
