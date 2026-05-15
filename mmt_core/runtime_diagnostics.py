"""Immediate, flushed runtime diagnostics for crash-prone native/model stages."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import threading
from typing import Any


def resolve_runtime_diagnostics_path(
    *,
    project_root: Path | str | None = None,
    workspace_root: Path | str | None = None,
) -> Path:
    if project_root is not None:
        return Path(project_root) / "cache" / "logs" / "runtime_diagnostics.log"
    if workspace_root is not None:
        return Path(workspace_root) / "logs" / "runtime_diagnostics.log"
    return Path.cwd() / "logs" / "runtime_diagnostics.log"


def write_runtime_diagnostic(
    message: str,
    *,
    log_path: Path | str | None = None,
    project_root: Path | str | None = None,
    workspace_root: Path | str | None = None,
    service: str = "",
    page: str = "",
    step: str = "",
    extra: dict[str, Any] | None = None,
) -> Path:
    target_path = Path(log_path) if log_path is not None else resolve_runtime_diagnostics_path(
        project_root=project_root,
        workspace_root=workspace_root,
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    thread = threading.current_thread()
    parts = [
        timestamp,
        f"service={str(service or '').strip() or 'runtime'}",
        f"thread={thread.name or 'unknown'}",
        f"tid={threading.get_ident()}",
    ]
    if page:
        parts.append(f"page={page}")
    if step:
        parts.append(f"step={step}")
    if extra:
        for key, value in extra.items():
            normalized_key = str(key or "").strip()
            if not normalized_key:
                continue
            parts.append(f"{normalized_key}={value}")
    parts.append(str(message or ""))
    line = " ".join(parts).rstrip() + "\n"

    with target_path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except Exception:
            pass
    return target_path


__all__ = [
    "resolve_runtime_diagnostics_path",
    "write_runtime_diagnostic",
]
