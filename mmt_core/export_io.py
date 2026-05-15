"""Disk I/O helpers for export manifests, filenames, and archive paths."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from .image_io import ensure_path
from .json_io import write_json_atomic

EXPORT_SCHEMA_VERSION = 1
INVALID_WINDOWS_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sanitize_windows_filename(value: str, fallback: str = "export") -> str:
    candidate = INVALID_WINDOWS_CHARS.sub("_", str(value or "")).strip()
    candidate = candidate.strip(". ").replace("\n", " ").replace("\r", " ")
    if not candidate:
        candidate = fallback
    return candidate


def ensure_export_dir(output_dir: Path | str) -> Path:
    path = ensure_path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    probe_path = path / ".mmt_export_write_test"
    probe_path.write_text("ok", encoding="utf-8")
    probe_path.unlink(missing_ok=True)
    return path


def ensure_unique_path(path: Path | str, overwrite: bool) -> Path:
    target_path = ensure_path(path)
    if overwrite or not target_path.exists():
        return target_path

    stem = target_path.stem
    suffix = target_path.suffix
    parent = target_path.parent
    counter = 1
    while True:
        candidate = parent / f"{stem}_{counter:03d}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def export_manifest_path(output_dir: Path | str, overwrite: bool) -> Path:
    base_path = ensure_export_dir(output_dir) / "export_manifest.json"
    return ensure_unique_path(base_path, overwrite=overwrite)


def export_zip_path(output_dir: Path | str, zip_name: str, overwrite: bool) -> Path:
    safe_name = sanitize_windows_filename(zip_name or "export.zip", fallback="export.zip")
    if not safe_name.lower().endswith(".zip"):
        safe_name = f"{safe_name}.zip"
    base_path = ensure_export_dir(output_dir) / safe_name
    return ensure_unique_path(base_path, overwrite=overwrite)


def save_export_manifest(path: Path | str, data: dict[str, Any]) -> Path:
    manifest_path = ensure_path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    return write_json_atomic(manifest_path, data, indent=2, ensure_ascii=False)


def build_export_basename(
    pattern: str,
    *,
    index: int,
    page_number: int,
    stem: str,
    source_stem: str,
    project_name: str,
    stage_name: str,
) -> tuple[str, bool]:
    values = {
        "index": int(index),
        "page_number": int(page_number),
        "stem": str(stem or ""),
        "source_stem": str(source_stem or ""),
        "project": str(project_name or ""),
        "stage": str(stage_name or ""),
    }
    try:
        rendered = str(pattern or "{index:04d}_{stem}").format(**values)
        return sanitize_windows_filename(rendered, fallback=f"{index:04d}_{source_stem or stem or 'page'}"), False
    except Exception:
        fallback = f"{index:04d}_{source_stem or stem or 'page'}"
        return sanitize_windows_filename(fallback, fallback="page"), True


__all__ = [
    "EXPORT_SCHEMA_VERSION",
    "build_export_basename",
    "ensure_export_dir",
    "ensure_unique_path",
    "export_manifest_path",
    "export_zip_path",
    "sanitize_windows_filename",
    "save_export_manifest",
    "timestamp",
]
