"""Disk I/O helpers for inpaint cache files and masks."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Protocol

from .image_io import ensure_path, project_relative_path

INPAINT_SCHEMA_VERSION = 1


class ProjectLike(Protocol):
    root_dir: Path
    cache_dir: Path


def inpaint_image_path(project: ProjectLike, image_relative_path: Path | str) -> Path:
    relative_path = Path(image_relative_path)
    return ensure_path(project.cache_dir) / "inpaint" / f"{relative_path.stem}.png"


def inpaint_json_path(project: ProjectLike, image_relative_path: Path | str) -> Path:
    relative_path = Path(image_relative_path)
    return ensure_path(project.cache_dir) / "inpaint" / f"{relative_path.stem}.json"


def text_mask_path(project: ProjectLike, image_relative_path: Path | str) -> Path:
    relative_path = Path(image_relative_path)
    return ensure_path(project.cache_dir) / "masks" / relative_path.stem / "text_removal_mask.png"


def bubble_mask_path(project: ProjectLike, image_relative_path: Path | str) -> Path:
    relative_path = Path(image_relative_path)
    return ensure_path(project.cache_dir) / "masks" / relative_path.stem / "bubble_mask.png"


def inpaint_preview_mask_path(project: ProjectLike, image_relative_path: Path | str) -> Path:
    relative_path = Path(image_relative_path)
    return ensure_path(project.cache_dir) / "masks" / relative_path.stem / "inpaint_preview_mask.png"


def load_inpaint_json(path: Path | str) -> dict[str, Any]:
    json_path = ensure_path(path)
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Inpaint cache is not valid JSON: {json_path}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Inpaint cache root must be an object: {json_path}")

    if payload.get("stage") != "inpaint":
        raise ValueError(f"Unsupported inpaint cache stage in {json_path}")

    settings = payload.get("settings", {})
    if not isinstance(settings, dict):
        settings = {}

    payload.setdefault("schema_version", INPAINT_SCHEMA_VERSION)
    payload.setdefault("source_image", "")
    payload.setdefault("ocr_cache_path", "")
    payload.setdefault("detection_cache_path", "")
    payload.setdefault("output_image_path", "")
    payload.setdefault("text_mask_path", "")
    payload.setdefault("bubble_mask_path", "")
    payload.setdefault("text_mask_hash", "")
    payload.setdefault("bubble_mask_hash", "")
    payload.setdefault("output_mask_hash", "")
    payload.setdefault("output_bubble_mask_hash", "")
    payload.setdefault("needs_inpaint", False)
    payload.setdefault("mask_created_at", "")
    payload.setdefault("mask_updated_at", "")
    payload.setdefault("inpaint_created_at", "")
    payload.setdefault("inpaint_updated_at", "")
    payload.setdefault("image_width", 0)
    payload.setdefault("image_height", 0)
    payload.setdefault("item_count", 0)
    payload.setdefault("active_item_count", 0)
    payload.setdefault("text_mask_box_count", 0)
    payload.setdefault("skipped_item_count", 0)
    payload.setdefault("masked_pixel_count", 0)
    payload.setdefault("bubble_mask_pixel_count", 0)
    payload.setdefault("device", "")
    payload.setdefault("status", "pending")
    payload.setdefault("error", "")
    payload.setdefault("created_at", "")
    payload.setdefault("updated_at", "")
    payload["settings"] = {
        "mask_padding": int(settings.get("mask_padding", 8) or 8),
        "use_bubble_mask": bool(settings.get("use_bubble_mask", True)),
        "use_crop_windows": bool(settings.get("use_crop_windows", True)),
        "crop_trigger_size": int(settings.get("crop_trigger_size", 800) or 800),
        "crop_margin": int(settings.get("crop_margin", 128) or 128),
        "resize_limit": int(settings.get("resize_limit", 1280) or 1280),
    }
    return payload


def save_inpaint_json(path: Path | str, data: dict[str, Any]) -> Path:
    json_path = ensure_path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    settings = data.get("settings", {}) if isinstance(data.get("settings", {}), dict) else {}
    payload = {
        "schema_version": int(data.get("schema_version", INPAINT_SCHEMA_VERSION)),
        "stage": "inpaint",
        "source_image": str(data.get("source_image", "")),
        "ocr_cache_path": str(data.get("ocr_cache_path", "")),
        "detection_cache_path": str(data.get("detection_cache_path", "")),
        "output_image_path": str(data.get("output_image_path", "")),
        "text_mask_path": str(data.get("text_mask_path", "")),
        "bubble_mask_path": str(data.get("bubble_mask_path", "")),
        "text_mask_hash": str(data.get("text_mask_hash", "")),
        "bubble_mask_hash": str(data.get("bubble_mask_hash", "")),
        "output_mask_hash": str(data.get("output_mask_hash", "")),
        "output_bubble_mask_hash": str(data.get("output_bubble_mask_hash", "")),
        "needs_inpaint": bool(data.get("needs_inpaint", False)),
        "mask_created_at": str(data.get("mask_created_at", "")),
        "mask_updated_at": str(data.get("mask_updated_at", "")),
        "inpaint_created_at": str(data.get("inpaint_created_at", "")),
        "inpaint_updated_at": str(data.get("inpaint_updated_at", "")),
        "image_width": int(data.get("image_width", 0) or 0),
        "image_height": int(data.get("image_height", 0) or 0),
        "item_count": int(data.get("item_count", 0) or 0),
        "active_item_count": int(data.get("active_item_count", 0) or 0),
        "text_mask_box_count": int(data.get("text_mask_box_count", 0) or 0),
        "skipped_item_count": int(data.get("skipped_item_count", 0) or 0),
        "masked_pixel_count": int(data.get("masked_pixel_count", 0) or 0),
        "bubble_mask_pixel_count": int(data.get("bubble_mask_pixel_count", 0) or 0),
        "device": str(data.get("device", "")),
        "status": str(data.get("status", "pending")),
        "error": str(data.get("error", "")),
        "created_at": str(data.get("created_at", "")),
        "updated_at": str(data.get("updated_at", "")),
        "settings": {
            "mask_padding": int(settings.get("mask_padding", 8) or 8),
            "use_bubble_mask": bool(settings.get("use_bubble_mask", True)),
            "use_crop_windows": bool(settings.get("use_crop_windows", True)),
            "crop_trigger_size": int(settings.get("crop_trigger_size", 800) or 800),
            "crop_margin": int(settings.get("crop_margin", 128) or 128),
            "resize_limit": int(settings.get("resize_limit", 1280) or 1280),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return json_path


def summarize_inpaint_json(data: dict[str, Any]) -> dict[str, Any]:
    status = str(data.get("status", "pending") or "pending")
    if bool(data.get("needs_inpaint", False)) and status.strip().lower() == "done":
        status = "prepared"
    return {
        "status": status,
        "item_count": int(data.get("item_count", 0) or 0),
        "active_item_count": int(data.get("active_item_count", 0) or 0),
        "text_mask_box_count": int(data.get("text_mask_box_count", 0) or 0),
        "skipped_item_count": int(data.get("skipped_item_count", 0) or 0),
        "masked_pixel_count": int(data.get("masked_pixel_count", 0) or 0),
        "bubble_mask_pixel_count": int(data.get("bubble_mask_pixel_count", 0) or 0),
        "has_output_image": bool(str(data.get("output_image_path", "")).strip()),
        "has_text_mask": bool(str(data.get("text_mask_path", "")).strip()),
        "has_bubble_mask": bool(str(data.get("bubble_mask_path", "")).strip()),
        "needs_inpaint": bool(data.get("needs_inpaint", False)),
        "error": str(data.get("error", "") or ""),
    }


def build_inpaint_metadata(
    *,
    project_root: Path | str,
    image_relative_path: Path | str,
    ocr_cache_path: Path | str,
    detection_cache_path: Path | str | None,
    output_image_path_value: Path | str,
    text_mask_path_value: Path | str,
    bubble_mask_path_value: Path | str | None,
    text_mask_hash: str,
    bubble_mask_hash: str,
    output_mask_hash: str,
    output_bubble_mask_hash: str,
    image_shape,
    item_count: int,
    active_item_count: int,
    text_mask_box_count: int,
    skipped_item_count: int,
    masked_pixel_count: int,
    bubble_mask_pixel_count: int,
    device: str,
    status: str,
    error: str,
    needs_inpaint: bool = False,
    created_at: str | None = None,
    updated_at: str | None = None,
    mask_created_at: str | None = None,
    mask_updated_at: str | None = None,
    inpaint_created_at: str | None = None,
    inpaint_updated_at: str | None = None,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root_path = ensure_path(project_root)
    return {
        "schema_version": INPAINT_SCHEMA_VERSION,
        "stage": "inpaint",
        "source_image": project_relative_path(root_path, root_path / Path(str(image_relative_path))),
        "ocr_cache_path": project_relative_path(root_path, ocr_cache_path),
        "detection_cache_path": project_relative_path(root_path, detection_cache_path) if detection_cache_path else "",
        "output_image_path": project_relative_path(root_path, output_image_path_value),
        "text_mask_path": project_relative_path(root_path, text_mask_path_value),
        "bubble_mask_path": project_relative_path(root_path, bubble_mask_path_value) if bubble_mask_path_value else "",
        "text_mask_hash": str(text_mask_hash or ""),
        "bubble_mask_hash": str(bubble_mask_hash or ""),
        "output_mask_hash": str(output_mask_hash or ""),
        "output_bubble_mask_hash": str(output_bubble_mask_hash or ""),
        "needs_inpaint": bool(needs_inpaint),
        "mask_created_at": str(mask_created_at or created_at or _timestamp()),
        "mask_updated_at": str(mask_updated_at or updated_at or _timestamp()),
        "inpaint_created_at": str(inpaint_created_at or ""),
        "inpaint_updated_at": str(inpaint_updated_at or ""),
        "image_width": int(image_shape[1]),
        "image_height": int(image_shape[0]),
        "item_count": int(item_count),
        "active_item_count": int(active_item_count),
        "text_mask_box_count": int(text_mask_box_count),
        "skipped_item_count": int(skipped_item_count),
        "masked_pixel_count": int(masked_pixel_count),
        "bubble_mask_pixel_count": int(bubble_mask_pixel_count),
        "device": str(device),
        "status": str(status),
        "error": str(error or ""),
        "created_at": str(created_at or _timestamp()),
        "updated_at": str(updated_at or _timestamp()),
        "settings": dict(settings or {}),
    }


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


__all__ = [
    "INPAINT_SCHEMA_VERSION",
    "bubble_mask_path",
    "build_inpaint_metadata",
    "inpaint_image_path",
    "inpaint_json_path",
    "inpaint_preview_mask_path",
    "load_inpaint_json",
    "save_inpaint_json",
    "summarize_inpaint_json",
    "text_mask_path",
]
