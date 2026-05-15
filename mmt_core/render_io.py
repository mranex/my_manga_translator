"""Disk I/O helpers for rendered page caches and render metadata."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Protocol

from .image_io import ensure_path
from .json_io import write_json_atomic

RENDER_SCHEMA_VERSION = 1


class ProjectLike(Protocol):
    root_dir: Path
    cache_dir: Path


def render_image_path(project: ProjectLike, image_relative_path: Path | str) -> Path:
    relative_path = Path(image_relative_path)
    return ensure_path(project.cache_dir) / "render" / f"{relative_path.stem}.png"


def render_json_path(project: ProjectLike, image_relative_path: Path | str) -> Path:
    relative_path = Path(image_relative_path)
    return ensure_path(project.cache_dir) / "render" / f"{relative_path.stem}.json"


def render_sprite_dir(project: ProjectLike, image_relative_path: Path | str) -> Path:
    relative_path = Path(image_relative_path)
    return ensure_path(project.cache_dir) / "render_sprites" / relative_path.stem


def load_render_json(path: Path | str) -> dict[str, Any]:
    json_path = ensure_path(path)
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Render cache is not valid JSON: {json_path}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Render cache root must be an object: {json_path}")
    if payload.get("stage") != "render":
        raise ValueError(f"Unsupported render cache stage in {json_path}")

    items = payload.get("items", [])
    if not isinstance(items, list):
        raise ValueError(f"Render cache field 'items' must be a list in {json_path}")

    payload.setdefault("schema_version", RENDER_SCHEMA_VERSION)
    payload.setdefault("source_image", "")
    payload.setdefault("inpaint_image_path", "")
    payload.setdefault("translation_cache_path", "")
    payload.setdefault("ocr_cache_path", "")
    payload.setdefault("output_image_path", "")
    payload.setdefault("image_width", 0)
    payload.setdefault("image_height", 0)
    payload.setdefault("item_count", 0)
    payload.setdefault("rendered_item_count", 0)
    payload.setdefault("skipped_item_count", 0)
    payload.setdefault("no_text_page", False)
    payload.setdefault("status", "pending")
    payload.setdefault("error", "")
    payload.setdefault("created_at", "")
    payload.setdefault("updated_at", "")
    payload.setdefault("edited", False)
    payload.setdefault("edited_at", "")
    payload.setdefault("needs_render", False)
    payload["downstream_stale"] = (
        list(payload.get("downstream_stale", []) or [])
        if isinstance(payload.get("downstream_stale", []), list)
        else []
    )
    settings = payload.get("settings", {})
    if not isinstance(settings, dict):
        settings = {}
    payload["settings"] = {
        "font_name": str(settings.get("font_name", "") or ""),
        "font_path": str(settings.get("font_path", "") or ""),
        "font_size_mode": str(settings.get("font_size_mode", "fit") or "fit"),
        "min_font_size": int(settings.get("min_font_size", 12) or 12),
        "max_font_size": int(settings.get("max_font_size", 72) or 72),
        "text_color": settings.get("text_color"),
        "stroke_enabled": bool(settings.get("stroke_enabled", True)),
        "stroke_color": settings.get("stroke_color"),
        "stroke_width": settings.get("stroke_width"),
        "auto_color": bool(settings.get("auto_color", True)),
        "auto_direction": bool(settings.get("auto_direction", True)),
        "vertical_cjk": bool(settings.get("vertical_cjk", True)),
        "save_sprites": bool(settings.get("save_sprites", True)),
        "force": bool(settings.get("force", False)),
    }
    payload["items"] = [normalize_render_item(item) for item in items]
    return payload


def save_render_json(path: Path | str, data: dict[str, Any]) -> Path:
    json_path = ensure_path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    settings = data.get("settings", {}) if isinstance(data.get("settings", {}), dict) else {}
    payload = {str(key): value for key, value in dict(data).items()}
    payload.update({
        "schema_version": int(data.get("schema_version", RENDER_SCHEMA_VERSION)),
        "stage": "render",
        "source_image": str(data.get("source_image", "")),
        "inpaint_image_path": str(data.get("inpaint_image_path", "")),
        "translation_cache_path": str(data.get("translation_cache_path", "")),
        "ocr_cache_path": str(data.get("ocr_cache_path", "")),
        "output_image_path": str(data.get("output_image_path", "")),
        "image_width": int(data.get("image_width", 0) or 0),
        "image_height": int(data.get("image_height", 0) or 0),
        "item_count": int(data.get("item_count", 0) or 0),
        "rendered_item_count": int(data.get("rendered_item_count", 0) or 0),
        "skipped_item_count": int(data.get("skipped_item_count", 0) or 0),
        "no_text_page": bool(data.get("no_text_page", False)),
        "status": str(data.get("status", "pending") or "pending"),
        "error": str(data.get("error", "") or ""),
        "created_at": str(data.get("created_at", "") or ""),
        "updated_at": str(data.get("updated_at", "") or ""),
        "edited": bool(data.get("edited", False)),
        "edited_at": str(data.get("edited_at", "") or ""),
        "needs_render": bool(data.get("needs_render", False)),
        "downstream_stale": list(data.get("downstream_stale", []) or []),
        "settings": {
            "font_name": str(settings.get("font_name", "") or ""),
            "font_path": str(settings.get("font_path", "") or ""),
            "font_size_mode": str(settings.get("font_size_mode", "fit") or "fit"),
            "min_font_size": int(settings.get("min_font_size", 12) or 12),
            "max_font_size": int(settings.get("max_font_size", 72) or 72),
            "text_color": settings.get("text_color"),
            "stroke_enabled": bool(settings.get("stroke_enabled", True)),
            "stroke_color": settings.get("stroke_color"),
            "stroke_width": settings.get("stroke_width"),
            "auto_color": bool(settings.get("auto_color", True)),
            "auto_direction": bool(settings.get("auto_direction", True)),
            "vertical_cjk": bool(settings.get("vertical_cjk", True)),
            "save_sprites": bool(settings.get("save_sprites", True)),
            "force": bool(settings.get("force", False)),
        },
        "items": [normalize_render_item(item) for item in data.get("items", [])],
    })
    return write_json_atomic(json_path, payload, indent=2, ensure_ascii=False)


def summarize_render_json(data: dict[str, Any]) -> dict[str, int | str]:
    summary: dict[str, int | str] = {
        "status": str(data.get("status", "pending") or "pending"),
        "total": 0,
        "rendered": 0,
        "skipped": 0,
        "error": 0,
        "excluded": 0,
    }
    for item in data.get("items", []):
        if not isinstance(item, dict):
            continue
        if bool(item.get("excluded", False)):
            summary["excluded"] = int(summary["excluded"]) + 1
            continue
        summary["total"] = int(summary["total"]) + 1
        status = str(item.get("status", "") or "").strip().lower()
        if status == "rendered":
            summary["rendered"] = int(summary["rendered"]) + 1
        elif status == "skipped":
            summary["skipped"] = int(summary["skipped"]) + 1
        elif status == "error":
            summary["error"] = int(summary["error"]) + 1
    return summary


def normalize_render_item(item: dict[str, Any] | Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        item = {}

    normalized = {str(key): value for key, value in item.items()}
    normalized.setdefault("id", 0)
    normalized.setdefault("translation_item_id", 0)
    normalized.setdefault("ocr_item_id", 0)
    normalized.setdefault("canon_id", "")
    normalized.setdefault("kind", "")
    normalized.setdefault("source_text", "")
    normalized.setdefault("translated_text", "")
    normalized.setdefault("bbox", None)
    normalized.setdefault("render_bbox", None)
    normalized.setdefault("writing_mode", "horizontal")
    normalized.setdefault("font_size", 0)
    normalized.setdefault("font_path", "")
    normalized.setdefault("text_color", None)
    normalized.setdefault("stroke_color", None)
    normalized.setdefault("stroke_width", 0.0)
    normalized.setdefault("sprite_path", "")
    normalized.setdefault("sprite_transform", {})
    normalized.setdefault("status", "pending")
    normalized.setdefault("error", "")
    normalized.setdefault("excluded", False)
    normalized.setdefault("bbox_edited", False)
    normalized.setdefault("bbox_edited_at", "")
    normalized.setdefault("needs_render", False)
    return normalized


def timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


__all__ = [
    "RENDER_SCHEMA_VERSION",
    "load_render_json",
    "normalize_render_item",
    "render_image_path",
    "render_json_path",
    "render_sprite_dir",
    "save_render_json",
    "summarize_render_json",
    "timestamp",
]
