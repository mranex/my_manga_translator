"""Helpers for loading and saving manually edited render boxes."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .canon_state import (
    canon_item_bbox,
    ensure_canon_state,
    get_canon_item,
    resolve_canon_item_for_stage_item,
    save_canon_state_to_detection_path,
    set_canon_item_enabled,
    update_canon_item_bbox,
)
from .detection_io import load_detection_json, save_detection_json
from .image_io import ensure_path
from .render_io import load_render_json, normalize_render_item, save_render_json

DOWNSTREAM_STALE_STAGES = ["render", "export"]
MIN_BOX_SIZE = 4


def load_render_edit_items(path: Path | str) -> list[dict[str, Any]]:
    """Return render items with canon-driven geometry for GUI bbox editing."""

    json_path = ensure_path(path)
    payload = load_render_json(json_path)
    detection_path, detection_data = _load_detection_cache_for_render(json_path)
    _ = detection_path

    items = payload.get("items", [])
    normalized_items: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        normalized_item = normalize_render_item(deepcopy(item))
        canon_item = resolve_canon_item_for_stage_item(detection_data["canon_state"], normalized_item, active_only=False)
        if canon_item is None:
            raise ValueError(
                f"Render cache item {normalized_item.get('id')} could not be matched to canon_state. "
                "Re-prepare Render first."
            )
        normalized_item["canon_id"] = str(canon_item.get("canon_id", "") or "")
        normalized_item["bbox"] = canon_item_bbox(canon_item, "bbox")
        normalized_item["render_bbox"] = canon_item_bbox(canon_item, "render_bbox")
        normalized_item["excluded"] = not bool(canon_item.get("enabled", True))
        normalized_item["_saved_render_bbox"] = deepcopy(normalized_item.get("render_bbox"))
        normalized_item["_saved_excluded"] = bool(normalized_item.get("excluded", False))
        normalized_items.append(normalized_item)
    return normalized_items


def save_render_edit_items(
    path: Path | str,
    items: list[dict[str, Any]],
    *,
    mark_edited: bool = True,
) -> dict[str, Any]:
    """Persist render box edits into canon_state and refresh render JSON snapshots."""

    json_path = ensure_path(path)
    payload = load_render_json(json_path)
    detection_path, detection_data = _load_detection_cache_for_render(json_path)
    canon_state = detection_data["canon_state"]
    image_width = _safe_int(payload.get("image_width"))
    image_height = _safe_int(payload.get("image_height"))
    timestamp_value = _timestamp()

    edited_by_canon_id: dict[str, dict[str, Any]] = {}
    needs_render = False
    for raw_item in items:
        if not isinstance(raw_item, dict):
            continue
        normalized_item = _normalize_render_edit_item(
            raw_item,
            image_width=image_width,
            image_height=image_height,
            timestamp_value=timestamp_value,
            canon_state=canon_state,
        )
        edited_by_canon_id[str(normalized_item.get("canon_id", "") or "")] = normalized_item
        needs_render = needs_render or bool(normalized_item.get("needs_render", False))

    if not edited_by_canon_id:
        raise ValueError("No valid render items are available to save.")

    save_canon_state_to_detection_path(detection_path, canon_state)

    normalized_items: list[dict[str, Any]] = []
    for raw_item in payload.get("items", []):
        if not isinstance(raw_item, dict):
            continue
        normalized_item = normalize_render_item(deepcopy(raw_item))
        canon_item = resolve_canon_item_for_stage_item(canon_state, normalized_item, active_only=False)
        if canon_item is None:
            continue
        canon_id = str(canon_item.get("canon_id", "") or "")
        edited_item = edited_by_canon_id.get(canon_id)
        normalized_item["canon_id"] = canon_id
        # Snapshot only. canon_state remains the source of truth.
        normalized_item["bbox"] = canon_item_bbox(canon_item, "bbox")
        normalized_item["render_bbox"] = canon_item_bbox(canon_item, "render_bbox")
        normalized_item["excluded"] = not bool(canon_item.get("enabled", True))
        if edited_item is not None and bool(edited_item.get("needs_render", False)):
            normalized_item["bbox_edited"] = True
            normalized_item["bbox_edited_at"] = timestamp_value
            normalized_item["needs_render"] = True
        normalized_items.append(normalized_item)

    payload["items"] = normalized_items
    if mark_edited:
        payload["edited"] = True
        payload["edited_at"] = timestamp_value
        payload["needs_render"] = True if needs_render else bool(payload.get("needs_render", False))
        payload["downstream_stale"] = list(DOWNSTREAM_STALE_STAGES)
    return _save_render_payload(payload, json_path)


def update_render_item_bbox(path: Path | str, item_id: int, render_bbox: list[int]) -> dict[str, Any]:
    """Update one render item render_bbox and persist it."""

    json_path = ensure_path(path)
    items = load_render_edit_items(json_path)
    target_id = int(item_id)
    for item in items:
        if _safe_int(item.get("id")) != target_id:
            continue
        item["render_bbox"] = list(render_bbox)
        return save_render_edit_items(json_path, items, mark_edited=True)
    raise ValueError(f"Render item {item_id} was not found in {json_path}")


def exclude_render_item(path: Path | str, item_id: int, excluded: bool = True) -> dict[str, Any]:
    """Soft-delete or restore one render item by toggling canon_state enablement."""

    json_path = ensure_path(path)
    items = load_render_edit_items(json_path)
    target_id = int(item_id)
    for item in items:
        if _safe_int(item.get("id")) != target_id:
            continue
        item["excluded"] = bool(excluded)
        item["needs_render"] = True
        return save_render_edit_items(json_path, items, mark_edited=True)
    raise ValueError(f"Render item {item_id} was not found in {json_path}")


def restore_render_item(path: Path | str, item_id: int) -> dict[str, Any]:
    """Restore one previously excluded render item."""

    return exclude_render_item(path, item_id, False)


def summarize_render_edit_state(data: dict[str, Any]) -> dict[str, Any]:
    """Return compact render edit counts and stale markers."""

    summary = {
        "active_items": 0,
        "excluded_items": 0,
        "needs_render_items": 0,
        "edited": bool(data.get("edited", False)),
        "edited_at": str(data.get("edited_at", "") or ""),
        "needs_render": bool(data.get("needs_render", False)),
        "downstream_stale": list(data.get("downstream_stale", []) or []),
    }
    for raw_item in data.get("items", []):
        if not isinstance(raw_item, dict):
            continue
        item = normalize_render_item(raw_item)
        if bool(item.get("excluded", False)):
            summary["excluded_items"] += 1
            continue
        summary["active_items"] += 1
        if bool(item.get("needs_render", False)):
            summary["needs_render_items"] += 1
    return summary


def _normalize_render_edit_item(
    raw_item: dict[str, Any],
    *,
    image_width: int,
    image_height: int,
    timestamp_value: str,
    canon_state: dict[str, Any],
) -> dict[str, Any]:
    normalized = normalize_render_item(deepcopy(raw_item))
    item_id = normalized.get("id")
    if item_id is None:
        raise ValueError("Render item is missing an id.")
    normalized["id"] = _safe_int(item_id)

    canon_id = str(normalized.get("canon_id", "") or "").strip()
    if not canon_id:
        canon_item = resolve_canon_item_for_stage_item(canon_state, normalized, active_only=False)
        if canon_item is None:
            raise ValueError(
                f"Render item {normalized['id']} could not be matched to canon_state. Re-prepare Render first."
            )
        canon_id = str(canon_item.get("canon_id", "") or "")
    normalized["canon_id"] = canon_id

    render_bbox = _sanitize_bbox(
        normalized.get("render_bbox"),
        image_width=image_width,
        image_height=image_height,
    )
    if render_bbox is None:
        raise ValueError(f"Render item {normalized['id']} has an invalid render bbox.")
    normalized["render_bbox"] = render_bbox
    normalized["excluded"] = bool(normalized.get("excluded", False))
    normalized["bbox_edited"] = bool(normalized.get("bbox_edited", False))
    normalized["needs_render"] = bool(normalized.get("needs_render", False))

    original_bbox = _sanitize_bbox(
        raw_item.get("_saved_render_bbox") if isinstance(raw_item, dict) else None,
        image_width=image_width,
        image_height=image_height,
    )
    if original_bbox is None:
        original_bbox = _sanitize_bbox(
            raw_item.get("_original_render_bbox") if isinstance(raw_item, dict) else None,
            image_width=image_width,
            image_height=image_height,
        )
    if original_bbox is None:
        original_bbox = _sanitize_bbox(
            raw_item.get("render_bbox"),
            image_width=image_width,
            image_height=image_height,
        )
    original_excluded = bool(raw_item.get("_saved_excluded", raw_item.get("excluded", False)))

    if original_bbox is not None and list(original_bbox) != list(render_bbox):
        normalized["bbox_edited"] = True
        normalized["bbox_edited_at"] = timestamp_value
        normalized["needs_render"] = True
        update_canon_item_bbox(
            canon_state,
            canon_id,
            field="render_bbox",
            bbox=render_bbox,
            image_shape=(image_height, image_width, 3),
        )

    if original_excluded != bool(normalized["excluded"]):
        normalized["needs_render"] = True
    set_canon_item_enabled(canon_state, canon_id, not bool(normalized["excluded"]))

    normalized.pop("_original_render_bbox", None)
    normalized.pop("_saved_render_bbox", None)
    normalized.pop("_saved_excluded", None)
    return normalized


def _sanitize_bbox(
    bbox: Any,
    *,
    image_width: int,
    image_height: int,
) -> list[int] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(value))) for value in bbox[:4]]
    except (TypeError, ValueError):
        return None

    x1 = max(0, min(x1, image_width))
    x2 = max(0, min(x2, image_width))
    y1 = max(0, min(y1, image_height))
    y2 = max(0, min(y2, image_height))

    left = min(x1, x2)
    top = min(y1, y2)
    right = max(x1, x2)
    bottom = max(y1, y2)
    if right - left < MIN_BOX_SIZE or bottom - top < MIN_BOX_SIZE:
        return None
    return [left, top, right, bottom]


def _load_detection_cache_for_render(render_json_path_value: Path) -> tuple[Path, dict[str, Any]]:
    detection_path = render_json_path_value.parents[1] / "detection" / render_json_path_value.name
    if not detection_path.exists():
        raise FileNotFoundError(
            f"Detection cache is missing for {render_json_path_value.stem}. Run Detection first."
        )
    detection_data = load_detection_json(detection_path)
    had_canon_state = isinstance(detection_data.get("canon_state"), dict)
    ensure_canon_state(detection_data)
    if not had_canon_state:
        save_detection_json(detection_path, detection_data)
    return detection_path, detection_data


def _save_render_payload(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    try:
        save_render_json(path, payload)
    except Exception as exc:
        raise RuntimeError(f"Failed to save render cache: {path}. {exc}") from exc
    return load_render_json(path)


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = [
    "DOWNSTREAM_STALE_STAGES",
    "exclude_render_item",
    "load_render_edit_items",
    "restore_render_item",
    "save_render_edit_items",
    "summarize_render_edit_state",
    "update_render_item_bbox",
]
