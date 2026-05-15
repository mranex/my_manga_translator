"""Helpers for loading and saving manually edited OCR boxes."""

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
from .ocr_io import load_ocr_json, normalize_ocr_item, save_ocr_payload

DOWNSTREAM_STALE_STAGES = ["translation", "inpaint", "render", "export"]
MIN_BOX_SIZE = 4


def load_ocr_edit_items(path: Path | str) -> list[dict[str, Any]]:
    """Return OCR items with canon-driven edit defaults."""

    json_path = ensure_path(path)
    payload = load_ocr_json(json_path)
    detection_path, detection_data = _load_detection_cache_for_ocr(json_path)
    _ = detection_path

    items = payload.get("items", [])
    normalized_items: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        normalized_item = normalize_ocr_item(deepcopy(item))
        canon_item = resolve_canon_item_for_stage_item(detection_data["canon_state"], normalized_item, active_only=False)
        if canon_item is None:
            raise ValueError(
                f"OCR cache item {normalized_item.get('id')} could not be matched to canon_state. "
                "Re-prepare OCR items first."
            )
        normalized_item["canon_id"] = str(canon_item.get("canon_id", "") or "")
        normalized_item["bbox"] = canon_item_bbox(canon_item, "bbox")
        normalized_item["ocr_bbox"] = canon_item_bbox(canon_item, "ocr_bbox")
        normalized_item["excluded"] = not bool(canon_item.get("enabled", True))
        normalized_item["_saved_bbox"] = deepcopy(normalized_item.get("bbox"))
        normalized_item["_saved_ocr_bbox"] = deepcopy(normalized_item.get("ocr_bbox"))
        normalized_item["_saved_excluded"] = bool(normalized_item.get("excluded", False))
        normalized_items.append(normalized_item)
    return normalized_items


def save_ocr_edit_items(
    path: Path | str,
    items: list[dict[str, Any]],
    *,
    mark_edited: bool = True,
) -> dict[str, Any]:
    """Persist OCR item bbox edits into canon_state and refresh OCR JSON snapshots."""

    json_path = ensure_path(path)
    payload = load_ocr_json(json_path)
    detection_path, detection_data = _load_detection_cache_for_ocr(json_path)
    canon_state = detection_data["canon_state"]
    image_width = _safe_int(payload.get("image_width"))
    image_height = _safe_int(payload.get("image_height"))
    timestamp_value = _timestamp()

    edited_by_canon_id: dict[str, dict[str, Any]] = {}
    for raw_item in items:
        if not isinstance(raw_item, dict):
            continue
        normalized_item = _normalize_ocr_edit_item(
            raw_item,
            image_width=image_width,
            image_height=image_height,
            timestamp_value=timestamp_value,
            canon_state=canon_state,
        )
        edited_by_canon_id[str(normalized_item.get("canon_id", "") or "")] = normalized_item

    if not edited_by_canon_id:
        raise ValueError("No valid OCR items are available to save.")

    save_canon_state_to_detection_path(detection_path, canon_state)

    normalized_items: list[dict[str, Any]] = []
    for raw_item in payload.get("items", []):
        if not isinstance(raw_item, dict):
            continue
        normalized_item = normalize_ocr_item(deepcopy(raw_item))
        canon_item = resolve_canon_item_for_stage_item(canon_state, normalized_item, active_only=False)
        if canon_item is None:
            continue
        canon_id = str(canon_item.get("canon_id", "") or "")
        edited_item = edited_by_canon_id.get(canon_id)
        normalized_item["canon_id"] = canon_id
        # Snapshot only. canon_state remains the source of truth.
        normalized_item["bbox"] = canon_item_bbox(canon_item, "bbox")
        normalized_item["ocr_bbox"] = canon_item_bbox(canon_item, "ocr_bbox")
        normalized_item["excluded"] = not bool(canon_item.get("enabled", True))
        if edited_item is not None:
            if bool(edited_item.get("bbox_edited", False)):
                normalized_item["bbox_edited"] = True
                normalized_item["bbox_edited_at"] = timestamp_value
            if bool(edited_item.get("needs_ocr", False)):
                normalized_item["needs_ocr"] = True
                if not bool(normalized_item.get("excluded", False)):
                    normalized_item["status"] = "prepared"
                normalized_item["updated_at"] = timestamp_value
        normalized_items.append(normalized_item)

    payload["items"] = normalized_items
    if mark_edited:
        payload["edited"] = True
        payload["edited_at"] = timestamp_value
        payload["downstream_stale"] = list(DOWNSTREAM_STALE_STAGES)
    return _save_ocr_payload(payload, json_path)


def update_ocr_item_bbox(
    path: Path | str,
    item_id: int,
    bbox: list[int] | None = None,
    ocr_bbox: list[int] | None = None,
) -> dict[str, Any]:
    """Update one OCR item bbox and/or OCR crop bbox."""

    json_path = ensure_path(path)
    items = load_ocr_edit_items(json_path)
    target_id = int(item_id)
    for item in items:
        if _safe_int(item.get("id")) != target_id:
            continue
        if bbox is not None:
            item["bbox"] = list(bbox)
        if ocr_bbox is not None:
            item["ocr_bbox"] = list(ocr_bbox)
        return save_ocr_edit_items(json_path, items, mark_edited=True)
    raise ValueError(f"OCR item {item_id} was not found in {json_path}")


def exclude_ocr_item(path: Path | str, item_id: int, excluded: bool = True) -> dict[str, Any]:
    """Soft-delete or restore one OCR item by toggling canon_state enablement."""

    json_path = ensure_path(path)
    items = load_ocr_edit_items(json_path)
    target_id = int(item_id)
    for item in items:
        if _safe_int(item.get("id")) != target_id:
            continue
        item["excluded"] = bool(excluded)
        return save_ocr_edit_items(json_path, items, mark_edited=True)
    raise ValueError(f"OCR item {item_id} was not found in {json_path}")


def restore_ocr_item(path: Path | str, item_id: int) -> dict[str, Any]:
    """Restore one previously excluded OCR item."""

    return exclude_ocr_item(path, item_id, False)


def summarize_ocr_edit_state(data: dict[str, Any]) -> dict[str, Any]:
    """Return compact OCR edit counts and stale markers."""

    summary = {
        "active_items": 0,
        "excluded_items": 0,
        "needs_ocr_items": 0,
        "edited": bool(data.get("edited", False)),
        "edited_at": str(data.get("edited_at", "") or ""),
        "downstream_stale": list(data.get("downstream_stale", []) or []),
    }
    for raw_item in data.get("items", []):
        if not isinstance(raw_item, dict):
            continue
        item = normalize_ocr_item(raw_item)
        if bool(item.get("excluded", False)):
            summary["excluded_items"] += 1
            continue
        summary["active_items"] += 1
        if bool(item.get("needs_ocr", False)):
            summary["needs_ocr_items"] += 1
    return summary


def _normalize_ocr_edit_item(
    raw_item: dict[str, Any],
    *,
    image_width: int,
    image_height: int,
    timestamp_value: str,
    canon_state: dict[str, Any],
) -> dict[str, Any]:
    normalized = normalize_ocr_item(deepcopy(raw_item))
    item_id = normalized.get("id")
    if item_id is None:
        raise ValueError("OCR item is missing an id.")
    normalized["id"] = _safe_int(item_id)

    canon_id = str(normalized.get("canon_id", "") or "").strip()
    if not canon_id:
        canon_item = resolve_canon_item_for_stage_item(canon_state, normalized, active_only=False)
        if canon_item is None:
            raise ValueError(
                f"OCR item {normalized['id']} could not be matched to canon_state. Re-prepare OCR items first."
            )
        canon_id = str(canon_item.get("canon_id", "") or "")
    normalized["canon_id"] = canon_id

    bbox = _sanitize_bbox(normalized.get("bbox"), image_width=image_width, image_height=image_height)
    if bbox is None:
        raise ValueError(f"OCR item {normalized['id']} has an invalid bbox.")
    normalized["bbox"] = bbox

    if normalized.get("ocr_bbox") is None:
        normalized["ocr_bbox"] = list(bbox)
    else:
        sanitized_ocr_bbox = _sanitize_bbox(
            normalized.get("ocr_bbox"),
            image_width=image_width,
            image_height=image_height,
        )
        if sanitized_ocr_bbox is None:
            raise ValueError(f"OCR item {normalized['id']} has an invalid OCR bbox.")
        normalized["ocr_bbox"] = sanitized_ocr_bbox

    normalized["excluded"] = bool(normalized.get("excluded", False))
    normalized["bbox_edited"] = bool(normalized.get("bbox_edited", False))
    normalized["needs_ocr"] = bool(normalized.get("needs_ocr", False))

    original_bbox = _sanitize_bbox(
        raw_item.get("_saved_bbox") if isinstance(raw_item, dict) else None,
        image_width=image_width,
        image_height=image_height,
    )
    if original_bbox is None:
        original_bbox = _sanitize_bbox(
            raw_item.get("_original_bbox") if isinstance(raw_item, dict) else None,
            image_width=image_width,
            image_height=image_height,
        )
    original_ocr_bbox = _sanitize_bbox(
        raw_item.get("_saved_ocr_bbox") if isinstance(raw_item, dict) else None,
        image_width=image_width,
        image_height=image_height,
    )
    if original_ocr_bbox is None:
        original_ocr_bbox = _sanitize_bbox(
            raw_item.get("_original_ocr_bbox") if isinstance(raw_item, dict) else None,
            image_width=image_width,
            image_height=image_height,
        )
    if original_bbox is None:
        original_bbox = _sanitize_bbox(
            raw_item.get("bbox"),
            image_width=image_width,
            image_height=image_height,
        )
    if original_ocr_bbox is None:
        original_ocr_bbox = _sanitize_bbox(
            raw_item.get("ocr_bbox"),
            image_width=image_width,
            image_height=image_height,
        )

    bbox_changed = original_bbox is not None and list(original_bbox) != list(normalized["bbox"])
    ocr_bbox_changed = original_ocr_bbox is not None and list(original_ocr_bbox) != list(normalized["ocr_bbox"])
    if bbox_changed or ocr_bbox_changed:
        normalized["bbox_edited"] = True
        normalized["bbox_edited_at"] = timestamp_value
    if bbox_changed:
        update_canon_item_bbox(
            canon_state,
            canon_id,
            field="bbox",
            bbox=normalized["bbox"],
            image_shape=(image_height, image_width, 3),
        )
    if ocr_bbox_changed:
        update_canon_item_bbox(
            canon_state,
            canon_id,
            field="ocr_bbox",
            bbox=normalized["ocr_bbox"],
            image_shape=(image_height, image_width, 3),
        )
        normalized["needs_ocr"] = True
        if not normalized["excluded"]:
            normalized["status"] = "prepared"
        normalized["updated_at"] = timestamp_value

    original_excluded = bool(raw_item.get("_saved_excluded", raw_item.get("excluded", False)))
    set_canon_item_enabled(canon_state, canon_id, not bool(normalized["excluded"]))
    if original_excluded != bool(normalized["excluded"]):
        normalized["needs_ocr"] = True
        normalized["updated_at"] = timestamp_value

    normalized.pop("_original_bbox", None)
    normalized.pop("_original_ocr_bbox", None)
    normalized.pop("_saved_bbox", None)
    normalized.pop("_saved_ocr_bbox", None)
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


def _load_detection_cache_for_ocr(ocr_json_path_value: Path) -> tuple[Path, dict[str, Any]]:
    detection_path = ocr_json_path_value.parents[1] / "detection" / ocr_json_path_value.name
    if not detection_path.exists():
        raise FileNotFoundError(
            f"Detection cache is missing for {ocr_json_path_value.stem}. Run Detection first."
        )
    detection_data = load_detection_json(detection_path)
    had_canon_state = isinstance(detection_data.get("canon_state"), dict)
    ensure_canon_state(detection_data)
    if not had_canon_state:
        save_detection_json(detection_path, detection_data)
    return detection_path, detection_data


def _save_ocr_payload(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    try:
        save_ocr_payload(payload, path)
    except Exception as exc:
        raise RuntimeError(f"Failed to save OCR cache: {path}. {exc}") from exc
    return load_ocr_json(path)


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = [
    "DOWNSTREAM_STALE_STAGES",
    "exclude_ocr_item",
    "load_ocr_edit_items",
    "restore_ocr_item",
    "save_ocr_edit_items",
    "summarize_ocr_edit_state",
    "update_ocr_item_bbox",
]
