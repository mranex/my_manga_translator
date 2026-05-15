"""Helpers for loading and saving manually edited detection boxes."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .canon_state import (
    add_manual_canon_item,
    canon_item_display_id,
    editor_category_for_kind,
    ensure_canon_state,
    get_canon_item,
    manual_kind_for_editor_category,
    save_canon_state_to_detection_path,
    set_canon_item_enabled,
    summarize_canon_state,
    update_canon_item_bbox,
)
from .detection_io import load_detection_json, save_detection_json
from .image_io import ensure_path

CATEGORY_TO_JSON_KEY = {
    "bubble": "bubbles",
    "text_region": "text_regions",
    "layout_region": "layout_regions",
}
JSON_KEY_TO_CATEGORY = {value: key for key, value in CATEGORY_TO_JSON_KEY.items()}
DOWNSTREAM_STALE_STAGES = ["ocr", "translation", "inpaint", "render", "export"]


def load_detection_edit_items(path: Path) -> list[dict[str, Any]]:
    """Return canon workflow items as GUI-friendly editable box items."""

    payload = load_detection_json(path)
    ensure_canon_state(payload)

    items: list[dict[str, Any]] = []
    for item_index, canon_item in enumerate(payload["canon_state"].get("items", [])):
        if not isinstance(canon_item, dict):
            continue
        items.append(_canon_item_to_edit_item(canon_item, default_index=item_index))
    return items


def save_detection_edit_items(
    path: Path,
    items: list[dict[str, Any]],
    *,
    mark_edited: bool = True,
) -> dict[str, Any]:
    """Persist editable detection boxes back into canon_state."""

    detection_path = ensure_path(path)
    payload = load_detection_json(detection_path)
    ensure_canon_state(payload)

    image_shape = (
        int(payload.get("image_height", 0) or 0),
        int(payload.get("image_width", 0) or 0),
        3,
    )
    canon_state = payload["canon_state"]
    valid_item_count = 0

    for raw_item in items:
        if not isinstance(raw_item, dict):
            continue

        canon_id = str(raw_item.get("canon_id", "") or "").strip()
        bbox = raw_item.get("bbox")
        if canon_id:
            update_canon_item_bbox(
                canon_state,
                canon_id,
                field="bbox",
                bbox=list(bbox or []),
                image_shape=image_shape,
            )
            set_canon_item_enabled(canon_state, canon_id, not bool(raw_item.get("excluded", False)))
            _apply_detection_overlay_fields_to_canon_item(
                get_canon_item(canon_state, canon_id),
                raw_item,
            )
            valid_item_count += 1
            continue

        metadata = _manual_detection_metadata(raw_item)
        add_manual_canon_item(
            canon_state,
            kind=manual_kind_for_editor_category(raw_item.get("category")),
            bbox=list(bbox or []),
            metadata=metadata,
            image_shape=image_shape,
        )
        added_item = canon_state["items"][-1]
        set_canon_item_enabled(canon_state, str(added_item.get("canon_id", "")), not bool(raw_item.get("excluded", False)))
        valid_item_count += 1

    if valid_item_count == 0:
        raise ValueError("No valid detection boxes are available to save.")

    payload["canon_state"] = canon_state
    if mark_edited:
        payload["edited"] = True
        payload["edited_at"] = _timestamp()
        payload["downstream_stale"] = list(DOWNSTREAM_STALE_STAGES)
    save_detection_json(detection_path, payload)
    return load_detection_json(detection_path)


def update_detection_item_bbox(
    path: Path,
    category: str,
    item_id: int,
    bbox: list[int],
) -> dict[str, Any]:
    """Update one detection item's canonical base bbox."""

    detection_path = ensure_path(path)
    items = load_detection_edit_items(detection_path)
    target_item = _find_edit_item(items, category=category, item_id=item_id)
    target_item["bbox"] = list(bbox)
    return save_detection_edit_items(detection_path, items, mark_edited=True)


def add_manual_detection_item(
    path: Path,
    category: str,
    bbox: list[int],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append a new manual canon item to the detection cache."""

    detection_path = ensure_path(path)
    payload = load_detection_json(detection_path)
    ensure_canon_state(payload)
    image_shape = (
        int(payload.get("image_height", 0) or 0),
        int(payload.get("image_width", 0) or 0),
        3,
    )
    canon_state = payload["canon_state"]
    add_manual_canon_item(
        canon_state,
        kind=manual_kind_for_editor_category(category),
        bbox=list(bbox),
        metadata=dict(metadata) if isinstance(metadata, dict) else None,
        image_shape=image_shape,
    )
    payload["canon_state"] = canon_state
    payload["edited"] = True
    payload["edited_at"] = _timestamp()
    payload["downstream_stale"] = list(DOWNSTREAM_STALE_STAGES)
    save_detection_json(detection_path, payload)
    return load_detection_json(detection_path)


def exclude_detection_item(
    path: Path,
    category: str,
    item_id: int,
    excluded: bool = True,
) -> dict[str, Any]:
    """Soft-delete or restore a canon item without removing it from JSON."""

    detection_path = ensure_path(path)
    items = load_detection_edit_items(detection_path)
    target_item = _find_edit_item(items, category=category, item_id=item_id)
    target_item["excluded"] = bool(excluded)
    return save_detection_edit_items(detection_path, items, mark_edited=True)


def summarize_detection_edit_state(data: dict[str, Any]) -> dict[str, Any]:
    """Return active/excluded counts and edit metadata for a detection cache."""

    if isinstance(data.get("canon_state"), dict):
        summary = summarize_canon_state(data["canon_state"])
    else:
        summary = _legacy_raw_detection_summary(data)

    summary["edited"] = bool(data.get("edited", False))
    summary["edited_at"] = data.get("edited_at")
    summary["downstream_stale"] = list(data.get("downstream_stale", []) or [])
    return summary


def _canon_item_to_edit_item(canon_item: dict[str, Any], *, default_index: int) -> dict[str, Any]:
    detector_refs = canon_item.get("detector_refs", {})
    metadata = canon_item.get("metadata", {})
    confidence = metadata.get("confidence") if isinstance(metadata, dict) else None

    return {
        **deepcopy(canon_item),
        "id": canon_item_display_id(canon_item, default_index=default_index),
        "canon_id": str(canon_item.get("canon_id", "") or ""),
        "category": editor_category_for_kind(canon_item.get("kind")),
        "bbox": deepcopy(canon_item.get("bbox")),
        "manual": bool(canon_item.get("manual", False)),
        "excluded": not bool(canon_item.get("enabled", True)),
        "source": str(canon_item.get("source", "") or "detector"),
        "detector": (
            str(metadata.get("detector", "") or "")
            if isinstance(metadata, dict)
            else str(canon_item.get("source", "") or "detector")
        ),
        "confidence": confidence,
        "bubble_id": (
            detector_refs.get("bubble_id")
            if isinstance(detector_refs, dict)
            else None
        ),
    }


def _apply_detection_overlay_fields_to_canon_item(canon_item: dict[str, Any], overlay_item: dict[str, Any]) -> None:
    if not isinstance(canon_item.get("metadata"), dict):
        canon_item["metadata"] = {}
    metadata = canon_item["metadata"]
    if "confidence" in overlay_item:
        metadata["confidence"] = overlay_item.get("confidence")
    if overlay_item.get("source_direction") not in (None, ""):
        canon_item["source_direction"] = str(overlay_item.get("source_direction"))
    if overlay_item.get("reading_order") is not None:
        canon_item["reading_order"] = overlay_item.get("reading_order")


def _manual_detection_metadata(raw_item: dict[str, Any]) -> dict[str, Any]:
    metadata = {}
    for key, value in raw_item.items():
        if key in {
            "id",
            "canon_id",
            "category",
            "kind",
            "bbox",
            "excluded",
            "manual",
            "enabled",
            "detector_refs",
            "metadata",
            "source",
            "detector",
        }:
            continue
        metadata[str(key)] = deepcopy(value)
    metadata.setdefault("detector", "manual")
    metadata.setdefault("display_id", _safe_int(raw_item.get("id")))
    return metadata


def _find_edit_item(items: list[dict[str, Any]], *, category: str, item_id: int) -> dict[str, Any]:
    normalized_category = _normalize_category(category)
    target_id = int(item_id)
    for item in items:
        if _normalize_category(item.get("category")) != normalized_category:
            continue
        if _safe_int(item.get("id")) != target_id:
            continue
        return item
    raise ValueError(f"Detection item not found: {normalized_category} #{target_id}")


def _legacy_raw_detection_summary(data: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "active_bubbles": 0,
        "excluded_bubbles": 0,
        "active_text_regions": 0,
        "excluded_text_regions": 0,
        "active_layout_regions": 0,
        "excluded_layout_regions": 0,
    }
    for json_key in ("bubbles", "text_regions", "layout_regions"):
        regions = data.get(json_key, [])
        if not isinstance(regions, list):
            continue
        active_key = f"active_{json_key}"
        excluded_key = f"excluded_{json_key}"
        for region in regions:
            if not isinstance(region, dict):
                continue
            if bool(region.get("excluded", False)):
                summary[excluded_key] += 1
            else:
                summary[active_key] += 1
    return summary


def _normalize_category(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace(" ", "_")
    if normalized in CATEGORY_TO_JSON_KEY:
        return normalized
    if normalized in CATEGORY_TO_JSON_KEY.values():
        return JSON_KEY_TO_CATEGORY[normalized]
    if normalized == "text":
        return "text_region"
    if normalized == "layout":
        return "layout_region"
    return "bubble"


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = [
    "CATEGORY_TO_JSON_KEY",
    "JSON_KEY_TO_CATEGORY",
    "DOWNSTREAM_STALE_STAGES",
    "add_manual_detection_item",
    "exclude_detection_item",
    "load_detection_edit_items",
    "save_detection_edit_items",
    "summarize_detection_edit_state",
    "update_detection_item_bbox",
]
