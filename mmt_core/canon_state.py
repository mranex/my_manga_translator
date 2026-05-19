"""Canonical workflow geometry helpers backed by detection JSON."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from .canon_overlap import resolve_largest_overlap_components
from .image_io import ensure_path
from .ocr_items import (
    bbox_to_list,
    build_ocr_items_from_detection,
    clamp_bbox_to_image as _ocr_items_clamp_bbox_to_image,
    infer_source_direction,
    intersect_bboxes,
    is_text_target_layout_region,
    region_belongs_to_bubble as _ocr_items_region_belongs_to_bubble,
)

CANON_STATE_SCHEMA_VERSION = 2
MIN_BOX_SIZE = 4
_CANON_ID_PREFIX = "item_"

EDITOR_CATEGORY_BY_KIND = {
    "bubble": "bubble",
    "layout_text": "layout_region",
}
MANUAL_KIND_BY_CATEGORY = {
    "bubble": "bubble",
    "layout_region": "layout_text",
}
VALID_CANON_KINDS = {"bubble", "layout_text"}


class ProjectLike(Protocol):
    root_dir: Path
    cache_dir: Path


def normalize_bbox(
    value: Any,
    image_shape: Sequence[int] | None = None,
) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None

    resolved_values: list[int] = []
    for raw_value in value[:4]:
        if raw_value is None:
            return None
        candidate_value = raw_value.strip() if isinstance(raw_value, str) else raw_value
        if candidate_value in {"", None}:
            return None
        try:
            resolved_values.append(int(round(float(candidate_value))))
        except (TypeError, ValueError):
            return None

    bbox = (
        min(resolved_values[0], resolved_values[2]),
        min(resolved_values[1], resolved_values[3]),
        max(resolved_values[0], resolved_values[2]),
        max(resolved_values[1], resolved_values[3]),
    )
    if image_shape is not None:
        bbox = clamp_bbox_to_image(bbox, image_shape)
    if bbox is None or not is_valid_bbox(bbox, min_size=MIN_BOX_SIZE):
        return None
    return bbox


def is_valid_bbox(bbox: Any, min_size: int = 2) -> bool:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return False
    try:
        x1, y1, x2, y2 = [int(value) for value in bbox[:4]]
    except Exception:
        return False
    return (x2 - x1) >= int(min_size) and (y2 - y1) >= int(min_size)


def clamp_bbox_to_image(
    bbox: Any,
    image_shape: Sequence[int] | None,
) -> tuple[int, int, int, int] | None:
    normalized = normalize_bbox(bbox, image_shape=None)
    if normalized is None:
        return None
    if image_shape is None or len(image_shape) < 2:
        return normalized

    image_height = max(0, int(image_shape[0]))
    image_width = max(0, int(image_shape[1]))
    x1, y1, x2, y2 = normalized
    x1 = max(0, min(x1, image_width))
    x2 = max(0, min(x2, image_width))
    y1 = max(0, min(y1, image_height))
    y2 = max(0, min(y2, image_height))
    clamped = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
    if not is_valid_bbox(clamped, min_size=MIN_BOX_SIZE):
        return None
    return clamped


def ensure_canon_state(
    detection_json: dict[str, Any],
    image_shape: Sequence[int] | None = None,
    *,
    logger: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if not isinstance(detection_json, dict):
        raise ValueError("Detection JSON root must be an object.")

    image_width, image_height = _resolve_image_size(
        image_shape=image_shape,
        image_width=detection_json.get("image_width"),
        image_height=detection_json.get("image_height"),
    )
    raw_canon_state = detection_json.get("canon_state")
    if _canon_state_schema_version(raw_canon_state) != CANON_STATE_SCHEMA_VERSION:
        detection_json["canon_state"] = build_canon_state_from_detection(
            detection_json,
            image_shape=(image_height, image_width, 3),
            logger=logger,
        )
        return detection_json

    detection_json["canon_state"] = _normalize_canon_state(
        raw_canon_state,
        image_width=image_width,
        image_height=image_height,
        detection_json=detection_json,
    )
    return detection_json


def load_canon_state_for_page(project: ProjectLike, image_relative_path: Path | str) -> dict[str, Any]:
    detection_path = detection_json_path(project, image_relative_path)
    if not detection_path.exists():
        raise FileNotFoundError(
            f"Detection cache is missing for {Path(str(image_relative_path)).name}. Run Detection first."
        )

    from .detection_io import load_detection_json, save_detection_json

    detection_data = load_detection_json(detection_path)
    existing_schema = _canon_state_schema_version(detection_data.get("canon_state"))
    ensure_canon_state(detection_data)
    if existing_schema != CANON_STATE_SCHEMA_VERSION:
        save_detection_json(detection_path, detection_data)
    return deepcopy(detection_data["canon_state"])


def save_canon_state_for_page(
    project: ProjectLike,
    image_relative_path: Path | str,
    canon_state: dict[str, Any],
) -> dict[str, Any]:
    detection_path = detection_json_path(project, image_relative_path)
    return save_canon_state_to_detection_path(detection_path, canon_state)


def build_canon_state_from_detection(
    detection_json: dict[str, Any],
    image_shape: Sequence[int] | None = None,
    *,
    logger: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if not isinstance(detection_json, dict):
        raise ValueError("Detection JSON root must be an object.")

    image_width, image_height = _resolve_image_size(
        image_shape=image_shape,
        image_width=detection_json.get("image_width"),
        image_height=detection_json.get("image_height"),
    )
    resolved_shape = (image_height, image_width, 3)
    timestamp_value = _timestamp()

    bubbles = [dict(item) for item in detection_json.get("bubbles", []) if isinstance(item, dict)]
    layout_regions = [dict(item) for item in detection_json.get("layout_regions", []) if isinstance(item, dict)]

    active_detection = dict(detection_json)
    active_detection["bubbles"] = [bubble for bubble in bubbles if not bool(bubble.get("excluded", False))]
    active_detection["layout_regions"] = [
        region for region in layout_regions if not bool(region.get("excluded", False))
    ]

    active_items = build_ocr_items_from_detection(active_detection, resolved_shape, logger=logger)
    active_items = resolve_largest_overlap_components(active_items, resolved_shape, logger=logger)

    canon_items: list[dict[str, Any]] = []
    used_layout_region_ids: set[int] = set()
    next_index = 0

    for workflow_item in active_items:
        canon_item = _build_canon_item_from_workflow_item(
            workflow_item,
            canon_index=next_index,
            image_width=image_width,
            image_height=image_height,
            enabled=not bool(workflow_item.get("excluded", False)),
        )
        canon_items.append(canon_item)
        for layout_region_id in canon_item.get("detector_refs", {}).get("layout_region_ids", []):
            resolved_id = _coerce_optional_int(layout_region_id)
            if resolved_id is not None:
                used_layout_region_ids.add(resolved_id)
        next_index += 1

    for layout_region in layout_regions:
        if not bool(layout_region.get("excluded", False)):
            continue
        layout_region_id = _coerce_optional_int(layout_region.get("id"))
        if layout_region_id is not None and layout_region_id in used_layout_region_ids:
            continue
        canon_item = _build_disabled_layout_item(
            layout_region,
            canon_index=next_index,
            image_width=image_width,
            image_height=image_height,
        )
        if canon_item is not None:
            canon_items.append(canon_item)
            next_index += 1

    return {
        "schema_version": CANON_STATE_SCHEMA_VERSION,
        "created_at": timestamp_value,
        "updated_at": timestamp_value,
        "created_from_detection": True,
        "items": canon_items,
    }


def get_active_canon_items(detection_json_or_state: dict[str, Any]) -> list[dict[str, Any]]:
    canon_state = _coerce_canon_state(detection_json_or_state)
    return [
        deepcopy(item)
        for item in canon_state.get("items", [])
        if isinstance(item, dict) and bool(item.get("enabled", True))
    ]


def get_canon_item(canon_state: dict[str, Any], canon_id: str) -> dict[str, Any]:
    normalized_id = str(canon_id or "").strip()
    if not normalized_id:
        raise ValueError("Canon item is missing canon_id.")
    for item in canon_state.get("items", []):
        if isinstance(item, dict) and str(item.get("canon_id", "") or "").strip() == normalized_id:
            return item
    raise ValueError(f"Canon item was not found: {normalized_id}")


def update_canon_item_bbox(
    canon_state: dict[str, Any],
    canon_id: str,
    *,
    field: str,
    bbox: list[int],
    image_shape: Sequence[int] | None = None,
) -> dict[str, Any]:
    if field not in {"bbox", "ocr_bbox", "render_bbox"}:
        raise ValueError(f"Unsupported canon bbox field: {field!r}")

    image_width, image_height = _resolve_image_size(
        image_shape=image_shape,
        image_width=None,
        image_height=None,
    )
    item = get_canon_item(canon_state, canon_id)
    previous_bbox = list(item.get("bbox", []) or [])
    sanitized_bbox = _sanitize_bbox(bbox, image_width=image_width, image_height=image_height)
    if sanitized_bbox is None:
        raise ValueError(f"Canon item {canon_id} has an invalid {field}.")

    item[field] = sanitized_bbox
    if field == "bbox":
        item["bbox_user_edited"] = True
        if not bool(item.get("ocr_bbox_user_edited", False)):
            item["ocr_bbox"] = list(sanitized_bbox)
        if not bool(item.get("render_bbox_user_edited", False)):
            item["render_bbox"] = list(sanitized_bbox)
        if normalize_canon_kind(item.get("kind")) == "layout_text":
            item["text_mask_bboxes"] = [list(sanitized_bbox)]
        else:
            item["text_mask_bboxes"] = _clip_bbox_list_to_bbox(
                _sanitize_bbox_list(
                    item.get("text_mask_bboxes"),
                    image_width=image_width,
                    image_height=image_height,
                ),
                clip_bbox=sanitized_bbox,
                image_width=image_width,
                image_height=image_height,
            ) or _fallback_bubble_mask_boxes(item, image_width=image_width, image_height=image_height)
        if previous_bbox == item.get("ocr_bbox") and not bool(item.get("ocr_bbox_user_edited", False)):
            item["ocr_bbox"] = list(sanitized_bbox)
    elif field == "ocr_bbox":
        item["ocr_bbox_user_edited"] = True
        if normalize_canon_kind(item.get("kind")) == "bubble":
            item["text_mask_bboxes"] = _fallback_bubble_mask_boxes(item, image_width=image_width, image_height=image_height)
    else:
        item["render_bbox_user_edited"] = True

    canon_state["updated_at"] = _timestamp()
    return canon_state


def set_canon_item_enabled(canon_state: dict[str, Any], canon_id: str, enabled: bool) -> dict[str, Any]:
    item = get_canon_item(canon_state, canon_id)
    item["enabled"] = bool(enabled)
    item["excluded"] = not bool(enabled)
    canon_state["updated_at"] = _timestamp()
    return canon_state


def add_manual_canon_item(
    canon_state: dict[str, Any],
    *,
    kind: str,
    bbox: list[int],
    metadata: dict[str, Any] | None = None,
    image_shape: Sequence[int] | None = None,
) -> dict[str, Any]:
    normalized_kind = normalize_canon_kind(kind)
    image_width, image_height = _resolve_image_size(
        image_shape=image_shape,
        image_width=None,
        image_height=None,
    )
    sanitized_bbox = _sanitize_bbox(bbox, image_width=image_width, image_height=image_height)
    if sanitized_bbox is None:
        raise ValueError("Manual canon item bbox is outside the image bounds or too small.")

    items = canon_state.setdefault("items", [])
    if not isinstance(items, list):
        raise ValueError("Canon state field 'items' must be a list.")

    canon_id = _next_canon_id(items)
    item_metadata = dict(metadata) if isinstance(metadata, dict) else {}
    detector_sources = item_metadata.pop("detector_sources", None)
    if not isinstance(detector_sources, list):
        detector_sources = ["manual"]

    new_item = {
        "canon_id": canon_id,
        "kind": normalized_kind,
        "enabled": True,
        "excluded": False,
        "manual": True,
        "source": "manual",
        "bbox": list(sanitized_bbox),
        "ocr_bbox": list(sanitized_bbox),
        "render_bbox": list(sanitized_bbox),
        "bbox_user_edited": False,
        "ocr_bbox_user_edited": False,
        "render_bbox_user_edited": False,
        "text_mask_bboxes": [list(sanitized_bbox)],
        "source_direction": str(item_metadata.pop("source_direction", "") or infer_source_direction(tuple(sanitized_bbox))),
        "reading_order": _coerce_optional_int(item_metadata.pop("reading_order", None)),
        "detector_refs": {
            "bubble_id": None,
            "layout_region_ids": [],
        },
        "metadata": {
            **item_metadata,
            "detector": "manual",
            "detector_sources": [str(value) for value in detector_sources if str(value).strip()],
        },
    }
    items.append(new_item)
    canon_state["updated_at"] = _timestamp()
    return canon_state


def summarize_canon_state(canon_state: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "total": 0,
        "active": 0,
        "excluded": 0,
        "active_bubbles": 0,
        "excluded_bubbles": 0,
        "active_layout_regions": 0,
        "excluded_layout_regions": 0,
    }

    for item in canon_state.get("items", []):
        if not isinstance(item, dict):
            continue
        summary["total"] += 1
        category = editor_category_for_kind(item.get("kind"))
        enabled = bool(item.get("enabled", True))
        if enabled:
            summary["active"] += 1
            if category == "bubble":
                summary["active_bubbles"] += 1
            else:
                summary["active_layout_regions"] += 1
        else:
            summary["excluded"] += 1
            if category == "bubble":
                summary["excluded_bubbles"] += 1
            else:
                summary["excluded_layout_regions"] += 1
    return summary


def detection_json_path(project: ProjectLike, image_relative_path: Path | str) -> Path:
    relative_path = Path(image_relative_path)
    return ensure_path(project.cache_dir) / "detection" / f"{relative_path.stem}.json"


def editor_category_for_kind(value: Any) -> str:
    return EDITOR_CATEGORY_BY_KIND.get(normalize_canon_kind(value), "layout_region")


def manual_kind_for_editor_category(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace(" ", "_")
    return MANUAL_KIND_BY_CATEGORY.get(normalized, "layout_text")


def normalize_canon_kind(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace(" ", "_")
    if normalized in VALID_CANON_KINDS:
        return normalized
    if normalized == "bubble":
        return "bubble"
    return "layout_text"


def canon_item_display_id(item: dict[str, Any], default_index: int = 0) -> int:
    if not isinstance(item, dict):
        return int(default_index)

    detector_refs = item.get("detector_refs", {})
    if isinstance(detector_refs, dict):
        bubble_id = _coerce_optional_int(detector_refs.get("bubble_id"))
        if bubble_id is not None:
            return bubble_id
        layout_region_ids = detector_refs.get("layout_region_ids", [])
        if isinstance(layout_region_ids, list):
            for value in layout_region_ids:
                resolved = _coerce_optional_int(value)
                if resolved is not None:
                    return resolved

    metadata = item.get("metadata", {})
    if isinstance(metadata, dict):
        display_id = _coerce_optional_int(metadata.get("display_id"))
        if display_id is not None:
            return display_id
    return int(default_index)


def resolve_canon_item_for_stage_item(
    canon_state: dict[str, Any],
    stage_item: dict[str, Any],
    *,
    active_only: bool = True,
) -> dict[str, Any] | None:
    items = canon_state.get("items", []) if isinstance(canon_state, dict) else []
    if not isinstance(items, list):
        return None

    explicit_canon_id = str(stage_item.get("canon_id", "") or "").strip()
    if explicit_canon_id:
        try:
            candidate = get_canon_item(canon_state, explicit_canon_id)
        except ValueError:
            candidate = None
        if candidate is not None and (not active_only or bool(candidate.get("enabled", True))):
            return candidate

    stage_kind = normalize_canon_kind(stage_item.get("kind"))
    detector_refs = stage_item.get("detector_refs", {})
    if not isinstance(detector_refs, dict):
        detector_refs = {}

    bubble_id = _coerce_optional_int(detector_refs.get("bubble_id", stage_item.get("bubble_id")))
    layout_region_ids = _coerce_int_list(
        detector_refs.get("layout_region_ids", stage_item.get("layout_region_ids"))
    )

    if bubble_id is None and stage_kind != "bubble" and not layout_region_ids:
        stage_kind = "layout_text"

    for candidate in items:
        if not isinstance(candidate, dict):
            continue
        if active_only and not bool(candidate.get("enabled", True)):
            continue
        if normalize_canon_kind(candidate.get("kind")) != stage_kind:
            continue

        candidate_refs = candidate.get("detector_refs", {})
        if not isinstance(candidate_refs, dict):
            candidate_refs = {}

        if stage_kind == "bubble":
            if bubble_id is not None and _coerce_optional_int(candidate_refs.get("bubble_id")) == bubble_id:
                return candidate
            continue

        candidate_layout_ids = _coerce_int_list(candidate_refs.get("layout_region_ids"))
        if layout_region_ids and set(candidate_layout_ids) & set(layout_region_ids):
            return candidate

    stage_boxes = [
        _bbox_key(stage_item.get("bbox")),
        _bbox_key(stage_item.get("ocr_bbox")),
        _bbox_key(stage_item.get("render_bbox")),
    ]
    for candidate in items:
        if not isinstance(candidate, dict):
            continue
        if active_only and not bool(candidate.get("enabled", True)):
            continue
        if normalize_canon_kind(candidate.get("kind")) != stage_kind:
            continue

        candidate_boxes = {
            _bbox_key(candidate.get("bbox")),
            _bbox_key(candidate.get("ocr_bbox")),
            _bbox_key(candidate.get("render_bbox")),
        }
        if any(box is not None and box in candidate_boxes for box in stage_boxes):
            return candidate

    return None


def canon_item_bbox(item: dict[str, Any], field: str) -> list[int] | None:
    if field not in {"bbox", "ocr_bbox", "render_bbox"}:
        raise ValueError(f"Unsupported canon bbox field: {field!r}")
    bbox = item.get(field)
    if bbox is None and field != "bbox":
        bbox = item.get("bbox")
    if bbox is None:
        return None
    return [int(value) for value in bbox[:4]]


def save_canon_state_to_detection_path(
    detection_path: Path | str,
    canon_state: dict[str, Any],
) -> dict[str, Any]:
    from .detection_io import load_detection_json, save_detection_json

    detection_path_value = ensure_path(detection_path)
    detection_data = load_detection_json(detection_path_value)
    image_shape = (
        int(detection_data.get("image_height", 0) or 0),
        int(detection_data.get("image_width", 0) or 0),
        3,
    )
    image_width = max(0, int(image_shape[1]))
    image_height = max(0, int(image_shape[0]))
    detection_data["canon_state"] = _normalize_canon_state(
        canon_state,
        image_width=image_width,
        image_height=image_height,
        detection_json=detection_data,
    )
    save_detection_json(detection_path_value, detection_data)
    return detection_data["canon_state"]


def _coerce_canon_state(detection_json_or_state: dict[str, Any]) -> dict[str, Any]:
    if isinstance(detection_json_or_state.get("items"), list):
        return detection_json_or_state
    canon_state = detection_json_or_state.get("canon_state", {})
    if isinstance(canon_state, dict):
        return canon_state
    raise ValueError("canon_state is missing.")


def _normalize_canon_state(
    raw_canon_state: dict[str, Any],
    *,
    image_width: int,
    image_height: int,
    detection_json: dict[str, Any] | None,
) -> dict[str, Any]:
    image_shape = (image_height, image_width, 3)
    normalized_items: list[dict[str, Any]] = []
    for item_index, raw_item in enumerate(raw_canon_state.get("items", [])):
        if not isinstance(raw_item, dict):
            continue
        normalized_items.append(
            _normalize_canon_item(
                raw_item,
                item_index=item_index,
                image_width=image_width,
                image_height=image_height,
                detection_json=detection_json,
                image_shape=image_shape,
            )
        )

    return {
        "schema_version": CANON_STATE_SCHEMA_VERSION,
        "created_at": str(raw_canon_state.get("created_at", "") or _timestamp()),
        "updated_at": str(raw_canon_state.get("updated_at", "") or _timestamp()),
        "created_from_detection": bool(raw_canon_state.get("created_from_detection", True)),
        "items": normalized_items,
    }


def _normalize_canon_item(
    raw_item: dict[str, Any],
    *,
    item_index: int,
    image_width: int,
    image_height: int,
    detection_json: dict[str, Any] | None,
    image_shape: Sequence[int],
) -> dict[str, Any]:
    normalized: dict[str, Any] = {str(key): deepcopy(value) for key, value in raw_item.items()}
    normalized["canon_id"] = str(normalized.get("canon_id", "") or f"{_CANON_ID_PREFIX}{item_index:04d}")
    normalized["kind"] = normalize_canon_kind(normalized.get("kind"))
    normalized["enabled"] = bool(normalized.get("enabled", True))
    normalized["excluded"] = not bool(normalized.get("enabled", True))
    normalized["manual"] = bool(normalized.get("manual", False))
    normalized["source"] = str(normalized.get("source", "") or ("manual" if normalized["manual"] else "detector"))

    bbox = _require_bbox(normalized.get("bbox"), image_width=image_width, image_height=image_height, label="bbox")
    ocr_bbox = _sanitize_bbox(normalized.get("ocr_bbox"), image_width=image_width, image_height=image_height) or list(bbox)
    render_bbox = _sanitize_bbox(normalized.get("render_bbox"), image_width=image_width, image_height=image_height) or list(bbox)

    normalized["bbox"] = bbox
    normalized["ocr_bbox"] = ocr_bbox
    normalized["render_bbox"] = render_bbox
    normalized["bbox_user_edited"] = bool(normalized.get("bbox_user_edited", False))
    normalized["ocr_bbox_user_edited"] = bool(normalized.get("ocr_bbox_user_edited", False))
    normalized["render_bbox_user_edited"] = bool(normalized.get("render_bbox_user_edited", False))
    normalized["reading_order"] = _coerce_optional_int(normalized.get("reading_order"))
    normalized["source_direction"] = str(
        normalized.get("source_direction", "") or infer_source_direction(tuple(ocr_bbox))
    )

    detector_refs = normalized.get("detector_refs", {})
    if not isinstance(detector_refs, dict):
        detector_refs = {}
    normalized["detector_refs"] = {
        "bubble_id": _coerce_optional_int(detector_refs.get("bubble_id")),
        "layout_region_ids": _coerce_int_list(detector_refs.get("layout_region_ids")),
    }

    metadata = normalized.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    detector_sources = metadata.get("detector_sources", [])
    if not isinstance(detector_sources, list):
        detector_sources = []
    normalized["metadata"] = {
        **metadata,
        "detector_sources": [str(value) for value in detector_sources if str(value).strip()],
    }

    normalized["text_mask_bboxes"] = _derive_text_mask_bboxes_for_item(
        normalized,
        image_width=image_width,
        image_height=image_height,
        detection_json=detection_json,
        image_shape=image_shape,
    )
    return normalized


def _build_canon_item_from_workflow_item(
    workflow_item: dict[str, Any],
    *,
    canon_index: int,
    image_width: int,
    image_height: int,
    enabled: bool,
) -> dict[str, Any]:
    bbox = _require_bbox(
        workflow_item.get("bbox"),
        image_width=image_width,
        image_height=image_height,
        label="bbox",
    )
    ocr_bbox = _sanitize_bbox(
        workflow_item.get("ocr_bbox"),
        image_width=image_width,
        image_height=image_height,
    ) or list(bbox)
    text_mask_bboxes = _sanitize_bbox_list(
        workflow_item.get("text_mask_bboxes"),
        image_width=image_width,
        image_height=image_height,
    )
    if not text_mask_bboxes:
        text_mask_bboxes = [list(ocr_bbox)]

    detector_refs = workflow_item.get("detector_refs", {})
    if not isinstance(detector_refs, dict):
        detector_refs = {}

    return {
        "canon_id": f"{_CANON_ID_PREFIX}{canon_index:04d}",
        "kind": normalize_canon_kind(workflow_item.get("kind")),
        "enabled": bool(enabled),
        "excluded": not bool(enabled),
        "manual": bool(workflow_item.get("manual", False)),
        "source": "manual" if bool(workflow_item.get("manual", False)) else "detector",
        "bbox": list(bbox),
        "ocr_bbox": list(ocr_bbox),
        "render_bbox": list(bbox),
        "bbox_user_edited": False,
        "ocr_bbox_user_edited": False,
        "render_bbox_user_edited": False,
        "text_mask_bboxes": text_mask_bboxes,
        "source_direction": str(workflow_item.get("source_direction", "") or infer_source_direction(tuple(ocr_bbox))),
        "reading_order": _coerce_optional_int(workflow_item.get("reading_order")),
        "detector_refs": {
            "bubble_id": _coerce_optional_int(detector_refs.get("bubble_id", workflow_item.get("bubble_id"))),
            "layout_region_ids": _coerce_int_list(
                detector_refs.get("layout_region_ids", workflow_item.get("layout_region_ids"))
            ),
        },
        "metadata": {
            "detector_sources": [
                str(value)
                for value in workflow_item.get("detector_sources", [])
                if str(value).strip()
            ],
            "ocr_bbox_source": str(workflow_item.get("ocr_bbox_source", "") or ""),
            "ocr_bbox_source_color": str(workflow_item.get("ocr_bbox_source_color", "") or ""),
        },
    }


def _build_disabled_layout_item(
    layout_region: dict[str, Any],
    *,
    canon_index: int,
    image_width: int,
    image_height: int,
) -> dict[str, Any] | None:
    if not _is_text_like_layout_region(layout_region):
        return None
    bbox = _sanitize_bbox(layout_region.get("bbox"), image_width=image_width, image_height=image_height)
    if bbox is None:
        return None
    return {
        "canon_id": f"{_CANON_ID_PREFIX}{canon_index:04d}",
        "kind": "layout_text",
        "enabled": False,
        "excluded": True,
        "manual": bool(layout_region.get("manual", False)),
        "source": "manual" if bool(layout_region.get("manual", False)) else "detector",
        "bbox": list(bbox),
        "ocr_bbox": list(bbox),
        "render_bbox": list(bbox),
        "bbox_user_edited": False,
        "ocr_bbox_user_edited": False,
        "render_bbox_user_edited": False,
        "text_mask_bboxes": [list(bbox)],
        "source_direction": infer_source_direction(tuple(bbox)),
        "reading_order": _coerce_optional_int(layout_region.get("reading_order")),
        "detector_refs": {
            "bubble_id": None,
            "layout_region_ids": _ids_from_regions([layout_region]),
        },
        "metadata": {
            "detector": str(layout_region.get("detector", "") or ""),
            "detector_sources": [
                str(layout_region.get("detector") or layout_region.get("source") or "pp_doclayout_v3")
            ],
            "label": str(layout_region.get("label", "") or ""),
        },
    }


def _derive_text_mask_bboxes_for_item(
    item: dict[str, Any],
    *,
    image_width: int,
    image_height: int,
    detection_json: dict[str, Any] | None,
    image_shape: Sequence[int],
) -> list[list[int]]:
    kind = normalize_canon_kind(item.get("kind"))
    bbox = _sanitize_bbox(item.get("bbox"), image_width=image_width, image_height=image_height)
    if bbox is None:
        return []

    existing_boxes = _sanitize_bbox_list(
        item.get("text_mask_bboxes"),
        image_width=image_width,
        image_height=image_height,
    )
    if kind == "layout_text":
        return existing_boxes or [list(bbox)]

    if existing_boxes:
        return _clip_bbox_list_to_bbox(
            existing_boxes,
            clip_bbox=bbox,
            image_width=image_width,
            image_height=image_height,
        )

    detector_refs = item.get("detector_refs", {})
    if not isinstance(detector_refs, dict):
        detector_refs = {}
    layout_region_ids = _coerce_int_list(detector_refs.get("layout_region_ids"))

    layout_regions_by_id: dict[int, dict[str, Any]] = {}
    if isinstance(detection_json, dict):
        for region in detection_json.get("layout_regions", []):
            if not isinstance(region, dict):
                continue
            region_id = _coerce_optional_int(region.get("id"))
            if region_id is not None:
                layout_regions_by_id[region_id] = region

    matched_boxes: list[list[int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for layout_region_id in layout_region_ids:
        region = layout_regions_by_id.get(layout_region_id)
        if not isinstance(region, dict):
            continue
        region_bbox = _ocr_items_clamp_bbox_to_image(region.get("bbox"), image_shape)
        if region_bbox is None:
            continue
        clipped_bbox = intersect_bboxes(tuple(region_bbox), tuple(bbox), image_shape)
        if clipped_bbox is None:
            continue
        bbox_key = tuple(int(value) for value in clipped_bbox)
        if bbox_key in seen:
            continue
        seen.add(bbox_key)
        matched_boxes.append([int(value) for value in clipped_bbox])

    if matched_boxes:
        return matched_boxes
    return []


def _matched_layout_regions_for_bubble(
    layout_regions: Sequence[dict[str, Any]],
    bubble_bbox: list[int],
    image_shape: Sequence[int],
) -> list[dict[str, Any]]:
    matched: list[dict[str, Any]] = []
    bubble_bbox_tuple = tuple(bubble_bbox)
    for layout_region in layout_regions:
        if bool(layout_region.get("excluded", False)):
            continue
        if not _is_text_like_layout_region(layout_region):
            continue
        layout_bbox = _ocr_items_clamp_bbox_to_image(layout_region.get("bbox"), image_shape)
        if layout_bbox is None:
            continue
        if _layout_region_belongs_to_bubble(layout_bbox, bubble_bbox_tuple):
            matched.append(dict(layout_region))
    return matched


def _matched_layout_boxes(
    layout_regions: Sequence[dict[str, Any]],
    bubble_bbox: list[int],
    image_shape: Sequence[int],
) -> list[list[int]]:
    boxes: list[list[int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    bubble_bbox_tuple = tuple(bubble_bbox)
    for layout_region in layout_regions:
        layout_bbox = _ocr_items_clamp_bbox_to_image(layout_region.get("bbox"), image_shape)
        if layout_bbox is None:
            continue
        clipped_bbox = intersect_bboxes(layout_bbox, bubble_bbox_tuple, image_shape)
        if clipped_bbox is None:
            continue
        bbox_key = tuple(int(value) for value in clipped_bbox)
        if bbox_key in seen:
            continue
        seen.add(bbox_key)
        boxes.append([int(value) for value in clipped_bbox])
    return boxes


def _layout_region_belongs_to_bubble(
    layout_bbox: tuple[int, int, int, int],
    bubble_bbox: tuple[int, int, int, int],
) -> bool:
    return _ocr_items_region_belongs_to_bubble(layout_bbox, bubble_bbox)


def _is_text_like_layout_region(layout_region: dict[str, Any]) -> bool:
    return is_text_target_layout_region(layout_region)


def _ids_from_regions(regions: Sequence[dict[str, Any]]) -> list[int]:
    resolved: list[int] = []
    for region in regions:
        value = _coerce_optional_int(region.get("id"))
        if value is not None:
            resolved.append(value)
    return resolved


def _union_bboxes(boxes: Sequence[list[int] | None]) -> list[int] | None:
    normalized_boxes = [box for box in boxes if isinstance(box, list) and len(box) >= 4]
    if not normalized_boxes:
        return None
    return [
        min(box[0] for box in normalized_boxes),
        min(box[1] for box in normalized_boxes),
        max(box[2] for box in normalized_boxes),
        max(box[3] for box in normalized_boxes),
    ]


def _sanitize_bbox_list(
    value: Any,
    *,
    image_width: int,
    image_height: int,
) -> list[list[int]]:
    if not isinstance(value, list):
        return []
    normalized: list[list[int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for bbox in value:
        sanitized_bbox = _sanitize_bbox(bbox, image_width=image_width, image_height=image_height)
        if sanitized_bbox is None:
            continue
        bbox_key = tuple(sanitized_bbox)
        if bbox_key in seen:
            continue
        seen.add(bbox_key)
        normalized.append(sanitized_bbox)
    return normalized


def _clip_bbox_list_to_bbox(
    boxes: Sequence[list[int]],
    *,
    clip_bbox: list[int],
    image_width: int,
    image_height: int,
) -> list[list[int]]:
    clipped_boxes: list[list[int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    image_shape = (image_height, image_width, 3)
    clip_box_tuple = tuple(clip_bbox)
    for box in boxes:
        sanitized_box = _sanitize_bbox(box, image_width=image_width, image_height=image_height)
        if sanitized_box is None:
            continue
        clipped_bbox = intersect_bboxes(tuple(sanitized_box), clip_box_tuple, image_shape)
        if clipped_bbox is None:
            continue
        bbox_key = tuple(int(value) for value in clipped_bbox)
        if bbox_key in seen:
            continue
        seen.add(bbox_key)
        clipped_boxes.append([int(value) for value in clipped_bbox])
    return clipped_boxes


def _sanitize_bbox(
    bbox: Any,
    *,
    image_width: int,
    image_height: int,
) -> list[int] | None:
    image_shape = (image_height, image_width, 3) if image_width > 0 or image_height > 0 else None
    normalized = normalize_bbox(bbox, image_shape=image_shape)
    if normalized is None:
        return None
    return [int(normalized[0]), int(normalized[1]), int(normalized[2]), int(normalized[3])]


def _require_bbox(
    bbox: Any,
    *,
    image_width: int,
    image_height: int,
    label: str,
) -> list[int]:
    sanitized = _sanitize_bbox(bbox, image_width=image_width, image_height=image_height)
    if sanitized is None:
        raise ValueError(f"Canon item has an invalid {label}.")
    return sanitized


def _resolve_image_size(
    *,
    image_shape: Sequence[int] | None,
    image_width: Any,
    image_height: Any,
) -> tuple[int, int]:
    if image_shape is not None and len(image_shape) >= 2:
        try:
            return max(0, int(image_shape[1])), max(0, int(image_shape[0]))
        except Exception:
            pass
    try:
        width = max(0, int(image_width or 0))
    except Exception:
        width = 0
    try:
        height = max(0, int(image_height or 0))
    except Exception:
        height = 0
    return width, height


def _coerce_optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _coerce_int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    normalized: list[int] = []
    for item in value:
        resolved = _coerce_optional_int(item)
        if resolved is not None:
            normalized.append(resolved)
    return normalized


def _bbox_key(bbox: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        return tuple(int(value) for value in bbox[:4])
    except Exception:
        return None


def _next_canon_id(items: Sequence[dict[str, Any]]) -> str:
    max_index = -1
    for item in items:
        if not isinstance(item, dict):
            continue
        canon_id = str(item.get("canon_id", "") or "")
        if canon_id.startswith(_CANON_ID_PREFIX):
            suffix = canon_id[len(_CANON_ID_PREFIX):]
            if suffix.isdigit():
                max_index = max(max_index, int(suffix))
    return f"{_CANON_ID_PREFIX}{max_index + 1:04d}"


def _is_conservative_sub_bbox(candidate_bbox: list[int], container_bbox: list[int]) -> bool:
    candidate_key = _bbox_key(candidate_bbox)
    container_key = _bbox_key(container_bbox)
    if candidate_key is None or container_key is None:
        return False
    candidate_area = max(1, (candidate_key[2] - candidate_key[0]) * (candidate_key[3] - candidate_key[1]))
    container_area = max(1, (container_key[2] - container_key[0]) * (container_key[3] - container_key[1]))
    return candidate_area <= container_area


def _canon_state_schema_version(value: Any) -> int | None:
    if not isinstance(value, dict):
        return None
    return _coerce_optional_int(value.get("schema_version"))


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = [
    "CANON_STATE_SCHEMA_VERSION",
    "add_manual_canon_item",
    "canon_item_bbox",
    "canon_item_display_id",
    "detection_json_path",
    "editor_category_for_kind",
    "ensure_canon_state",
    "get_active_canon_items",
    "get_canon_item",
    "load_canon_state_for_page",
    "manual_kind_for_editor_category",
    "normalize_bbox",
    "normalize_canon_kind",
    "is_valid_bbox",
    "clamp_bbox_to_image",
    "resolve_canon_item_for_stage_item",
    "save_canon_state_for_page",
    "save_canon_state_to_detection_path",
    "set_canon_item_enabled",
    "summarize_canon_state",
    "update_canon_item_bbox",
    "build_canon_state_from_detection",
]
