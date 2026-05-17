"""Canonical workflow geometry helpers backed by detection JSON."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from .canon_overlap import resolve_largest_overlap_components
from .image_io import ensure_path
from .ocr_items import (
    bbox_to_list,
    build_ocr_items_from_detection,
    clamp_bbox_to_image as _ocr_items_clamp_bbox_to_image,
    expand_bbox,
    infer_source_direction,
    intersect_bboxes,
    is_huge_bbox,
    is_text_like_layout_label,
    overlap_ratio_against_many,
    region_belongs_to_bubble,
    sort_key_for_region,
    text_region_belongs_to_bubble,
)

CANON_STATE_SCHEMA_VERSION = 1
MIN_BOX_SIZE = 4
_CANON_ID_PREFIX = "item_"

EDITOR_CATEGORY_BY_KIND = {
    "bubble": "bubble",
    "outside_text": "text_region",
    "text_region": "text_region",
    "layout_text": "layout_region",
}
MANUAL_KIND_BY_CATEGORY = {
    "bubble": "bubble",
    "text_region": "text_region",
    "layout_region": "layout_text",
}
VALID_CANON_KINDS = set(EDITOR_CATEGORY_BY_KIND)


class ProjectLike(Protocol):
    root_dir: Path
    cache_dir: Path


def normalize_bbox(
    value: Any,
    image_shape: Sequence[int] | None = None,
) -> tuple[int, int, int, int] | None:
    """Best-effort bbox normalization for detector/model outputs."""

    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None

    resolved_values: list[int] = []
    for raw_value in value[:4]:
        if raw_value is None:
            return None

        candidate_value = raw_value
        if isinstance(candidate_value, str):
            candidate_value = candidate_value.strip()
            if not candidate_value:
                return None

        try:
            numeric_value = float(candidate_value)
        except (TypeError, ValueError):
            return None

        if not math.isfinite(numeric_value):
            return None

        resolved_values.append(int(round(numeric_value)))

    bbox = (
        min(resolved_values[0], resolved_values[2]),
        min(resolved_values[1], resolved_values[3]),
        max(resolved_values[0], resolved_values[2]),
        max(resolved_values[1], resolved_values[3]),
    )
    if image_shape is not None:
        bbox = clamp_bbox_to_image(bbox, image_shape)
        if bbox is None:
            return None

    if not is_valid_bbox(bbox, min_size=MIN_BOX_SIZE):
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

    try:
        image_height = max(0, int(image_shape[0]))
        image_width = max(0, int(image_shape[1]))
    except Exception:
        return normalized

    x1, y1, x2, y2 = normalized
    if image_width > 0:
        x1 = max(0, min(x1, image_width))
        x2 = max(0, min(x2, image_width))
    if image_height > 0:
        y1 = max(0, min(y1, image_height))
        y2 = max(0, min(y2, image_height))

    clamped = (
        min(x1, x2),
        min(y1, y2),
        max(x1, x2),
        max(y1, y2),
    )
    if not is_valid_bbox(clamped, min_size=MIN_BOX_SIZE):
        return None
    return clamped


def ensure_canon_state(
    detection_json: dict[str, Any],
    image_shape: Sequence[int] | None = None,
    *,
    logger: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Ensure detection JSON contains a normalized canon_state object."""

    if not isinstance(detection_json, dict):
        raise ValueError("Detection JSON root must be an object.")

    image_width, image_height = _resolve_image_size(
        image_shape=image_shape,
        image_width=detection_json.get("image_width"),
        image_height=detection_json.get("image_height"),
    )

    raw_canon_state = detection_json.get("canon_state")
    if not isinstance(raw_canon_state, dict):
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
        logger=logger,
        page_name=_page_name_from_detection(detection_json),
    )
    return detection_json


def load_canon_state_for_page(project: ProjectLike, image_relative_path: Path | str) -> dict[str, Any]:
    """Load and lazily create canon_state for one project page."""

    detection_path = detection_json_path(project, image_relative_path)
    if not detection_path.exists():
        raise FileNotFoundError(
            f"Detection cache is missing for {Path(str(image_relative_path)).name}. Run Detection first."
        )

    from .detection_io import load_detection_json, save_detection_json

    detection_data = load_detection_json(detection_path)
    had_canon_state = isinstance(detection_data.get("canon_state"), dict)
    ensure_canon_state(detection_data)
    if not had_canon_state:
        save_detection_json(detection_path, detection_data)
    return deepcopy(detection_data["canon_state"])


def save_canon_state_for_page(
    project: ProjectLike,
    image_relative_path: Path | str,
    canon_state: dict[str, Any],
) -> dict[str, Any]:
    """Persist canon_state inside the detection cache for one project page."""

    detection_path = detection_json_path(project, image_relative_path)
    return save_canon_state_to_detection_path(detection_path, canon_state)


def build_canon_state_from_detection(
    detection_json: dict[str, Any],
    image_shape: Sequence[int] | None = None,
    *,
    logger: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Create canon_state from detection output using OCR-prep-like workflow units."""

    if not isinstance(detection_json, dict):
        raise ValueError("Detection JSON root must be an object.")

    image_width, image_height = _resolve_image_size(
        image_shape=image_shape,
        image_width=detection_json.get("image_width"),
        image_height=detection_json.get("image_height"),
    )
    resolved_shape = (image_height, image_width, 3)
    timestamp_value = _timestamp()
    page_name = _page_name_from_detection(detection_json)

    bubbles = [dict(item) for item in detection_json.get("bubbles", []) if isinstance(item, dict)]
    text_regions = [dict(item) for item in detection_json.get("text_regions", []) if isinstance(item, dict)]
    layout_regions = [dict(item) for item in detection_json.get("layout_regions", []) if isinstance(item, dict)]

    active_detection = dict(detection_json)
    active_detection["bubbles"] = [bubble for bubble in bubbles if not bool(bubble.get("excluded", False))]
    active_detection["text_regions"] = [
        region for region in text_regions if not bool(region.get("excluded", False))
    ]
    active_detection["layout_regions"] = [
        region for region in layout_regions if not bool(region.get("excluded", False))
    ]

    active_items = build_ocr_items_from_detection(active_detection, resolved_shape, logger=logger)
    active_items = resolve_largest_overlap_components(active_items, resolved_shape, logger=logger)
    canon_items: list[dict[str, Any]] = []
    used_text_region_ids: set[int] = set()
    used_layout_region_ids: set[int] = set()
    next_index = 0

    for workflow_item in active_items:
        canon_item = _build_active_canon_item(
            workflow_item,
            canon_index=next_index,
            image_width=image_width,
            image_height=image_height,
            bubbles=bubbles,
            text_regions=text_regions,
            layout_regions=layout_regions,
            used_text_region_ids=used_text_region_ids,
            used_layout_region_ids=used_layout_region_ids,
            page_name=page_name,
            logger=logger,
        )
        if canon_item is not None:
            canon_items.append(canon_item)
            next_index += 1

    for bubble in bubbles:
        if not bool(bubble.get("excluded", False)):
            continue
        bubble_id = _coerce_optional_int(bubble.get("id"))
        matched_text_regions = [
            dict(region)
            for region in text_regions
            if bubble_id is not None and _coerce_optional_int(region.get("bubble_id")) == bubble_id
        ]
        disabled_item = _build_disabled_canon_item_from_bubble(
            bubble,
            matched_text_regions=matched_text_regions,
            canon_index=next_index,
            image_width=image_width,
            image_height=image_height,
            page_name=page_name,
            logger=logger,
        )
        if disabled_item is not None:
            canon_items.append(disabled_item)
            next_index += 1

    for region in text_regions:
        if not bool(region.get("excluded", False)):
            continue
        if region.get("bubble_id") not in (None, ""):
            continue
        disabled_item = _build_disabled_canon_item_from_text_region(
            region,
            canon_index=next_index,
            image_width=image_width,
            image_height=image_height,
            page_name=page_name,
            logger=logger,
        )
        if disabled_item is not None:
            canon_items.append(disabled_item)
            next_index += 1

    for region in layout_regions:
        if not bool(region.get("excluded", False)):
            continue
        disabled_item = _build_disabled_canon_item_from_layout_region(
            region,
            canon_index=next_index,
            image_width=image_width,
            image_height=image_height,
            bubbles=bubbles,
            layout_regions=layout_regions,
            page_name=page_name,
            logger=logger,
        )
        if disabled_item is not None:
            canon_items.append(disabled_item)
            next_index += 1

    _attach_suppressed_by_canon_ids(canon_items)

    if logger is not None and not canon_items:
        logger(f"[canon_state] Created an empty canon_state for {page_name}. OCR will have no items for this page.")

    return {
        "schema_version": CANON_STATE_SCHEMA_VERSION,
        "created_at": timestamp_value,
        "updated_at": timestamp_value,
        "created_from_detection": True,
        "items": canon_items,
    }


def get_active_canon_items(detection_json_or_state: dict[str, Any]) -> list[dict[str, Any]]:
    """Return canon items that are currently enabled."""

    canon_state = _coerce_canon_state(detection_json_or_state)
    return [
        deepcopy(item)
        for item in canon_state.get("items", [])
        if isinstance(item, dict) and bool(item.get("enabled", True))
    ]


def get_canon_item(canon_state: dict[str, Any], canon_id: str) -> dict[str, Any]:
    """Return one canon item by canon_id."""

    normalized_id = str(canon_id or "").strip()
    if not normalized_id:
        raise ValueError("Canon item is missing canon_id.")

    for item in canon_state.get("items", []):
        if not isinstance(item, dict):
            continue
        if str(item.get("canon_id", "") or "").strip() == normalized_id:
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
    """Update one canon item bbox field and keep dependent boxes in sync."""

    if field not in {"bbox", "ocr_bbox", "render_bbox"}:
        raise ValueError(f"Unsupported canon bbox field: {field!r}")

    image_width, image_height = _resolve_image_size(
        image_shape=image_shape,
        image_width=None,
        image_height=None,
    )
    item = get_canon_item(canon_state, canon_id)
    previous_bbox = list(item.get("bbox", []) or [])
    sanitized_bbox = _sanitize_bbox(
        bbox,
        image_width=image_width,
        image_height=image_height,
    )
    if sanitized_bbox is None:
        raise ValueError(f"Canon item {canon_id} has an invalid {field}.")

    item[field] = sanitized_bbox
    if field == "bbox":
        item["bbox_user_edited"] = True
        if not bool(item.get("ocr_bbox_user_edited", False)):
            item["ocr_bbox"] = list(sanitized_bbox)
        if not bool(item.get("render_bbox_user_edited", False)):
            item["render_bbox"] = list(sanitized_bbox)
        if normalize_canon_kind(item.get("kind")) != "bubble":
            existing_masks = _sanitize_bbox_list(
                item.get("text_mask_bboxes"),
                image_width=image_width,
                image_height=image_height,
            )
            if (
                not existing_masks
                or (
                    len(existing_masks) == 1
                    and _bbox_key(existing_masks[0]) == _bbox_key(previous_bbox)
                )
            ):
                item["text_mask_bboxes"] = [list(sanitized_bbox)]
    elif field == "ocr_bbox":
        item["ocr_bbox_user_edited"] = True
    elif field == "render_bbox":
        item["render_bbox_user_edited"] = True

    canon_state["updated_at"] = _timestamp()
    return canon_state


def set_canon_item_enabled(canon_state: dict[str, Any], canon_id: str, enabled: bool) -> dict[str, Any]:
    """Soft-delete or restore one canon item."""

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
    """Append a new manual canon item."""

    normalized_kind = normalize_canon_kind(kind)
    image_width, image_height = _resolve_image_size(
        image_shape=image_shape,
        image_width=None,
        image_height=None,
    )
    sanitized_bbox = _sanitize_bbox(
        bbox,
        image_width=image_width,
        image_height=image_height,
    )
    if sanitized_bbox is None:
        raise ValueError("Manual canon item bbox is outside the image bounds or too small.")

    items = canon_state.setdefault("items", [])
    if not isinstance(items, list):
        raise ValueError("Canon state field 'items' must be a list.")

    canon_id = _next_canon_id(items)
    item_metadata = dict(metadata) if isinstance(metadata, dict) else {}
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
        "text_mask_bboxes": [] if normalized_kind == "bubble" else [list(sanitized_bbox)],
        "source_direction": str(item_metadata.pop("source_direction", "") or infer_source_direction(sanitized_bbox)),
        "reading_order": _coerce_optional_int(item_metadata.pop("reading_order", None)),
        "detector_refs": {
            "bubble_id": None,
            "text_region_ids": [],
            "layout_region_ids": [],
        },
        "metadata": item_metadata,
    }
    items.append(new_item)
    canon_state["updated_at"] = _timestamp()
    return canon_state


def summarize_canon_state(canon_state: dict[str, Any]) -> dict[str, Any]:
    """Return compact counts for canon_state items."""

    summary = {
        "total": 0,
        "active": 0,
        "excluded": 0,
        "active_bubbles": 0,
        "excluded_bubbles": 0,
        "active_text_regions": 0,
        "excluded_text_regions": 0,
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
            elif category == "layout_region":
                summary["active_layout_regions"] += 1
            else:
                summary["active_text_regions"] += 1
        else:
            summary["excluded"] += 1
            if category == "bubble":
                summary["excluded_bubbles"] += 1
            elif category == "layout_region":
                summary["excluded_layout_regions"] += 1
            else:
                summary["excluded_text_regions"] += 1
    return summary


def detection_json_path(project: ProjectLike, image_relative_path: Path | str) -> Path:
    relative_path = Path(image_relative_path)
    return ensure_path(project.cache_dir) / "detection" / f"{relative_path.stem}.json"


def editor_category_for_kind(value: Any) -> str:
    return EDITOR_CATEGORY_BY_KIND.get(normalize_canon_kind(value), "text_region")


def manual_kind_for_editor_category(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace(" ", "_")
    return MANUAL_KIND_BY_CATEGORY.get(normalized, "text_region")


def normalize_canon_kind(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace(" ", "_")
    if normalized in VALID_CANON_KINDS:
        return normalized
    if normalized in {"layout_region", "layout"}:
        return "layout_text"
    if normalized in {"text", "outside"}:
        return "outside_text"
    if normalized in {"text_item", "ocr_item"}:
        return "text_region"
    return "text_region"


def canon_item_display_id(item: dict[str, Any], default_index: int = 0) -> int:
    if not isinstance(item, dict):
        return int(default_index)

    detector_refs = item.get("detector_refs", {})
    if isinstance(detector_refs, dict):
        bubble_id = _coerce_optional_int(detector_refs.get("bubble_id"))
        if bubble_id is not None:
            return bubble_id

        text_region_ids = detector_refs.get("text_region_ids", [])
        if isinstance(text_region_ids, list):
            for value in text_region_ids:
                resolved = _coerce_optional_int(value)
                if resolved is not None:
                    return resolved

        layout_region_ids = detector_refs.get("layout_region_ids", [])
        if isinstance(layout_region_ids, list):
            for value in layout_region_ids:
                resolved = _coerce_optional_int(value)
                if resolved is not None:
                    return resolved

    metadata = item.get("metadata", {})
    if isinstance(metadata, dict):
        resolved = _coerce_optional_int(metadata.get("display_id"))
        if resolved is not None:
            return resolved

    return int(default_index)


def resolve_canon_item_for_stage_item(
    canon_state: dict[str, Any],
    stage_item: dict[str, Any],
    *,
    active_only: bool = False,
) -> dict[str, Any] | None:
    """Best-effort mapping for older stage cache items without canon_id."""

    if not isinstance(stage_item, dict):
        return None

    explicit_canon_id = str(stage_item.get("canon_id", "") or "").strip()
    if explicit_canon_id:
        try:
            candidate = get_canon_item(canon_state, explicit_canon_id)
        except ValueError:
            candidate = None
        if candidate is not None and (not active_only or bool(candidate.get("enabled", True))):
            return candidate

    candidates = [
        item
        for item in canon_state.get("items", [])
        if isinstance(item, dict)
        and (not active_only or bool(item.get("enabled", True)))
    ]
    if not candidates:
        return None

    stage_kind = normalize_canon_kind(stage_item.get("kind"))
    stage_boxes = [
        _bbox_key(stage_item.get("bbox")),
        _bbox_key(stage_item.get("ocr_bbox")),
        _bbox_key(stage_item.get("render_bbox")),
    ]
    stage_boxes = [box for box in stage_boxes if box is not None]

    for candidate in candidates:
        if normalize_canon_kind(candidate.get("kind")) != stage_kind:
            continue
        candidate_boxes = {
            _bbox_key(candidate.get("bbox")),
            _bbox_key(candidate.get("ocr_bbox")),
            _bbox_key(candidate.get("render_bbox")),
        }
        if any(box in candidate_boxes for box in stage_boxes):
            return candidate

    legacy_id = _coerce_optional_int(stage_item.get("id"))
    if legacy_id is None:
        legacy_id = _coerce_optional_int(stage_item.get("ocr_item_id"))
    if legacy_id is None:
        legacy_id = _coerce_optional_int(stage_item.get("translation_item_id"))
    ordered_candidates = sorted(
        candidates,
        key=lambda item: sort_key_for_region(
            item.get("ocr_bbox") or item.get("bbox"),
            item.get("reading_order"),
        ),
    )
    if legacy_id is not None and 0 <= legacy_id < len(ordered_candidates):
        candidate = ordered_candidates[legacy_id]
        if normalize_canon_kind(candidate.get("kind")) == stage_kind or stage_kind == "text_region":
            return candidate

    return None


def canon_item_bbox(item: dict[str, Any], field: str) -> list[int] | None:
    """Resolve one canon item bbox with fallback rules."""

    if not isinstance(item, dict):
        return None
    if field == "ocr_bbox":
        return deepcopy(item.get("ocr_bbox") or item.get("bbox"))
    if field == "render_bbox":
        return deepcopy(item.get("render_bbox") or item.get("bbox"))
    return deepcopy(item.get("bbox"))


def save_canon_state_to_detection_path(
    detection_path: Path | str,
    canon_state: dict[str, Any],
    *,
    image_shape: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Persist canon_state directly to a detection cache path."""

    path = ensure_path(detection_path)
    if not path.exists():
        raise FileNotFoundError(f"Detection cache is missing: {path}")

    from .detection_io import load_detection_json, save_detection_json

    detection_data = load_detection_json(path)
    image_width, image_height = _resolve_image_size(
        image_shape=image_shape,
        image_width=detection_data.get("image_width"),
        image_height=detection_data.get("image_height"),
    )
    detection_data["canon_state"] = _normalize_canon_state(
        canon_state,
        image_width=image_width,
        image_height=image_height,
        preserve_created_at_from=detection_data.get("canon_state"),
    )
    try:
        save_detection_json(path, detection_data)
    except Exception as exc:
        raise RuntimeError(f"Failed to save canon state: {path}. {exc}") from exc
    return deepcopy(detection_data["canon_state"])


def _coerce_canon_state(detection_json_or_state: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(detection_json_or_state, dict):
        raise ValueError("Canon state root must be an object.")
    if isinstance(detection_json_or_state.get("items"), list):
        return detection_json_or_state
    canon_state = detection_json_or_state.get("canon_state")
    if not isinstance(canon_state, dict):
        raise ValueError("Detection JSON is missing canon_state. Run Detection first.")
    return canon_state


def _normalize_canon_state(
    canon_state: dict[str, Any],
    *,
    image_width: int,
    image_height: int,
    preserve_created_at_from: Any = None,
    detection_json: dict[str, Any] | None = None,
    logger: Callable[[str], None] | None = None,
    page_name: str | None = None,
) -> dict[str, Any]:
    if not isinstance(canon_state, dict):
        raise ValueError("canon_state must be an object.")

    normalized = deepcopy(canon_state)
    raw_items = normalized.get("items", [])
    if not isinstance(raw_items, list):
        raise ValueError("canon_state.items must be a list.")

    normalized_items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item_index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, dict):
            continue
        try:
            normalized_item = _normalize_canon_item(
                raw_item,
                item_index=item_index,
                image_width=image_width,
                image_height=image_height,
                logger=logger,
                page_name=page_name,
            )
        except ValueError as exc:
            _log_invalid_bbox_candidate(
                logger,
                page_name=page_name,
                kind=normalize_canon_kind(raw_item.get("kind")),
                candidate_source=_candidate_source_label(raw_item),
                original_bbox=raw_item.get("bbox"),
                sanitized_bbox=None,
                reason=str(exc),
                item_id=str(raw_item.get("canon_id", "") or f"#{item_index}"),
            )
            continue
        canon_id = str(normalized_item["canon_id"])
        if canon_id in seen_ids:
            _log_canon_warning(
                logger,
                f"[canon_state] Skipping duplicate canon_id on {page_name or 'page'}: {canon_id}",
            )
            continue
        seen_ids.add(canon_id)
        normalized_items.append(normalized_item)

    _backfill_text_mask_bboxes_for_items(
        normalized_items,
        detection_json=detection_json,
        image_width=image_width,
        image_height=image_height,
    )
    for item in normalized_items:
        if item.get("text_mask_bboxes") is None:
            item["text_mask_bboxes"] = []

    created_at = str(normalized.get("created_at", "") or "").strip()
    if not created_at and isinstance(preserve_created_at_from, dict):
        created_at = str(preserve_created_at_from.get("created_at", "") or "").strip()

    normalized["schema_version"] = CANON_STATE_SCHEMA_VERSION
    normalized["created_at"] = created_at or _timestamp()
    normalized["updated_at"] = _timestamp()
    normalized["created_from_detection"] = bool(normalized.get("created_from_detection", True))
    normalized["items"] = normalized_items
    return normalized


def _normalize_canon_item(
    raw_item: dict[str, Any],
    *,
    item_index: int,
    image_width: int,
    image_height: int,
    logger: Callable[[str], None] | None = None,
    page_name: str | None = None,
) -> dict[str, Any]:
    normalized = deepcopy(raw_item)
    canon_id = str(normalized.get("canon_id", "") or "").strip()
    if not canon_id:
        raise ValueError(f"Canon item #{item_index} is missing canon_id.")

    normalized["canon_id"] = canon_id
    normalized["kind"] = normalize_canon_kind(normalized.get("kind"))
    enabled = normalized.get("enabled")
    if enabled is None:
        enabled = not bool(normalized.get("excluded", False))
    normalized["enabled"] = bool(enabled)
    normalized["excluded"] = not bool(normalized["enabled"])
    normalized["manual"] = bool(normalized.get("manual", False))
    normalized["source"] = str(normalized.get("source", "") or ("manual" if normalized["manual"] else "detector"))

    bbox = _sanitize_bbox(normalized.get("bbox"), image_width=image_width, image_height=image_height)
    if bbox is None:
        raise ValueError(f"Canon item {canon_id} has an invalid bbox.")
    normalized["bbox"] = bbox

    ocr_bbox = normalized.get("ocr_bbox")
    if ocr_bbox is None:
        normalized["ocr_bbox"] = list(bbox)
    else:
        sanitized_ocr_bbox = _sanitize_bbox(
            ocr_bbox,
            image_width=image_width,
            image_height=image_height,
        )
        if sanitized_ocr_bbox is None:
            _log_invalid_bbox_candidate(
                logger,
                page_name=page_name,
                kind=normalize_canon_kind(normalized.get("kind")),
                candidate_source=_candidate_source_label(normalized),
                original_bbox=ocr_bbox,
                sanitized_bbox=bbox,
                reason=f"Canon item {canon_id} has an invalid ocr_bbox. Falling back to bbox.",
                item_id=canon_id,
            )
            sanitized_ocr_bbox = list(bbox)
        normalized["ocr_bbox"] = sanitized_ocr_bbox

    render_bbox = normalized.get("render_bbox")
    if render_bbox is None:
        normalized["render_bbox"] = list(bbox)
    else:
        sanitized_render_bbox = _sanitize_bbox(
            render_bbox,
            image_width=image_width,
            image_height=image_height,
        )
        if sanitized_render_bbox is None:
            _log_invalid_bbox_candidate(
                logger,
                page_name=page_name,
                kind=normalize_canon_kind(normalized.get("kind")),
                candidate_source=_candidate_source_label(normalized),
                original_bbox=render_bbox,
                sanitized_bbox=bbox,
                reason=f"Canon item {canon_id} has an invalid render_bbox. Falling back to bbox.",
                item_id=canon_id,
            )
            sanitized_render_bbox = list(bbox)
        normalized["render_bbox"] = sanitized_render_bbox

    normalized["bbox_user_edited"] = bool(normalized.get("bbox_user_edited", False))
    normalized["ocr_bbox_user_edited"] = bool(normalized.get("ocr_bbox_user_edited", False))
    normalized["render_bbox_user_edited"] = bool(normalized.get("render_bbox_user_edited", False))
    raw_text_mask_bboxes = normalized.get("text_mask_bboxes", None) if "text_mask_bboxes" in normalized else None
    if "text_mask_bboxes" not in normalized:
        normalized["text_mask_bboxes"] = None
    else:
        original_text_mask_count = len(raw_text_mask_bboxes) if isinstance(raw_text_mask_bboxes, list) else 0
        normalized["text_mask_bboxes"] = _sanitize_bbox_list(
            raw_text_mask_bboxes,
            image_width=image_width,
            image_height=image_height,
        )
        if logger is not None and original_text_mask_count and len(normalized["text_mask_bboxes"]) < original_text_mask_count:
            _log_canon_warning(
                logger,
                f"[canon_state] Filtered {original_text_mask_count - len(normalized['text_mask_bboxes'])} invalid text_mask_bboxes "
                f"for {canon_id} on {page_name or 'page'}.",
            )
    normalized["source_direction"] = str(
        normalized.get("source_direction", "") or infer_source_direction(normalized["ocr_bbox"])
    )
    normalized["reading_order"] = _coerce_optional_int(normalized.get("reading_order"))
    normalized["suppressed"] = bool(normalized.get("suppressed", False))
    normalized["suppression_reason"] = str(normalized.get("suppression_reason", "") or "")
    normalized["suppressed_by_workflow_index"] = _coerce_optional_int(normalized.get("suppressed_by_workflow_index"))
    normalized["suppressed_by"] = str(normalized.get("suppressed_by", "") or "")
    normalized["overlap_group_id"] = str(normalized.get("overlap_group_id", "") or "")

    detector_refs = normalized.get("detector_refs", {})
    if not isinstance(detector_refs, dict):
        detector_refs = {}
    normalized["detector_refs"] = {
        "bubble_id": _coerce_optional_int(detector_refs.get("bubble_id")),
        "text_region_ids": _coerce_int_list(detector_refs.get("text_region_ids")),
        "layout_region_ids": _coerce_int_list(detector_refs.get("layout_region_ids")),
    }
    metadata = normalized.get("metadata", {})
    normalized["metadata"] = dict(metadata) if isinstance(metadata, dict) else {}
    if normalized["suppressed"]:
        normalized["metadata"]["suppressed"] = True
    if normalized["suppression_reason"]:
        normalized["metadata"]["suppression_reason"] = normalized["suppression_reason"]
    if normalized["suppressed_by_workflow_index"] is not None:
        normalized["metadata"]["suppressed_by_workflow_index"] = normalized["suppressed_by_workflow_index"]
    if normalized["suppressed_by"]:
        normalized["metadata"]["suppressed_by"] = normalized["suppressed_by"]
    if normalized["overlap_group_id"]:
        normalized["metadata"]["overlap_group_id"] = normalized["overlap_group_id"]
    return normalized


def _build_active_canon_item(
    workflow_item: dict[str, Any],
    *,
    canon_index: int,
    image_width: int,
    image_height: int,
    bubbles: list[dict[str, Any]],
    text_regions: list[dict[str, Any]],
    layout_regions: list[dict[str, Any]],
    used_text_region_ids: set[int],
    used_layout_region_ids: set[int],
    page_name: str | None = None,
    logger: Callable[[str], None] | None = None,
) -> dict[str, Any] | None:
    kind = normalize_canon_kind(workflow_item.get("kind"))
    suppressed = bool(workflow_item.get("suppressed", False))
    suppression_reason = str(workflow_item.get("suppression_reason", "") or "")
    suppressed_by_workflow_index = _coerce_optional_int(workflow_item.get("suppressed_by_workflow_index"))
    overlap_group_id = str(workflow_item.get("overlap_group_id", "") or "")
    workflow_index = _coerce_optional_int(workflow_item.get("workflow_index"))
    bbox = _sanitize_bbox(
        workflow_item.get("bbox"),
        image_width=image_width,
        image_height=image_height,
    )
    if bbox is None:
        _log_invalid_bbox_candidate(
            logger,
            page_name=page_name,
            kind=kind,
            candidate_source=_candidate_source_label(workflow_item),
            original_bbox=workflow_item.get("bbox"),
            sanitized_bbox=None,
            reason="Workflow item bbox is invalid and was skipped.",
            item_id=f"{_CANON_ID_PREFIX}{canon_index:04d}",
        )
        return None

    ocr_bbox = _sanitize_bbox(
        workflow_item.get("ocr_bbox") or workflow_item.get("bbox"),
        image_width=image_width,
        image_height=image_height,
    )
    if ocr_bbox is None:
        _log_invalid_bbox_candidate(
            logger,
            page_name=page_name,
            kind=kind,
            candidate_source=_candidate_source_label(workflow_item),
            original_bbox=workflow_item.get("ocr_bbox") or workflow_item.get("bbox"),
            sanitized_bbox=bbox,
            reason="Workflow item ocr_bbox is invalid. Falling back to bbox.",
            item_id=f"{_CANON_ID_PREFIX}{canon_index:04d}",
        )
        ocr_bbox = list(bbox)
    if kind == "bubble":
        clipped_ocr_bbox = intersect_bboxes(tuple(ocr_bbox), tuple(bbox), (image_height, image_width, 3))
        if clipped_ocr_bbox is None:
            ocr_bbox = list(bbox)
        else:
            if tuple(ocr_bbox) != clipped_ocr_bbox and logger is not None:
                bubble_id = _coerce_optional_int(workflow_item.get("bubble_id"))
                logger(
                    f"[canon_state] Clipped bubble OCR bbox to bubble bbox for bubble "
                    f"{bubble_id if bubble_id is not None else canon_index}."
                )
            ocr_bbox = list(clipped_ocr_bbox)

    detector_refs = {
        "bubble_id": None,
        "text_region_ids": [],
        "layout_region_ids": [],
    }
    manual = False
    source = "detector"
    metadata: dict[str, Any] = {}

    if kind == "bubble":
        bubble_id = _coerce_optional_int(workflow_item.get("bubble_id"))
        detector_refs["bubble_id"] = bubble_id
        matched_bubble = _find_by_id(bubbles, bubble_id)
        workflow_layout_region_ids = _coerce_int_list(workflow_item.get("layout_region_ids"))
        detector_refs["layout_region_ids"] = workflow_layout_region_ids
        if workflow_layout_region_ids and not suppressed:
            used_layout_region_ids.update(workflow_layout_region_ids)
        matched_text_regions = [
            region
            for region in text_regions
            if bubble_id is not None and _coerce_optional_int(region.get("bubble_id")) == bubble_id
        ]
        filtered_text_regions: list[dict[str, Any]] = []
        for region in matched_text_regions:
            text_bbox = _ocr_items_clamp_bbox_to_image(region.get("bbox"), (image_height, image_width, 3))
            if text_bbox is None:
                continue
            if text_region_belongs_to_bubble(text_bbox, tuple(bbox)):
                filtered_text_regions.append(region)
                continue
            if logger is not None:
                logger(
                    f"[canon_state] Rejected text region {_coerce_optional_int(region.get('id'))} "
                    f"for bubble {bubble_id}: outside bubble bbox."
                )
        matched_text_regions = filtered_text_regions
        detector_refs["text_region_ids"] = _ids_from_regions(matched_text_regions)
        if not suppressed:
            used_text_region_ids.update(detector_refs["text_region_ids"])
        manual = bool((matched_bubble or {}).get("manual", False)) or any(
            bool(region.get("manual", False)) for region in matched_text_regions
        )
        source = "manual" if manual else "detector"
        if matched_bubble is not None:
            metadata = _metadata_from_detection_region(
                matched_bubble,
                exclude_keys={"id", "bbox", "manual", "excluded", "source"},
            )
        if workflow_item.get("ocr_bbox_source"):
            metadata["ocr_bbox_source"] = str(workflow_item.get("ocr_bbox_source") or "")
        if workflow_item.get("ocr_bbox_source_color"):
            metadata["ocr_bbox_source_color"] = str(workflow_item.get("ocr_bbox_source_color") or "")
        text_mask_bboxes = _clipped_bboxes_from_detection_regions(
            matched_text_regions,
            clip_bbox=bbox,
            image_width=image_width,
            image_height=image_height,
        )
        if not text_mask_bboxes and not manual and _is_conservative_sub_bbox(ocr_bbox, bbox):
            text_mask_bboxes = [list(ocr_bbox)]
    elif kind in {"outside_text", "text_region"}:
        matched_text_region = _match_region_to_workflow_item(
            workflow_item,
            text_regions,
            used_region_ids=used_text_region_ids,
            allowed_bubble_id=None,
        )
        if matched_text_region is not None:
            detector_refs["text_region_ids"] = _ids_from_regions([matched_text_region])
            if not suppressed:
                used_text_region_ids.update(detector_refs["text_region_ids"])
            manual = bool(matched_text_region.get("manual", False))
            source = "manual" if manual else "detector"
            metadata = _metadata_from_detection_region(
                matched_text_region,
                exclude_keys={"id", "bbox", "manual", "excluded", "source", "bubble_id"},
            )
        text_mask_bboxes = [list(bbox)]
    elif kind == "layout_text":
        matched_layout_region = _match_region_to_workflow_item(
            workflow_item,
            layout_regions,
            used_region_ids=used_layout_region_ids,
            allowed_bubble_id=None,
        )
        if matched_layout_region is not None:
            detector_refs["layout_region_ids"] = _ids_from_regions([matched_layout_region])
            if not suppressed:
                used_layout_region_ids.update(detector_refs["layout_region_ids"])
            manual = bool(matched_layout_region.get("manual", False))
            source = "manual" if manual else "detector"
            metadata = _metadata_from_detection_region(
                matched_layout_region,
                exclude_keys={"id", "bbox", "manual", "excluded", "source"},
            )
        text_mask_bboxes = [list(bbox)]
    else:
        text_mask_bboxes = [list(bbox)]

    detector_sources = workflow_item.get("detector_sources", [])
    if isinstance(detector_sources, list) and detector_sources:
        metadata.setdefault("detector_sources", [str(value) for value in detector_sources if str(value).strip()])
    if workflow_index is not None:
        metadata["workflow_index"] = workflow_index
    if overlap_group_id:
        metadata["overlap_group_id"] = overlap_group_id
    if suppressed:
        metadata["suppressed"] = True
        metadata["suppression_reason"] = suppression_reason
        if suppressed_by_workflow_index is not None:
            metadata["suppressed_by_workflow_index"] = suppressed_by_workflow_index

    try:
        return _normalize_canon_item(
            {
                "canon_id": f"{_CANON_ID_PREFIX}{canon_index:04d}",
                "kind": kind,
                "enabled": not suppressed,
                "excluded": suppressed,
                "manual": manual,
                "source": source,
                "bbox": bbox,
                "ocr_bbox": ocr_bbox,
                "render_bbox": bbox,
                "bbox_user_edited": False,
                "ocr_bbox_user_edited": False,
                "render_bbox_user_edited": False,
                "text_mask_bboxes": text_mask_bboxes,
                "source_direction": str(workflow_item.get("source_direction", "") or infer_source_direction(ocr_bbox)),
                "reading_order": _coerce_optional_int(workflow_item.get("reading_order")),
                "detector_refs": detector_refs,
                "metadata": metadata,
                "suppressed": suppressed,
                "suppression_reason": suppression_reason,
                "suppressed_by_workflow_index": suppressed_by_workflow_index,
                "overlap_group_id": overlap_group_id,
            },
            item_index=canon_index,
            image_width=image_width,
            image_height=image_height,
            logger=logger,
            page_name=page_name,
        )
    except ValueError as exc:
        _log_invalid_bbox_candidate(
            logger,
            page_name=page_name,
            kind=kind,
            candidate_source=source,
            original_bbox=workflow_item.get("bbox"),
            sanitized_bbox=bbox,
            reason=str(exc),
            item_id=f"{_CANON_ID_PREFIX}{canon_index:04d}",
        )
        return None


def _attach_suppressed_by_canon_ids(canon_items: list[dict[str, Any]]) -> None:
    workflow_index_to_canon_id: dict[int, str] = {}
    for item in canon_items:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        workflow_index = _coerce_optional_int(metadata.get("workflow_index"))
        canon_id = str(item.get("canon_id", "") or "").strip()
        if workflow_index is None or not canon_id:
            continue
        workflow_index_to_canon_id[workflow_index] = canon_id

    for item in canon_items:
        if not isinstance(item, dict) or not bool(item.get("suppressed", False)):
            continue
        metadata = item.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
            item["metadata"] = metadata
        winner_index = _coerce_optional_int(
            item.get("suppressed_by_workflow_index", metadata.get("suppressed_by_workflow_index"))
        )
        if winner_index is None:
            continue
        winner_canon_id = workflow_index_to_canon_id.get(winner_index, "")
        item["suppressed_by_workflow_index"] = winner_index
        metadata["suppressed_by_workflow_index"] = winner_index
        item["suppressed_by"] = winner_canon_id
        metadata["suppressed_by"] = winner_canon_id


def _build_disabled_canon_item_from_bubble(
    bubble: dict[str, Any],
    *,
    matched_text_regions: list[dict[str, Any]],
    canon_index: int,
    image_width: int,
    image_height: int,
    page_name: str | None = None,
    logger: Callable[[str], None] | None = None,
) -> dict[str, Any] | None:
    bbox = _sanitize_bbox(bubble.get("bbox"), image_width=image_width, image_height=image_height)
    if bbox is None:
        _log_invalid_bbox_candidate(
            logger,
            page_name=page_name,
            kind="bubble",
            candidate_source=_candidate_source_label(bubble),
            original_bbox=bubble.get("bbox"),
            sanitized_bbox=None,
            reason="Excluded bubble candidate has an invalid bbox and was skipped.",
            item_id=f"{_CANON_ID_PREFIX}{canon_index:04d}",
        )
        return None

    refined_bbox = _union_bboxes(
        [
            _sanitize_bbox(region.get("bbox"), image_width=image_width, image_height=image_height)
            for region in matched_text_regions
            if not bool(region.get("excluded", False))
        ],
        image_width=image_width,
        image_height=image_height,
    )
    ocr_bbox = (
        bbox_to_list(expand_bbox(tuple(refined_bbox), (image_height, image_width, 3), 24))
        if refined_bbox is not None
        else list(bbox)
    )
    if ocr_bbox is None:
        ocr_bbox = list(bbox)

    manual = bool(bubble.get("manual", False)) or any(bool(region.get("manual", False)) for region in matched_text_regions)
    reading_order = min(
        [
            value
            for value in (_coerce_optional_int(region.get("reading_order")) for region in matched_text_regions)
            if value is not None
        ],
        default=None,
    )
    source_direction = next(
        (
            str(region.get("source_direction", "") or "").strip()
            for region in matched_text_regions
            if str(region.get("source_direction", "") or "").strip()
        ),
        infer_source_direction(ocr_bbox),
    )
    try:
        return _normalize_canon_item(
            {
                "canon_id": f"{_CANON_ID_PREFIX}{canon_index:04d}",
                "kind": "bubble",
                "enabled": False,
                "manual": manual,
                "source": "manual" if manual else "detector",
                "bbox": bbox,
                "ocr_bbox": ocr_bbox,
                "render_bbox": bbox,
                "bbox_user_edited": False,
                "ocr_bbox_user_edited": False,
                "render_bbox_user_edited": False,
                "text_mask_bboxes": _bboxes_from_detection_regions(
                    matched_text_regions,
                    image_width=image_width,
                    image_height=image_height,
                ),
                "source_direction": source_direction,
                "reading_order": reading_order,
                "detector_refs": {
                    "bubble_id": _coerce_optional_int(bubble.get("id")),
                    "text_region_ids": _ids_from_regions(matched_text_regions),
                    "layout_region_ids": [],
                },
                "metadata": _metadata_from_detection_region(
                    bubble,
                    exclude_keys={"id", "bbox", "manual", "excluded", "source"},
                ),
            },
            item_index=canon_index,
            image_width=image_width,
            image_height=image_height,
            logger=logger,
            page_name=page_name,
        )
    except ValueError as exc:
        _log_invalid_bbox_candidate(
            logger,
            page_name=page_name,
            kind="bubble",
            candidate_source=_candidate_source_label(bubble),
            original_bbox=bubble.get("bbox"),
            sanitized_bbox=bbox,
            reason=str(exc),
            item_id=f"{_CANON_ID_PREFIX}{canon_index:04d}",
        )
        return None


def _build_disabled_canon_item_from_text_region(
    region: dict[str, Any],
    *,
    canon_index: int,
    image_width: int,
    image_height: int,
    page_name: str | None = None,
    logger: Callable[[str], None] | None = None,
) -> dict[str, Any] | None:
    bbox = _sanitize_bbox(region.get("bbox"), image_width=image_width, image_height=image_height)
    if bbox is None:
        _log_invalid_bbox_candidate(
            logger,
            page_name=page_name,
            kind="outside_text" if not bool(region.get("manual", False)) else "text_region",
            candidate_source=_candidate_source_label(region),
            original_bbox=region.get("bbox"),
            sanitized_bbox=None,
            reason="Excluded text-region candidate has an invalid bbox and was skipped.",
            item_id=f"{_CANON_ID_PREFIX}{canon_index:04d}",
        )
        return None
    ocr_bbox = bbox_to_list(expand_bbox(tuple(bbox), (image_height, image_width, 3), 16)) or list(bbox)
    kind = "text_region" if bool(region.get("manual", False)) else "outside_text"
    try:
        return _normalize_canon_item(
            {
                "canon_id": f"{_CANON_ID_PREFIX}{canon_index:04d}",
                "kind": kind,
                "enabled": False,
                "manual": bool(region.get("manual", False)),
                "source": "manual" if bool(region.get("manual", False)) else "detector",
                "bbox": bbox,
                "ocr_bbox": ocr_bbox,
                "render_bbox": bbox,
                "bbox_user_edited": False,
                "ocr_bbox_user_edited": False,
                "render_bbox_user_edited": False,
                "text_mask_bboxes": [list(bbox)],
                "source_direction": str(region.get("source_direction", "") or infer_source_direction(ocr_bbox)),
                "reading_order": _coerce_optional_int(region.get("reading_order")),
                "detector_refs": {
                    "bubble_id": None,
                    "text_region_ids": _ids_from_regions([region]),
                    "layout_region_ids": [],
                },
                "metadata": _metadata_from_detection_region(
                    region,
                    exclude_keys={"id", "bbox", "manual", "excluded", "source", "bubble_id"},
                ),
            },
            item_index=canon_index,
            image_width=image_width,
            image_height=image_height,
            logger=logger,
            page_name=page_name,
        )
    except ValueError as exc:
        _log_invalid_bbox_candidate(
            logger,
            page_name=page_name,
            kind=kind,
            candidate_source=_candidate_source_label(region),
            original_bbox=region.get("bbox"),
            sanitized_bbox=bbox,
            reason=str(exc),
            item_id=f"{_CANON_ID_PREFIX}{canon_index:04d}",
        )
        return None


def _build_disabled_canon_item_from_layout_region(
    region: dict[str, Any],
    *,
    canon_index: int,
    image_width: int,
    image_height: int,
    bubbles: list[dict[str, Any]],
    layout_regions: list[dict[str, Any]],
    page_name: str | None = None,
    logger: Callable[[str], None] | None = None,
) -> dict[str, Any] | None:
    label = str(region.get("label", "") or "").strip().lower()
    if not is_text_like_layout_label(label):
        return None

    bbox = clamp_bbox_to_image(region.get("bbox"), (image_height, image_width, 3))
    if bbox is None:
        _log_invalid_bbox_candidate(
            logger,
            page_name=page_name,
            kind="layout_text",
            candidate_source=_candidate_source_label(region),
            original_bbox=region.get("bbox"),
            sanitized_bbox=None,
            reason="Excluded layout-text candidate has an invalid bbox and was skipped.",
            item_id=f"{_CANON_ID_PREFIX}{canon_index:04d}",
        )
        return None
    if is_huge_bbox(bbox, (image_height, image_width, 3), max_region_ratio=0.35):
        return None
    if overlap_ratio_against_many(bbox, (bubble.get("bbox") for bubble in bubbles), (image_height, image_width, 3)) >= 0.35:
        return None
    if overlap_ratio_against_many(
        bbox,
        (layout_region.get("bbox") for layout_region in layout_regions if not bool(layout_region.get("excluded", False))),
        (image_height, image_width, 3),
    ) >= 0.99:
        pass

    bbox_list = list(bbox)
    ocr_bbox = bbox_to_list(expand_bbox(bbox, (image_height, image_width, 3), 16)) or bbox_list
    try:
        return _normalize_canon_item(
            {
                "canon_id": f"{_CANON_ID_PREFIX}{canon_index:04d}",
                "kind": "layout_text",
                "enabled": False,
                "manual": bool(region.get("manual", False)),
                "source": "manual" if bool(region.get("manual", False)) else "detector",
                "bbox": bbox_list,
                "ocr_bbox": ocr_bbox,
                "render_bbox": bbox_list,
                "bbox_user_edited": False,
                "ocr_bbox_user_edited": False,
                "render_bbox_user_edited": False,
                "text_mask_bboxes": [list(bbox_list)],
                "source_direction": infer_source_direction(ocr_bbox),
                "reading_order": _coerce_optional_int(region.get("reading_order")),
                "detector_refs": {
                    "bubble_id": None,
                    "text_region_ids": [],
                    "layout_region_ids": _ids_from_regions([region]),
                },
                "metadata": _metadata_from_detection_region(
                    region,
                    exclude_keys={"id", "bbox", "manual", "excluded", "source"},
                ),
            },
            item_index=canon_index,
            image_width=image_width,
            image_height=image_height,
            logger=logger,
            page_name=page_name,
        )
    except ValueError as exc:
        _log_invalid_bbox_candidate(
            logger,
            page_name=page_name,
            kind="layout_text",
            candidate_source=_candidate_source_label(region),
            original_bbox=region.get("bbox"),
            sanitized_bbox=bbox_list,
            reason=str(exc),
            item_id=f"{_CANON_ID_PREFIX}{canon_index:04d}",
        )
        return None


def _match_region_to_workflow_item(
    workflow_item: dict[str, Any],
    candidate_regions: list[dict[str, Any]],
    *,
    used_region_ids: set[int],
    allowed_bubble_id: int | None,
) -> dict[str, Any] | None:
    workflow_bbox = _bbox_key(workflow_item.get("bbox"))
    if workflow_bbox is not None:
        for region in candidate_regions:
            region_id = _coerce_optional_int(region.get("id"))
            if region_id is None or region_id in used_region_ids:
                continue
            if allowed_bubble_id is None and region.get("bubble_id") not in (None, ""):
                continue
            if _bbox_key(region.get("bbox")) == workflow_bbox:
                return region

    best_region: dict[str, Any] | None = None
    best_order = (10**9, 10**9, 10**9)
    for region in candidate_regions:
        region_id = _coerce_optional_int(region.get("id"))
        if region_id is None or region_id in used_region_ids:
            continue
        if allowed_bubble_id is None and region.get("bubble_id") not in (None, ""):
            continue
        region_order = sort_key_for_region(region.get("bbox"), region.get("reading_order"))
        if best_region is None or region_order < best_order:
            best_region = region
            best_order = region_order
    return best_region


def _find_by_id(items: list[dict[str, Any]], item_id: int | None) -> dict[str, Any] | None:
    if item_id is None:
        return None
    for item in items:
        if _coerce_optional_int(item.get("id")) == item_id:
            return item
    return None


def _metadata_from_detection_region(region: dict[str, Any], *, exclude_keys: set[str]) -> dict[str, Any]:
    metadata = {}
    for key, value in region.items():
        if key in exclude_keys:
            continue
        metadata[str(key)] = deepcopy(value)
    return metadata


def _ids_from_regions(regions: Sequence[dict[str, Any]]) -> list[int]:
    resolved: list[int] = []
    for region in regions:
        value = _coerce_optional_int(region.get("id"))
        if value is not None:
            resolved.append(value)
    return resolved


def _union_bboxes(
    boxes: Sequence[list[int] | None],
    *,
    image_width: int,
    image_height: int,
) -> list[int] | None:
    normalized_boxes = [box for box in boxes if isinstance(box, list) and len(box) >= 4]
    if not normalized_boxes:
        return None
    merged = [
        min(box[0] for box in normalized_boxes),
        min(box[1] for box in normalized_boxes),
        max(box[2] for box in normalized_boxes),
        max(box[3] for box in normalized_boxes),
    ]
    return _sanitize_bbox(merged, image_width=image_width, image_height=image_height)


def _backfill_text_mask_bboxes_for_items(
    items: list[dict[str, Any]],
    *,
    detection_json: dict[str, Any] | None,
    image_width: int,
    image_height: int,
) -> None:
    text_regions_by_id: dict[int, dict[str, Any]] = {}
    layout_regions_by_id: dict[int, dict[str, Any]] = {}
    text_regions_by_bubble_id: dict[int, list[dict[str, Any]]] = {}
    if isinstance(detection_json, dict):
        for region in detection_json.get("text_regions", []):
            if not isinstance(region, dict):
                continue
            region_id = _coerce_optional_int(region.get("id"))
            if region_id is not None:
                text_regions_by_id[region_id] = region
            bubble_id = _coerce_optional_int(region.get("bubble_id"))
            if bubble_id is not None:
                text_regions_by_bubble_id.setdefault(bubble_id, []).append(region)
        for region in detection_json.get("layout_regions", []):
            if not isinstance(region, dict):
                continue
            region_id = _coerce_optional_int(region.get("id"))
            if region_id is not None:
                layout_regions_by_id[region_id] = region

    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("text_mask_bboxes") is not None:
            continue
        item["text_mask_bboxes"] = _derive_text_mask_bboxes_for_item(
            item,
            image_width=image_width,
            image_height=image_height,
            text_regions_by_id=text_regions_by_id,
            layout_regions_by_id=layout_regions_by_id,
            text_regions_by_bubble_id=text_regions_by_bubble_id,
        )


def _derive_text_mask_bboxes_for_item(
    item: dict[str, Any],
    *,
    image_width: int,
    image_height: int,
    text_regions_by_id: dict[int, dict[str, Any]],
    layout_regions_by_id: dict[int, dict[str, Any]],
    text_regions_by_bubble_id: dict[int, list[dict[str, Any]]],
) -> list[list[int]]:
    kind = normalize_canon_kind(item.get("kind"))
    bbox = _sanitize_bbox(item.get("bbox"), image_width=image_width, image_height=image_height)
    metadata = item.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    for key in ("text_mask_bboxes", "inpaint_bboxes", "text_bboxes"):
        legacy_boxes = _sanitize_bbox_list(
            metadata.get(key),
            image_width=image_width,
            image_height=image_height,
        )
        if legacy_boxes:
            if kind == "bubble" and bbox is not None:
                return _clip_bbox_list_to_bbox(
                    legacy_boxes,
                    clip_bbox=bbox,
                    image_width=image_width,
                    image_height=image_height,
                )
            return legacy_boxes

    detector_refs = item.get("detector_refs", {})
    if not isinstance(detector_refs, dict):
        detector_refs = {}

    text_region_ids = _coerce_int_list(detector_refs.get("text_region_ids"))
    layout_region_ids = _coerce_int_list(detector_refs.get("layout_region_ids"))
    bubble_id = _coerce_optional_int(detector_refs.get("bubble_id"))

    if kind == "bubble":
        matched_regions = [text_regions_by_id[region_id] for region_id in text_region_ids if region_id in text_regions_by_id]
        if not matched_regions and bubble_id is not None:
            matched_regions = list(text_regions_by_bubble_id.get(bubble_id, []))
        filtered_regions = []
        bubble_bbox_tuple = tuple(bbox) if bbox is not None else None
        for region in matched_regions:
            if bubble_bbox_tuple is None:
                break
            text_bbox = _ocr_items_clamp_bbox_to_image(region.get("bbox"), (image_height, image_width, 3))
            if text_bbox is None:
                continue
            if text_region_belongs_to_bubble(text_bbox, bubble_bbox_tuple):
                filtered_regions.append(region)
        matched_boxes = []
        if bbox is not None:
            matched_boxes = _clipped_bboxes_from_detection_regions(
                filtered_regions,
                clip_bbox=bbox,
                image_width=image_width,
                image_height=image_height,
            )
        if matched_boxes:
            return matched_boxes

        ocr_bbox = _sanitize_bbox(item.get("ocr_bbox"), image_width=image_width, image_height=image_height)
        if (
            not bool(item.get("manual", False))
            and bbox is not None
            and ocr_bbox is not None
            and _is_conservative_sub_bbox(ocr_bbox, bbox)
        ):
            return [list(ocr_bbox)]
        return []

    if kind == "layout_text":
        matched_boxes = _bboxes_from_detection_regions(
            [layout_regions_by_id[region_id] for region_id in layout_region_ids if region_id in layout_regions_by_id],
            image_width=image_width,
            image_height=image_height,
        )
        if matched_boxes:
            return matched_boxes
    else:
        matched_boxes = _bboxes_from_detection_regions(
            [text_regions_by_id[region_id] for region_id in text_region_ids if region_id in text_regions_by_id],
            image_width=image_width,
            image_height=image_height,
        )
        if matched_boxes:
            return matched_boxes

    if bbox is not None:
        return [list(bbox)]

    ocr_bbox = _sanitize_bbox(item.get("ocr_bbox"), image_width=image_width, image_height=image_height)
    if ocr_bbox is not None and kind != "bubble":
        return [list(ocr_bbox)]
    return []


def _bboxes_from_detection_regions(
    regions: Sequence[dict[str, Any]],
    *,
    image_width: int,
    image_height: int,
) -> list[list[int]]:
    boxes: list[list[int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for region in regions:
        if not isinstance(region, dict):
            continue
        bbox = _sanitize_bbox(region.get("bbox"), image_width=image_width, image_height=image_height)
        if bbox is None:
            continue
        bbox_key = tuple(bbox)
        if bbox_key in seen:
            continue
        seen.add(bbox_key)
        boxes.append(list(bbox))
    return boxes


def _clipped_bboxes_from_detection_regions(
    regions: Sequence[dict[str, Any]],
    *,
    clip_bbox: list[int],
    image_width: int,
    image_height: int,
) -> list[list[int]]:
    boxes: list[list[int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    clip_box_tuple = tuple(clip_bbox)
    for region in regions:
        if not isinstance(region, dict):
            continue
        bbox = _sanitize_bbox(region.get("bbox"), image_width=image_width, image_height=image_height)
        if bbox is None:
            continue
        clipped_bbox = intersect_bboxes(tuple(bbox), clip_box_tuple, (image_height, image_width, 3))
        if clipped_bbox is None:
            continue
        bbox_key = tuple(clipped_bbox)
        if bbox_key in seen:
            continue
        seen.add(bbox_key)
        boxes.append(list(clipped_bbox))
    return boxes


def _clip_bbox_list_to_bbox(
    boxes: Sequence[list[int]],
    *,
    clip_bbox: list[int],
    image_width: int,
    image_height: int,
) -> list[list[int]]:
    clipped_boxes: list[list[int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    clip_box_tuple = tuple(clip_bbox)
    for box in boxes:
        sanitized_box = _sanitize_bbox(box, image_width=image_width, image_height=image_height)
        if sanitized_box is None:
            continue
        clipped_bbox = intersect_bboxes(tuple(sanitized_box), clip_box_tuple, (image_height, image_width, 3))
        if clipped_bbox is None:
            continue
        bbox_key = tuple(clipped_bbox)
        if bbox_key in seen:
            continue
        seen.add(bbox_key)
        clipped_boxes.append(list(clipped_bbox))
    return clipped_boxes


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
        sanitized_bbox = _sanitize_bbox(
            bbox,
            image_width=image_width,
            image_height=image_height,
        )
        if sanitized_bbox is None:
            continue
        bbox_key = tuple(sanitized_bbox)
        if bbox_key in seen:
            continue
        seen.add(bbox_key)
        normalized.append(sanitized_bbox)
    return normalized


def _is_conservative_sub_bbox(candidate_bbox: list[int], container_bbox: list[int]) -> bool:
    candidate_key = _bbox_key(candidate_bbox)
    container_key = _bbox_key(container_bbox)
    if candidate_key is None or container_key is None:
        return False
    if candidate_key == container_key:
        return False

    candidate_area = max(1, (candidate_key[2] - candidate_key[0]) * (candidate_key[3] - candidate_key[1]))
    container_area = max(1, (container_key[2] - container_key[0]) * (container_key[3] - container_key[1]))
    return candidate_area < int(container_area * 0.95)


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


def _page_name_from_detection(detection_json: dict[str, Any]) -> str:
    source_image = str(detection_json.get("source_image", "") or "").strip()
    if source_image:
        return Path(source_image).name
    return "page"


def _candidate_source_label(candidate: dict[str, Any], default: str = "detector") -> str:
    if not isinstance(candidate, dict):
        return default

    detector_sources = candidate.get("detector_sources")
    if isinstance(detector_sources, list):
        normalized_sources = [str(value).strip() for value in detector_sources if str(value).strip()]
        if normalized_sources:
            return ", ".join(normalized_sources)

    for key in ("source", "detector", "label"):
        value = str(candidate.get(key, "") or "").strip()
        if value:
            return value
    return default


def _log_canon_warning(logger: Callable[[str], None] | None, message: str) -> None:
    if logger is not None:
        logger(message)


def _log_invalid_bbox_candidate(
    logger: Callable[[str], None] | None,
    *,
    page_name: str | None,
    kind: str,
    candidate_source: str,
    original_bbox: Any,
    sanitized_bbox: Any,
    reason: str,
    item_id: str | None = None,
) -> None:
    if logger is None:
        return

    details = [
        "[canon_state] Skipped invalid candidate",
        f"page={page_name or 'page'}",
        f"kind={kind or 'unknown'}",
        f"source={candidate_source or 'unknown'}",
    ]
    if item_id:
        details.append(f"item={item_id}")
    details.append(f"original_bbox={original_bbox!r}")
    details.append(f"sanitized_bbox={sanitized_bbox!r}")
    details.append(f"reason={reason}")
    logger(" | ".join(details))


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
