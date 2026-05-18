"""Disk I/O helpers for detection cache files."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from detectors.base import BubbleRegion, LayoutRegion, PageDetectionResult
from .detection_config import DetectionConfig
from .image_io import ensure_path, project_relative_path, save_png_image
from .json_io import write_json_atomic

DETECTION_SCHEMA_VERSION = 2
_OBSOLETE_REGION_KEY = "text" + "_regions"


class ProjectLike(Protocol):
    root_dir: Path
    cache_dir: Path


def detection_json_path(project: ProjectLike, image_relative_path: Path | str) -> Path:
    """Return the canonical detection cache path for a project page."""

    relative_path = Path(image_relative_path)
    return ensure_path(project.cache_dir) / "detection" / f"{relative_path.stem}.json"


def mask_dir_for_page(project: ProjectLike, image_relative_path: Path | str) -> Path:
    """Return the canonical bubble mask cache directory for a project page."""

    relative_path = Path(image_relative_path)
    return ensure_path(project.cache_dir) / "masks" / relative_path.stem


def save_detection_result(
    result: PageDetectionResult,
    *,
    image_path: Path | str,
    image_shape: Sequence[int],
    detection_json_output_path: Path | str,
    mask_output_dir: Path | str,
    project_root: Path | str | None = None,
    logger: Callable[[str], None] | None = None,
    detection_config: DetectionConfig | dict[str, Any] | None = None,
) -> Path:
    """Serialize a detection result to JSON and any bubble masks to disk."""

    output_path = ensure_path(detection_json_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    project_root_path = ensure_path(project_root) if project_root is not None else output_path.parents[2]
    mask_dir = ensure_path(mask_output_dir)
    if mask_dir.exists():
        for existing_mask in mask_dir.glob("bubble_*.png"):
            existing_mask.unlink()

    payload = {
        "schema_version": DETECTION_SCHEMA_VERSION,
        "stage": "detection",
        "source_image": project_relative_path(project_root_path, image_path),
        "image_width": int(image_shape[1]),
        "image_height": int(image_shape[0]),
        "method": str(result.method),
        "stats": _json_safe(result.stats),
        "detection_config": DetectionConfig.from_value(detection_config).to_settings_dict(),
        "bubbles": [],
        "layout_regions": [],
    }

    for bubble_index, bubble in enumerate(result.bubbles):
        mask_path = _save_bubble_mask(
            bubble=bubble,
            bubble_index=bubble_index,
            mask_output_dir=mask_dir,
            project_root=project_root_path,
        )
        payload["bubbles"].append(
            _serialize_bubble(
                bubble,
                bubble_id=bubble_index,
                mask_path=mask_path,
            )
        )

    for layout_index, layout_region in enumerate(result.layout_regions):
        payload["layout_regions"].append(_serialize_layout_region(layout_region, region_id=layout_index))

    from .canon_state import ensure_canon_state

    ensure_canon_state(payload, image_shape=image_shape, logger=logger)
    return save_detection_json(output_path, payload)


def load_detection_json(path: Path | str) -> dict[str, Any]:
    """Load and lightly validate a detection cache JSON file."""

    detection_path = ensure_path(path)
    try:
        payload = json.loads(detection_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Detection cache is not valid JSON: {detection_path}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Detection cache root must be an object: {detection_path}")

    if payload.get("stage") != "detection":
        raise ValueError(f"Unsupported detection cache stage in {detection_path}")

    for key in ("bubbles", "layout_regions"):
        value = payload.get(key, [])
        if not isinstance(value, list):
            raise ValueError(f"Detection cache field '{key}' must be a list in {detection_path}")
        payload[key] = [_normalize_detection_region(item) for item in value if isinstance(item, dict)]
    payload.pop(_OBSOLETE_REGION_KEY, None)

    payload.setdefault("schema_version", DETECTION_SCHEMA_VERSION)
    payload.setdefault("method", "")
    payload.setdefault("stats", {})
    payload.setdefault("source_image", "")
    payload.setdefault("image_width", 0)
    payload.setdefault("image_height", 0)
    detection_config = payload.get("detection_config")
    payload["detection_config"] = detection_config if isinstance(detection_config, dict) else {}
    payload.setdefault("edited", False)
    payload.setdefault("edited_at", "")
    downstream_stale = payload.get("downstream_stale", [])
    payload["downstream_stale"] = downstream_stale if isinstance(downstream_stale, list) else []
    canon_state = payload.get("canon_state")
    if isinstance(canon_state, dict):
        from .canon_state import ensure_canon_state

        ensure_canon_state(payload)
    return payload


def save_detection_json(path: Path | str, data: dict[str, Any]) -> Path:
    """Persist a full detection JSON payload to disk."""

    output_path = ensure_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        str(key): _json_safe(value)
        for key, value in dict(data).items()
    }
    payload.pop(_OBSOLETE_REGION_KEY, None)
    payload.update(
        {
            "schema_version": int(data.get("schema_version", DETECTION_SCHEMA_VERSION)),
            "stage": "detection",
            "source_image": str(data.get("source_image", "") or ""),
            "image_width": int(data.get("image_width", 0) or 0),
            "image_height": int(data.get("image_height", 0) or 0),
            "method": str(data.get("method", "") or ""),
            "stats": _json_safe(data.get("stats", {})),
            "detection_config": DetectionConfig.from_value(data.get("detection_config", {})).to_settings_dict(),
            "bubbles": [_normalize_detection_region(item) for item in data.get("bubbles", []) if isinstance(item, dict)],
            "layout_regions": [
                _normalize_detection_region(item)
                for item in data.get("layout_regions", [])
                if isinstance(item, dict)
            ],
            "edited": bool(data.get("edited", False)),
            "edited_at": str(data.get("edited_at", "") or ""),
            "downstream_stale": list(data.get("downstream_stale", []) or []),
        }
    )
    canon_state = data.get("canon_state")
    if isinstance(canon_state, dict):
        payload["canon_state"] = _json_safe(canon_state)
    return write_json_atomic(output_path, _json_safe(payload), indent=2, ensure_ascii=False)


def _serialize_bubble(
    bubble: BubbleRegion,
    *,
    bubble_id: int,
    mask_path: str | None,
) -> dict[str, Any]:
    return {
        "id": bubble_id,
        "bbox": _bbox_to_list(bubble.bbox),
        "confidence": float(bubble.score),
        "detector": str(bubble.detector or "yolov8_seg_bubble"),
        "class_id": bubble.class_id,
        "is_dark": bool(bubble.is_dark),
        "mask_path": mask_path,
    }


def _serialize_layout_region(layout_region: LayoutRegion, *, region_id: int) -> dict[str, Any]:
    return {
        "id": region_id,
        "bbox": _bbox_to_list(layout_region.bbox),
        "confidence": float(layout_region.score),
        "label": layout_region.label,
        "detector": str(layout_region.detector or "pp_doclayout_v3"),
        "reading_order": layout_region.reading_order,
        "label_id": layout_region.label_id,
    }


def _save_bubble_mask(
    *,
    bubble: BubbleRegion,
    bubble_index: int,
    mask_output_dir: Path,
    project_root: Path,
) -> str | None:
    if bubble.mask is None:
        return None

    mask_output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = mask_output_dir / f"bubble_{bubble_index:03d}.png"
    save_png_image(bubble.mask, mask_path)
    return project_relative_path(project_root, mask_path)


def _bbox_to_list(bbox: Sequence[int | float]) -> list[int]:
    return [int(value) for value in bbox[:4]]


def _normalize_detection_region(region: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(region)
    normalized["manual"] = bool(normalized.get("manual", False))
    normalized["excluded"] = bool(normalized.get("excluded", False))
    normalized.setdefault("source", normalized.get("detector") or "unknown")
    return normalized


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value

    if isinstance(value, Path):
        return value.as_posix()

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if is_dataclass(value):
        return _json_safe(asdict(value))

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return str(value)
