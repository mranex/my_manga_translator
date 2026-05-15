"""Lightweight helpers for page-level workflow status in the studio shell."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mmt_core import (
    detection_json_path,
    inpaint_json_path,
    ocr_json_path,
    render_image_path,
    render_json_path,
    translation_json_path,
)

PAGE_STATUS_UNTOUCHED = "missing"
PAGE_STATUS_WORKING = "ready"
PAGE_STATUS_COMPLETE = "done"
PAGE_STATUS_ERROR = "error"

_ERROR_STATUSES = {"error", "failed"}
_WORKING_STAGE_CACHE_BUILDERS = (
    ocr_json_path,
    translation_json_path,
    inpaint_json_path,
    render_json_path,
)


@dataclass(slots=True, frozen=True)
class PageWorkflowStatus:
    """Compact workflow status payload for a single project page."""

    status: str
    has_canon_state: bool = False
    has_render_output: bool = False
    is_processing: bool = False
    has_error: bool = False


def get_page_workflow_status(
    project: Any,
    image_relative_path: str,
    *,
    current_detection_data: dict[str, Any] | None = None,
    processing_page: str | None = None,
) -> PageWorkflowStatus:
    """Return a lightweight page workflow status for filmstrip-style UI state."""

    if project is None:
        return PageWorkflowStatus(status=PAGE_STATUS_UNTOUCHED)

    normalized_relative_path = str(image_relative_path or "").strip()
    if not normalized_relative_path:
        return PageWorkflowStatus(status=PAGE_STATUS_UNTOUCHED)

    if _page_has_error_status(project, normalized_relative_path):
        return PageWorkflowStatus(
            status=PAGE_STATUS_ERROR,
            has_error=True,
        )

    detection_path = detection_json_path(project, normalized_relative_path)
    if not detection_path.exists():
        return PageWorkflowStatus(status=PAGE_STATUS_UNTOUCHED)

    try:
        has_canon_state = _detection_has_canon_state(
            detection_path,
            current_detection_data=current_detection_data,
        )
    except Exception:
        return PageWorkflowStatus(
            status=PAGE_STATUS_ERROR,
            has_error=True,
        )

    if not has_canon_state:
        return PageWorkflowStatus(status=PAGE_STATUS_UNTOUCHED)

    is_processing = normalized_relative_path == str(processing_page or "").strip()
    if is_processing:
        return PageWorkflowStatus(
            status=PAGE_STATUS_WORKING,
            has_canon_state=True,
            has_render_output=render_image_path(project, normalized_relative_path).exists(),
            is_processing=True,
        )

    render_metadata = project.stage_metadata(normalized_relative_path, "render")
    render_stale = bool(render_metadata.get("stale")) if isinstance(render_metadata, dict) else False
    has_render_output = render_image_path(project, normalized_relative_path).exists()
    if has_render_output and not render_stale:
        return PageWorkflowStatus(
            status=PAGE_STATUS_COMPLETE,
            has_canon_state=True,
            has_render_output=True,
        )

    if _page_has_any_working_cache(project, normalized_relative_path):
        return PageWorkflowStatus(
            status=PAGE_STATUS_WORKING,
            has_canon_state=True,
            has_render_output=has_render_output,
        )

    return PageWorkflowStatus(
        status=PAGE_STATUS_WORKING,
        has_canon_state=True,
        has_render_output=has_render_output,
    )


def _page_has_error_status(project: Any, image_relative_path: str) -> bool:
    for stage_name in ("detection", "ocr", "translation", "inpaint", "render", "export"):
        stage_metadata = project.stage_metadata(image_relative_path, stage_name)
        if not isinstance(stage_metadata, dict):
            continue
        normalized_status = str(stage_metadata.get("status", "") or "").strip().lower()
        if normalized_status in _ERROR_STATUSES:
            return True
    return False


def _page_has_any_working_cache(project: Any, image_relative_path: str) -> bool:
    for builder in _WORKING_STAGE_CACHE_BUILDERS:
        try:
            if builder(project, image_relative_path).exists():
                return True
        except Exception:
            continue
    return False


def _detection_has_canon_state(
    detection_path: Path,
    *,
    current_detection_data: dict[str, Any] | None = None,
) -> bool:
    if isinstance(current_detection_data, dict):
        return isinstance(current_detection_data.get("canon_state"), dict)

    payload = json.loads(detection_path.read_text(encoding="utf-8"))
    return isinstance(payload.get("canon_state"), dict)


__all__ = [
    "PAGE_STATUS_COMPLETE",
    "PAGE_STATUS_ERROR",
    "PAGE_STATUS_UNTOUCHED",
    "PAGE_STATUS_WORKING",
    "PageWorkflowStatus",
    "get_page_workflow_status",
]
