"""Preview mode decisions for the persistent desktop workbench."""

from __future__ import annotations

from dataclasses import dataclass

PREVIEW_SOURCE = "Source"
PREVIEW_DETECTION = "Detection Overlay"
PREVIEW_MASK = "Mask Overlay"
PREVIEW_INPAINT = "Inpaint Result"
PREVIEW_RENDER = "Render Result"


@dataclass(slots=True)
class PreviewPreferences:
    auto_preview_result: bool = True
    follow_batch_progress: bool = False


class PreviewController:
    """Encapsulates stage-to-preview behavior for smart result switching."""

    def __init__(self, preferences: PreviewPreferences | None = None) -> None:
        self._preferences = preferences or PreviewPreferences()

    @property
    def preferences(self) -> PreviewPreferences:
        return self._preferences

    def set_preferences(self, *, auto_preview_result: bool, follow_batch_progress: bool) -> None:
        self._preferences = PreviewPreferences(
            auto_preview_result=bool(auto_preview_result),
            follow_batch_progress=bool(follow_batch_progress),
        )

    def result_preview_mode(
        self,
        stage_name: str,
        *,
        export_source: str | None = None,
        current_mode: str | None = None,
    ) -> str | None:
        if not self._preferences.auto_preview_result:
            return None

        normalized_stage = str(stage_name or "").strip().lower()
        if normalized_stage == "detection":
            return PREVIEW_DETECTION
        if normalized_stage in {"ocr_prepare", "ocr"}:
            return PREVIEW_SOURCE
        if normalized_stage == "translation":
            return current_mode or PREVIEW_SOURCE
        if normalized_stage == "inpaint_mask":
            return PREVIEW_MASK
        if normalized_stage == "inpaint":
            return PREVIEW_INPAINT
        if normalized_stage == "render":
            return PREVIEW_RENDER
        if normalized_stage == "export":
            if str(export_source or "").strip().lower() == "render":
                return PREVIEW_RENDER
            return current_mode or PREVIEW_SOURCE
        return current_mode

    def process_preview_mode(
        self,
        process_stage: str,
        *,
        event_name: str | None = None,
        current_mode: str | None = None,
    ) -> str | None:
        """Return explicit Process preview behavior independent of auto-preview preference."""

        normalized_stage = str(process_stage or "").strip().lower()
        normalized_event = str(event_name or "").strip().lower()

        if normalized_stage == "detection":
            return PREVIEW_DETECTION
        if normalized_stage in {"ocr_prepare", "ocr", "translation_init", "translation"}:
            return PREVIEW_SOURCE
        if normalized_stage == "inpaint_mask":
            return PREVIEW_MASK
        if normalized_stage == "inpaint":
            if normalized_event in {"page_done", "process_stage_completed", "process_finished"}:
                return PREVIEW_INPAINT
            if normalized_event in {"mask_ready", "process_stage_started", "page_start"}:
                return PREVIEW_MASK
            return current_mode or PREVIEW_MASK
        if normalized_stage == "render":
            if normalized_event in {"page_done", "process_stage_completed", "process_finished"}:
                return PREVIEW_RENDER
            return current_mode or PREVIEW_INPAINT
        return current_mode

    def should_follow_batch(self) -> bool:
        return self._preferences.follow_batch_progress


__all__ = [
    "PREVIEW_DETECTION",
    "PREVIEW_INPAINT",
    "PREVIEW_MASK",
    "PREVIEW_RENDER",
    "PREVIEW_SOURCE",
    "PreviewController",
    "PreviewPreferences",
]
