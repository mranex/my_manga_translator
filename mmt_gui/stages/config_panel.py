"""Shared workflow configuration stage."""

from __future__ import annotations

from typing import Any

from mmt_core import OCRConfig, RenderConfig, TranslationConfig
from PyQt6.QtWidgets import QLabel, QWidget

from mmt_gui.widgets import CollapsibleSection

from .base_panel import StagePanel
from .detection_panel import DetectionPanel
from .inpaint_panel import InpaintPanel
from .ocr_panel import OCRPanel
from .render_panel import RenderPanel
from .translation_panel import TranslationPanel


class ConfigPanel(StagePanel):
    """Hosts workflow settings that were previously spread across stage panels."""

    def __init__(
        self,
        *,
        detection_panel: DetectionPanel,
        ocr_panel: OCRPanel,
        translation_panel: TranslationPanel,
        inpaint_panel: InpaintPanel,
        render_panel: RenderPanel,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("Config", parent)
        self._detection_panel = detection_panel
        self._ocr_panel = ocr_panel
        self._translation_panel = translation_panel
        self._inpaint_panel = inpaint_panel
        self._render_panel = render_panel

        self.set_stage_note(
            "Workflow settings live here. Detection, OCR, Translation, Inpaint, and Render actions use these controls."
        )

        detection_section = CollapsibleSection("Detection Config", expanded=True)
        detection_note = QLabel(
            "This build does not expose separate detection settings. Detection actions keep the existing pipeline behavior."
        )
        detection_note.setWordWrap(True)
        detection_note.setProperty("role", "muted")
        detection_section.content_layout.addWidget(detection_note)
        self.content_layout.addWidget(detection_section)

        self._add_config_group("OCR Config", self._ocr_panel.config_sections(), expanded=True)
        self._add_config_group("Translation Config", self._translation_panel.config_sections(), expanded=True)
        self._add_config_group("Inpaint Config", self._inpaint_panel.config_sections(), expanded=True)
        self._add_config_group("Render Config", self._render_panel.config_sections(), expanded=True)

    def apply_ocr_settings(self, settings: dict[str, Any]) -> None:
        self._ocr_panel.apply_settings(settings)

    def apply_translation_settings(self, settings: dict[str, Any]) -> None:
        self._translation_panel.apply_settings(settings)

    def apply_inpaint_settings(self, settings: dict[str, Any]) -> None:
        self._inpaint_panel.apply_settings(settings)

    def apply_render_settings(self, settings: dict[str, Any]) -> None:
        self._render_panel.apply_settings(settings)

    def ocr_config(self) -> OCRConfig:
        return self._ocr_panel.ocr_config()

    def ocr_server_values(self) -> dict[str, Any]:
        return self._ocr_panel.server_values()

    def ocr_settings_snapshot(self) -> dict[str, Any]:
        return self._ocr_panel.settings_snapshot()

    def ocr_provider_label(self) -> str:
        return self.ocr_config().provider_label

    def translation_config(self) -> TranslationConfig:
        return self._translation_panel.config()

    def translation_settings_snapshot(self) -> dict[str, Any]:
        return self._translation_panel.settings_snapshot()

    def translation_target_language(self) -> str:
        return self.translation_config().target_language.strip() or "en"

    def translation_provider_name(self) -> str:
        return self.translation_config().translator.strip() or "Google"

    def inpaint_settings(self, *, force_override: bool | None = None) -> dict[str, Any]:
        return self._inpaint_panel.settings(force_override=force_override)

    def inpaint_settings_snapshot(self) -> dict[str, Any]:
        return self._inpaint_panel.settings_snapshot()

    def inpaint_device_label(self) -> str:
        return self._inpaint_panel.device_input.currentText().strip() or "auto"

    def render_config(self, *, force_override: bool | None = None) -> RenderConfig:
        return self._render_panel.config(force_override=force_override)

    def render_settings_snapshot(self) -> dict[str, Any]:
        return self._render_panel.settings_snapshot()

    def render_style_label(self) -> str:
        font_name = self._render_panel.font_name_input.currentText().strip()
        if font_name:
            return font_name
        font_path = self._render_panel.font_path_input.text().strip()
        if font_path:
            return font_path.replace("\\", "/").split("/")[-1]
        return "Auto"

    def _add_config_group(self, title: str, widgets: list[QWidget], *, expanded: bool) -> None:
        section = CollapsibleSection(title, expanded=expanded)
        for widget in widgets:
            widget.show()
            section.content_layout.addWidget(widget)
        self.content_layout.addWidget(section)


__all__ = ["ConfigPanel"]
