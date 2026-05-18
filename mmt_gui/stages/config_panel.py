"""Shared workflow configuration stage."""

from __future__ import annotations

from typing import Any

from mmt_core import DEFAULT_MANGA_MODEL_ID, DetectionConfig, OCRConfig, RenderConfig, TranslationConfig
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)

from mmt_gui.widgets import CollapsibleSection
from mmt_gui.widgets.settings_card import style_button

from .base_panel import StagePanel
from .detection_panel import DetectionPanel
from .inpaint_panel import InpaintPanel
from .ocr_panel import OCRPanel
from .render_panel import RenderPanel
from .translation_panel import TranslationPanel


class ConfigPanel(StagePanel):
    """Hosts workflow settings that were previously spread across stage panels."""

    load_manga_detector_requested = pyqtSignal()
    reload_manga_detector_requested = pyqtSignal()
    unload_manga_detector_requested = pyqtSignal()
    manga_detector_status_requested = pyqtSignal()

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
        detection_form = QFormLayout()
        detection_form.setContentsMargins(0, 0, 0, 0)
        detection_form.setSpacing(8)

        self.detection_engine_input = QComboBox()
        self.detection_engine_input.addItem("Manga RT-DETR: OGK Comic Text/Bubble Detector", "manga_rtdetr")
        self.detection_engine_input.addItem("Classic: PP-DocLayoutV3 + YOLOv8 Bubble", "classic")

        self.manga_model_id_input = QLineEdit(DEFAULT_MANGA_MODEL_ID)
        self.manga_model_id_input.setPlaceholderText(DEFAULT_MANGA_MODEL_ID)

        self.manga_device_input = QComboBox()
        self.manga_device_input.setEditable(True)
        self.manga_device_input.addItems(["auto", "cuda", "cuda:0", "cpu"])
        self.manga_device_input.setCurrentText("auto")

        self.manga_confidence_threshold_input = QDoubleSpinBox()
        self.manga_confidence_threshold_input.setRange(0.01, 0.99)
        self.manga_confidence_threshold_input.setSingleStep(0.05)
        self.manga_confidence_threshold_input.setDecimals(2)
        self.manga_confidence_threshold_input.setValue(0.35)

        detection_form.addRow("Engine:", self.detection_engine_input)
        detection_form.addRow("Manga Model ID / Path:", self.manga_model_id_input)
        detection_form.addRow("Manga Device:", self.manga_device_input)
        detection_form.addRow("Confidence Threshold:", self.manga_confidence_threshold_input)
        detection_section.content_layout.addLayout(detection_form)

        runtime_label = QLabel("Manga Detector Runtime")
        runtime_label.setProperty("role", "muted")
        detection_section.content_layout.addWidget(runtime_label)

        runtime_actions = QGridLayout()
        runtime_actions.setContentsMargins(0, 0, 0, 0)
        runtime_actions.setHorizontalSpacing(8)
        runtime_actions.setVerticalSpacing(8)

        self.load_manga_detector_button = QPushButton("Load Manga Detector")
        style_button(self.load_manga_detector_button, "primary")
        self.load_manga_detector_button.clicked.connect(self.load_manga_detector_requested.emit)
        runtime_actions.addWidget(self.load_manga_detector_button, 0, 0)

        self.reload_manga_detector_button = QPushButton("Reload Manga Detector")
        style_button(self.reload_manga_detector_button, "rerun")
        self.reload_manga_detector_button.clicked.connect(self.reload_manga_detector_requested.emit)
        runtime_actions.addWidget(self.reload_manga_detector_button, 0, 1)

        self.unload_manga_detector_button = QPushButton("Unload Manga Detector")
        style_button(self.unload_manga_detector_button, "danger")
        self.unload_manga_detector_button.clicked.connect(self.unload_manga_detector_requested.emit)
        runtime_actions.addWidget(self.unload_manga_detector_button, 1, 0)

        self.manga_detector_status_button = QPushButton("Check Status")
        style_button(self.manga_detector_status_button, "secondary")
        self.manga_detector_status_button.clicked.connect(self.manga_detector_status_requested.emit)
        runtime_actions.addWidget(self.manga_detector_status_button, 1, 1)

        detection_section.content_layout.addLayout(runtime_actions)

        self.manga_detector_status_value = QLabel("Not loaded")
        self.manga_detector_status_value.setWordWrap(True)
        self.manga_detector_status_value.setTextInteractionFlags(
            self.manga_detector_status_value.textInteractionFlags()
        )
        runtime_status_form = QFormLayout()
        runtime_status_form.setContentsMargins(0, 0, 0, 0)
        runtime_status_form.addRow("Status:", self.manga_detector_status_value)
        detection_section.content_layout.addLayout(runtime_status_form)
        self.content_layout.addWidget(detection_section)
        self._detection_section = detection_section

        self._add_config_group("OCR Config", self._ocr_panel.config_sections(), expanded=True)
        self._add_config_group("Translation Config", self._translation_panel.config_sections(), expanded=True)
        self._add_config_group("Inpaint Config", self._inpaint_panel.config_sections(), expanded=True)
        self._add_config_group("Render Config", self._render_panel.config_sections(), expanded=True)

    def detection_config(self) -> DetectionConfig:
        return DetectionConfig.from_value(
            {
                "engine": self.detection_engine_input.currentData(),
                "manga_model_id": self.manga_model_id_input.text().strip() or DEFAULT_MANGA_MODEL_ID,
                "manga_device": self.manga_device_input.currentText().strip() or "auto",
                "manga_confidence_threshold": self.manga_confidence_threshold_input.value(),
            }
        )

    def detection_settings_snapshot(self) -> dict[str, Any]:
        return self.detection_config().to_settings_dict()

    def apply_detection_settings(self, settings: dict[str, Any]) -> None:
        config = DetectionConfig.from_value(settings)
        engine_index = self.detection_engine_input.findData(config.engine)
        if engine_index >= 0:
            self.detection_engine_input.setCurrentIndex(engine_index)
        self.manga_model_id_input.setText(config.manga_model_id)
        self.manga_device_input.setCurrentText(config.manga_device)
        self.manga_confidence_threshold_input.setValue(config.manga_confidence_threshold)

    def set_manga_detector_status(self, status_text: str) -> None:
        self.manga_detector_status_value.setText(str(status_text or "Not loaded"))

    def set_detection_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.detection_engine_input,
            self.manga_model_id_input,
            self.manga_device_input,
            self.manga_confidence_threshold_input,
            self.load_manga_detector_button,
            self.reload_manga_detector_button,
            self.unload_manga_detector_button,
            self.manga_detector_status_button,
        ):
            widget.setEnabled(bool(enabled))

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
