"""Inpaint stage inspector panel."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QGridLayout, QLabel, QPushButton, QSpinBox

from mmt_core import summarize_inpaint_json
from mmt_gui.widgets import CollapsibleSection
from mmt_gui.widgets.settings_card import style_button

from .base_panel import StagePanel


class InpaintPanel(StagePanel):
    """Inspector panel for LaMa model actions and inpaint metadata."""

    SETTINGS_VERSION = 2
    SETTINGS_VERSION_KEY = "inpaint_settings_version"

    prepare_selected_requested = pyqtSignal()
    reprepare_selected_requested = pyqtSignal()
    prepare_all_requested = pyqtSignal()
    reprepare_all_requested = pyqtSignal()
    run_selected_requested = pyqtSignal()
    rerun_selected_requested = pyqtSignal()
    run_all_requested = pyqtSignal()
    rerun_all_requested = pyqtSignal()
    reload_requested = pyqtSignal()
    clear_preview_requested = pyqtSignal()
    load_model_requested = pyqtSignal()
    unload_model_requested = pyqtSignal()

    def __init__(self, parent: object | None = None) -> None:
        super().__init__("Inpaint", parent)
        self._pending_settings_migration_message: str | None = None

        model_card = CollapsibleSection("LaMa Model", expanded=False)
        self.load_model_button = QPushButton("Load LaMa Model")
        style_button(self.load_model_button, "primary")
        self.load_model_button.clicked.connect(self.load_model_requested.emit)
        model_card.content_layout.addWidget(self.load_model_button)

        self.unload_model_button = QPushButton("Unload LaMa Model")
        style_button(self.unload_model_button, "danger")
        self.unload_model_button.clicked.connect(self.unload_model_requested.emit)
        model_card.content_layout.addWidget(self.unload_model_button)

        self.lama_model_status_value = QLabel("Not loaded")
        model_form = QFormLayout()
        model_form.setContentsMargins(0, 0, 0, 0)
        model_form.addRow("LaMa Model:", self.lama_model_status_value)
        model_card.content_layout.addLayout(model_form)
        self.content_layout.addWidget(model_card)

        actions_card = CollapsibleSection("Inpaint Actions", expanded=True)
        actions_layout = QGridLayout()
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setHorizontalSpacing(8)
        actions_layout.setVerticalSpacing(8)

        mask_current_label = QLabel("Mask Current")
        mask_current_label.setProperty("role", "muted")
        actions_layout.addWidget(mask_current_label, 0, 0)

        self.prepare_selected_button = QPushButton("Prepare Mask")
        style_button(self.prepare_selected_button, "primary")
        self.prepare_selected_button.clicked.connect(self.prepare_selected_requested.emit)
        self.prepare_selected_button.setToolTip("Prepare the current page mask and reuse cached output when available.")
        actions_layout.addWidget(self.prepare_selected_button, 0, 1)

        self.reprepare_selected_button = QPushButton("Re-prepare Mask")
        style_button(self.reprepare_selected_button, "rerun")
        self.reprepare_selected_button.clicked.connect(self.reprepare_selected_requested.emit)
        self.reprepare_selected_button.setToolTip("Force the current page to rebuild its inpaint mask.")
        actions_layout.addWidget(self.reprepare_selected_button, 0, 2)

        mask_all_label = QLabel("Mask All")
        mask_all_label.setProperty("role", "muted")
        actions_layout.addWidget(mask_all_label, 1, 0)

        self.prepare_all_button = QPushButton("Prepare All")
        style_button(self.prepare_all_button, "primary")
        self.prepare_all_button.clicked.connect(self.prepare_all_requested.emit)
        self.prepare_all_button.setToolTip("Prepare masks for every page and reuse cached output when available.")
        actions_layout.addWidget(self.prepare_all_button, 1, 1)

        self.reprepare_all_button = QPushButton("Re-prepare All")
        style_button(self.reprepare_all_button, "rerun")
        self.reprepare_all_button.clicked.connect(self.reprepare_all_requested.emit)
        self.reprepare_all_button.setToolTip("Force every page to rebuild its inpaint mask.")
        actions_layout.addWidget(self.reprepare_all_button, 1, 2)

        run_current_label = QLabel("Inpaint Current")
        run_current_label.setProperty("role", "muted")
        actions_layout.addWidget(run_current_label, 2, 0)

        self.run_selected_button = QPushButton("Inpaint")
        style_button(self.run_selected_button, "primary")
        self.run_selected_button.clicked.connect(self.run_selected_requested.emit)
        self.run_selected_button.setToolTip("Inpaint the current page and reuse cached output when available.")
        actions_layout.addWidget(self.run_selected_button, 2, 1)

        self.rerun_selected_button = QPushButton("Re-inpaint")
        style_button(self.rerun_selected_button, "rerun")
        self.rerun_selected_button.clicked.connect(self.rerun_selected_requested.emit)
        self.rerun_selected_button.setToolTip("Force the current page to regenerate the inpaint result.")
        actions_layout.addWidget(self.rerun_selected_button, 2, 2)

        run_all_label = QLabel("Inpaint All")
        run_all_label.setProperty("role", "muted")
        actions_layout.addWidget(run_all_label, 3, 0)

        self.run_all_button = QPushButton("Inpaint All")
        style_button(self.run_all_button, "primary")
        self.run_all_button.clicked.connect(self.run_all_requested.emit)
        self.run_all_button.setToolTip("Inpaint every page and reuse cached output when available.")
        actions_layout.addWidget(self.run_all_button, 3, 1)

        self.rerun_all_button = QPushButton("Re-inpaint All")
        style_button(self.rerun_all_button, "rerun")
        self.rerun_all_button.clicked.connect(self.rerun_all_requested.emit)
        self.rerun_all_button.setToolTip("Force every page to regenerate the inpaint result.")
        actions_layout.addWidget(self.rerun_all_button, 3, 2)

        cache_label = QLabel("Cache")
        cache_label.setProperty("role", "muted")
        actions_layout.addWidget(cache_label, 4, 0)

        self.reload_button = QPushButton("Reload Cache")
        style_button(self.reload_button, "secondary")
        self.reload_button.clicked.connect(self.reload_requested.emit)
        self.reload_button.setToolTip("Reload inpaint metadata from disk.")
        actions_layout.addWidget(self.reload_button, 4, 1)

        self.clear_preview_button = QPushButton("Clear Inpaint Preview")
        style_button(self.clear_preview_button, "danger")
        self.clear_preview_button.clicked.connect(self.clear_preview_requested.emit)
        self.clear_preview_button.setToolTip("Return the preview to the source page.")
        actions_layout.addWidget(self.clear_preview_button, 4, 2)
        actions_card.content_layout.addLayout(actions_layout)
        self.content_layout.addWidget(actions_card)

        settings_card = CollapsibleSection("Mask & Device Settings", expanded=False)
        settings_form = QFormLayout()
        settings_form.setContentsMargins(0, 0, 0, 0)
        settings_form.setSpacing(8)

        self.mask_padding_input = QSpinBox()
        self.mask_padding_input.setRange(0, 128)
        self.mask_padding_input.setValue(0)
        self.use_bubble_mask_checkbox = QCheckBox("Use bubble mask guidance")
        self.use_bubble_mask_checkbox.setChecked(True)
        self.use_crop_windows_checkbox = QCheckBox("Use crop windows")
        self.use_crop_windows_checkbox.setChecked(True)
        self.device_input = QComboBox()
        self.device_input.setEditable(True)
        self.device_input.addItems(["auto", "cpu", "cuda", "cuda:0"])
        self.device_input.setCurrentText("auto")

        settings_form.addRow("Mask Padding:", self.mask_padding_input)
        settings_form.addRow("Use Bubble Mask:", self.use_bubble_mask_checkbox)
        settings_form.addRow("Use Crop Windows:", self.use_crop_windows_checkbox)
        settings_form.addRow("Device:", self.device_input)
        settings_card.content_layout.addLayout(settings_form)
        self.content_layout.addWidget(settings_card)

        details_card = CollapsibleSection("Inpaint Details", expanded=True)
        details_form = QFormLayout()
        details_form.setContentsMargins(0, 0, 0, 0)
        details_form.setSpacing(8)

        self.source_path_value = QLabel("-")
        self.source_path_value.setWordWrap(True)
        self.source_path_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.ocr_cache_path_value = QLabel("-")
        self.ocr_cache_path_value.setWordWrap(True)
        self.ocr_cache_path_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.text_mask_path_value = QLabel("-")
        self.text_mask_path_value.setWordWrap(True)
        self.text_mask_path_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.bubble_mask_path_value = QLabel("-")
        self.bubble_mask_path_value.setWordWrap(True)
        self.bubble_mask_path_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.output_path_value = QLabel("-")
        self.output_path_value.setWordWrap(True)
        self.output_path_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.item_count_value = QLabel("0")
        self.masked_pixels_value = QLabel("0")
        self.status_value = QLabel("-")
        self.error_value = QLabel("-")
        self.error_value.setWordWrap(True)
        self.error_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        details_form.addRow("Source Image:", self.source_path_value)
        details_form.addRow("OCR Cache:", self.ocr_cache_path_value)
        details_form.addRow("Text Mask:", self.text_mask_path_value)
        details_form.addRow("Bubble Mask:", self.bubble_mask_path_value)
        details_form.addRow("Output Image:", self.output_path_value)
        details_form.addRow("Item Count:", self.item_count_value)
        details_form.addRow("Masked Pixels:", self.masked_pixels_value)
        details_form.addRow("Status:", self.status_value)
        details_form.addRow("Error:", self.error_value)
        details_card.content_layout.addLayout(details_form)
        self.content_layout.addWidget(details_card)

    def settings(self, *, force_override: bool | None = None) -> dict[str, Any]:
        return {
            "mask_padding": self.mask_padding_input.value(),
            "use_bubble_mask": self.use_bubble_mask_checkbox.isChecked(),
            "use_crop_windows": self.use_crop_windows_checkbox.isChecked(),
            "force": bool(force_override) if force_override is not None else False,
            "device": self.device_value(),
            "strict_ocr_bbox_mask": True,
        }

    def device_value(self) -> str | None:
        device_value = self.device_input.currentText().strip()
        if not device_value or device_value.lower() == "auto":
            return None
        return device_value

    def set_model_status(self, status_text: str) -> None:
        self.lama_model_status_value.setText(status_text or "Not loaded")

    def set_metadata(self, inpaint_data: dict[str, Any] | None, output_image: str | None = None) -> None:
        payload = inpaint_data or {}
        summary = summarize_inpaint_json(payload)
        self.source_path_value.setText(str(payload.get("source_image", "") or "-"))
        self.ocr_cache_path_value.setText(str(payload.get("ocr_cache_path", "") or "-"))
        self.text_mask_path_value.setText(str(payload.get("text_mask_path", "") or "-"))
        self.bubble_mask_path_value.setText(str(payload.get("bubble_mask_path", "") or "-"))
        self.output_path_value.setText(str(payload.get("output_image_path", "") or output_image or "-"))
        self.item_count_value.setText(str(summary.get("item_count", 0)))
        self.masked_pixels_value.setText(str(summary.get("masked_pixel_count", 0)))
        self.status_value.setText(str(summary.get("status", "-") or "-"))
        self.error_value.setText(str(summary.get("error", "") or "-"))

    def clear_view(self, *, source_image: str | None = None, output_image: str | None = None) -> None:
        self.source_path_value.setText(source_image or "-")
        self.ocr_cache_path_value.setText("-")
        self.text_mask_path_value.setText("-")
        self.bubble_mask_path_value.setText("-")
        self.output_path_value.setText(output_image or "-")
        self.item_count_value.setText("0")
        self.masked_pixels_value.setText("0")
        self.status_value.setText("-")
        self.error_value.setText("-")

    def set_actions_enabled(self, enabled: bool) -> None:
        for widget in (
            self.prepare_selected_button,
            self.reprepare_selected_button,
            self.prepare_all_button,
            self.reprepare_all_button,
            self.run_selected_button,
            self.rerun_selected_button,
            self.run_all_button,
            self.rerun_all_button,
            self.reload_button,
            self.clear_preview_button,
            self.load_model_button,
            self.unload_model_button,
            self.mask_padding_input,
            self.use_bubble_mask_checkbox,
            self.use_crop_windows_checkbox,
            self.device_input,
        ):
            widget.setEnabled(enabled)

    def settings_snapshot(self) -> dict[str, Any]:
        return {
            self.SETTINGS_VERSION_KEY: self.SETTINGS_VERSION,
            "settings_version": self.SETTINGS_VERSION,
            "mask_padding": self.mask_padding_input.value(),
            "use_bubble_mask": self.use_bubble_mask_checkbox.isChecked(),
            "use_crop_windows": self.use_crop_windows_checkbox.isChecked(),
            "device": self.device_input.currentText().strip() or "auto",
            "strict_ocr_bbox_mask": True,
        }

    def apply_settings(self, settings: dict[str, Any]) -> None:
        if not isinstance(settings, dict):
            return
        version = 0
        try:
            version = int(
                settings.get(self.SETTINGS_VERSION_KEY, settings.get("settings_version", 0)) or 0
            )
        except Exception:
            version = 0
        target_mask_padding = settings.get("mask_padding", self.mask_padding_input.value())
        if settings and version < self.SETTINGS_VERSION:
            target_mask_padding = 0
            self._pending_settings_migration_message = "Migrated Inpaint mask padding default to 0."
        try:
            self.mask_padding_input.setValue(int(target_mask_padding))
        except Exception:
            pass
        self.use_bubble_mask_checkbox.setChecked(
            bool(settings.get("use_bubble_mask", self.use_bubble_mask_checkbox.isChecked()))
        )
        self.use_crop_windows_checkbox.setChecked(
            bool(settings.get("use_crop_windows", self.use_crop_windows_checkbox.isChecked()))
        )
        self.device_input.setCurrentText(str(settings.get("device", "") or self.device_input.currentText()))

    def consume_settings_migration_message(self) -> str | None:
        message = self._pending_settings_migration_message
        self._pending_settings_migration_message = None
        return message


__all__ = ["InpaintPanel"]
