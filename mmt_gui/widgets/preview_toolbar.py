"""Central preview toolbar with mode and zoom controls."""

from __future__ import annotations

from collections.abc import Sequence

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton, QWidget

from .settings_card import style_button


class PreviewToolbar(QFrame):
    """Toolbar that controls preview mode and simple zoom actions."""

    preview_mode_changed = pyqtSignal(str)
    fit_requested = pyqtSignal()
    zoom_in_requested = pyqtSignal()
    zoom_out_requested = pyqtSignal()
    reset_zoom_requested = pyqtSignal()
    auto_preview_changed = pyqtSignal(bool)
    follow_batch_progress_changed = pyqtSignal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("PreviewToolbar")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        title_label = QLabel("Preview")
        title_label.setProperty("role", "sectionTitle")
        layout.addWidget(title_label)

        layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.currentTextChanged.connect(self.preview_mode_changed.emit)
        layout.addWidget(self.mode_combo, 1)

        self.auto_preview_checkbox = QCheckBox("Auto Preview Result")
        self.auto_preview_checkbox.setChecked(True)
        self.auto_preview_checkbox.toggled.connect(self.auto_preview_changed.emit)
        layout.addWidget(self.auto_preview_checkbox)

        self.follow_batch_checkbox = QCheckBox("Follow Batch Progress")
        self.follow_batch_checkbox.setChecked(False)
        self.follow_batch_checkbox.toggled.connect(self.follow_batch_progress_changed.emit)
        layout.addWidget(self.follow_batch_checkbox)

        self.fit_button = QPushButton("Fit")
        self.fit_button.clicked.connect(self.fit_requested.emit)
        style_button(self.fit_button, "secondary")
        layout.addWidget(self.fit_button)

        self.zoom_out_button = QPushButton("Zoom Out")
        self.zoom_out_button.clicked.connect(self.zoom_out_requested.emit)
        style_button(self.zoom_out_button, "secondary")
        layout.addWidget(self.zoom_out_button)

        self.reset_button = QPushButton("100%")
        self.reset_button.clicked.connect(self.reset_zoom_requested.emit)
        style_button(self.reset_button, "secondary")
        layout.addWidget(self.reset_button)

        self.zoom_in_button = QPushButton("Zoom In")
        self.zoom_in_button.clicked.connect(self.zoom_in_requested.emit)
        style_button(self.zoom_in_button, "secondary")
        layout.addWidget(self.zoom_in_button)

    def set_modes(self, modes: Sequence[str], current_mode: str | None = None) -> None:
        unique_modes = [str(mode) for mode in modes if str(mode).strip()]
        if not unique_modes:
            unique_modes = ["Source"]

        preferred_mode = str(current_mode or "").strip()
        if preferred_mode and preferred_mode not in unique_modes:
            preferred_mode = ""

        if not preferred_mode:
            current_text = self.mode_combo.currentText().strip()
            if current_text in unique_modes:
                preferred_mode = current_text

        if not preferred_mode:
            preferred_mode = unique_modes[0]

        self.mode_combo.blockSignals(True)
        self.mode_combo.clear()
        self.mode_combo.addItems(unique_modes)
        self.mode_combo.setCurrentText(preferred_mode)
        self.mode_combo.blockSignals(False)

    def current_mode(self) -> str:
        return self.mode_combo.currentText().strip() or "Source"

    def set_current_mode(self, mode: str) -> None:
        normalized = str(mode or "").strip()
        if not normalized:
            return
        self.mode_combo.setCurrentText(normalized)

    def auto_preview_enabled(self) -> bool:
        return self.auto_preview_checkbox.isChecked()

    def set_auto_preview_enabled(self, enabled: bool) -> None:
        self.auto_preview_checkbox.setChecked(bool(enabled))

    def follow_batch_progress_enabled(self) -> bool:
        return self.follow_batch_checkbox.isChecked()

    def set_follow_batch_progress_enabled(self, enabled: bool) -> None:
        self.follow_batch_checkbox.setChecked(bool(enabled))


__all__ = ["PreviewToolbar"]
