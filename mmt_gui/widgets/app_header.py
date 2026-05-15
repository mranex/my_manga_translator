"""Top application header bar."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from mmt_gui.styles import AVAILABLE_THEMES


class AppHeader(QFrame):
    """Displays project/page identity, progress, status, and theme controls."""

    theme_changed = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("AppHeader")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(16)

        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(2)

        self.project_name_label = QLabel("No Project Open")
        self.project_name_label.setProperty("role", "title")
        text_layout.addWidget(self.project_name_label)

        self.page_name_label = QLabel("No page selected")
        self.page_name_label.setProperty("role", "muted")
        text_layout.addWidget(self.page_name_label)
        layout.addLayout(text_layout, 1)

        status_layout = QVBoxLayout()
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(6)

        self.status_label = QLabel("Ready")
        self.status_label.setProperty("role", "muted")
        status_layout.addWidget(self.status_label)

        self.service_status_label = QLabel("Services: starting...")
        self.service_status_label.setProperty("role", "muted")
        status_layout.addWidget(self.service_status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)
        status_layout.addWidget(self.progress_bar)
        layout.addLayout(status_layout, 1)

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(AVAILABLE_THEMES)
        self.theme_combo.textActivated.connect(self.theme_changed.emit)
        layout.addWidget(self.theme_combo)

    def set_project_name(self, project_name: str) -> None:
        self.project_name_label.setText(str(project_name or "No Project Open"))

    def set_page_name(self, page_name: str) -> None:
        self.page_name_label.setText(str(page_name or "No page selected"))

    def set_status_text(self, message: str) -> None:
        self.status_label.setText(str(message or "Ready"))

    def set_service_status_text(self, message: str) -> None:
        self.service_status_label.setText(str(message or "Services: -"))

    def set_progress_value(self, value: int | None) -> None:
        if value is None:
            self.progress_bar.setVisible(False)
            self.progress_bar.setValue(0)
            return
        clamped = max(0, min(100, int(value)))
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(clamped)
        if clamped >= 100:
            self.progress_bar.setVisible(False)

    def set_theme_name(self, theme_name: str) -> None:
        index = self.theme_combo.findText(theme_name)
        if index >= 0:
            self.theme_combo.blockSignals(True)
            self.theme_combo.setCurrentIndex(index)
            self.theme_combo.blockSignals(False)


__all__ = ["AppHeader"]
