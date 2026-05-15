"""Collapsible developer log panel."""

from __future__ import annotations

from datetime import datetime

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .settings_card import style_button


class LogPanel(QFrame):
    """Reusable developer log surface for docked or embedded use."""

    expanded_changed = pyqtSignal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("LogPanel")
        self._expanded = False
        self._dock_mode = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(14, 10, 14, 10)
        header_layout.setSpacing(10)

        self.toggle_button = QToolButton()
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle_button.clicked.connect(self.toggle_expanded)
        header_layout.addWidget(self.toggle_button)

        title_label = QLabel("Developer Log")
        title_label.setProperty("role", "sectionTitle")
        header_layout.addWidget(title_label)

        header_layout.addStretch(1)

        self.clear_button = QPushButton("Clear Log")
        style_button(self.clear_button, "secondary")
        self.clear_button.clicked.connect(self.clear)
        header_layout.addWidget(self.clear_button)
        layout.addWidget(header_widget)

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setPlaceholderText("Worker messages, warnings, and pipeline errors will appear here.")
        self.output.document().setMaximumBlockCount(1500)
        self.output.setMinimumHeight(120)
        layout.addWidget(self.output)

        self.set_expanded(False)

    def append_message(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.output.appendPlainText(f"[{timestamp}] {message}")

    def clear(self) -> None:
        self.output.clear()

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = bool(expanded)
        self.output.setVisible(self._expanded)
        if not self._dock_mode:
            self.toggle_button.setText("Hide Log" if self._expanded else "Show Log")
            self.toggle_button.setArrowType(
                Qt.ArrowType.DownArrow if self._expanded else Qt.ArrowType.RightArrow
            )
        self.expanded_changed.emit(self._expanded)

    def is_expanded(self) -> bool:
        return self._expanded

    def toggle_expanded(self) -> None:
        self.set_expanded(not self._expanded)

    def set_dock_mode(self, enabled: bool) -> None:
        self._dock_mode = bool(enabled)
        self.toggle_button.setVisible(not self._dock_mode)
        if self._dock_mode and not self._expanded:
            self.set_expanded(True)


__all__ = ["LogPanel"]
