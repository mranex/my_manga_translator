"""Shared helpers for right-side inspector stage panels."""

from __future__ import annotations

from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from mmt_gui.widgets.stage_status import StatusLabel


class StagePanel(QScrollArea):
    """Scrollable inspector panel with a common header."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)

        container = QFrame()
        container.setObjectName("StagePanel")
        root_layout = QVBoxLayout(container)
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(14)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(10)

        title_label = QLabel(title)
        title_label.setProperty("role", "title")
        header_layout.addWidget(title_label)

        header_layout.addStretch(1)

        self.status_label = StatusLabel()
        header_layout.addWidget(self.status_label)
        root_layout.addLayout(header_layout)

        self.note_label = QLabel("")
        self.note_label.setProperty("role", "muted")
        self.note_label.setWordWrap(True)
        self.note_label.setVisible(False)
        root_layout.addWidget(self.note_label)

        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(14)
        root_layout.addLayout(self.content_layout)
        root_layout.addStretch(1)

        self.setWidget(container)

    def set_stage_status_text(self, text: str) -> None:
        self.status_label.set_status_text(text)

    def set_stage_note(self, text: str | None) -> None:
        normalized = str(text or "").strip()
        self.note_label.setText(normalized)
        self.note_label.setVisible(bool(normalized))

    def detach_widget(self, widget: QWidget | None) -> None:
        if widget is None:
            return
        widget.hide()
        widget.setParent(None)


__all__ = ["StagePanel"]
