"""Reusable multiline text editor widget for OCR/translation item editing."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QLabel, QPlainTextEdit, QVBoxLayout, QWidget


class TextItemEditorWidget(QWidget):
    """Large text editor with dirty-state tracking and friendly empty state."""

    dirty_changed = pyqtSignal(bool)

    def __init__(
        self,
        title: str,
        *,
        placeholder: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("TextItemEditorWidget")
        self._loaded_text = ""
        self._suppress_dirty_signal = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        header_label = QLabel(title)
        header_label.setProperty("role", "sectionTitle")
        layout.addWidget(header_label)

        self.status_label = QLabel("No item selected.")
        self.status_label.setProperty("role", "muted")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.editor = QPlainTextEdit()
        self.editor.setPlaceholderText(placeholder)
        self.editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.editor.setMinimumHeight(150)
        self.editor.textChanged.connect(self._on_text_changed)
        layout.addWidget(self.editor)

        self.set_enabled_for_item(False)

    def set_enabled_for_item(self, enabled: bool, *, message: str | None = None) -> None:
        self.editor.setEnabled(enabled)
        if message is not None:
            self.status_label.setText(str(message))

    def set_loaded_text(self, text: str, *, status_text: str | None = None) -> None:
        self._loaded_text = str(text or "")
        self._suppress_dirty_signal = True
        try:
            self.editor.setPlainText(self._loaded_text)
        finally:
            self._suppress_dirty_signal = False
        self.editor.document().setModified(False)
        if status_text is not None:
            self.status_label.setText(str(status_text))
        self.dirty_changed.emit(False)

    def set_status_text(self, text: str) -> None:
        self.status_label.setText(str(text or ""))

    def text(self) -> str:
        return self.editor.toPlainText()

    def is_dirty(self) -> bool:
        return self.text() != self._loaded_text

    def revert(self) -> None:
        self.set_loaded_text(self._loaded_text, status_text=self.status_label.text())

    def _on_text_changed(self) -> None:
        if self._suppress_dirty_signal:
            return
        self.dirty_changed.emit(self.is_dirty())


__all__ = ["TextItemEditorWidget"]
