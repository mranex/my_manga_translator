"""Reusable collapsible section widget for stage inspector panels."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QToolButton, QVBoxLayout, QWidget


class CollapsibleSection(QFrame):
    """Compact spoiler/card section with a clickable header."""

    expanded_changed = pyqtSignal(bool)

    def __init__(
        self,
        title: str,
        *,
        badge_text: str = "",
        expanded: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("CollapsibleSection")
        self._expanded = False

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.header_button = QToolButton()
        self.header_button.setObjectName("CollapsibleHeader")
        self.header_button.setProperty("sectionToggle", True)
        self.header_button.setCheckable(True)
        self.header_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.header_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.header_button.clicked.connect(self.set_expanded)

        self.badge_label = QLabel()
        self.badge_label.setProperty("sectionBadge", True)
        self.badge_label.setVisible(False)

        header_row = QWidget()
        header_layout = QHBoxLayout(header_row)
        header_layout.setContentsMargins(12, 10, 12, 10)
        header_layout.setSpacing(10)
        header_layout.addWidget(self.header_button, 1)
        header_layout.addWidget(self.badge_label)
        root_layout.addWidget(header_row)

        self.body = QFrame()
        self.body.setObjectName("CollapsibleSectionBody")
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(12, 0, 12, 12)
        self.body_layout.setSpacing(10)
        root_layout.addWidget(self.body)

        self.set_title(title)
        self.set_badge_text(badge_text)
        self.set_expanded(expanded)

    @property
    def content_layout(self) -> QVBoxLayout:
        return self.body_layout

    def set_title(self, title: str) -> None:
        self.header_button.setText(str(title or "Section"))

    def set_badge_text(self, text: str) -> None:
        normalized = str(text or "").strip()
        self.badge_label.setText(normalized)
        self.badge_label.setVisible(bool(normalized))

    def set_content(self, widget: QWidget) -> None:
        self.body_layout.addWidget(widget)

    def set_expanded(self, expanded: bool) -> None:
        normalized = bool(expanded)
        self._expanded = normalized
        self.header_button.blockSignals(True)
        self.header_button.setChecked(normalized)
        self.header_button.blockSignals(False)
        self.header_button.setArrowType(
            Qt.ArrowType.DownArrow if normalized else Qt.ArrowType.RightArrow
        )
        self.body.setVisible(normalized)
        self.expanded_changed.emit(normalized)

    def is_expanded(self) -> bool:
        return self._expanded


class StaticSection(QFrame):
    """Always-visible card section used for flattened workflow stages."""

    expanded_changed = pyqtSignal(bool)

    def __init__(
        self,
        title: str,
        *,
        badge_text: str = "",
        expanded: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("SettingsCard")
        self._expanded = True

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(10)

        header_row = QWidget()
        header_layout = QHBoxLayout(header_row)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(10)

        self.title_label = QLabel()
        self.title_label.setProperty("role", "sectionTitle")
        header_layout.addWidget(self.title_label, 1)

        self.badge_label = QLabel()
        self.badge_label.setProperty("sectionBadge", True)
        self.badge_label.setVisible(False)
        header_layout.addWidget(self.badge_label)
        root_layout.addWidget(header_row)

        self.body_layout = QVBoxLayout()
        self.body_layout.setContentsMargins(0, 0, 0, 0)
        self.body_layout.setSpacing(10)
        root_layout.addLayout(self.body_layout)

        self.set_title(title)
        self.set_badge_text(badge_text)
        self.set_expanded(expanded)

    @property
    def content_layout(self) -> QVBoxLayout:
        return self.body_layout

    def set_title(self, title: str) -> None:
        self.title_label.setText(str(title or "Section"))

    def set_badge_text(self, text: str) -> None:
        normalized = str(text or "").strip()
        self.badge_label.setText(normalized)
        self.badge_label.setVisible(bool(normalized))

    def set_content(self, widget: QWidget) -> None:
        self.body_layout.addWidget(widget)

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = True
        self.setVisible(True)
        self.expanded_changed.emit(True)

    def is_expanded(self) -> bool:
        return True


__all__ = ["CollapsibleSection", "StaticSection"]
