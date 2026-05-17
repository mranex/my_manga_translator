from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QWidget,
)


@dataclass(slots=True)
class StageToolButton:
    button: QPushButton
    label: str | None = None


@dataclass(slots=True)
class StageToolGroup:
    title: str
    items: Sequence[StageToolButton]


class _NoWheelScrollArea(QScrollArea):
    def wheelEvent(self, event) -> None:  # type: ignore[override]
        event.accept()


class StageSubToolBar(QFrame):
    """Compact horizontal action strip shown below the workflow stage tabs."""

    _EXPANDED_MIN_HEIGHT = 56
    _EXPANDED_MAX_HEIGHT = 64

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("StageSubToolBar")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._active_buttons: list[QPushButton] = []

        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(10, 8, 10, 8)
        root_layout.setSpacing(0)

        self.scroll_area = _NoWheelScrollArea(self)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        root_layout.addWidget(self.scroll_area)

        self.content_widget = QWidget(self.scroll_area)
        self.content_widget.setObjectName("StageSubToolContent")
        self.content_widget.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)

        self.content_layout = QHBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(8)
        self.content_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.scroll_area.setWidget(self.content_widget)
        self._set_collapsed(True)

    def set_stage_groups(self, _stage_label: str, groups: Sequence[StageToolGroup]) -> None:
        normalized_groups = [group for group in groups if list(group.items)]
        self._clear_groups()
        if not normalized_groups:
            self._set_collapsed(True)
            return

        for group_index, group in enumerate(normalized_groups):
            group_frame = QFrame(self.content_widget)
            group_frame.setObjectName("StageSubToolGroup")
            group_frame.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)

            group_layout = QHBoxLayout(group_frame)
            group_layout.setContentsMargins(10, 6, 10, 6)
            group_layout.setSpacing(8)
            group_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

            group_label = QLabel(str(group.title).upper(), group_frame)
            group_label.setProperty("stageSubtoolGroupTitle", True)
            group_label.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
            group_layout.addWidget(group_label)

            for tool_button in group.items:
                button = tool_button.button
                if tool_button.label:
                    button.setText(str(tool_button.label))
                button.setParent(group_frame)
                button.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
                button.setMinimumWidth(96)
                button.setMaximumWidth(200)
                button.setFixedHeight(34)
                group_layout.addWidget(button, 0, Qt.AlignmentFlag.AlignVCenter)
                button.show()
                
                # Force QSS to re-evaluate ancestor selectors (like QFrame#StageSubToolGroup)
                button.style().unpolish(button)
                button.style().polish(button)
                
                self._active_buttons.append(button)

            self.content_layout.addWidget(group_frame, 0, Qt.AlignmentFlag.AlignVCenter)

            if group_index < len(normalized_groups) - 1:
                separator = QFrame(self.content_widget)
                separator.setObjectName("StageSubToolSeparator")
                separator.setFrameShape(QFrame.Shape.VLine)
                separator.setFrameShadow(QFrame.Shadow.Plain)
                separator.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
                separator.setFixedHeight(26)
                self.content_layout.addWidget(separator, 0, Qt.AlignmentFlag.AlignVCenter)

        self.content_layout.addStretch(1)
        self._set_collapsed(False)
        self.content_widget.adjustSize()
        self.updateGeometry()

    def _clear_groups(self) -> None:
        for button in self._active_buttons:
            button.hide()
            button.setParent(None)
        self._active_buttons.clear()

        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _set_collapsed(self, collapsed: bool) -> None:
        self.setVisible(not collapsed)
        if collapsed:
            self.setMinimumHeight(0)
            self.setMaximumHeight(0)
        else:
            self.setMinimumHeight(self._EXPANDED_MIN_HEIGHT)
            self.setMaximumHeight(self._EXPANDED_MAX_HEIGHT)


__all__ = ["StageSubToolBar", "StageToolButton", "StageToolGroup"]
