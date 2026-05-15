"""Top workflow tab strip with folder-like buttons and line status indicators."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .stage_status import StageStatusLine

STAGE_ORDER = (
    ("process", "Process"),
    ("project", "Project"),
    ("detection", "Detection"),
    ("ocr", "OCR"),
    ("translation", "Translation"),
    ("inpaint", "Inpaint"),
    ("render", "Render"),
    ("export", "Export"),
)


class WorkflowTabs(QFrame):
    """Folder-like workflow tabs shown above the studio workspace."""

    stage_selected = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("WorkflowTabs")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMaximumHeight(84)
        self._current_stage_key = "project"
        self._suppress_emits = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 4)
        layout.setSpacing(8)

        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True)
        self.stage_rows: dict[str, tuple[QPushButton, StageStatusLine]] = {}
        self._glow_effects: dict[str, QGraphicsDropShadowEffect] = {}

        for stage_key, stage_label in STAGE_ORDER:
            tab_container = QWidget(self)
            tab_container.setObjectName("WorkflowTabContainer")
            tab_layout = QVBoxLayout(tab_container)
            tab_layout.setContentsMargins(0, 0, 0, 0)
            tab_layout.setSpacing(4)

            button = QPushButton(stage_label, tab_container)
            button.setCheckable(True)
            button.setProperty("folderTab", True)
            button.setToolTip(stage_label)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setAutoDefault(False)
            button.setDefault(False)
            button.setMinimumWidth(96)
            button.setMaximumWidth(140)
            button.setFixedHeight(42)
            button.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            button.clicked.connect(lambda checked=False, key=stage_key: self._emit_stage_selected(key))

            glow_effect = QGraphicsDropShadowEffect(button)
            glow_effect.setBlurRadius(28)
            glow_effect.setOffset(0, 4)
            glow_effect.setEnabled(False)
            button.setGraphicsEffect(glow_effect)

            self.button_group.addButton(button)
            tab_layout.addWidget(button)

            status_line = StageStatusLine(tab_container)
            status_line.setProperty("workflowStageLine", True)
            tab_layout.addWidget(status_line)

            layout.addWidget(tab_container)
            self.stage_rows[stage_key] = (button, status_line)
            self._glow_effects[stage_key] = glow_effect
            button.setProperty("stageKey", stage_key)

        layout.addStretch(1)
        self.set_current_stage(self._current_stage_key)

    def set_current_stage(self, stage_key: str) -> None:
        if stage_key not in self.stage_rows:
            stage_key = "project"
        self._current_stage_key = stage_key
        self._suppress_emits = True
        try:
            for key, (button, _line) in self.stage_rows.items():
                is_active = key == stage_key
                button.setChecked(is_active)
                button.setProperty("activeTab", is_active)
                button.style().unpolish(button)
                button.style().polish(button)
                self._update_button_glow(key, active=is_active)
        finally:
            self._suppress_emits = False

    def set_stage_status(self, stage_key: str, status: str) -> None:
        row = self.stage_rows.get(stage_key)
        if row is None:
            return
        button, status_line = row
        status_line.set_status(status)
        button.setToolTip(f"{button.text()}\nStatus: {status_line.toolTip()}")

    def set_stage_statuses(self, statuses: dict[str, str]) -> None:
        for stage_key, _label in STAGE_ORDER:
            self.set_stage_status(stage_key, statuses.get(stage_key, "missing"))

    def _emit_stage_selected(self, stage_key: str) -> None:
        if self._suppress_emits:
            return
        self._current_stage_key = stage_key
        self.stage_selected.emit(stage_key)

    def refresh_theme_state(self) -> None:
        self.set_current_stage(self._current_stage_key)
        self.update()

    def _update_button_glow(self, stage_key: str, *, active: bool) -> None:
        effect = self._glow_effects.get(stage_key)
        if effect is None:
            return

        is_dark = self.palette().window().color().lightness() < 128
        glow_color = QColor("#38bdf8" if is_dark else "#60a5fa")
        glow_color.setAlpha(95 if active and is_dark else (70 if active else 0))
        effect.setColor(glow_color)
        effect.setEnabled(active)
        effect.setBlurRadius(28 if active else 0)
        effect.setOffset(0, 5 if active else 0)


__all__ = ["STAGE_ORDER", "WorkflowTabs"]
