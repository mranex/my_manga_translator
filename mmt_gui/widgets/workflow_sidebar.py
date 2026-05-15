"""Left workflow sidebar with stage navigation and page list."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .page_list import PageListWidget
from .settings_card import SettingsCard
from .stage_status import StageStatusDot

STAGE_ORDER = (
    ("project", "Project"),
    ("detection", "Detection"),
    ("ocr", "OCR"),
    ("translation", "Translation"),
    ("inpaint", "Inpaint"),
    ("render", "Render"),
    ("export", "Export"),
)


class WorkflowSidebar(QFrame):
    """Sidebar containing workflow stage buttons and the page list."""

    stage_selected = pyqtSignal(str)
    page_selected = pyqtSignal(int)
    remove_current_page_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("WorkflowSidebar")
        self._current_stage_key = "project"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(14)

        workflow_card = SettingsCard("Workflow")
        self.stage_button_group = QButtonGroup(self)
        self.stage_button_group.setExclusive(True)
        self.stage_rows: dict[str, tuple[QPushButton, StageStatusDot]] = {}

        for stage_key, stage_label in STAGE_ORDER:
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)

            button = QPushButton(stage_label)
            button.setCheckable(True)
            button.setProperty("navButton", True)
            button.clicked.connect(lambda checked=False, key=stage_key: self._emit_stage_selected(key))
            self.stage_button_group.addButton(button)
            row_layout.addWidget(button, 1)

            dot = StageStatusDot()
            row_layout.addWidget(dot)
            workflow_card.content_layout.addWidget(row_widget)
            self.stage_rows[stage_key] = (button, dot)

        workflow_card.content_layout.addStretch(1)
        layout.addWidget(workflow_card)

        pages_card = SettingsCard("Pages")
        self.page_list = PageListWidget()
        self.page_list.page_selected.connect(self.page_selected.emit)
        self.page_list.remove_page_requested.connect(self.remove_current_page_requested.emit)
        pages_card.content_layout.addWidget(self.page_list, 1)
        layout.addWidget(pages_card, 1)

        self.set_current_stage("project")

    def set_current_stage(self, stage_key: str) -> None:
        self._current_stage_key = stage_key
        button = self.stage_rows.get(stage_key, (None, None))[0]
        if button is not None:
            button.setChecked(True)

    def set_stage_status(self, stage_key: str, status: str) -> None:
        row = self.stage_rows.get(stage_key)
        if row is None:
            return
        row[1].set_status(status)

    def set_stage_statuses(self, statuses: dict[str, str]) -> None:
        for stage_key, _label in STAGE_ORDER:
            self.set_stage_status(stage_key, statuses.get(stage_key, "missing"))

    def set_pages(self, page_names: list[str], selected_index: int | None = None) -> None:
        self.page_list.set_pages(page_names, selected_index)

    def current_page_row(self) -> int:
        return self.page_list.currentRow()

    def _emit_stage_selected(self, stage_key: str) -> None:
        self._current_stage_key = stage_key
        self.stage_selected.emit(stage_key)


__all__ = ["STAGE_ORDER", "WorkflowSidebar"]
