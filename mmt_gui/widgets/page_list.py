"""Page list widget for project source images."""

from __future__ import annotations

from collections.abc import Sequence

from PyQt6.QtCore import QPoint, Qt, pyqtSignal
from PyQt6.QtWidgets import QListWidget, QListWidgetItem, QMenu


class PageListWidget(QListWidget):
    """Lists imported page filenames and emits row selections."""

    page_selected = pyqtSignal(int)
    remove_page_requested = pyqtSignal()

    def __init__(self, parent: QListWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("SidebarPageList")
        self.setProperty("pageList", True)
        self.setAlternatingRowColors(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.setUniformItemSizes(True)
        self.currentRowChanged.connect(self._emit_page_selected)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

    def set_pages(self, page_names: Sequence[str], selected_index: int | None = None) -> None:
        self.clear()

        for page_name in page_names:
            item = QListWidgetItem(page_name)
            item.setToolTip(page_name)
            self.addItem(item)

        if not page_names:
            return

        if selected_index is None or selected_index < 0 or selected_index >= len(page_names):
            selected_index = 0

        self.setCurrentRow(selected_index)

    def _emit_page_selected(self, row: int) -> None:
        if row >= 0:
            self.page_selected.emit(row)

    def _show_context_menu(self, position: QPoint) -> None:
        item = self.itemAt(position)
        if item is None:
            return

        row = self.row(item)
        if row >= 0 and row != self.currentRow():
            self.setCurrentRow(row)
            if self.currentRow() != row:
                return

        menu = QMenu(self)
        remove_action = menu.addAction("Remove Page from Project")
        chosen_action = menu.exec(self.viewport().mapToGlobal(position))
        if chosen_action == remove_action and self.currentRow() >= 0:
            self.remove_page_requested.emit()
