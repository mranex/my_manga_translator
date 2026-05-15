"""Project stage inspector panel."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFormLayout, QLabel, QPushButton, QWidget

from mmt_gui.widgets import CollapsibleSection
from mmt_gui.widgets.settings_card import style_button

from .base_panel import StagePanel


class ProjectPanel(StagePanel):
    """Inspector panel for project-level actions and metadata."""

    new_project_requested = pyqtSignal()
    open_project_requested = pyqtSignal()
    save_project_requested = pyqtSignal()
    import_images_requested = pyqtSignal()
    remove_current_page_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Project", parent)

        actions_card = CollapsibleSection("Project Actions", expanded=True)
        self.new_project_button = QPushButton("New Project")
        style_button(self.new_project_button, "primary")
        self.new_project_button.clicked.connect(self.new_project_requested.emit)
        actions_card.content_layout.addWidget(self.new_project_button)

        self.open_project_button = QPushButton("Open Project")
        style_button(self.open_project_button, "secondary")
        self.open_project_button.clicked.connect(self.open_project_requested.emit)
        actions_card.content_layout.addWidget(self.open_project_button)

        self.save_project_button = QPushButton("Save Project")
        style_button(self.save_project_button, "secondary")
        self.save_project_button.clicked.connect(self.save_project_requested.emit)
        actions_card.content_layout.addWidget(self.save_project_button)

        self.import_images_button = QPushButton("Import Images")
        style_button(self.import_images_button, "secondary")
        self.import_images_button.clicked.connect(self.import_images_requested.emit)
        actions_card.content_layout.addWidget(self.import_images_button)

        self.remove_current_page_button = QPushButton("Remove Current Page")
        style_button(self.remove_current_page_button, "danger")
        self.remove_current_page_button.clicked.connect(self.remove_current_page_requested.emit)
        actions_card.content_layout.addWidget(self.remove_current_page_button)
        self.content_layout.addWidget(actions_card)

        info_card = CollapsibleSection("Project Details", expanded=True)
        info_form = QFormLayout()
        info_form.setContentsMargins(0, 0, 0, 0)
        info_form.setSpacing(8)

        self.project_name_value = QLabel("No Project Open")
        self.project_root_value = QLabel("-")
        self.project_root_value.setWordWrap(True)
        self.project_root_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.source_dir_value = QLabel("-")
        self.source_dir_value.setWordWrap(True)
        self.source_dir_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.page_count_value = QLabel("0")
        self.current_page_value = QLabel("No page selected")
        self.current_page_value.setWordWrap(True)

        info_form.addRow("Name:", self.project_name_value)
        info_form.addRow("Project Root:", self.project_root_value)
        info_form.addRow("Source Folder:", self.source_dir_value)
        info_form.addRow("Pages:", self.page_count_value)
        info_form.addRow("Current Page:", self.current_page_value)
        info_card.content_layout.addLayout(info_form)
        self.content_layout.addWidget(info_card)

        overview_card = CollapsibleSection("Overview", expanded=False)
        overview_label = QLabel(
            "Use the workflow sidebar to move page-by-page through detection, OCR, translation, inpaint, and render."
        )
        overview_label.setWordWrap(True)
        overview_card.content_layout.addWidget(overview_label)
        self.content_layout.addWidget(overview_card)

    def set_project_details(
        self,
        *,
        project_name: str | None,
        project_root: str | None,
        source_dir: str | None,
        page_count: int,
        current_page_name: str | None,
    ) -> None:
        self.project_name_value.setText(project_name or "No Project Open")
        self.project_root_value.setText(project_root or "-")
        self.source_dir_value.setText(source_dir or "-")
        self.page_count_value.setText(str(max(0, int(page_count))))
        self.current_page_value.setText(current_page_name or "No page selected")

    def set_actions_enabled(self, enabled: bool) -> None:
        for button in (
            self.new_project_button,
            self.open_project_button,
            self.save_project_button,
            self.import_images_button,
            self.remove_current_page_button,
        ):
            button.setEnabled(enabled)


__all__ = ["ProjectPanel"]
