"""Export stage inspector panel."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from mmt_core import DEFAULT_NAMING_PATTERN, ExportConfig
from mmt_gui.widgets import CollapsibleSection
from mmt_gui.widgets.settings_card import style_button

from .base_panel import StagePanel

EXPORT_SOURCE_LABELS = {
    "render": "Render Result",
    "inpaint": "Inpaint Result",
    "source": "Source Image",
}
EXPORT_SOURCE_VALUES = {value: key for key, value in EXPORT_SOURCE_LABELS.items()}

PAGE_SCOPE_LABELS = {
    "current": "Current Page",
    "all": "All Pages",
    "selected": "Selected Pages",
}
PAGE_SCOPE_VALUES = {value: key for key, value in PAGE_SCOPE_LABELS.items()}

OUTPUT_FORMAT_LABELS = {
    "original": "Original",
    "png": "PNG",
    "jpeg": "JPEG",
    "webp": "WebP",
}
OUTPUT_FORMAT_VALUES = {value: key for key, value in OUTPUT_FORMAT_LABELS.items()}


class ExportPanel(StagePanel):
    """Functional export UI for copying/converting cached page outputs."""

    browse_output_requested = pyqtSignal()
    export_current_requested = pyqtSignal()
    export_all_requested = pyqtSignal()
    export_selected_requested = pyqtSignal()
    open_output_folder_requested = pyqtSignal()
    reload_summary_requested = pyqtSignal()

    def __init__(self, parent: object | None = None) -> None:
        super().__init__("Export", parent)
        self._last_summary: dict[str, Any] | None = None
        self._actions_enabled = True

        settings_section = CollapsibleSection("Export Settings", expanded=True)
        settings_form = QFormLayout()
        settings_form.setContentsMargins(0, 0, 0, 0)
        settings_form.setSpacing(8)

        self.export_source_input = QComboBox()
        for value in ("render", "inpaint", "source"):
            self.export_source_input.addItem(EXPORT_SOURCE_LABELS[value], value)
        self.export_source_input.setCurrentIndex(0)

        self.pages_input = QComboBox()
        for value in ("current", "all"):
            self.pages_input.addItem(PAGE_SCOPE_LABELS[value], value)
        self.pages_input.setCurrentText("Current Page")

        output_folder_widget = QWidget()
        output_folder_layout = QGridLayout(output_folder_widget)
        output_folder_layout.setContentsMargins(0, 0, 0, 0)
        output_folder_layout.setHorizontalSpacing(8)
        output_folder_layout.setVerticalSpacing(0)

        self.output_folder_input = QLineEdit()
        self.output_folder_input.setPlaceholderText("Choose an output folder.")
        self.browse_output_button = QPushButton("Browse")
        style_button(self.browse_output_button, "secondary")
        self.browse_output_button.clicked.connect(self.browse_output_requested.emit)
        output_folder_layout.addWidget(self.output_folder_input, 0, 0)
        output_folder_layout.addWidget(self.browse_output_button, 0, 1)

        settings_form.addRow("Export Source:", self.export_source_input)
        settings_form.addRow("Pages:", self.pages_input)
        settings_form.addRow("Output Folder:", output_folder_widget)
        settings_section.content_layout.addLayout(settings_form)

        self.selected_pages_note = QLabel(
            "Selected Pages is reserved for a future multi-select page list. Current and All Pages are available now."
        )
        self.selected_pages_note.setProperty("role", "muted")
        self.selected_pages_note.setWordWrap(True)
        settings_section.content_layout.addWidget(self.selected_pages_note)
        self.content_layout.addWidget(settings_section)

        format_section = CollapsibleSection("Naming & Format", expanded=True)
        format_form = QFormLayout()
        format_form.setContentsMargins(0, 0, 0, 0)
        format_form.setSpacing(8)

        self.format_input = QComboBox()
        for value in ("original", "png", "jpeg", "webp"):
            self.format_input.addItem(OUTPUT_FORMAT_LABELS[value], value)
        self.format_input.currentTextChanged.connect(self._update_quality_enabled_state)

        self.quality_input = QSpinBox()
        self.quality_input.setRange(1, 100)
        self.quality_input.setValue(95)

        self.naming_pattern_input = QLineEdit(DEFAULT_NAMING_PATTERN)
        self.naming_pattern_input.setPlaceholderText(DEFAULT_NAMING_PATTERN)

        format_form.addRow("Format:", self.format_input)
        format_form.addRow("Quality:", self.quality_input)
        format_form.addRow("Naming Pattern:", self.naming_pattern_input)
        format_section.content_layout.addLayout(format_form)
        self.content_layout.addWidget(format_section)

        options_section = CollapsibleSection("Advanced Options", expanded=False)
        self.create_zip_checkbox = QCheckBox("Create ZIP")
        self.include_manifest_checkbox = QCheckBox("Include manifest")
        self.include_manifest_checkbox.setChecked(True)
        self.overwrite_checkbox = QCheckBox("Overwrite existing files")
        self.open_output_folder_checkbox = QCheckBox("Open output folder after export")
        self.zip_name_input = QLineEdit()
        self.zip_name_input.setPlaceholderText("Optional ZIP name, e.g. my_project_export.zip")

        options_form = QFormLayout()
        options_form.setContentsMargins(0, 0, 0, 0)
        options_form.setSpacing(8)
        options_form.addRow("ZIP Name:", self.zip_name_input)
        options_form.addRow("Options:", self.create_zip_checkbox)
        options_section.content_layout.addLayout(options_form)
        options_section.content_layout.addWidget(self.include_manifest_checkbox)
        options_section.content_layout.addWidget(self.overwrite_checkbox)
        options_section.content_layout.addWidget(self.open_output_folder_checkbox)
        self.content_layout.addWidget(options_section)

        actions_section = CollapsibleSection("Export Actions", expanded=True)
        self.export_current_button = QPushButton("Export Current Page")
        style_button(self.export_current_button, "primary")
        self.export_current_button.clicked.connect(self.export_current_requested.emit)
        self.export_current_button.setToolTip("Export only the current page.")
        actions_section.content_layout.addWidget(self.export_current_button)

        self.export_all_button = QPushButton("Export All Pages")
        style_button(self.export_all_button, "primary")
        self.export_all_button.clicked.connect(self.export_all_requested.emit)
        self.export_all_button.setToolTip("Export every page in the project.")
        actions_section.content_layout.addWidget(self.export_all_button)

        self.export_selected_button = QPushButton("Export Selected Pages")
        self.export_selected_button.setEnabled(False)
        style_button(self.export_selected_button, "secondary")
        self.export_selected_button.clicked.connect(self.export_selected_requested.emit)
        actions_section.content_layout.addWidget(self.export_selected_button)

        self.open_output_folder_button = QPushButton("Open Output Folder")
        style_button(self.open_output_folder_button, "secondary")
        self.open_output_folder_button.clicked.connect(self.open_output_folder_requested.emit)
        actions_section.content_layout.addWidget(self.open_output_folder_button)

        self.reload_summary_button = QPushButton("Reload Last Export Summary")
        style_button(self.reload_summary_button, "secondary")
        self.reload_summary_button.clicked.connect(self.reload_summary_requested.emit)
        actions_section.content_layout.addWidget(self.reload_summary_button)
        self.content_layout.addWidget(actions_section)

        summary_section = CollapsibleSection("Export Summary", expanded=True)
        summary_form = QFormLayout()
        summary_form.setContentsMargins(0, 0, 0, 0)
        summary_form.setSpacing(8)

        self.total_pages_value = QLabel("0")
        self.exported_count_value = QLabel("0")
        self.skipped_count_value = QLabel("0")
        self.error_count_value = QLabel("0")
        self.output_dir_value = QLabel("-")
        self.output_dir_value.setWordWrap(True)
        self.output_dir_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.zip_path_value = QLabel("-")
        self.zip_path_value.setWordWrap(True)
        self.zip_path_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.manifest_path_value = QLabel("-")
        self.manifest_path_value.setWordWrap(True)
        self.manifest_path_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.export_note_value = QLabel("Run an export to populate the summary.")
        self.export_note_value.setWordWrap(True)
        self.export_note_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        summary_form.addRow("Total Pages:", self.total_pages_value)
        summary_form.addRow("Exported:", self.exported_count_value)
        summary_form.addRow("Skipped:", self.skipped_count_value)
        summary_form.addRow("Errors:", self.error_count_value)
        summary_form.addRow("Output Folder:", self.output_dir_value)
        summary_form.addRow("ZIP Path:", self.zip_path_value)
        summary_form.addRow("Manifest Path:", self.manifest_path_value)
        summary_form.addRow("Notes:", self.export_note_value)
        summary_section.content_layout.addLayout(summary_form)

        self.summary_table = QTableWidget(0, 5)
        self.summary_table.setProperty("stageTable", True)
        self.summary_table.setHorizontalHeaderLabels(["Page", "Source", "Output", "Status", "Error"])
        self.summary_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.summary_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.summary_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.summary_table.setAlternatingRowColors(True)
        summary_header = self.summary_table.horizontalHeader()
        summary_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        summary_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        summary_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        summary_header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        summary_header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.summary_table.setMinimumHeight(260)
        summary_section.content_layout.addWidget(self.summary_table)
        self.content_layout.addWidget(summary_section)

        self._update_quality_enabled_state()

    def build_config(self, *, page_scope_override: str | None = None) -> ExportConfig:
        page_scope = str(page_scope_override or self.pages_input.currentData() or "current")
        output_format = str(self.format_input.currentData() or "original")
        return ExportConfig(
            export_source=str(self.export_source_input.currentData() or "render"),
            page_scope=page_scope,
            selected_image_relative_paths=[],
            output_dir=self.output_folder_input.text().strip(),
            output_format=output_format,
            quality=self.quality_input.value(),
            create_zip=self.create_zip_checkbox.isChecked(),
            zip_name=self.zip_name_input.text().strip(),
            naming_pattern=self.naming_pattern_input.text().strip() or DEFAULT_NAMING_PATTERN,
            overwrite=self.overwrite_checkbox.isChecked(),
            open_output_folder=self.open_output_folder_checkbox.isChecked(),
            include_manifest=self.include_manifest_checkbox.isChecked(),
        )

    def set_output_dir(self, output_dir: str) -> None:
        self.output_folder_input.setText(str(output_dir or ""))

    def output_dir(self) -> str:
        return self.output_folder_input.text().strip()

    def last_summary(self) -> dict[str, Any] | None:
        return dict(self._last_summary) if isinstance(self._last_summary, dict) else None

    def set_summary(self, summary: dict[str, Any] | None) -> None:
        self._last_summary = dict(summary) if isinstance(summary, dict) else None
        payload = self._last_summary or {}
        self.total_pages_value.setText(str(payload.get("total_pages", 0) or 0))
        self.exported_count_value.setText(str(payload.get("exported_count", 0) or 0))
        self.skipped_count_value.setText(str(payload.get("skipped_count", 0) or 0))
        self.error_count_value.setText(str(payload.get("error_count", 0) or 0))
        self.output_dir_value.setText(str(payload.get("output_dir", "") or "-"))
        self.zip_path_value.setText(str(payload.get("zip_path", "") or "-"))
        self.manifest_path_value.setText(str(payload.get("manifest_path", "") or "-"))

        note_lines: list[str] = []
        if payload.get("manifest_error"):
            note_lines.append(f"Manifest: {payload['manifest_error']}")
        if payload.get("zip_error"):
            note_lines.append(f"ZIP: {payload['zip_error']}")
        if not note_lines:
            note_lines.append("Export summary loaded." if payload else "Run an export to populate the summary.")
        self.export_note_value.setText("\n".join(note_lines))

        items = payload.get("items", [])
        if not isinstance(items, list):
            items = []
        self.summary_table.setRowCount(len(items))
        for row_index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            row_values = [
                str(item.get("image_relative_path", "")),
                str(item.get("source_path", "")),
                str(item.get("output_path", "")),
                str(item.get("status", "")),
                str(item.get("error", "")),
            ]
            for column_index, value in enumerate(row_values):
                table_item = QTableWidgetItem(value)
                if column_index in (1, 2, 4):
                    table_item.setToolTip(value)
                self.summary_table.setItem(row_index, column_index, table_item)

    def clear_summary(self) -> None:
        self._last_summary = None
        self.summary_table.setRowCount(0)
        self.total_pages_value.setText("0")
        self.exported_count_value.setText("0")
        self.skipped_count_value.setText("0")
        self.error_count_value.setText("0")
        self.output_dir_value.setText("-")
        self.zip_path_value.setText("-")
        self.manifest_path_value.setText("-")
        self.export_note_value.setText("Run an export to populate the summary.")

    def set_actions_enabled(self, enabled: bool) -> None:
        self._actions_enabled = bool(enabled)
        for widget in (
            self.export_current_button,
            self.export_all_button,
            self.browse_output_button,
            self.open_output_folder_button,
            self.reload_summary_button,
            self.export_source_input,
            self.pages_input,
            self.output_folder_input,
            self.format_input,
            self.quality_input,
            self.naming_pattern_input,
            self.create_zip_checkbox,
            self.include_manifest_checkbox,
            self.overwrite_checkbox,
            self.open_output_folder_checkbox,
            self.zip_name_input,
        ):
            widget.setEnabled(enabled)
        self.export_selected_button.setEnabled(False)
        self._update_quality_enabled_state()

    def _update_quality_enabled_state(self) -> None:
        selected_format = str(self.format_input.currentData() or "original")
        self.quality_input.setEnabled(self._actions_enabled and selected_format in {"jpeg", "webp"})

    def settings_snapshot(self) -> dict[str, Any]:
        return {
            "export_source": str(self.export_source_input.currentData() or "render"),
            "page_scope": str(self.pages_input.currentData() or "current"),
            "output_dir": self.output_folder_input.text().strip(),
            "output_format": str(self.format_input.currentData() or "original"),
            "quality": self.quality_input.value(),
            "naming_pattern": self.naming_pattern_input.text().strip() or DEFAULT_NAMING_PATTERN,
            "create_zip": self.create_zip_checkbox.isChecked(),
            "zip_name": self.zip_name_input.text().strip(),
            "include_manifest": self.include_manifest_checkbox.isChecked(),
            "overwrite": self.overwrite_checkbox.isChecked(),
            "open_output_folder": self.open_output_folder_checkbox.isChecked(),
        }

    def apply_settings(self, settings: dict[str, Any]) -> None:
        if not isinstance(settings, dict):
            return
        self.export_source_input.setCurrentIndex(
            max(0, self.export_source_input.findData(str(settings.get("export_source", "render") or "render")))
        )
        page_scope = str(settings.get("page_scope", "current") or "current")
        page_scope_index = self.pages_input.findData(page_scope)
        if page_scope_index >= 0:
            self.pages_input.setCurrentIndex(page_scope_index)
        self.output_folder_input.setText(str(settings.get("output_dir", "") or ""))
        format_index = self.format_input.findData(str(settings.get("output_format", "original") or "original"))
        if format_index >= 0:
            self.format_input.setCurrentIndex(format_index)
        try:
            self.quality_input.setValue(int(settings.get("quality", self.quality_input.value())))
        except Exception:
            pass
        self.naming_pattern_input.setText(
            str(settings.get("naming_pattern", "") or DEFAULT_NAMING_PATTERN)
        )
        self.create_zip_checkbox.setChecked(bool(settings.get("create_zip", self.create_zip_checkbox.isChecked())))
        self.zip_name_input.setText(str(settings.get("zip_name", "") or ""))
        self.include_manifest_checkbox.setChecked(
            bool(settings.get("include_manifest", self.include_manifest_checkbox.isChecked()))
        )
        self.overwrite_checkbox.setChecked(bool(settings.get("overwrite", self.overwrite_checkbox.isChecked())))
        self.open_output_folder_checkbox.setChecked(
            bool(settings.get("open_output_folder", self.open_output_folder_checkbox.isChecked()))
        )
        self._update_quality_enabled_state()


__all__ = ["ExportPanel"]
