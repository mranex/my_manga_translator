"""Render stage inspector panel."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
)

from mmt_core import (
    RenderConfig,
    list_project_fonts,
    parse_color_value,
    summarize_render_edit_state,
    summarize_render_json,
)
from mmt_gui.widgets import CollapsibleSection, StaticSection
from mmt_gui.widgets.settings_card import style_button

from .base_panel import StagePanel


class RenderPanel(StagePanel):
    """Inspector panel for render settings, metadata, and render box editing."""

    prepare_selected_requested = pyqtSignal()
    reprepare_selected_requested = pyqtSignal()
    prepare_all_requested = pyqtSignal()
    reprepare_all_requested = pyqtSignal()
    run_selected_requested = pyqtSignal()
    rerun_selected_requested = pyqtSignal()
    run_all_requested = pyqtSignal()
    rerun_all_requested = pyqtSignal()
    reload_requested = pyqtSignal()
    clear_preview_requested = pyqtSignal()
    box_edit_mode_toggled = pyqtSignal(bool)
    save_box_edits_requested = pyqtSignal()
    cancel_box_edits_requested = pyqtSignal()
    exclude_selected_box_requested = pyqtSignal()
    restore_selected_box_requested = pyqtSignal()
    show_excluded_items_toggled = pyqtSignal(bool)
    reload_box_cache_requested = pyqtSignal()
    current_item_changed = pyqtSignal(int)

    def __init__(self, workspace_root: Path, parent: object | None = None) -> None:
        super().__init__("Render", parent)
        self.workspace_root = workspace_root
        self._all_items: list[dict[str, Any]] = []
        self._items: list[dict[str, Any]] = []
        self._actions_enabled = True
        self._box_edit_dirty = False
        self._selected_box: dict[str, Any] | None = None

        actions_card = StaticSection("Render Action", expanded=True)
        self.actions_section = actions_card
        actions_layout = QGridLayout()
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setHorizontalSpacing(8)
        actions_layout.setVerticalSpacing(8)

        prepare_current_label = QLabel("Prepare Current")
        prepare_current_label.setProperty("role", "muted")
        actions_layout.addWidget(prepare_current_label, 0, 0)

        self.prepare_selected_button = QPushButton("Prepare")
        style_button(self.prepare_selected_button, "primary")
        self.prepare_selected_button.clicked.connect(self.prepare_selected_requested.emit)
        self.prepare_selected_button.setToolTip("Prepare render metadata for the current page and reuse cache when available.")
        actions_layout.addWidget(self.prepare_selected_button, 0, 1)

        self.reprepare_selected_button = QPushButton("Re-prepare")
        style_button(self.reprepare_selected_button, "rerun")
        self.reprepare_selected_button.clicked.connect(self.reprepare_selected_requested.emit)
        self.reprepare_selected_button.setToolTip("Force the current page to rebuild render metadata.")
        actions_layout.addWidget(self.reprepare_selected_button, 0, 2)

        prepare_all_label = QLabel("Prepare All")
        prepare_all_label.setProperty("role", "muted")
        actions_layout.addWidget(prepare_all_label, 1, 0)

        self.prepare_all_button = QPushButton("Prepare All")
        style_button(self.prepare_all_button, "primary")
        self.prepare_all_button.clicked.connect(self.prepare_all_requested.emit)
        self.prepare_all_button.setToolTip("Prepare render metadata for every page and reuse cache when available.")
        actions_layout.addWidget(self.prepare_all_button, 1, 1)

        self.reprepare_all_button = QPushButton("Re-prepare All")
        style_button(self.reprepare_all_button, "rerun")
        self.reprepare_all_button.clicked.connect(self.reprepare_all_requested.emit)
        self.reprepare_all_button.setToolTip("Force every page to rebuild render metadata.")
        actions_layout.addWidget(self.reprepare_all_button, 1, 2)

        render_current_label = QLabel("Render Current")
        render_current_label.setProperty("role", "muted")
        actions_layout.addWidget(render_current_label, 2, 0)

        self.run_selected_button = QPushButton("Render")
        style_button(self.run_selected_button, "primary")
        self.run_selected_button.clicked.connect(self.run_selected_requested.emit)
        self.run_selected_button.setToolTip("Render the current page and reuse cached output when available.")
        actions_layout.addWidget(self.run_selected_button, 2, 1)

        self.rerun_selected_button = QPushButton("Re-render")
        style_button(self.rerun_selected_button, "rerun")
        self.rerun_selected_button.clicked.connect(self.rerun_selected_requested.emit)
        self.rerun_selected_button.setToolTip("Force the current page to regenerate rendered output.")
        actions_layout.addWidget(self.rerun_selected_button, 2, 2)

        render_all_label = QLabel("Render All")
        render_all_label.setProperty("role", "muted")
        actions_layout.addWidget(render_all_label, 3, 0)

        self.run_all_button = QPushButton("Render All")
        style_button(self.run_all_button, "primary")
        self.run_all_button.clicked.connect(self.run_all_requested.emit)
        self.run_all_button.setToolTip("Render every page and reuse cached output when available.")
        actions_layout.addWidget(self.run_all_button, 3, 1)

        self.rerun_all_button = QPushButton("Re-render All")
        style_button(self.rerun_all_button, "rerun")
        self.rerun_all_button.clicked.connect(self.rerun_all_requested.emit)
        self.rerun_all_button.setToolTip("Force every page to regenerate rendered output.")
        actions_layout.addWidget(self.rerun_all_button, 3, 2)

        cache_label = QLabel("Cache")
        cache_label.setProperty("role", "muted")
        actions_layout.addWidget(cache_label, 4, 0)

        self.reload_button = QPushButton("Reload Cache")
        style_button(self.reload_button, "secondary")
        self.reload_button.clicked.connect(self.reload_requested.emit)
        self.reload_button.setToolTip("Reload render data from disk.")
        actions_layout.addWidget(self.reload_button, 4, 1)

        self.clear_preview_button = QPushButton("Clear Render Preview")
        style_button(self.clear_preview_button, "danger")
        self.clear_preview_button.clicked.connect(self.clear_preview_requested.emit)
        self.clear_preview_button.setToolTip("Return the preview to the source page.")
        actions_layout.addWidget(self.clear_preview_button, 4, 2)
        actions_card.content_layout.addLayout(actions_layout)
        self.content_layout.addWidget(actions_card)

        settings_card = CollapsibleSection("Render Settings", expanded=False)
        self.settings_section = settings_card
        settings_form = QFormLayout()
        settings_form.setContentsMargins(0, 0, 0, 0)
        settings_form.setSpacing(8)

        self.font_name_input = QComboBox()
        self.font_name_input.setEditable(True)
        self.font_name_input.addItem("")
        for display_name, _font_path in list_project_fonts(workspace_root):
            self.font_name_input.addItem(display_name)
        if self.font_name_input.count() > 1:
            self.font_name_input.setCurrentIndex(1)

        self.font_path_input = QLineEdit()
        self.font_path_input.setPlaceholderText("Optional explicit font file path.")

        self.min_font_size_input = QSpinBox()
        self.min_font_size_input.setRange(6, 256)
        self.min_font_size_input.setValue(12)
        self.max_font_size_input = QSpinBox()
        self.max_font_size_input.setRange(6, 512)
        self.max_font_size_input.setValue(72)

        self.stroke_enabled_checkbox = QCheckBox("Stroke enabled")
        self.stroke_enabled_checkbox.setChecked(True)
        self.stroke_width_input = QDoubleSpinBox()
        self.stroke_width_input.setRange(0.0, 20.0)
        self.stroke_width_input.setSingleStep(0.5)
        self.stroke_width_input.setValue(0.0)
        self.stroke_width_input.setSpecialValueText("Auto")

        self.text_color_input = QLineEdit("auto")
        self.stroke_color_input = QLineEdit("auto")

        self.auto_color_checkbox = QCheckBox("Auto color")
        self.auto_color_checkbox.setChecked(True)
        self.auto_direction_checkbox = QCheckBox("Auto direction")
        self.auto_direction_checkbox.setChecked(True)
        self.vertical_cjk_checkbox = QCheckBox("Vertical CJK")
        self.vertical_cjk_checkbox.setChecked(True)
        self.save_sprites_checkbox = QCheckBox("Save sprites")
        self.save_sprites_checkbox.setChecked(True)

        settings_form.addRow("Font:", self.font_name_input)
        settings_form.addRow("Font Path:", self.font_path_input)
        settings_form.addRow("Min Font Size:", self.min_font_size_input)
        settings_form.addRow("Max Font Size:", self.max_font_size_input)
        settings_form.addRow("Stroke Enabled:", self.stroke_enabled_checkbox)
        settings_form.addRow("Stroke Width:", self.stroke_width_input)
        settings_form.addRow("Text Color:", self.text_color_input)
        settings_form.addRow("Stroke Color:", self.stroke_color_input)
        settings_form.addRow("Auto Color:", self.auto_color_checkbox)
        settings_form.addRow("Auto Direction:", self.auto_direction_checkbox)
        settings_form.addRow("Vertical CJK:", self.vertical_cjk_checkbox)
        settings_form.addRow("Save Sprites:", self.save_sprites_checkbox)
        settings_card.content_layout.addLayout(settings_form)
        self.content_layout.addWidget(settings_card)

        details_card = CollapsibleSection("Render Details", expanded=True)
        self.details_section = details_card
        details_form = QFormLayout()
        details_form.setContentsMargins(0, 0, 0, 0)
        details_form.setSpacing(8)
        self.translation_cache_path_value = QLabel("-")
        self.translation_cache_path_value.setWordWrap(True)
        self.translation_cache_path_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.inpaint_image_path_value = QLabel("-")
        self.inpaint_image_path_value.setWordWrap(True)
        self.inpaint_image_path_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.output_path_value = QLabel("-")
        self.output_path_value.setWordWrap(True)
        self.output_path_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.item_count_value = QLabel("0")
        self.rendered_item_count_value = QLabel("0")
        self.skipped_item_count_value = QLabel("0")
        self.no_text_page_value = QLabel("No")
        self.status_value = QLabel("-")
        self.error_value = QLabel("-")
        self.error_value.setWordWrap(True)
        self.error_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        details_form.addRow("Translation Cache:", self.translation_cache_path_value)
        details_form.addRow("Inpaint Image:", self.inpaint_image_path_value)
        details_form.addRow("Render Output:", self.output_path_value)
        details_form.addRow("Item Count:", self.item_count_value)
        details_form.addRow("Rendered Items:", self.rendered_item_count_value)
        details_form.addRow("Skipped Items:", self.skipped_item_count_value)
        details_form.addRow("No Text Page:", self.no_text_page_value)
        details_form.addRow("Status:", self.status_value)
        details_form.addRow("Error:", self.error_value)
        details_card.content_layout.addLayout(details_form)
        self.content_layout.addWidget(details_card)

        items_card = CollapsibleSection("Render Items", expanded=True)
        self.items_section = items_card
        self.items_table = QTableWidget(0, 8)
        self.items_table.setProperty("stageTable", True)
        self.items_table.setHorizontalHeaderLabels(
            ["id", "kind", "writing_mode", "font_size", "status", "translated_text", "render_bbox", "sprite_path"]
        )
        self.items_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.items_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.items_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.items_table.setAlternatingRowColors(True)
        self.items_table.itemSelectionChanged.connect(self._on_item_selected)
        header = self.items_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)
        self.items_table.setMinimumHeight(280)
        items_card.content_layout.addWidget(self.items_table)
        self.content_layout.addWidget(items_card)

        self.box_editor_section = CollapsibleSection("Render Box Editor", expanded=False)
        box_editor_layout = QGridLayout()
        box_editor_layout.setContentsMargins(0, 0, 0, 0)
        box_editor_layout.setHorizontalSpacing(8)
        box_editor_layout.setVerticalSpacing(8)

        self.enable_box_edit_checkbox = QCheckBox("Enable Render Box Editing")
        self.enable_box_edit_checkbox.toggled.connect(self.box_edit_mode_toggled.emit)
        box_editor_layout.addWidget(self.enable_box_edit_checkbox, 0, 0, 1, 2)

        self.show_excluded_items_checkbox = QCheckBox("Show Excluded Render Items")
        self.show_excluded_items_checkbox.toggled.connect(self._on_show_excluded_toggled)
        box_editor_layout.addWidget(self.show_excluded_items_checkbox, 0, 2)

        self.save_box_edits_button = QPushButton("Save Box Edits")
        style_button(self.save_box_edits_button, "primary")
        self.save_box_edits_button.clicked.connect(self.save_box_edits_requested.emit)
        self.save_box_edits_button.setToolTip("Write edited render boxes back to the cached render JSON.")
        box_editor_layout.addWidget(self.save_box_edits_button, 1, 0)

        self.cancel_box_edits_button = QPushButton("Cancel Unsaved Edits")
        style_button(self.cancel_box_edits_button, "secondary")
        self.cancel_box_edits_button.clicked.connect(self.cancel_box_edits_requested.emit)
        self.cancel_box_edits_button.setToolTip("Discard in-memory render box edits and reload the cached render JSON.")
        box_editor_layout.addWidget(self.cancel_box_edits_button, 1, 1)

        self.reload_box_cache_button = QPushButton("Reload Boxes")
        style_button(self.reload_box_cache_button, "secondary")
        self.reload_box_cache_button.clicked.connect(self.reload_box_cache_requested.emit)
        self.reload_box_cache_button.setToolTip("Reload editable render boxes from the cached render JSON.")
        box_editor_layout.addWidget(self.reload_box_cache_button, 1, 2)

        self.exclude_selected_box_button = QPushButton("Delete / Exclude")
        style_button(self.exclude_selected_box_button, "danger")
        self.exclude_selected_box_button.clicked.connect(self.exclude_selected_box_requested.emit)
        self.exclude_selected_box_button.setToolTip("Soft-delete the selected render item by marking it excluded.")
        box_editor_layout.addWidget(self.exclude_selected_box_button, 2, 0)

        self.restore_selected_box_button = QPushButton("Restore Selected")
        style_button(self.restore_selected_box_button, "secondary")
        self.restore_selected_box_button.clicked.connect(self.restore_selected_box_requested.emit)
        self.restore_selected_box_button.setToolTip("Restore an excluded render item when Show Excluded Render Items is enabled.")
        box_editor_layout.addWidget(self.restore_selected_box_button, 2, 1)
        self.box_editor_section.content_layout.addLayout(box_editor_layout)

        self.box_dirty_label = QLabel("")
        self.box_dirty_label.setProperty("role", "muted")
        self.box_dirty_label.setVisible(False)
        self.box_editor_section.content_layout.addWidget(self.box_dirty_label)

        self.box_warning_label = QLabel("")
        self.box_warning_label.setProperty("role", "muted")
        self.box_warning_label.setWordWrap(True)
        self.box_warning_label.setVisible(False)
        self.box_editor_section.content_layout.addWidget(self.box_warning_label)

        selected_form = QFormLayout()
        selected_form.setContentsMargins(0, 0, 0, 0)
        selected_form.setSpacing(8)
        self.selected_box_id_value = QLabel("-")
        self.selected_box_translation_id_value = QLabel("-")
        self.selected_box_ocr_id_value = QLabel("-")
        self.selected_box_kind_value = QLabel("-")
        self.selected_box_render_bbox_value = QLabel("-")
        self.selected_box_writing_mode_value = QLabel("-")
        self.selected_box_font_size_value = QLabel("-")
        self.selected_box_status_value = QLabel("-")
        self.selected_box_sprite_value = QLabel("-")
        self.selected_box_excluded_value = QLabel("-")
        self.selected_box_needs_render_value = QLabel("-")
        self.selected_box_error_value = QLabel("-")
        self.selected_box_error_value.setWordWrap(True)

        selected_form.addRow("Selected ID:", self.selected_box_id_value)
        selected_form.addRow("Translation Item:", self.selected_box_translation_id_value)
        selected_form.addRow("OCR Item:", self.selected_box_ocr_id_value)
        selected_form.addRow("Kind:", self.selected_box_kind_value)
        selected_form.addRow("Render BBox:", self.selected_box_render_bbox_value)
        selected_form.addRow("Writing Mode:", self.selected_box_writing_mode_value)
        selected_form.addRow("Font Size:", self.selected_box_font_size_value)
        selected_form.addRow("Status:", self.selected_box_status_value)
        selected_form.addRow("Sprite Path:", self.selected_box_sprite_value)
        selected_form.addRow("Excluded:", self.selected_box_excluded_value)
        selected_form.addRow("Needs Render:", self.selected_box_needs_render_value)
        selected_form.addRow("Error:", self.selected_box_error_value)
        self.box_editor_section.content_layout.addLayout(selected_form)
        self.content_layout.addWidget(self.box_editor_section)

        self._update_box_editor_state()

    def config_sections(self) -> list[QWidget]:
        return [self.settings_section]

    def simplify_for_config_stage(self) -> None:
        self.detach_widget(self.details_section)
        self.detach_widget(self.items_section)
        self.detach_widget(self.box_editor_section)

    def config(self, *, force_override: bool | None = None) -> RenderConfig:
        try:
            parsed_text_color = parse_color_value(self.text_color_input.text())
        except ValueError as exc:
            raise ValueError(f"Invalid text color value. {exc}") from exc

        try:
            parsed_stroke_color = parse_color_value(self.stroke_color_input.text())
        except ValueError as exc:
            raise ValueError(f"Invalid stroke color value. {exc}") from exc

        stroke_width_value = self.stroke_width_input.value()
        return RenderConfig(
            font_name=self.font_name_input.currentText().strip(),
            font_path=self.font_path_input.text().strip(),
            min_font_size=self.min_font_size_input.value(),
            max_font_size=max(self.min_font_size_input.value(), self.max_font_size_input.value()),
            stroke_enabled=self.stroke_enabled_checkbox.isChecked(),
            stroke_width=stroke_width_value if stroke_width_value > 0 else None,
            text_color=parsed_text_color,
            stroke_color=parsed_stroke_color,
            auto_color=self.auto_color_checkbox.isChecked(),
            auto_direction=self.auto_direction_checkbox.isChecked(),
            vertical_cjk=self.vertical_cjk_checkbox.isChecked(),
            save_sprites=self.save_sprites_checkbox.isChecked(),
            force=bool(force_override) if force_override is not None else False,
        )

    def set_data(self, render_data: dict[str, Any] | None, output_image: str | None = None) -> None:
        payload = render_data or {}
        previous_item_id = self.current_table_item_id()
        self._all_items = list(payload.get("items", [])) if isinstance(payload.get("items"), list) else []
        self._rebuild_items_table(previous_item_id=previous_item_id)

        summary = summarize_render_json(payload if payload else {"items": self._all_items, "status": "pending"})
        edit_summary = summarize_render_edit_state(payload if payload else {"items": self._all_items})
        self.translation_cache_path_value.setText(str(payload.get("translation_cache_path", "") or "-"))
        self.inpaint_image_path_value.setText(str(payload.get("inpaint_image_path", "") or "-"))
        self.output_path_value.setText(str(payload.get("output_image_path", "") or output_image or "-"))
        self.item_count_value.setText(str(int(payload.get("item_count", summary.get("total", 0)) or 0)))
        self.rendered_item_count_value.setText(str(summary.get("rendered", 0)))
        skipped_display = int(summary.get("skipped", 0) or 0) + int(summary.get("excluded", 0) or 0)
        self.skipped_item_count_value.setText(
            f"{skipped_display} ({int(summary.get('excluded', 0) or 0)} excluded)"
        )
        self.no_text_page_value.setText("Yes" if bool(payload.get("no_text_page", False)) else "No")
        self.status_value.setText(str(summary.get("status", "-") or "-"))
        self.error_value.setText(str(payload.get("error", "") or "-"))
        if edit_summary.get("needs_render"):
            self.set_box_warning("Render boxes changed. Re-render is recommended.")
        else:
            self.set_box_warning(None)

    def clear_view(self, *, output_image: str | None = None) -> None:
        self._all_items = []
        self._items = []
        self.items_table.blockSignals(True)
        self.items_table.setRowCount(0)
        self.items_table.blockSignals(False)
        self.translation_cache_path_value.setText("-")
        self.inpaint_image_path_value.setText("-")
        self.output_path_value.setText(output_image or "-")
        self.item_count_value.setText("0")
        self.rendered_item_count_value.setText("0")
        self.skipped_item_count_value.setText("0")
        self.no_text_page_value.setText("No")
        self.status_value.setText("-")
        self.error_value.setText("-")
        self.set_selected_box(None)
        self.set_box_dirty(False)
        self.set_box_warning(None)
        self.box_editor_section.set_expanded(False)

    def set_actions_enabled(self, enabled: bool) -> None:
        self._actions_enabled = bool(enabled)
        for widget in (
            self.prepare_selected_button,
            self.reprepare_selected_button,
            self.prepare_all_button,
            self.reprepare_all_button,
            self.run_selected_button,
            self.rerun_selected_button,
            self.run_all_button,
            self.rerun_all_button,
            self.reload_button,
            self.clear_preview_button,
            self.font_name_input,
            self.font_path_input,
            self.min_font_size_input,
            self.max_font_size_input,
            self.stroke_enabled_checkbox,
            self.stroke_width_input,
            self.text_color_input,
            self.stroke_color_input,
            self.auto_color_checkbox,
            self.auto_direction_checkbox,
            self.vertical_cjk_checkbox,
            self.save_sprites_checkbox,
            self.items_table,
            self.enable_box_edit_checkbox,
            self.show_excluded_items_checkbox,
            self.save_box_edits_button,
            self.cancel_box_edits_button,
            self.reload_box_cache_button,
            self.exclude_selected_box_button,
            self.restore_selected_box_button,
        ):
            widget.setEnabled(enabled)
        self._update_box_editor_state()

    def set_box_edit_mode_checked(self, enabled: bool) -> None:
        self.enable_box_edit_checkbox.blockSignals(True)
        self.enable_box_edit_checkbox.setChecked(bool(enabled))
        self.enable_box_edit_checkbox.blockSignals(False)
        self._update_box_editor_state()

    def box_edit_mode_enabled(self) -> bool:
        return self.enable_box_edit_checkbox.isChecked()

    def set_show_excluded_items_checked(self, enabled: bool) -> None:
        self.show_excluded_items_checkbox.blockSignals(True)
        self.show_excluded_items_checkbox.setChecked(bool(enabled))
        self.show_excluded_items_checkbox.blockSignals(False)
        self._rebuild_items_table(previous_item_id=self.current_table_item_id())
        self._update_box_editor_state()

    def show_excluded_items_enabled(self) -> bool:
        return self.show_excluded_items_checkbox.isChecked()

    def set_box_dirty(self, dirty: bool) -> None:
        self._box_edit_dirty = bool(dirty)
        self.box_dirty_label.setVisible(self._box_edit_dirty)
        self.box_dirty_label.setText("Unsaved render box edits")
        self._update_box_editor_state()

    def has_unsaved_box_edits(self) -> bool:
        return self._box_edit_dirty

    def select_item_by_id(self, item_id: int | None) -> bool:
        row_index = self._row_for_item_id(item_id)
        if row_index is not None:
            self.items_table.setCurrentCell(row_index, 0)
            return self.current_table_item_id() == item_id
        return False

    def current_table_item_id(self) -> int | None:
        row_index = self.items_table.currentRow()
        if row_index < 0 or row_index >= len(self._items):
            return None
        return int(self._items[row_index].get("id", row_index))

    def set_selected_box(self, box_data: dict[str, Any] | None) -> None:
        self._selected_box = dict(box_data) if isinstance(box_data, dict) else None
        if self._selected_box is None:
            self.selected_box_id_value.setText("-")
            self.selected_box_translation_id_value.setText("-")
            self.selected_box_ocr_id_value.setText("-")
            self.selected_box_kind_value.setText("-")
            self.selected_box_render_bbox_value.setText("-")
            self.selected_box_writing_mode_value.setText("-")
            self.selected_box_font_size_value.setText("-")
            self.selected_box_status_value.setText("-")
            self.selected_box_sprite_value.setText("-")
            self.selected_box_excluded_value.setText("-")
            self.selected_box_needs_render_value.setText("-")
            self.selected_box_error_value.setText("-")
            self._update_box_editor_state()
            return

        item = self._selected_box
        self.selected_box_id_value.setText(str(item.get("id", "-")))
        self.selected_box_translation_id_value.setText(str(item.get("translation_item_id", "-")))
        self.selected_box_ocr_id_value.setText(str(item.get("ocr_item_id", "-")))
        self.selected_box_kind_value.setText(str(item.get("kind", "-")))
        self.selected_box_render_bbox_value.setText(self._format_bbox(item.get("render_bbox")))
        self.selected_box_writing_mode_value.setText(str(item.get("writing_mode", "-") or "-"))
        self.selected_box_font_size_value.setText(str(item.get("font_size", "-") or "-"))
        self.selected_box_status_value.setText(str(item.get("status", "-") or "-"))
        self.selected_box_sprite_value.setText(str(item.get("sprite_path", "-") or "-"))
        self.selected_box_excluded_value.setText("Yes" if bool(item.get("excluded", False)) else "No")
        self.selected_box_needs_render_value.setText("Yes" if bool(item.get("needs_render", False)) else "No")
        self.selected_box_error_value.setText(str(item.get("error", "-") or "-"))
        self._update_box_editor_state()

    def selected_box(self) -> dict[str, Any] | None:
        return dict(self._selected_box) if self._selected_box is not None else None

    def set_box_warning(self, text: str | None) -> None:
        normalized = str(text or "").strip()
        self.box_warning_label.setText(normalized)
        self.box_warning_label.setVisible(bool(normalized))

    def settings_snapshot(self) -> dict[str, Any]:
        return {
            "font_name": self.font_name_input.currentText().strip(),
            "font_path": self.font_path_input.text().strip(),
            "min_font_size": self.min_font_size_input.value(),
            "max_font_size": self.max_font_size_input.value(),
            "stroke_enabled": self.stroke_enabled_checkbox.isChecked(),
            "stroke_width": self.stroke_width_input.value(),
            "text_color": self.text_color_input.text().strip(),
            "stroke_color": self.stroke_color_input.text().strip(),
            "auto_color": self.auto_color_checkbox.isChecked(),
            "auto_direction": self.auto_direction_checkbox.isChecked(),
            "vertical_cjk": self.vertical_cjk_checkbox.isChecked(),
            "save_sprites": self.save_sprites_checkbox.isChecked(),
        }

    def apply_settings(self, settings: dict[str, Any]) -> None:
        if not isinstance(settings, dict):
            return
        self.font_name_input.setCurrentText(str(settings.get("font_name", "") or self.font_name_input.currentText()))
        self.font_path_input.setText(str(settings.get("font_path", "") or ""))
        try:
            self.min_font_size_input.setValue(int(settings.get("min_font_size", self.min_font_size_input.value())))
        except Exception:
            pass
        try:
            self.max_font_size_input.setValue(int(settings.get("max_font_size", self.max_font_size_input.value())))
        except Exception:
            pass
        self.stroke_enabled_checkbox.setChecked(
            bool(settings.get("stroke_enabled", self.stroke_enabled_checkbox.isChecked()))
        )
        try:
            self.stroke_width_input.setValue(float(settings.get("stroke_width", self.stroke_width_input.value())))
        except Exception:
            pass
        self.text_color_input.setText(str(settings.get("text_color", "") or self.text_color_input.text()))
        self.stroke_color_input.setText(str(settings.get("stroke_color", "") or self.stroke_color_input.text()))
        self.auto_color_checkbox.setChecked(bool(settings.get("auto_color", self.auto_color_checkbox.isChecked())))
        self.auto_direction_checkbox.setChecked(
            bool(settings.get("auto_direction", self.auto_direction_checkbox.isChecked()))
        )
        self.vertical_cjk_checkbox.setChecked(
            bool(settings.get("vertical_cjk", self.vertical_cjk_checkbox.isChecked()))
        )
        self.save_sprites_checkbox.setChecked(
            bool(settings.get("save_sprites", self.save_sprites_checkbox.isChecked()))
        )

    def _rebuild_items_table(self, *, previous_item_id: int | None) -> None:
        show_excluded = self.show_excluded_items_checkbox.isChecked()
        self._items = [
            dict(item)
            for item in self._all_items
            if show_excluded or not bool(item.get("excluded", False))
        ]

        self.items_table.blockSignals(True)
        self.items_table.setRowCount(len(self._items))
        for row_index, item in enumerate(self._items):
            status_text = "excluded" if bool(item.get("excluded", False)) else str(item.get("status", ""))
            row_values = [
                str(item.get("id", "")),
                str(item.get("kind", "")),
                str(item.get("writing_mode", "")),
                str(item.get("font_size", "")),
                status_text,
                self._display_text(item.get("translated_text")),
                self._format_bbox(item.get("render_bbox")),
                str(item.get("sprite_path", "")),
            ]
            error_text = str(item.get("error", "") or "").strip()
            tooltip_lines = []
            translated_text = str(item.get("translated_text", "") or "").strip()
            if translated_text:
                tooltip_lines.append(translated_text)
            if bool(item.get("needs_render", False)):
                tooltip_lines.append("Render box changed. Re-render is recommended.")
            if bool(item.get("excluded", False)):
                tooltip_lines.append("This render item is excluded.")
            if error_text:
                tooltip_lines.append(f"Error: {error_text}")
            tooltip = "\n\n".join(tooltip_lines) if tooltip_lines else ""
            for column_index, value in enumerate(row_values):
                table_item = QTableWidgetItem(value)
                if tooltip:
                    table_item.setToolTip(tooltip)
                if column_index == 0:
                    table_item.setData(Qt.ItemDataRole.UserRole, int(item.get("id", row_index)))
                self.items_table.setItem(row_index, column_index, table_item)
        self.items_table.blockSignals(False)

        self.box_editor_section.set_expanded(bool(self._all_items))
        if self._items:
            target_row = self._row_for_item_id(previous_item_id)
            if target_row is None:
                target_row = 0
            self.items_table.setCurrentCell(target_row, 0)
            self._update_item_details(target_row)
        else:
            self._update_item_details(None)

    def _on_show_excluded_toggled(self, enabled: bool) -> None:
        self._rebuild_items_table(previous_item_id=self.current_table_item_id())
        self.show_excluded_items_toggled.emit(bool(enabled))
        self._update_box_editor_state()

    def _on_item_selected(self) -> None:
        row_index = self.items_table.currentRow()
        self._update_item_details(row_index)
        current_id = self.current_table_item_id()
        if current_id is not None:
            self.current_item_changed.emit(current_id)

    def _update_item_details(self, row_index: int | None) -> None:
        if row_index is None or row_index < 0 or row_index >= len(self._items):
            self.set_selected_box(None)
            return
        self.set_selected_box(self._items[row_index])

    def _row_for_item_id(self, item_id: int | None) -> int | None:
        if item_id is None:
            return None
        for row_index, item in enumerate(self._items):
            if int(item.get("id", row_index)) == int(item_id):
                return row_index
        return None

    def _update_box_editor_state(self) -> None:
        edit_enabled = self.box_edit_mode_enabled() and self._actions_enabled and bool(self._all_items)
        has_selection = self._selected_box is not None
        selected_excluded = bool(self._selected_box and self._selected_box.get("excluded", False))
        self.save_box_edits_button.setEnabled(edit_enabled and self._box_edit_dirty)
        self.cancel_box_edits_button.setEnabled(edit_enabled and self._box_edit_dirty)
        self.reload_box_cache_button.setEnabled(edit_enabled)
        self.exclude_selected_box_button.setEnabled(edit_enabled and has_selection and not selected_excluded)
        self.restore_selected_box_button.setEnabled(edit_enabled and has_selection and selected_excluded)

    @staticmethod
    def _display_text(value: Any, limit: int = 72) -> str:
        text = " ".join(str(value or "").split())
        if len(text) <= limit:
            return text
        return f"{text[: limit - 1]}..."

    @staticmethod
    def _format_bbox(bbox: Any) -> str:
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return "-"
        return f"[{int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}]"


__all__ = ["RenderPanel"]
