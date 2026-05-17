"""Detection stage inspector panel."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from mmt_core import summarize_detection_edit_state
from mmt_gui.widgets import CollapsibleSection, StaticSection
from mmt_gui.widgets.settings_card import style_button

from .base_panel import StagePanel

BOX_TYPE_OPTIONS = (
    ("Bubble", "bubble"),
    ("Layout Region", "layout_region"),
)


class DetectionPanel(StagePanel):
    """Inspector panel for cached detection runs and manual box editing."""

    run_selected_requested = pyqtSignal()
    rerun_selected_requested = pyqtSignal()
    run_all_requested = pyqtSignal()
    rerun_all_requested = pyqtSignal()
    reload_requested = pyqtSignal()
    clear_overlay_requested = pyqtSignal()
    edit_mode_toggled = pyqtSignal(bool)
    box_type_changed = pyqtSignal(str)
    create_box_toggled = pyqtSignal(bool)
    save_box_edits_requested = pyqtSignal()
    cancel_box_edits_requested = pyqtSignal()
    exclude_selected_requested = pyqtSignal()
    restore_selected_requested = pyqtSignal()
    show_excluded_toggled = pyqtSignal(bool)
    reload_boxes_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Detection", parent)
        self._actions_enabled = True
        self._edit_dirty = False
        self._selected_box: dict[str, Any] | None = None

        actions_card = StaticSection("Detection Action", expanded=True)
        self.actions_section = actions_card
        actions_layout = QGridLayout()
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setHorizontalSpacing(8)
        actions_layout.setVerticalSpacing(8)

        current_label = QLabel("Current Page")
        current_label.setProperty("role", "muted")
        actions_layout.addWidget(current_label, 0, 0)

        self.run_selected_button = QPushButton("Run")
        style_button(self.run_selected_button, "primary")
        self.run_selected_button.clicked.connect(self.run_selected_requested.emit)
        self.run_selected_button.setToolTip("Process the current page and reuse cached detection when available.")
        actions_layout.addWidget(self.run_selected_button, 0, 1)

        self.rerun_selected_button = QPushButton("Re-run")
        style_button(self.rerun_selected_button, "rerun")
        self.rerun_selected_button.clicked.connect(self.rerun_selected_requested.emit)
        self.rerun_selected_button.setToolTip("Force the current page to regenerate detection output.")
        actions_layout.addWidget(self.rerun_selected_button, 0, 2)

        all_label = QLabel("All Pages")
        all_label.setProperty("role", "muted")
        actions_layout.addWidget(all_label, 1, 0)

        self.run_all_button = QPushButton("Run All")
        style_button(self.run_all_button, "primary")
        self.run_all_button.clicked.connect(self.run_all_requested.emit)
        self.run_all_button.setToolTip("Process every page and reuse cached detection when available.")
        actions_layout.addWidget(self.run_all_button, 1, 1)

        self.rerun_all_button = QPushButton("Re-run All")
        style_button(self.rerun_all_button, "rerun")
        self.rerun_all_button.clicked.connect(self.rerun_all_requested.emit)
        self.rerun_all_button.setToolTip("Force every page to regenerate detection output.")
        actions_layout.addWidget(self.rerun_all_button, 1, 2)

        cache_label = QLabel("Cache")
        cache_label.setProperty("role", "muted")
        actions_layout.addWidget(cache_label, 2, 0)

        self.reload_button = QPushButton("Reload Cache")
        style_button(self.reload_button, "secondary")
        self.reload_button.clicked.connect(self.reload_requested.emit)
        self.reload_button.setToolTip("Reload cached detection data from disk.")
        actions_layout.addWidget(self.reload_button, 2, 1)

        self.clear_overlay_button = QPushButton("Clear Overlay")
        style_button(self.clear_overlay_button, "danger")
        self.clear_overlay_button.clicked.connect(self.clear_overlay_requested.emit)
        self.clear_overlay_button.setToolTip("Hide detection overlays in the preview.")
        actions_layout.addWidget(self.clear_overlay_button, 2, 2)
        actions_card.content_layout.addLayout(actions_layout)
        self.content_layout.addWidget(actions_card)

        stats_card = CollapsibleSection("Detection Stats", expanded=True)
        self.stats_section = stats_card
        stats_form = QFormLayout()
        stats_form.setContentsMargins(0, 0, 0, 0)
        stats_form.setSpacing(8)

        self.bubbles_value = QLabel("0")
        self.layout_regions_value = QLabel("0")
        self.method_value = QLabel("-")
        self.cache_path_value = QLabel("-")
        self.cache_path_value.setWordWrap(True)
        self.cache_path_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.edit_state_value = QLabel("No manual edits")
        self.edit_state_value.setWordWrap(True)

        stats_form.addRow("Bubbles:", self.bubbles_value)
        stats_form.addRow("Layout Regions:", self.layout_regions_value)
        stats_form.addRow("Method:", self.method_value)
        stats_form.addRow("Cache JSON:", self.cache_path_value)
        stats_form.addRow("Edit State:", self.edit_state_value)
        stats_card.content_layout.addLayout(stats_form)
        self.content_layout.addWidget(stats_card)

        editor_card = StaticSection("Detection Box Editor", expanded=True)
        self.editor_section = editor_card
        editor_layout = QGridLayout()
        editor_layout.setContentsMargins(0, 0, 0, 0)
        editor_layout.setHorizontalSpacing(8)
        editor_layout.setVerticalSpacing(8)

        self.enable_edit_checkbox = QCheckBox("Enable Box Editing")
        self.enable_edit_checkbox.toggled.connect(self.edit_mode_toggled.emit)
        editor_layout.addWidget(self.enable_edit_checkbox, 0, 0, 1, 2)

        self.show_excluded_checkbox = QCheckBox("Show Excluded Boxes")
        self.show_excluded_checkbox.toggled.connect(self.show_excluded_toggled.emit)
        editor_layout.addWidget(self.show_excluded_checkbox, 0, 2)

        editor_layout.addWidget(QLabel("Box Type:"), 1, 0)
        self.box_type_input = QComboBox()
        for label, value in BOX_TYPE_OPTIONS:
            self.box_type_input.addItem(label, value)
        self.box_type_input.currentIndexChanged.connect(self._emit_box_type_changed)
        editor_layout.addWidget(self.box_type_input, 1, 1)

        self.create_box_button = QPushButton("Create Box")
        self.create_box_button.setCheckable(True)
        style_button(self.create_box_button, "secondary")
        self.create_box_button.toggled.connect(self.create_box_toggled.emit)
        self.create_box_button.setToolTip("Click-drag on the preview to create a manual box of the selected type.")
        editor_layout.addWidget(self.create_box_button, 1, 2)

        self.save_box_edits_button = QPushButton("Save Box Edits")
        style_button(self.save_box_edits_button, "primary")
        self.save_box_edits_button.clicked.connect(self.save_box_edits_requested.emit)
        self.save_box_edits_button.setToolTip("Write edited detection boxes back to the cached detection JSON.")
        editor_layout.addWidget(self.save_box_edits_button, 2, 0)

        self.cancel_box_edits_button = QPushButton("Cancel Unsaved Edits")
        style_button(self.cancel_box_edits_button, "secondary")
        self.cancel_box_edits_button.clicked.connect(self.cancel_box_edits_requested.emit)
        self.cancel_box_edits_button.setToolTip("Discard in-memory box changes and reload the cached detection JSON.")
        editor_layout.addWidget(self.cancel_box_edits_button, 2, 1)

        self.reload_boxes_button = QPushButton("Reload Boxes")
        style_button(self.reload_boxes_button, "secondary")
        self.reload_boxes_button.clicked.connect(self.reload_boxes_requested.emit)
        self.reload_boxes_button.setToolTip("Reload editable boxes from the cached detection JSON.")
        editor_layout.addWidget(self.reload_boxes_button, 2, 2)

        self.exclude_selected_button = QPushButton("Delete / Exclude")
        style_button(self.exclude_selected_button, "danger")
        self.exclude_selected_button.clicked.connect(self.exclude_selected_requested.emit)
        self.exclude_selected_button.setToolTip("Soft-delete the selected box by marking it excluded.")
        editor_layout.addWidget(self.exclude_selected_button, 3, 0)

        self.restore_selected_button = QPushButton("Restore Selected")
        style_button(self.restore_selected_button, "secondary")
        self.restore_selected_button.clicked.connect(self.restore_selected_requested.emit)
        self.restore_selected_button.setToolTip("Restore an excluded detection item when Show Excluded Boxes is enabled.")
        editor_layout.addWidget(self.restore_selected_button, 3, 1)

        editor_card.content_layout.addLayout(editor_layout)

        self.editor_dirty_label = QLabel("")
        self.editor_dirty_label.setProperty("role", "muted")
        self.editor_dirty_label.setWordWrap(True)
        self.editor_dirty_label.setVisible(False)
        editor_card.content_layout.addWidget(self.editor_dirty_label)

        self.stale_warning_label = QLabel("")
        self.stale_warning_label.setProperty("role", "muted")
        self.stale_warning_label.setWordWrap(True)
        self.stale_warning_label.setVisible(False)
        editor_card.content_layout.addWidget(self.stale_warning_label)

        selected_form = QFormLayout()
        selected_form.setContentsMargins(0, 0, 0, 0)
        selected_form.setSpacing(8)
        self.selected_id_value = QLabel("-")
        self.selected_type_value = QLabel("-")
        self.selected_bbox_value = QLabel("-")
        self.selected_detector_value = QLabel("-")
        self.selected_confidence_value = QLabel("-")
        self.selected_bubble_value = QLabel("-")
        self.selected_flags_value = QLabel("-")
        self.selected_error_value = QLabel("-")
        self.selected_error_value.setWordWrap(True)

        selected_form.addRow("Selected ID:", self.selected_id_value)
        selected_form.addRow("Type:", self.selected_type_value)
        selected_form.addRow("BBox:", self.selected_bbox_value)
        selected_form.addRow("Detector:", self.selected_detector_value)
        selected_form.addRow("Confidence:", self.selected_confidence_value)
        selected_form.addRow("Bubble ID:", self.selected_bubble_value)
        selected_form.addRow("Flags:", self.selected_flags_value)
        selected_form.addRow("Error:", self.selected_error_value)
        editor_card.content_layout.addLayout(selected_form)
        self.content_layout.addWidget(editor_card)

        self._update_editor_button_state()

    def simplify_for_config_stage(self) -> None:
        self.detach_widget(self.stats_section)

    def set_detection_data(self, detection_data: dict[str, Any] | None, cache_path: str | None) -> None:
        payload = detection_data or {}
        summary = summarize_detection_edit_state(payload)
        self.bubbles_value.setText(
            f"{summary.get('active_bubbles', 0)} active / {summary.get('excluded_bubbles', 0)} excluded"
        )
        self.layout_regions_value.setText(
            f"{summary.get('active_layout_regions', 0)} active / {summary.get('excluded_layout_regions', 0)} excluded"
        )
        self.method_value.setText(str(payload.get("method") or "-"))
        self.cache_path_value.setText(cache_path or "-")

        if summary.get("edited"):
            stale_targets = summary.get("downstream_stale", [])
            stale_text = ", ".join(str(value) for value in stale_targets if str(value).strip())
            if stale_text:
                self.edit_state_value.setText(f"Edited manually; stale: {stale_text}")
            else:
                self.edit_state_value.setText("Edited manually")
        else:
            self.edit_state_value.setText("No manual edits")

    def clear_detection_data(self, cache_path: str | None = None) -> None:
        self.set_detection_data(None, cache_path)
        self.set_selected_box(None)
        self.set_dirty(False)
        self.set_stale_warning(None)

    def set_actions_enabled(self, enabled: bool) -> None:
        self._actions_enabled = bool(enabled)
        for button in (
            self.run_selected_button,
            self.rerun_selected_button,
            self.run_all_button,
            self.rerun_all_button,
            self.reload_button,
            self.clear_overlay_button,
            self.save_box_edits_button,
            self.cancel_box_edits_button,
            self.reload_boxes_button,
            self.exclude_selected_button,
            self.restore_selected_button,
            self.create_box_button,
        ):
            button.setEnabled(self._actions_enabled)
        self.enable_edit_checkbox.setEnabled(self._actions_enabled)
        self.show_excluded_checkbox.setEnabled(self._actions_enabled)
        self.box_type_input.setEnabled(self._actions_enabled)
        self._update_editor_button_state()

    def set_edit_mode_checked(self, enabled: bool) -> None:
        self.enable_edit_checkbox.blockSignals(True)
        self.enable_edit_checkbox.setChecked(bool(enabled))
        self.enable_edit_checkbox.blockSignals(False)
        self._update_editor_button_state()

    def edit_mode_enabled(self) -> bool:
        return self.enable_edit_checkbox.isChecked()

    def set_show_excluded_checked(self, enabled: bool) -> None:
        self.show_excluded_checkbox.blockSignals(True)
        self.show_excluded_checkbox.setChecked(bool(enabled))
        self.show_excluded_checkbox.blockSignals(False)
        self._update_editor_button_state()

    def show_excluded_enabled(self) -> bool:
        return self.show_excluded_checkbox.isChecked()

    def set_create_box_checked(self, enabled: bool) -> None:
        self.create_box_button.blockSignals(True)
        self.create_box_button.setChecked(bool(enabled))
        self.create_box_button.blockSignals(False)
        self._update_editor_button_state()

    def create_box_enabled(self) -> bool:
        return self.create_box_button.isChecked()

    def selected_box_category(self) -> str:
        return str(self.box_type_input.currentData() or "bubble")

    def set_selected_box_category(self, category: str) -> None:
        normalized = str(category or "").strip().lower()
        for index in range(self.box_type_input.count()):
            if str(self.box_type_input.itemData(index) or "") == normalized:
                self.box_type_input.blockSignals(True)
                self.box_type_input.setCurrentIndex(index)
                self.box_type_input.blockSignals(False)
                break

    def set_dirty(self, dirty: bool) -> None:
        self._edit_dirty = bool(dirty)
        self.editor_dirty_label.setVisible(self._edit_dirty)
        self.editor_dirty_label.setText("Unsaved detection box edits")
        self._update_editor_button_state()

    def has_unsaved_box_edits(self) -> bool:
        return self._edit_dirty

    def set_selected_box(self, box_data: dict[str, Any] | None) -> None:
        self._selected_box = dict(box_data) if isinstance(box_data, dict) else None
        if self._selected_box is None:
            self.selected_id_value.setText("-")
            self.selected_type_value.setText("-")
            self.selected_bbox_value.setText("-")
            self.selected_detector_value.setText("-")
            self.selected_confidence_value.setText("-")
            self.selected_bubble_value.setText("-")
            self.selected_flags_value.setText("-")
            self.selected_error_value.setText("-")
            self._update_editor_button_state()
            return

        box = self._selected_box
        bbox = box.get("bbox") if isinstance(box.get("bbox"), list) else None
        bbox_text = ", ".join(str(int(value)) for value in bbox[:4]) if bbox else "-"
        flags = []
        if bool(box.get("manual", False)):
            flags.append("manual")
        if bool(box.get("excluded", False)):
            flags.append("excluded")
        self.selected_id_value.setText(str(box.get("id", "-")))
        self.selected_type_value.setText(str(box.get("category", "-")).replace("_", " "))
        self.selected_bbox_value.setText(bbox_text)
        self.selected_detector_value.setText(str(box.get("detector") or box.get("source") or "-"))
        confidence = box.get("confidence")
        self.selected_confidence_value.setText("-" if confidence in (None, "") else str(confidence))
        self.selected_bubble_value.setText(
            "-" if box.get("bubble_id") in (None, "") else str(box.get("bubble_id"))
        )
        self.selected_flags_value.setText(", ".join(flags) if flags else "-")
        self.selected_error_value.setText(str(box.get("error") or "-"))
        self._update_editor_button_state()

    def selected_box(self) -> dict[str, Any] | None:
        return dict(self._selected_box) if self._selected_box is not None else None

    def set_stale_warning(self, text: str | None) -> None:
        normalized = str(text or "").strip()
        self.stale_warning_label.setText(normalized)
        self.stale_warning_label.setVisible(bool(normalized))

    def _emit_box_type_changed(self) -> None:
        self.box_type_changed.emit(self.selected_box_category())

    def _update_editor_button_state(self) -> None:
        edit_enabled = self.edit_mode_enabled() and self._actions_enabled
        has_selection = self._selected_box is not None
        selected_excluded = bool(self._selected_box and self._selected_box.get("excluded", False))

        self.show_excluded_checkbox.setEnabled(self._actions_enabled)
        self.box_type_input.setEnabled(edit_enabled)
        self.create_box_button.setEnabled(edit_enabled)
        self.save_box_edits_button.setEnabled(edit_enabled and self._edit_dirty)
        self.cancel_box_edits_button.setEnabled(edit_enabled and self._edit_dirty)
        self.reload_boxes_button.setEnabled(edit_enabled)
        self.exclude_selected_button.setEnabled(edit_enabled and has_selection and not selected_excluded)
        self.restore_selected_button.setEnabled(edit_enabled and has_selection and selected_excluded)


__all__ = ["DetectionPanel"]
