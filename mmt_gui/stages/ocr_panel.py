"""OCR stage inspector panel."""

from __future__ import annotations

from pathlib import Path
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
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mmt_core import (
    OCRConfig,
    OCR_PROVIDER_CHOICES,
    OCR_PROVIDER_CHROME_LENS,
    OCR_PROVIDER_DEEPSEEK_OCR_LLAMA,
    OCR_PROVIDER_PADDLE_VL_LLAMA,
    normalize_ocr_provider_name,
    summarize_ocr_edit_state,
    summarize_ocr_items,
    update_ocr_item_text,
)
from mmt_gui.widgets import CollapsibleSection, CropPreviewPanel, StaticSection, TextItemEditorWidget
from mmt_gui.widgets.settings_card import style_button

from .base_panel import StagePanel


class OCRPanel(StagePanel):
    """Inspector panel for server controls, OCR items, and OCR box editing."""

    start_server_requested = pyqtSignal()
    check_server_requested = pyqtSignal()
    stop_server_requested = pyqtSignal()
    prepare_selected_requested = pyqtSignal()
    reprepare_selected_requested = pyqtSignal()
    prepare_all_requested = pyqtSignal()
    reprepare_all_requested = pyqtSignal()
    run_selected_requested = pyqtSignal()
    rerun_selected_requested = pyqtSignal()
    run_all_requested = pyqtSignal()
    rerun_all_requested = pyqtSignal()
    run_selected_items_requested = pyqtSignal()
    rerun_selected_items_requested = pyqtSignal()
    reload_requested = pyqtSignal()
    save_text_requested = pyqtSignal()
    box_edit_mode_toggled = pyqtSignal(bool)
    box_field_changed = pyqtSignal(str)
    save_box_edits_requested = pyqtSignal()
    cancel_box_edits_requested = pyqtSignal()
    exclude_selected_box_requested = pyqtSignal()
    restore_selected_box_requested = pyqtSignal()
    show_excluded_items_toggled = pyqtSignal(bool)
    reload_box_cache_requested = pyqtSignal()
    current_item_changed = pyqtSignal(int)
    ocr_provider_changed = pyqtSignal(str)
    cache_updated = pyqtSignal(object)
    error_emitted = pyqtSignal(str, str)
    warning_emitted = pyqtSignal(str)
    message_emitted = pyqtSignal(str)

    def __init__(self, parent: object | None = None) -> None:
        super().__init__("OCR", parent)
        self._actions_enabled = True
        self._project_root: Path | None = None
        self._all_items: list[dict[str, Any]] = []
        self._items: list[dict[str, Any]] = []
        self._cache_path: Path | None = None
        self._editor_row: int | None = None
        self._selection_guard = False
        self._box_edit_dirty = False
        self._selected_box: dict[str, Any] | None = None

        self.provider_section = CollapsibleSection("OCR Provider", expanded=True)
        provider_form = QFormLayout()
        provider_form.setContentsMargins(0, 0, 0, 0)
        provider_form.setSpacing(8)
        self.ocr_provider_input = QComboBox()
        for provider_key, provider_text in OCR_PROVIDER_CHOICES:
            self.ocr_provider_input.addItem(provider_text, provider_key)
        self.ocr_provider_input.currentIndexChanged.connect(self._on_provider_changed)
        provider_form.addRow("Provider:", self.ocr_provider_input)
        self.provider_section.content_layout.addLayout(provider_form)
        self.content_layout.addWidget(self.provider_section)

        self.chrome_lens_section = CollapsibleSection("Chrome Lens Settings", expanded=True)
        self.chrome_lens_info_label = QLabel(
            "Chrome Lens uses a browser-based OCR flow. It may require Chrome/browser access and can be slower or less predictable than PaddleOCR-VL Local."
        )
        self.chrome_lens_info_label.setWordWrap(True)
        self.chrome_lens_info_label.setProperty("role", "muted")
        self.chrome_lens_section.content_layout.addWidget(self.chrome_lens_info_label)

        self.chrome_lens_timeout_input = QSpinBox()
        self.chrome_lens_timeout_input.setRange(5, 1800)
        self.chrome_lens_timeout_input.setValue(120)
        self.chrome_lens_headless_checkbox = QCheckBox("Headless browser (if supported)")
        self.chrome_lens_path_input = QLineEdit()
        self.chrome_lens_user_data_dir_input = QLineEdit()
        self.chrome_lens_language_input = QLineEdit("ja")
        self.chrome_lens_max_retries_input = QSpinBox()
        self.chrome_lens_max_retries_input.setRange(1, 10)
        self.chrome_lens_max_retries_input.setValue(5)

        chrome_form = QFormLayout()
        chrome_form.setContentsMargins(0, 0, 0, 0)
        chrome_form.setSpacing(8)
        chrome_form.addRow("Timeout (s):", self.chrome_lens_timeout_input)
        chrome_form.addRow("Chrome Path:", self.chrome_lens_path_input)
        chrome_form.addRow("User Data Dir:", self.chrome_lens_user_data_dir_input)
        chrome_form.addRow("Source Language:", self.chrome_lens_language_input)
        chrome_form.addRow("Max Retries:", self.chrome_lens_max_retries_input)
        chrome_form.addRow("", self.chrome_lens_headless_checkbox)
        self.chrome_lens_section.content_layout.addLayout(chrome_form)
        self.content_layout.addWidget(self.chrome_lens_section)

        self.server_section = CollapsibleSection("OCR llama.cpp Server", expanded=True)
        self.server_url_input = QLineEdit()
        self.server_model_path_input = QLineEdit()
        self.server_mmproj_path_input = QLineEdit()
        self.server_llama_cpp_dir_input = QLineEdit()

        self.server_gpu_layers_input = QSpinBox()
        self.server_gpu_layers_input.setRange(-1, 999)
        self.server_ctx_size_input = QSpinBox()
        self.server_ctx_size_input.setRange(512, 131072)
        self.server_ctx_size_input.setSingleStep(512)

        self.check_server_button = QPushButton("Check Server")
        style_button(self.check_server_button, "secondary")
        self.check_server_button.clicked.connect(self.check_server_requested.emit)
        self.server_section.content_layout.addWidget(self.check_server_button)

        self.start_server_button = QPushButton("Start Server")
        style_button(self.start_server_button, "primary")
        self.start_server_button.clicked.connect(self.start_server_requested.emit)
        self.server_section.content_layout.addWidget(self.start_server_button)

        self.stop_server_button = QPushButton("Stop Server")
        style_button(self.stop_server_button, "danger")
        self.stop_server_button.clicked.connect(self.stop_server_requested.emit)
        self.server_section.content_layout.addWidget(self.stop_server_button)

        self.server_status_value = QLabel("Unknown")
        server_form2 = QFormLayout()
        server_form2.setContentsMargins(0, 0, 0, 0)
        server_form2.addRow("Status:", self.server_status_value)
        self.server_section.content_layout.addLayout(server_form2)

        self.server_settings_section = CollapsibleSection("Advanced Server Settings", expanded=False)
        server_form = QFormLayout()
        server_form.setContentsMargins(0, 0, 0, 0)
        server_form.setSpacing(8)
        server_form.addRow("Server URL:", self.server_url_input)
        server_form.addRow("Model Path:", self.server_model_path_input)
        server_form.addRow("mmproj Path:", self.server_mmproj_path_input)
        server_form.addRow("llama.cpp Dir:", self.server_llama_cpp_dir_input)
        server_form.addRow("GPU Layers:", self.server_gpu_layers_input)
        server_form.addRow("Context Size:", self.server_ctx_size_input)
        self.server_settings_section.content_layout.addLayout(server_form)
        self.server_section.content_layout.addWidget(self.server_settings_section)
        self.content_layout.addWidget(self.server_section)

        actions_card = StaticSection("OCR Action", expanded=True)
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
        self.prepare_selected_button.setToolTip("Prepare OCR items for the current page and reuse cache when available.")
        actions_layout.addWidget(self.prepare_selected_button, 0, 1)

        self.reprepare_selected_button = QPushButton("Re-prepare")
        style_button(self.reprepare_selected_button, "rerun")
        self.reprepare_selected_button.clicked.connect(self.reprepare_selected_requested.emit)
        self.reprepare_selected_button.setToolTip("Force the current page to rebuild OCR items and crops.")
        actions_layout.addWidget(self.reprepare_selected_button, 0, 2)

        prepare_all_label = QLabel("Prepare All")
        prepare_all_label.setProperty("role", "muted")
        actions_layout.addWidget(prepare_all_label, 1, 0)

        self.prepare_all_button = QPushButton("Prepare All")
        style_button(self.prepare_all_button, "primary")
        self.prepare_all_button.clicked.connect(self.prepare_all_requested.emit)
        self.prepare_all_button.setToolTip("Prepare OCR items for every page and reuse cache when available.")
        actions_layout.addWidget(self.prepare_all_button, 1, 1)

        self.reprepare_all_button = QPushButton("Re-prepare All")
        style_button(self.reprepare_all_button, "rerun")
        self.reprepare_all_button.clicked.connect(self.reprepare_all_requested.emit)
        self.reprepare_all_button.setToolTip("Force every page to rebuild OCR items and crops.")
        actions_layout.addWidget(self.reprepare_all_button, 1, 2)

        run_current_label = QLabel("OCR Current")
        run_current_label.setProperty("role", "muted")
        actions_layout.addWidget(run_current_label, 2, 0)

        self.run_selected_button = QPushButton("Run OCR")
        style_button(self.run_selected_button, "primary")
        self.run_selected_button.clicked.connect(self.run_selected_requested.emit)
        self.run_selected_button.setToolTip("Run OCR for the current page and reuse existing recognized items when available.")
        actions_layout.addWidget(self.run_selected_button, 2, 1)

        self.rerun_selected_button = QPushButton("Re-run OCR")
        style_button(self.rerun_selected_button, "rerun")
        self.rerun_selected_button.clicked.connect(self.rerun_selected_requested.emit)
        self.rerun_selected_button.setToolTip("Force the current page to regenerate OCR text.")
        actions_layout.addWidget(self.rerun_selected_button, 2, 2)

        run_all_label = QLabel("OCR All")
        run_all_label.setProperty("role", "muted")
        actions_layout.addWidget(run_all_label, 3, 0)

        self.run_all_button = QPushButton("Run OCR All")
        style_button(self.run_all_button, "primary")
        self.run_all_button.clicked.connect(self.run_all_requested.emit)
        self.run_all_button.setToolTip("Run OCR for every page and reuse recognized items when available.")
        actions_layout.addWidget(self.run_all_button, 3, 1)

        self.rerun_all_button = QPushButton("Re-run OCR All")
        style_button(self.rerun_all_button, "rerun")
        self.rerun_all_button.clicked.connect(self.rerun_all_requested.emit)
        self.rerun_all_button.setToolTip("Force every page to regenerate OCR text.")
        actions_layout.addWidget(self.rerun_all_button, 3, 2)

        selected_items_label = QLabel("Selected Items")
        selected_items_label.setProperty("role", "muted")
        actions_layout.addWidget(selected_items_label, 4, 0)

        self.run_selected_items_button = QPushButton("Run Selected")
        style_button(self.run_selected_items_button, "primary")
        self.run_selected_items_button.clicked.connect(self.run_selected_items_requested.emit)
        self.run_selected_items_button.setToolTip("Run OCR for the selected table rows and reuse existing results when available.")
        actions_layout.addWidget(self.run_selected_items_button, 4, 1)

        self.rerun_selected_items_button = QPushButton("Re-run Selected")
        style_button(self.rerun_selected_items_button, "rerun")
        self.rerun_selected_items_button.clicked.connect(self.rerun_selected_items_requested.emit)
        self.rerun_selected_items_button.setToolTip("Force the selected OCR items to regenerate text.")
        actions_layout.addWidget(self.rerun_selected_items_button, 4, 2)

        cache_label = QLabel("Cache")
        cache_label.setProperty("role", "muted")
        actions_layout.addWidget(cache_label, 5, 0)

        self.save_text_button = QPushButton("Save OCR Text")
        style_button(self.save_text_button, "secondary")
        self.save_text_button.clicked.connect(self.save_text_requested.emit)
        self.save_text_button.setToolTip("Save the currently selected OCR editor text to disk.")
        actions_layout.addWidget(self.save_text_button, 5, 1)

        self.reload_button = QPushButton("Reload Cache")
        style_button(self.reload_button, "secondary")
        self.reload_button.clicked.connect(self.reload_requested.emit)
        self.reload_button.setToolTip("Reload OCR items from disk.")
        actions_layout.addWidget(self.reload_button, 5, 2)
        actions_card.content_layout.addLayout(actions_layout)
        self.content_layout.addWidget(actions_card)

        details_card = CollapsibleSection("OCR Summary", expanded=True)
        self.summary_section = details_card
        details_form = QFormLayout()
        details_form.setContentsMargins(0, 0, 0, 0)
        details_form.setSpacing(8)
        self.total_items_value = QLabel("0")
        self.prepared_items_value = QLabel("0")
        self.done_items_value = QLabel("0")
        self.error_items_value = QLabel("0")
        self.needs_ocr_items_value = QLabel("0")
        self.cache_path_value = QLabel("-")
        self.cache_path_value.setWordWrap(True)
        self.cache_path_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        details_form.addRow("Items:", self.total_items_value)
        details_form.addRow("Prepared:", self.prepared_items_value)
        details_form.addRow("Done:", self.done_items_value)
        details_form.addRow("Error:", self.error_items_value)
        details_form.addRow("Needs OCR:", self.needs_ocr_items_value)
        details_form.addRow("OCR JSON:", self.cache_path_value)
        details_card.content_layout.addLayout(details_form)
        self.content_layout.addWidget(details_card)

        items_card = StaticSection("OCR Items", expanded=True)
        self.items_section = items_card
        self.items_table = QTableWidget(0, 7)
        self.items_table.setProperty("stageTable", True)
        self.items_table.setHorizontalHeaderLabels(["id", "kind", "bbox", "ocr_bbox", "status", "provider", "text"])
        self.items_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.items_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.items_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.items_table.setAlternatingRowColors(True)
        self.items_table.currentCellChanged.connect(self._on_current_cell_changed)
        header = self.items_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        self.items_table.setMinimumHeight(240)
        items_card.content_layout.addWidget(self.items_table)
        self.content_layout.addWidget(items_card)

        self.box_editor_section = CollapsibleSection("OCR Box Editor", expanded=False)
        box_editor_layout = QGridLayout()
        box_editor_layout.setContentsMargins(0, 0, 0, 0)
        box_editor_layout.setHorizontalSpacing(8)
        box_editor_layout.setVerticalSpacing(8)

        self.enable_box_edit_checkbox = QCheckBox("Enable OCR Box Editing")
        self.enable_box_edit_checkbox.toggled.connect(self.box_edit_mode_toggled.emit)
        box_editor_layout.addWidget(self.enable_box_edit_checkbox, 0, 0, 1, 2)

        self.show_excluded_items_checkbox = QCheckBox("Show Excluded OCR Items")
        self.show_excluded_items_checkbox.toggled.connect(self._on_show_excluded_toggled)
        box_editor_layout.addWidget(self.show_excluded_items_checkbox, 0, 2)

        box_editor_layout.addWidget(QLabel("Box Field:"), 1, 0)
        self.box_field_input = QComboBox()
        self.box_field_input.addItem("OCR Crop Box", "ocr_bbox")
        self.box_field_input.addItem("Item Box", "bbox")
        self.box_field_input.currentIndexChanged.connect(self._emit_box_field_changed)
        box_editor_layout.addWidget(self.box_field_input, 1, 1, 1, 2)

        self.save_box_edits_button = QPushButton("Save Box Edits")
        style_button(self.save_box_edits_button, "primary")
        self.save_box_edits_button.clicked.connect(self.save_box_edits_requested.emit)
        self.save_box_edits_button.setToolTip("Write edited OCR boxes back to the cached OCR JSON.")
        box_editor_layout.addWidget(self.save_box_edits_button, 2, 0)

        self.cancel_box_edits_button = QPushButton("Cancel Unsaved Edits")
        style_button(self.cancel_box_edits_button, "secondary")
        self.cancel_box_edits_button.clicked.connect(self.cancel_box_edits_requested.emit)
        self.cancel_box_edits_button.setToolTip("Discard in-memory OCR box edits and reload the cached OCR JSON.")
        box_editor_layout.addWidget(self.cancel_box_edits_button, 2, 1)

        self.reload_box_cache_button = QPushButton("Reload Boxes")
        style_button(self.reload_box_cache_button, "secondary")
        self.reload_box_cache_button.clicked.connect(self.reload_box_cache_requested.emit)
        self.reload_box_cache_button.setToolTip("Reload editable OCR boxes from the cached OCR JSON.")
        box_editor_layout.addWidget(self.reload_box_cache_button, 2, 2)

        self.exclude_selected_box_button = QPushButton("Delete / Exclude")
        style_button(self.exclude_selected_box_button, "danger")
        self.exclude_selected_box_button.clicked.connect(self.exclude_selected_box_requested.emit)
        self.exclude_selected_box_button.setToolTip("Soft-delete the selected OCR item by marking it excluded.")
        box_editor_layout.addWidget(self.exclude_selected_box_button, 3, 0)

        self.restore_selected_box_button = QPushButton("Restore Selected")
        style_button(self.restore_selected_box_button, "secondary")
        self.restore_selected_box_button.clicked.connect(self.restore_selected_box_requested.emit)
        self.restore_selected_box_button.setToolTip("Restore an excluded OCR item when Show Excluded OCR Items is enabled.")
        box_editor_layout.addWidget(self.restore_selected_box_button, 3, 1)
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

        selected_box_form = QFormLayout()
        selected_box_form.setContentsMargins(0, 0, 0, 0)
        selected_box_form.setSpacing(8)
        self.selected_box_id_value = QLabel("-")
        self.selected_box_kind_value = QLabel("-")
        self.selected_box_bbox_value = QLabel("-")
        self.selected_box_ocr_bbox_value = QLabel("-")
        self.selected_box_status_value = QLabel("-")
        self.selected_box_engine_value = QLabel("-")
        self.selected_box_detector_value = QLabel("-")
        self.selected_box_crop_value = QLabel("-")
        self.selected_box_excluded_value = QLabel("-")
        self.selected_box_needs_ocr_value = QLabel("-")
        self.selected_box_error_value = QLabel("-")
        self.selected_box_error_value.setWordWrap(True)
        selected_box_form.addRow("Selected ID:", self.selected_box_id_value)
        selected_box_form.addRow("Kind:", self.selected_box_kind_value)
        selected_box_form.addRow("BBox:", self.selected_box_bbox_value)
        selected_box_form.addRow("OCR BBox:", self.selected_box_ocr_bbox_value)
        selected_box_form.addRow("Status:", self.selected_box_status_value)
        selected_box_form.addRow("OCR Provider:", self.selected_box_engine_value)
        selected_box_form.addRow("Detector Sources:", self.selected_box_detector_value)
        selected_box_form.addRow("Crop Path:", self.selected_box_crop_value)
        selected_box_form.addRow("Excluded:", self.selected_box_excluded_value)
        selected_box_form.addRow("Needs OCR:", self.selected_box_needs_ocr_value)
        selected_box_form.addRow("Error:", self.selected_box_error_value)
        self.box_editor_section.content_layout.addLayout(selected_box_form)
        self.content_layout.addWidget(self.box_editor_section)

        self.editor_section = StaticSection("OCR Item Editor", expanded=True)
        editor_info_layout = QVBoxLayout()
        editor_info_layout.setContentsMargins(0, 0, 0, 0)
        editor_info_layout.setSpacing(6)
        self.editor_details_label = QLabel("Select an OCR item to edit its text.")
        self.editor_details_label.setWordWrap(True)
        self.editor_details_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.editor_dirty_label = QLabel("Saved")
        self.editor_dirty_label.setProperty("role", "muted")
        editor_info_layout.addWidget(self.editor_details_label)
        editor_info_layout.addWidget(self.editor_dirty_label)
        self.editor_section.content_layout.addLayout(editor_info_layout)

        editor_body = QWidget()
        editor_body_layout = QGridLayout(editor_body)
        editor_body_layout.setContentsMargins(0, 0, 0, 0)
        editor_body_layout.setHorizontalSpacing(12)
        editor_body_layout.setVerticalSpacing(8)

        self.crop_preview_panel = CropPreviewPanel()
        editor_body_layout.addWidget(self.crop_preview_panel, 0, 0)

        self.text_editor = TextItemEditorWidget(
            "OCR Text",
            placeholder="Select an OCR item to edit its recognized text.",
        )
        self.text_editor.dirty_changed.connect(self._update_editor_button_state)
        editor_body_layout.addWidget(self.text_editor, 0, 1)
        editor_body_layout.setColumnStretch(0, 1)
        editor_body_layout.setColumnStretch(1, 2)
        self.editor_section.content_layout.addWidget(editor_body)

        editor_actions_layout = QGridLayout()
        editor_actions_layout.setContentsMargins(0, 0, 0, 0)
        editor_actions_layout.setHorizontalSpacing(8)
        editor_actions_layout.setVerticalSpacing(8)

        self.editor_previous_button = QPushButton("Previous Item")
        style_button(self.editor_previous_button, "secondary")
        self.editor_previous_button.clicked.connect(self.select_previous_item)
        editor_actions_layout.addWidget(self.editor_previous_button, 0, 0)

        self.editor_next_button = QPushButton("Next Item")
        style_button(self.editor_next_button, "secondary")
        self.editor_next_button.clicked.connect(self.select_next_item)
        editor_actions_layout.addWidget(self.editor_next_button, 0, 1)

        self.editor_revert_button = QPushButton("Revert")
        style_button(self.editor_revert_button, "secondary")
        self.editor_revert_button.clicked.connect(self.revert_current_item)
        editor_actions_layout.addWidget(self.editor_revert_button, 0, 2)

        self.editor_save_button = QPushButton("Save OCR Text")
        style_button(self.editor_save_button, "primary")
        self.editor_save_button.clicked.connect(self.save_current_item)
        editor_actions_layout.addWidget(self.editor_save_button, 0, 3)
        self.editor_section.content_layout.addLayout(editor_actions_layout)
        self.content_layout.addWidget(self.editor_section)

        self._set_editor_enabled(False)

    def config_sections(self) -> list[QWidget]:
        return [self.provider_section, self.chrome_lens_section, self.server_section]

    def simplify_for_config_stage(self) -> None:
        self.detach_widget(self.summary_section)
        self.detach_widget(self.box_editor_section)
        self._update_provider_sections()
        self._update_box_editor_state()

    def set_server_values(self, manager: Any) -> None:
        self.server_url_input.setText(str(getattr(manager, "server_url", "") or ""))
        self.server_model_path_input.setText(str(getattr(manager, "model_path", "") or ""))
        self.server_mmproj_path_input.setText(str(getattr(manager, "mmproj_path", "") or ""))
        self.server_llama_cpp_dir_input.setText(str(getattr(manager, "llama_cpp_dir", "") or ""))
        self.server_gpu_layers_input.setValue(int(getattr(manager, "gpu_layers", 99) or 99))
        self.server_ctx_size_input.setValue(int(getattr(manager, "ctx_size", 8192) or 8192))

    def selected_ocr_provider(self) -> str:
        return normalize_ocr_provider_name(
            str(self.ocr_provider_input.currentData() or OCR_PROVIDER_PADDLE_VL_LLAMA)
        )

    def set_selected_ocr_provider(self, provider_name: str) -> None:
        normalized = normalize_ocr_provider_name(provider_name)
        for index in range(self.ocr_provider_input.count()):
            if str(self.ocr_provider_input.itemData(index) or "") == normalized:
                self.ocr_provider_input.blockSignals(True)
                self.ocr_provider_input.setCurrentIndex(index)
                self.ocr_provider_input.blockSignals(False)
                break
        self._update_provider_sections()

    def chrome_lens_values(self) -> dict[str, Any]:
        return {
            "timeout": self.chrome_lens_timeout_input.value(),
            "chrome_lens_headless": self.chrome_lens_headless_checkbox.isChecked(),
            "chrome_lens_chrome_path": self.chrome_lens_path_input.text().strip(),
            "chrome_lens_user_data_dir": self.chrome_lens_user_data_dir_input.text().strip(),
            "chrome_lens_language": self.chrome_lens_language_input.text().strip() or "ja",
            "chrome_lens_max_retries": self.chrome_lens_max_retries_input.value(),
        }

    def ocr_config(self) -> OCRConfig:
        payload = self.server_values()
        payload.update(self.chrome_lens_values())
        payload["ocr_provider"] = self.selected_ocr_provider()
        return OCRConfig.from_value(payload)

    def server_values(self) -> dict[str, Any]:
        return {
            "server_url": self.server_url_input.text().strip(),
            "model_path": self.server_model_path_input.text().strip(),
            "mmproj_path": self.server_mmproj_path_input.text().strip(),
            "llama_cpp_dir": self.server_llama_cpp_dir_input.text().strip(),
            "gpu_layers": self.server_gpu_layers_input.value(),
            "ctx_size": self.server_ctx_size_input.value(),
        }

    def set_server_status(self, status: str) -> None:
        normalized = str(status or "Unknown")
        self.server_status_value.setText(normalized)
        self.server_section.set_badge_text(normalized)
        if normalized == "Ready":
            self.server_settings_section.set_expanded(False)
            self.server_section.set_expanded(False)
        else:
            self.server_section.set_expanded(True)

    def _on_provider_changed(self) -> None:
        self._update_provider_sections()
        self.ocr_provider_changed.emit(self.selected_ocr_provider())

    def _update_provider_sections(self) -> None:
        provider = self.selected_ocr_provider()
        uses_llama_server = provider in {
            OCR_PROVIDER_PADDLE_VL_LLAMA,
            OCR_PROVIDER_DEEPSEEK_OCR_LLAMA,
        }
        self.server_section.setVisible(uses_llama_server)
        self.chrome_lens_section.setVisible(provider == OCR_PROVIDER_CHROME_LENS)

    def set_project_root(self, project_root: Path | None) -> None:
        self._project_root = project_root.resolve() if isinstance(project_root, Path) else None
        self.crop_preview_panel.set_project_root(self._project_root)
        self._refresh_editor_view()

    def set_items(self, items: list[dict[str, Any]], cache_path: Path | None) -> None:
        previous_item_id = self.current_editor_item_id()
        self._all_items = [dict(item) for item in items]
        self._cache_path = cache_path.resolve() if isinstance(cache_path, Path) else None
        self._rebuild_items_table(previous_item_id=previous_item_id)

    def clear_view(self) -> None:
        self._all_items = []
        self._items = []
        self._cache_path = None
        self._editor_row = None
        self.items_table.blockSignals(True)
        self.items_table.setRowCount(0)
        self.items_table.blockSignals(False)
        self.total_items_value.setText("0")
        self.prepared_items_value.setText("0")
        self.done_items_value.setText("0")
        self.error_items_value.setText("0")
        self.needs_ocr_items_value.setText("0")
        self.cache_path_value.setText("-")
        self.items_section.set_badge_text("0 items")
        self.box_editor_section.set_expanded(False)
        self.editor_section.set_expanded(False)
        self.set_selected_box(None)
        self.set_box_dirty(False)
        self.set_box_warning(None)
        self._set_editor_enabled(False)
        self.crop_preview_panel.clear("Select an OCR item to preview its crop.")

    def selected_item_ids(self) -> list[int]:
        selection_model = self.items_table.selectionModel()
        if selection_model is None:
            return []
        ids: list[int] = []
        for model_index in selection_model.selectedRows():
            row_index = model_index.row()
            if row_index < 0 or row_index >= len(self._items):
                continue
            ids.append(int(self._items[row_index].get("id", row_index)))
        return ids

    def current_editor_item_id(self) -> int | None:
        if self._editor_row is None or self._editor_row < 0 or self._editor_row >= len(self._items):
            return None
        return int(self._items[self._editor_row].get("id", self._editor_row))

    def current_table_item_id(self) -> int | None:
        row_index = self.items_table.currentRow()
        if row_index < 0 or row_index >= len(self._items):
            return None
        return int(self._items[row_index].get("id", row_index))

    def select_item_by_id(self, item_id: int | None) -> bool:
        row_index = self._row_for_item_id(item_id)
        if row_index is None:
            return False
        self._set_current_row(row_index)
        return self.current_table_item_id() == item_id

    def has_unsaved_changes(self) -> bool:
        return self.text_editor.is_dirty()

    def ensure_pending_changes_resolved(self, parent: QWidget | None = None) -> bool:
        if not self.has_unsaved_changes():
            return True

        message_box = QMessageBox(parent or self)
        message_box.setIcon(QMessageBox.Icon.Question)
        message_box.setWindowTitle("Unsaved OCR edits")
        message_box.setText("Save OCR text changes before leaving this item?")
        save_button = message_box.addButton("Save changes", QMessageBox.ButtonRole.AcceptRole)
        discard_button = message_box.addButton("Discard", QMessageBox.ButtonRole.DestructiveRole)
        cancel_button = message_box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        message_box.setDefaultButton(save_button)
        message_box.exec()
        clicked = message_box.clickedButton()
        if clicked is save_button:
            return self.save_current_item()
        if clicked is discard_button:
            self.revert_current_item()
            return True
        if clicked is cancel_button:
            self.warning_emitted.emit("OCR row selection change canceled because edits are still unsaved.")
        return False

    def save_current_item(self) -> bool:
        if self._cache_path is None:
            self.error_emitted.emit("OCR JSON missing", "Reload OCR items before saving text.")
            return False
        if self._editor_row is None or self._editor_row < 0 or self._editor_row >= len(self._items):
            self.error_emitted.emit("No OCR item selected", "Select an OCR item before saving text.")
            return False

        item = self._items[self._editor_row]
        item_id = int(item.get("id", self._editor_row))
        try:
            payload = update_ocr_item_text(self._cache_path, item_id, self.text_editor.text())
        except Exception as exc:
            self.error_emitted.emit("Failed to save OCR text", str(exc))
            return False

        self.set_items(payload.get("items", []), self._cache_path)
        target_row = self._row_for_item_id(item_id)
        if target_row is not None:
            self._load_editor_for_row(target_row)
        self.cache_updated.emit(
            {
                "stage": "ocr",
                "cache_path": str(self._cache_path),
                "ocr_data": payload,
                "message": f"Saved OCR text for item {item_id}.",
            }
        )
        self.message_emitted.emit(f"Saved OCR text for item {item_id}.")
        return True

    def revert_current_item(self) -> None:
        self._refresh_editor_view()
        self.message_emitted.emit("Reverted OCR editor changes.")

    def select_previous_item(self) -> None:
        if self._editor_row is None or self._editor_row <= 0:
            return
        self._set_current_row(self._editor_row - 1)

    def select_next_item(self) -> None:
        if self._editor_row is None or self._editor_row >= len(self._items) - 1:
            return
        self._set_current_row(self._editor_row + 1)

    def set_actions_enabled(self, enabled: bool) -> None:
        self._actions_enabled = bool(enabled)
        for widget in (
            self.ocr_provider_input,
            self.chrome_lens_timeout_input,
            self.chrome_lens_headless_checkbox,
            self.chrome_lens_path_input,
            self.chrome_lens_user_data_dir_input,
            self.chrome_lens_language_input,
            self.chrome_lens_max_retries_input,
            self.prepare_selected_button,
            self.reprepare_selected_button,
            self.prepare_all_button,
            self.reprepare_all_button,
            self.run_selected_button,
            self.rerun_selected_button,
            self.run_all_button,
            self.rerun_all_button,
            self.run_selected_items_button,
            self.rerun_selected_items_button,
            self.save_text_button,
            self.reload_button,
            self.items_table,
            self.editor_previous_button,
            self.editor_next_button,
            self.editor_revert_button,
            self.editor_save_button,
            self.text_editor.editor,
            self.enable_box_edit_checkbox,
            self.show_excluded_items_checkbox,
            self.box_field_input,
            self.save_box_edits_button,
            self.cancel_box_edits_button,
            self.reload_box_cache_button,
            self.exclude_selected_box_button,
            self.restore_selected_box_button,
        ):
            widget.setEnabled(enabled)
        self._update_editor_button_state(self.text_editor.is_dirty())
        self._update_box_editor_state()

    def set_server_actions_enabled(self, enabled: bool) -> None:
        for widget in (self.check_server_button, self.start_server_button, self.stop_server_button):
            widget.setEnabled(enabled)

    def set_box_edit_mode_checked(self, enabled: bool) -> None:
        self.enable_box_edit_checkbox.blockSignals(True)
        self.enable_box_edit_checkbox.setChecked(bool(enabled))
        self.enable_box_edit_checkbox.blockSignals(False)
        self._update_box_editor_state()

    def box_edit_mode_enabled(self) -> bool:
        return self.enable_box_edit_checkbox.isChecked()

    def selected_box_field(self) -> str:
        return str(self.box_field_input.currentData() or "ocr_bbox")

    def set_selected_box_field(self, value: str) -> None:
        normalized = str(value or "ocr_bbox").strip().lower()
        for index in range(self.box_field_input.count()):
            if str(self.box_field_input.itemData(index) or "") == normalized:
                self.box_field_input.blockSignals(True)
                self.box_field_input.setCurrentIndex(index)
                self.box_field_input.blockSignals(False)
                break

    def set_show_excluded_items_checked(self, enabled: bool) -> None:
        self.show_excluded_items_checkbox.blockSignals(True)
        self.show_excluded_items_checkbox.setChecked(bool(enabled))
        self.show_excluded_items_checkbox.blockSignals(False)
        self._rebuild_items_table(previous_item_id=self.current_editor_item_id())
        self._update_box_editor_state()

    def show_excluded_items_enabled(self) -> bool:
        return self.show_excluded_items_checkbox.isChecked()

    def set_box_dirty(self, dirty: bool) -> None:
        self._box_edit_dirty = bool(dirty)
        self.box_dirty_label.setVisible(self._box_edit_dirty)
        self.box_dirty_label.setText("Unsaved OCR box edits")
        self._update_box_editor_state()

    def has_unsaved_box_edits(self) -> bool:
        return self._box_edit_dirty

    def set_selected_box(self, box_data: dict[str, Any] | None) -> None:
        self._selected_box = dict(box_data) if isinstance(box_data, dict) else None
        if self._selected_box is None:
            self.selected_box_id_value.setText("-")
            self.selected_box_kind_value.setText("-")
            self.selected_box_bbox_value.setText("-")
            self.selected_box_ocr_bbox_value.setText("-")
            self.selected_box_status_value.setText("-")
            self.selected_box_engine_value.setText("-")
            self.selected_box_detector_value.setText("-")
            self.selected_box_crop_value.setText("-")
            self.selected_box_excluded_value.setText("-")
            self.selected_box_needs_ocr_value.setText("-")
            self.selected_box_error_value.setText("-")
            self._update_box_editor_state()
            return

        box = self._selected_box
        self.selected_box_id_value.setText(str(box.get("id", "-")))
        self.selected_box_kind_value.setText(str(box.get("kind", "-")))
        self.selected_box_bbox_value.setText(self._format_bbox(box.get("bbox")))
        self.selected_box_ocr_bbox_value.setText(self._format_bbox(box.get("ocr_bbox")))
        self.selected_box_status_value.setText(str(box.get("status", "-") or "-"))
        self.selected_box_engine_value.setText(str(box.get("ocr_provider", "") or box.get("ocr_engine", "-") or "-"))
        detector_sources = box.get("detector_sources")
        if isinstance(detector_sources, list):
            detector_text = ", ".join(str(value) for value in detector_sources if str(value).strip()) or "-"
        else:
            detector_text = str(detector_sources or "-")
        self.selected_box_detector_value.setText(detector_text)
        self.selected_box_crop_value.setText(str(box.get("crop_path", "-") or "-"))
        self.selected_box_excluded_value.setText("Yes" if bool(box.get("excluded", False)) else "No")
        self.selected_box_needs_ocr_value.setText("Yes" if bool(box.get("needs_ocr", False)) else "No")
        self.selected_box_error_value.setText(str(box.get("error", "-") or "-"))
        self._update_box_editor_state()

    def selected_box(self) -> dict[str, Any] | None:
        return dict(self._selected_box) if self._selected_box is not None else None

    def set_box_warning(self, text: str | None) -> None:
        normalized = str(text or "").strip()
        self.box_warning_label.setText(normalized)
        self.box_warning_label.setVisible(bool(normalized))

    def settings_snapshot(self) -> dict[str, Any]:
        values = self.server_values()
        values.update(self.chrome_lens_values())
        values["ocr_provider"] = self.selected_ocr_provider()
        values["server_status"] = self.server_status_value.text().strip()
        return values

    def apply_settings(self, settings: dict[str, Any]) -> None:
        if not isinstance(settings, dict):
            return
        self.set_selected_ocr_provider(str(settings.get("ocr_provider", OCR_PROVIDER_PADDLE_VL_LLAMA) or OCR_PROVIDER_PADDLE_VL_LLAMA))
        self.server_url_input.setText(str(settings.get("server_url", "") or self.server_url_input.text()))
        self.server_model_path_input.setText(
            str(settings.get("model_path", "") or self.server_model_path_input.text())
        )
        self.server_mmproj_path_input.setText(
            str(settings.get("mmproj_path", "") or self.server_mmproj_path_input.text())
        )
        self.server_llama_cpp_dir_input.setText(
            str(settings.get("llama_cpp_dir", "") or self.server_llama_cpp_dir_input.text())
        )
        try:
            self.server_gpu_layers_input.setValue(int(settings.get("gpu_layers", self.server_gpu_layers_input.value())))
        except Exception:
            pass
        try:
            self.server_ctx_size_input.setValue(int(settings.get("ctx_size", self.server_ctx_size_input.value())))
        except Exception:
            pass
        try:
            self.chrome_lens_timeout_input.setValue(
                int(settings.get("timeout", self.chrome_lens_timeout_input.value()))
            )
        except Exception:
            pass
        self.chrome_lens_headless_checkbox.setChecked(bool(settings.get("chrome_lens_headless", False)))
        self.chrome_lens_path_input.setText(
            str(settings.get("chrome_lens_chrome_path", "") or self.chrome_lens_path_input.text())
        )
        self.chrome_lens_user_data_dir_input.setText(
            str(settings.get("chrome_lens_user_data_dir", "") or self.chrome_lens_user_data_dir_input.text())
        )
        self.chrome_lens_language_input.setText(
            str(settings.get("chrome_lens_language", "") or self.chrome_lens_language_input.text() or "ja")
        )
        try:
            self.chrome_lens_max_retries_input.setValue(
                int(settings.get("chrome_lens_max_retries", self.chrome_lens_max_retries_input.value()))
            )
        except Exception:
            pass
        server_status = str(settings.get("server_status", "") or "").strip()
        if server_status:
            self.set_server_status(server_status)
        self._update_provider_sections()

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
            provider_text = str(item.get("ocr_provider", "") or item.get("ocr_engine", "") or "-")
            row_values = [
                str(item.get("id", "")),
                str(item.get("kind", "")),
                self._format_bbox(item.get("bbox")),
                self._format_bbox(item.get("ocr_bbox")),
                status_text,
                provider_text,
                self._display_text(item.get("text")),
            ]
            full_text = str(item.get("text", "") or "")
            error_text = str(item.get("error", "") or "").strip()
            needs_ocr = bool(item.get("needs_ocr", False))
            for column_index, value in enumerate(row_values):
                table_item = QTableWidgetItem(value)
                if column_index == 0:
                    table_item.setData(Qt.ItemDataRole.UserRole, int(item.get("id", row_index)))
                tooltip_lines = []
                if column_index == 5 and full_text:
                    tooltip_lines.append(full_text)
                if needs_ocr:
                    tooltip_lines.append("OCR crop box changed. Re-run OCR is recommended.")
                if bool(item.get("excluded", False)):
                    tooltip_lines.append("This OCR item is excluded.")
                if error_text:
                    tooltip_lines.append(f"Error: {error_text}")
                if tooltip_lines:
                    table_item.setToolTip("\n\n".join(tooltip_lines))
                table_item.setFlags(table_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.items_table.setItem(row_index, column_index, table_item)
        self.items_table.blockSignals(False)

        summary = summarize_ocr_items(self._all_items)
        edit_summary = summarize_ocr_edit_state({"items": self._all_items})
        self.total_items_value.setText(
            f"{summary.get('total', 0)} active / {summary.get('excluded', 0)} excluded"
        )
        self.items_section.set_badge_text(
            f"{summary.get('total', 0)} active / {summary.get('excluded', 0)} excluded"
        )
        self.prepared_items_value.setText(str(summary.get("prepared", 0)))
        self.done_items_value.setText(str(summary.get("done", 0)))
        self.error_items_value.setText(str(summary.get("error", 0)))
        self.needs_ocr_items_value.setText(str(edit_summary.get("needs_ocr_items", 0)))
        self.cache_path_value.setText(str(self._cache_path) if self._cache_path is not None else "-")
        self.box_editor_section.set_expanded(bool(self._all_items))
        self.editor_section.set_expanded(bool(self._items))

        if self._items:
            target_row = self._row_for_item_id(previous_item_id)
            if target_row is None:
                target_row = 0
            self._selection_guard = True
            try:
                self.items_table.setCurrentCell(target_row, 0)
            finally:
                self._selection_guard = False
            self._load_editor_for_row(target_row)
        else:
            self._editor_row = None
            self._set_editor_enabled(False)
            self.crop_preview_panel.clear("Select an OCR item to preview its crop.")

    def _on_current_cell_changed(self, current_row: int, _current_column: int, previous_row: int, _previous_column: int) -> None:
        if self._selection_guard:
            return
        if current_row < 0:
            return
        if self._editor_row == current_row:
            item_id = self.current_editor_item_id()
            if item_id is not None:
                self.current_item_changed.emit(item_id)
            return
        if not self.ensure_pending_changes_resolved(self):
            restore_row = previous_row if previous_row >= 0 else self._editor_row
            if restore_row is not None and restore_row >= 0:
                self._selection_guard = True
                try:
                    self.items_table.setCurrentCell(int(restore_row), 0)
                finally:
                    self._selection_guard = False
            return
        self._load_editor_for_row(current_row)
        item_id = self.current_editor_item_id()
        if item_id is not None:
            self.current_item_changed.emit(item_id)

    def _load_editor_for_row(self, row_index: int | None) -> None:
        self._editor_row = row_index
        self._refresh_editor_view()

    def _refresh_editor_view(self) -> None:
        if self._editor_row is None or self._editor_row < 0 or self._editor_row >= len(self._items):
            self._set_editor_enabled(False)
            self.crop_preview_panel.clear("Select an OCR item to preview its crop.")
            return

        item = self._items[self._editor_row]
        item_id = int(item.get("id", self._editor_row))
        kind = str(item.get("kind", "") or "-")
        status = str(item.get("status", "") or "-")
        provider_text = str(item.get("ocr_provider", "") or item.get("ocr_engine", "") or "-")
        bbox = self._format_bbox(item.get("bbox"))
        ocr_bbox = self._format_bbox(item.get("ocr_bbox"))
        error_text = str(item.get("error", "") or "").strip()

        details = [
            f"Item {item_id}",
            f"Kind: {kind}",
            f"Status: {status}",
            f"OCR Provider: {provider_text}",
            f"BBox: {bbox}",
            f"OCR BBox: {ocr_bbox}",
        ]
        if error_text:
            details.append(f"Error: {error_text}")
        detail_text = "\n".join(details)

        self._set_editor_enabled(True)
        self.editor_details_label.setText(detail_text)
        self.text_editor.set_loaded_text(
            str(item.get("text", "") or ""),
            status_text="Edit the OCR result below.",
        )
        self.crop_preview_panel.set_crop(
            item.get("crop_path"),
            details=detail_text,
        )
        self._update_editor_button_state(False)

    def _set_editor_enabled(self, enabled: bool) -> None:
        active = bool(enabled and self._actions_enabled)
        if enabled:
            self.text_editor.set_enabled_for_item(active, message=self.text_editor.status_label.text())
        else:
            self.editor_details_label.setText("Select an OCR item to edit its text.")
            self.editor_dirty_label.setText("Saved")
            self.text_editor.set_enabled_for_item(False, message="No item selected.")
            self.text_editor.set_loaded_text("", status_text="No item selected.")
        self.editor_previous_button.setEnabled(bool(active and self._editor_row not in (None, 0)))
        self.editor_next_button.setEnabled(
            bool(active and self._editor_row is not None and self._editor_row < len(self._items) - 1)
        )
        self.editor_revert_button.setEnabled(False)
        self.editor_save_button.setEnabled(False)

    def _update_editor_button_state(self, dirty: bool) -> None:
        has_item = self._actions_enabled and self._editor_row is not None and 0 <= self._editor_row < len(self._items)
        self.editor_dirty_label.setText("Unsaved changes" if dirty else "Saved")
        self.editor_save_button.setEnabled(bool(has_item and dirty))
        self.editor_revert_button.setEnabled(bool(has_item and dirty))
        self.editor_previous_button.setEnabled(bool(has_item and self._editor_row not in (None, 0)))
        self.editor_next_button.setEnabled(
            bool(has_item and self._editor_row is not None and self._editor_row < len(self._items) - 1)
        )

    def _set_current_row(self, row_index: int) -> None:
        if row_index < 0 or row_index >= len(self._items):
            return
        self.items_table.setCurrentCell(row_index, 0)

    def _row_for_item_id(self, item_id: int | None) -> int | None:
        if item_id is None:
            return None
        for index, item in enumerate(self._items):
            if int(item.get("id", index)) == int(item_id):
                return index
        return None

    def _emit_box_field_changed(self) -> None:
        self.box_field_changed.emit(self.selected_box_field())

    def _on_show_excluded_toggled(self, enabled: bool) -> None:
        if not self.ensure_pending_changes_resolved(self):
            self.show_excluded_items_checkbox.blockSignals(True)
            self.show_excluded_items_checkbox.setChecked(not bool(enabled))
            self.show_excluded_items_checkbox.blockSignals(False)
            return
        previous_item_id = self.current_editor_item_id()
        self._rebuild_items_table(previous_item_id=previous_item_id)
        self.show_excluded_items_toggled.emit(bool(enabled))
        self._update_box_editor_state()

    def _update_box_editor_state(self) -> None:
        edit_enabled = self.box_edit_mode_enabled() and self._actions_enabled and bool(self._all_items)
        has_selection = self._selected_box is not None
        selected_excluded = bool(self._selected_box and self._selected_box.get("excluded", False))

        self.box_field_input.setEnabled(edit_enabled)
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


__all__ = ["OCRPanel"]
