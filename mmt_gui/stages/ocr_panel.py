"""OCR stage inspector panel."""

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
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mmt_core import (
    DEFAULT_LOCAL_OCR_PROVIDER,
    OCRConfig,
    OCR_PROVIDER_CHROME_LENS,
    OCR_PROVIDER_DEEPSEEK_OCR_LLAMA,
    OCR_PROVIDER_MODE_CHROME_LENS,
    OCR_PROVIDER_MODE_CHOICES,
    OCR_PROVIDER_MODE_LOCAL,
    OCR_PROVIDER_PADDLE_VL_LLAMA,
    LOCAL_OCR_PROVIDER_CHOICES,
    normalize_local_ocr_provider,
    normalize_ocr_provider_name,
    normalize_ocr_provider_mode,
    resolve_ocr_provider,
    selection_from_provider,
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
    create_run_server_bat_requested = pyqtSignal()
    check_run_server_bat_requested = pyqtSignal()
    open_server_folder_requested = pyqtSignal()
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

    def __init__(self, workspace_root: Path | None = None, parent: object | None = None) -> None:
        super().__init__("OCR", parent)
        self._actions_enabled = True
        self._workspace_root = workspace_root.resolve() if isinstance(workspace_root, Path) else None
        self._project_root: Path | None = None
        self._all_items: list[dict[str, Any]] = []
        self._items: list[dict[str, Any]] = []
        self._cache_path: Path | None = None
        self._editor_row: int | None = None
        self._selection_guard = False
        self._box_edit_dirty = False
        self._selected_box: dict[str, Any] | None = None
        self._loaded_local_provider_key = DEFAULT_LOCAL_OCR_PROVIDER
        self._syncing_server_address = False
        self._local_provider_settings: dict[str, dict[str, Any]] = self._default_local_provider_settings()

        self.provider_section = CollapsibleSection("OCR Provider", expanded=True)
        provider_form = QFormLayout()
        provider_form.setContentsMargins(0, 0, 0, 0)
        provider_form.setSpacing(8)
        self.ocr_provider_mode_label = QLabel("OCR Provider:")
        self.ocr_provider_mode_input = QComboBox()
        for provider_key, provider_text in OCR_PROVIDER_MODE_CHOICES:
            self.ocr_provider_mode_input.addItem(provider_text, provider_key)
        self.ocr_provider_mode_input.currentIndexChanged.connect(self._on_provider_changed)
        provider_form.addRow(self.ocr_provider_mode_label, self.ocr_provider_mode_input)

        self.local_ocr_provider_label = QLabel("OCR Local:")
        self.local_ocr_provider_input = QComboBox()
        for provider_key, provider_text in LOCAL_OCR_PROVIDER_CHOICES:
            self.local_ocr_provider_input.addItem(provider_text, provider_key)
        self.local_ocr_provider_input.currentIndexChanged.connect(self._on_local_provider_changed)
        provider_form.addRow(self.local_ocr_provider_label, self.local_ocr_provider_input)
        self.provider_section.content_layout.addLayout(provider_form)
        self.content_layout.addWidget(self.provider_section)

        self.server_section = CollapsibleSection("OCR Runtime", expanded=True)
        runtime_actions = QGridLayout()
        runtime_actions.setContentsMargins(0, 0, 0, 0)
        runtime_actions.setHorizontalSpacing(8)
        runtime_actions.setVerticalSpacing(8)

        self.check_server_button = QPushButton("Check health server")
        style_button(self.check_server_button, "secondary")
        self.check_server_button.clicked.connect(self.check_server_requested.emit)
        runtime_actions.addWidget(self.check_server_button, 0, 0)

        self.start_server_button = QPushButton("Start local server")
        style_button(self.start_server_button, "primary")
        self.start_server_button.clicked.connect(self.start_server_requested.emit)
        runtime_actions.addWidget(self.start_server_button, 0, 1)
        self.server_section.content_layout.addLayout(runtime_actions)

        self.server_status_value = QLabel("Unknown")
        runtime_status_form = QFormLayout()
        runtime_status_form.setContentsMargins(0, 0, 0, 0)
        runtime_status_form.addRow("Status:", self.server_status_value)
        self.server_section.content_layout.addLayout(runtime_status_form)
        self.content_layout.addWidget(self.server_section)

        self.local_server_section = CollapsibleSection("Advanced Local Server Settings", expanded=False)
        local_server_actions = QGridLayout()
        local_server_actions.setContentsMargins(0, 0, 0, 0)
        local_server_actions.setHorizontalSpacing(8)
        local_server_actions.setVerticalSpacing(8)

        self.create_run_server_bat_button = QPushButton("Create run_server.bat")
        style_button(self.create_run_server_bat_button, "secondary")
        self.create_run_server_bat_button.clicked.connect(self.create_run_server_bat_requested.emit)
        local_server_actions.addWidget(self.create_run_server_bat_button, 0, 0)

        self.check_run_server_bat_button = QPushButton("Check run_server.bat")
        style_button(self.check_run_server_bat_button, "secondary")
        self.check_run_server_bat_button.clicked.connect(self.check_run_server_bat_requested.emit)
        local_server_actions.addWidget(self.check_run_server_bat_button, 0, 1)

        self.open_server_folder_button = QPushButton("Open server folder")
        style_button(self.open_server_folder_button, "secondary")
        self.open_server_folder_button.clicked.connect(self.open_server_folder_requested.emit)
        local_server_actions.addWidget(self.open_server_folder_button, 0, 2)
        self.local_server_section.content_layout.addLayout(local_server_actions)

        self.server_url_input = QLineEdit()
        self.server_host_input = QLineEdit("127.0.0.1")
        self.server_port_input = QSpinBox()
        self.server_port_input.setRange(1, 65535)
        self.server_port_input.setValue(8080)
        self.server_model_path_input = QLineEdit()
        self.server_mmproj_path_input = QLineEdit()
        self.server_llama_cpp_dir_input = QLineEdit()
        self.server_gpu_layers_input = QSpinBox()
        self.server_gpu_layers_input.setRange(-1, 999)
        self.server_ctx_size_input = QSpinBox()
        self.server_ctx_size_input.setRange(512, 131072)
        self.server_ctx_size_input.setSingleStep(512)
        self.server_ctx_size_input.setValue(8192)
        self.server_temperature_input = QDoubleSpinBox()
        self.server_temperature_input.setRange(0.0, 2.0)
        self.server_temperature_input.setSingleStep(0.05)
        self.server_temperature_input.setDecimals(2)
        self.server_temperature_input.setValue(0.0)
        self.server_extra_args_input = QLineEdit()

        self.server_url_input.editingFinished.connect(self._on_server_url_changed)
        self.server_host_input.editingFinished.connect(self._on_server_host_port_changed)
        self.server_port_input.valueChanged.connect(lambda _value: self._on_server_host_port_changed())

        local_server_form = QFormLayout()
        local_server_form.setContentsMargins(0, 0, 0, 0)
        local_server_form.setSpacing(8)
        local_server_form.addRow("Server URL:", self.server_url_input)
        local_server_form.addRow("Host:", self.server_host_input)
        local_server_form.addRow("Port:", self.server_port_input)
        local_server_form.addRow("Model Path:", self.server_model_path_input)
        local_server_form.addRow("mmproj Path:", self.server_mmproj_path_input)
        local_server_form.addRow("llama.cpp Folder / llama-server.exe:", self.server_llama_cpp_dir_input)
        local_server_form.addRow("GPU Layers:", self.server_gpu_layers_input)
        local_server_form.addRow("Context Size:", self.server_ctx_size_input)
        local_server_form.addRow("Temperature:", self.server_temperature_input)
        local_server_form.addRow("Extra Args:", self.server_extra_args_input)
        self.local_server_section.content_layout.addLayout(local_server_form)
        self.content_layout.addWidget(self.local_server_section)

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
        self._load_local_provider_settings(self.selected_local_ocr_provider())
        self._update_provider_sections()

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
        editor_body.setObjectName("EditorBodyWidget")
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
        return [self.provider_section, self.server_section, self.local_server_section, self.chrome_lens_section]

    def simplify_for_config_stage(self) -> None:
        self.detach_widget(self.summary_section)
        self.detach_widget(self.box_editor_section)
        self._update_provider_sections()
        self._update_box_editor_state()

    def set_server_values(self, manager: Any) -> None:
        provider_key = normalize_local_ocr_provider(
            str(getattr(manager, "provider_key", "") or self.selected_local_ocr_provider())
        )
        provider_settings = self._local_provider_settings.setdefault(
            provider_key,
            self._default_local_provider_settings().get(provider_key, {}),
        )
        provider_settings.update(
            {
                "server_url": str(getattr(manager, "server_url", "") or provider_settings.get("server_url", "")),
                "host": str(getattr(manager, "host", "") or provider_settings.get("host", "")),
                "port": int(getattr(manager, "port", provider_settings.get("port", 8080)) or 8080),
                "model_path": str(getattr(manager, "model_path", "") or provider_settings.get("model_path", "")),
                "mmproj_path": str(getattr(manager, "mmproj_path", "") or provider_settings.get("mmproj_path", "")),
                "llama_cpp_dir": str(
                    getattr(manager, "llama_cpp_dir", "") or provider_settings.get("llama_cpp_dir", "")
                ),
                "gpu_layers": int(getattr(manager, "gpu_layers", provider_settings.get("gpu_layers", 99)) or 99),
                "ctx_size": int(getattr(manager, "ctx_size", provider_settings.get("ctx_size", 8192)) or 8192),
                "temperature": float(
                    getattr(manager, "temperature", provider_settings.get("temperature", 0.0)) or 0.0
                ),
                "extra_args": str(getattr(manager, "extra_args", "") or provider_settings.get("extra_args", "")),
            }
        )
        self._load_local_provider_settings(self.selected_local_ocr_provider())

    def selected_ocr_provider_mode(self) -> str:
        return normalize_ocr_provider_mode(
            str(self.ocr_provider_mode_input.currentData() or OCR_PROVIDER_MODE_LOCAL)
        )

    def selected_local_ocr_provider(self) -> str:
        return normalize_local_ocr_provider(
            str(self.local_ocr_provider_input.currentData() or DEFAULT_LOCAL_OCR_PROVIDER)
        )

    def selected_ocr_provider(self) -> str:
        return resolve_ocr_provider(
            ocr_provider_mode=self.selected_ocr_provider_mode(),
            local_ocr_provider=self.selected_local_ocr_provider(),
        )

    def set_selected_ocr_provider(self, provider_name: str) -> None:
        mode, local_provider = selection_from_provider(provider_name)
        self.set_selected_provider_mode(mode)
        self.set_selected_local_ocr_provider(local_provider)

    def set_selected_provider_mode(self, provider_mode: str) -> None:
        normalized = normalize_ocr_provider_mode(provider_mode)
        for index in range(self.ocr_provider_mode_input.count()):
            if str(self.ocr_provider_mode_input.itemData(index) or "") == normalized:
                self.ocr_provider_mode_input.blockSignals(True)
                self.ocr_provider_mode_input.setCurrentIndex(index)
                self.ocr_provider_mode_input.blockSignals(False)
                break
        self._update_provider_sections()

    def set_selected_local_ocr_provider(self, provider_name: str) -> None:
        normalized = normalize_local_ocr_provider(provider_name)
        for index in range(self.local_ocr_provider_input.count()):
            if str(self.local_ocr_provider_input.itemData(index) or "") == normalized:
                self.local_ocr_provider_input.blockSignals(True)
                self.local_ocr_provider_input.setCurrentIndex(index)
                self.local_ocr_provider_input.blockSignals(False)
                break
        self._load_local_provider_settings(normalized)
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
        payload["ocr_provider_mode"] = self.selected_ocr_provider_mode()
        payload["local_ocr_provider"] = self.selected_local_ocr_provider()
        return OCRConfig.from_value(payload)

    def server_values(self) -> dict[str, Any]:
        host = self.server_host_input.text().strip() or "127.0.0.1"
        port = int(self.server_port_input.value())
        return {
            "server_url": f"http://{host}:{port}",
            "host": host,
            "port": port,
            "model_path": self.server_model_path_input.text().strip(),
            "mmproj_path": self.server_mmproj_path_input.text().strip(),
            "llama_cpp_dir": self.server_llama_cpp_dir_input.text().strip(),
            "gpu_layers": self.server_gpu_layers_input.value(),
            "ctx_size": self.server_ctx_size_input.value(),
            "temperature": float(self.server_temperature_input.value()),
            "extra_args": self.server_extra_args_input.text().strip(),
        }

    def set_server_status(self, status: str) -> None:
        normalized = str(status or "Unknown")
        self.server_status_value.setText(normalized)
        self.server_section.set_badge_text(normalized)
        if normalized == "Ready":
            self.server_section.set_expanded(False)
        else:
            self.server_section.set_expanded(True)

    def _on_provider_changed(self) -> None:
        self._update_provider_sections()
        self.ocr_provider_changed.emit(self.selected_ocr_provider())

    def _update_provider_sections(self) -> None:
        local_mode = self.selected_ocr_provider_mode() == OCR_PROVIDER_MODE_LOCAL
        self.local_ocr_provider_label.setVisible(local_mode)
        self.local_ocr_provider_input.setVisible(local_mode)
        self.local_server_section.setVisible(local_mode)
        self.chrome_lens_section.setVisible(not local_mode)
        self.check_server_button.setEnabled(self._actions_enabled)
        self.start_server_button.setVisible(local_mode)
        self.start_server_button.setEnabled(local_mode and self._actions_enabled)
        self.create_run_server_bat_button.setEnabled(local_mode and self._actions_enabled)
        self.check_run_server_bat_button.setEnabled(local_mode and self._actions_enabled)
        self.open_server_folder_button.setEnabled(local_mode and self._actions_enabled)

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
            self.ocr_provider_mode_input,
            self.local_ocr_provider_input,
            self.chrome_lens_timeout_input,
            self.chrome_lens_headless_checkbox,
            self.chrome_lens_path_input,
            self.chrome_lens_user_data_dir_input,
            self.chrome_lens_language_input,
            self.chrome_lens_max_retries_input,
            self.server_url_input,
            self.server_host_input,
            self.server_port_input,
            self.server_model_path_input,
            self.server_mmproj_path_input,
            self.server_llama_cpp_dir_input,
            self.server_gpu_layers_input,
            self.server_ctx_size_input,
            self.server_temperature_input,
            self.server_extra_args_input,
            self.check_server_button,
            self.start_server_button,
            self.create_run_server_bat_button,
            self.check_run_server_bat_button,
            self.open_server_folder_button,
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
        self._update_provider_sections()
        self._update_editor_button_state(self.text_editor.is_dirty())
        self._update_box_editor_state()

    def set_server_actions_enabled(self, enabled: bool) -> None:
        self.check_server_button.setEnabled(bool(enabled and self._actions_enabled))
        local_mode = self.selected_ocr_provider_mode() == OCR_PROVIDER_MODE_LOCAL
        for widget in (
            self.start_server_button,
            self.create_run_server_bat_button,
            self.check_run_server_bat_button,
            self.open_server_folder_button,
        ):
            widget.setEnabled(bool(enabled and self._actions_enabled and local_mode))

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
        self._save_current_local_provider_settings()
        values = self.server_values()
        values.update(self.chrome_lens_values())
        values["ocr_provider"] = self.selected_ocr_provider()
        values["ocr_provider_mode"] = self.selected_ocr_provider_mode()
        values["local_ocr_provider"] = self.selected_local_ocr_provider()
        values["local_provider_settings"] = self._local_provider_settings_snapshot()
        values["server_status"] = self.server_status_value.text().strip()
        return values

    def apply_settings(self, settings: dict[str, Any]) -> None:
        if not isinstance(settings, dict):
            return
        provider_name = str(settings.get("ocr_provider", OCR_PROVIDER_PADDLE_VL_LLAMA) or OCR_PROVIDER_PADDLE_VL_LLAMA)
        provider_mode, local_provider = selection_from_provider(provider_name)
        provider_mode = normalize_ocr_provider_mode(
            str(settings.get("ocr_provider_mode", "") or ""),
            fallback=provider_mode,
        )
        local_provider = normalize_local_ocr_provider(
            str(settings.get("local_ocr_provider", "") or ""),
            fallback=local_provider,
        )
        self._apply_local_provider_settings_payload(settings, target_provider=local_provider)

        local_provider_settings = settings.get("local_provider_settings")
        if isinstance(local_provider_settings, dict):
            for provider_key, provider_settings in local_provider_settings.items():
                normalized_provider = normalize_local_ocr_provider(str(provider_key or ""))
                if not isinstance(provider_settings, dict):
                    continue
                self._apply_local_provider_settings_payload(provider_settings, target_provider=normalized_provider)

        self.set_selected_provider_mode(provider_mode)
        self.set_selected_local_ocr_provider(local_provider)
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

    def _on_local_provider_changed(self) -> None:
        self._save_current_local_provider_settings(provider_key=self._loaded_local_provider_key)
        self._load_local_provider_settings(self.selected_local_ocr_provider())
        self._update_provider_sections()
        self.ocr_provider_changed.emit(self.selected_ocr_provider())

    def _on_server_url_changed(self) -> None:
        if self._syncing_server_address:
            return
        raw_url = self.server_url_input.text().strip()
        if not raw_url:
            return
        if "://" not in raw_url:
            raw_url = f"http://{raw_url}"
        config = OCRConfig.from_value({"server_url": raw_url})
        self._syncing_server_address = True
        try:
            self.server_url_input.setText(config.server_url)
            self.server_host_input.setText(config.host)
            self.server_port_input.setValue(int(config.port))
        finally:
            self._syncing_server_address = False

    def _on_server_host_port_changed(self) -> None:
        if self._syncing_server_address:
            return
        self._syncing_server_address = True
        try:
            host = self.server_host_input.text().strip() or "127.0.0.1"
            port = int(self.server_port_input.value())
            self.server_host_input.setText(host)
            self.server_url_input.setText(f"http://{host}:{port}")
        finally:
            self._syncing_server_address = False

    def _default_local_provider_settings(self) -> dict[str, dict[str, Any]]:
        root = self._workspace_root or Path.cwd()
        llama_cpp_dir = str((root / "tools" / "llama.cpp").resolve())
        return {
            OCR_PROVIDER_PADDLE_VL_LLAMA: {
                "server_url": "http://127.0.0.1:8080",
                "host": "127.0.0.1",
                "port": 8080,
                "model_path": str((root / "model" / "paddleocr_vl" / "model.gguf").resolve()),
                "mmproj_path": str((root / "model" / "paddleocr_vl" / "mmproj.gguf").resolve()),
                "llama_cpp_dir": llama_cpp_dir,
                "gpu_layers": 99,
                "ctx_size": 8192,
                "temperature": 0.0,
                "extra_args": "",
            },
            OCR_PROVIDER_DEEPSEEK_OCR_LLAMA: {
                "server_url": "http://127.0.0.1:8080",
                "host": "127.0.0.1",
                "port": 8080,
                "model_path": str((root / "model" / "deepseek_ocr" / "deepseek-ocr-Q4_K_M.gguf").resolve()),
                "mmproj_path": str((root / "model" / "deepseek_ocr" / "mmproj-deepseek-ocr-bf16.gguf").resolve()),
                "llama_cpp_dir": llama_cpp_dir,
                "gpu_layers": 99,
                "ctx_size": 8192,
                "temperature": 0.0,
                "extra_args": "",
            },
        }

    def _local_provider_settings_snapshot(self) -> dict[str, dict[str, Any]]:
        snapshot: dict[str, dict[str, Any]] = {}
        for provider_key, provider_settings in self._local_provider_settings.items():
            snapshot[str(provider_key)] = dict(provider_settings)
        return snapshot

    def _save_current_local_provider_settings(self, provider_key: str | None = None) -> None:
        normalized_provider = normalize_local_ocr_provider(provider_key or self.selected_local_ocr_provider())
        self._local_provider_settings[normalized_provider] = dict(self.server_values())

    def _load_local_provider_settings(self, provider_key: str) -> None:
        normalized_provider = normalize_local_ocr_provider(provider_key)
        provider_settings = dict(
            self._local_provider_settings.get(
                normalized_provider,
                self._default_local_provider_settings().get(normalized_provider, {}),
            )
        )
        config = OCRConfig.from_value(provider_settings)
        self._syncing_server_address = True
        try:
            self.server_url_input.setText(config.server_url)
            self.server_host_input.setText(config.host)
            self.server_port_input.setValue(int(config.port))
        finally:
            self._syncing_server_address = False
        self.server_model_path_input.setText(config.model_path)
        self.server_mmproj_path_input.setText(config.mmproj_path)
        self.server_llama_cpp_dir_input.setText(config.llama_cpp_dir)
        self.server_gpu_layers_input.setValue(int(config.gpu_layers))
        self.server_ctx_size_input.setValue(int(config.ctx_size))
        self.server_temperature_input.setValue(float(config.temperature))
        self.server_extra_args_input.setText(config.extra_args)
        self._loaded_local_provider_key = normalized_provider

    def _apply_local_provider_settings_payload(
        self,
        payload: dict[str, Any],
        *,
        target_provider: str,
    ) -> None:
        config = OCRConfig.from_value(
            {
                "ocr_provider": target_provider,
                "ocr_provider_mode": OCR_PROVIDER_MODE_LOCAL,
                "local_ocr_provider": target_provider,
                "server_url": payload.get("server_url"),
                "host": payload.get("host"),
                "port": payload.get("port"),
                "model_path": payload.get("model_path"),
                "mmproj_path": payload.get("mmproj_path"),
                "llama_cpp_dir": payload.get("llama_cpp_dir"),
                "gpu_layers": payload.get("gpu_layers"),
                "ctx_size": payload.get("ctx_size"),
                "temperature": payload.get("temperature"),
                "extra_args": payload.get("extra_args"),
            }
        )
        self._local_provider_settings[normalize_local_ocr_provider(target_provider)] = {
            "server_url": config.server_url,
            "host": config.host,
            "port": int(config.port),
            "model_path": config.model_path,
            "mmproj_path": config.mmproj_path,
            "llama_cpp_dir": config.llama_cpp_dir,
            "gpu_layers": int(config.gpu_layers),
            "ctx_size": int(config.ctx_size),
            "temperature": float(config.temperature),
            "extra_args": config.extra_args,
        }

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
