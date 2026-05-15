"""Translation stage inspector panel."""

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
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mmt_core import (
    DEFAULT_PROMPT_STYLE_ID,
    LANGUAGE_CHOICES,
    OPENAI_COMPATIBLE_PRESET_BASE_URLS,
    OPENAI_COMPATIBLE_PRESET_CHOICES,
    PROMPT_MODE_BUILT_IN_PLUS_USER,
    PROMPT_MODE_CHOICES,
    TRANSLATOR_CHOICES,
    TranslationConfig,
    create_custom_style,
    default_output_contract,
    delete_custom_style,
    list_builtin_prompt_styles,
    normalize_custom_styles,
    normalize_translator_name,
    summarize_translation_json,
    update_ocr_item_text,
    update_translation_item_source_text,
    update_translation_item_translated_text,
)
from mmt_gui.widgets import CollapsibleSection, CropPreviewPanel, TextItemEditorWidget
from mmt_gui.widgets.settings_card import style_button

from .base_panel import StagePanel


class TranslationPanel(StagePanel):
    """Inspector panel for translation settings and cached items."""

    initialize_selected_requested = pyqtSignal()
    reinitialize_selected_requested = pyqtSignal()
    initialize_all_requested = pyqtSignal()
    reinitialize_all_requested = pyqtSignal()
    run_selected_requested = pyqtSignal()
    rerun_selected_requested = pyqtSignal()
    run_all_requested = pyqtSignal()
    rerun_all_requested = pyqtSignal()
    run_selected_items_requested = pyqtSignal()
    rerun_selected_items_requested = pyqtSignal()
    reload_requested = pyqtSignal()
    save_text_requested = pyqtSignal()
    cache_updated = pyqtSignal(object)
    error_emitted = pyqtSignal(str, str)
    warning_emitted = pyqtSignal(str)
    message_emitted = pyqtSignal(str)

    def __init__(self, parent: object | None = None) -> None:
        super().__init__("Translation", parent)
        self._actions_enabled = True
        self._project_root: Path | None = None
        self._ocr_cache_path: Path | None = None
        self._ocr_items_by_id: dict[int, dict[str, Any]] = {}
        self._items: list[dict[str, Any]] = []
        self._cache_path: Path | None = None
        self._editor_row: int | None = None
        self._selection_guard = False
        self._custom_styles: list[dict[str, Any]] = []
        self._last_built_in_style_id = DEFAULT_PROMPT_STYLE_ID
        self._ui_refresh_guard = False

        settings_card = CollapsibleSection("Translation Settings", expanded=False)
        settings_form = QFormLayout()
        settings_form.setContentsMargins(0, 0, 0, 0)
        settings_form.setSpacing(8)

        self.source_language_input = QComboBox()
        self.source_language_input.setEditable(True)
        self.source_language_input.addItems(LANGUAGE_CHOICES)
        self.source_language_input.setCurrentText("ja")

        self.target_language_input = QComboBox()
        self.target_language_input.setEditable(True)
        self.target_language_input.addItems(LANGUAGE_CHOICES)
        self.target_language_input.setCurrentText("en")

        self.translator_input = QComboBox()
        self.translator_input.addItems(TRANSLATOR_CHOICES)
        self.translator_input.setCurrentText("Google")

        self.batch_size_input = QSpinBox()
        self.batch_size_input.setRange(1, 50)
        self.batch_size_input.setValue(3)

        self.use_context_memory_checkbox = QCheckBox("Use context memory")

        self.gemini_api_key_input = QLineEdit()
        self.gemini_api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.local_llm_server_url_input = QLineEdit("http://127.0.0.1:8080")
        self.local_llm_model_input = QLineEdit("gpt-4o")
        self.deepseek_api_key_input = QLineEdit()
        self.deepseek_api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.deepseek_model_input = QLineEdit("deepseek-v4-flash")
        self.deepseek_thinking_checkbox = QCheckBox("Enable DeepSeek thinking")
        self.openai_provider_preset_input = QComboBox()
        self.openai_provider_preset_input.addItems(OPENAI_COMPATIBLE_PRESET_CHOICES)
        self.openai_provider_preset_input.setCurrentText("OpenAI")
        self.openai_base_url_input = QLineEdit(
            OPENAI_COMPATIBLE_PRESET_BASE_URLS.get("OpenAI", "https://api.openai.com/v1")
        )
        self.openai_api_key_input = QLineEdit()
        self.openai_api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.openai_model_input = QLineEdit()
        self.openai_temperature_input = QDoubleSpinBox()
        self.openai_temperature_input.setDecimals(2)
        self.openai_temperature_input.setRange(0.0, 2.0)
        self.openai_temperature_input.setSingleStep(0.05)
        self.openai_temperature_input.setValue(0.3)
        self.openai_max_tokens_input = QSpinBox()
        self.openai_max_tokens_input.setRange(0, 200000)
        self.openai_max_tokens_input.setValue(0)
        self.openai_max_tokens_input.setSpecialValueText("Auto")
        self.openai_timeout_input = QSpinBox()
        self.openai_timeout_input.setRange(5, 3600)
        self.openai_timeout_input.setValue(120)
        self.openai_json_mode_checkbox = QCheckBox("Request JSON mode")

        settings_form.addRow("Source Language:", self.source_language_input)
        settings_form.addRow("Target Language:", self.target_language_input)
        settings_form.addRow("Translator:", self.translator_input)
        settings_form.addRow("Batch Size (pages):", self.batch_size_input)
        settings_form.addRow("Use Context Memory:", self.use_context_memory_checkbox)
        settings_card.content_layout.addLayout(settings_form)
        self.content_layout.addWidget(settings_card)

        prompt_card = CollapsibleSection("Prompt Studio", expanded=True)
        prompt_form = QFormLayout()
        prompt_form.setContentsMargins(0, 0, 0, 0)
        prompt_form.setSpacing(8)

        self.style_input = QComboBox()
        self._rebuild_style_dropdown(DEFAULT_PROMPT_STYLE_ID)
        self.style_description_label = QLabel("")
        self.style_description_label.setWordWrap(True)
        self.style_description_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.prompt_mode_input = QComboBox()
        for mode_value, mode_label in PROMPT_MODE_CHOICES:
            self.prompt_mode_input.addItem(mode_label, mode_value)

        self.user_instructions_input = QPlainTextEdit()
        self.user_instructions_input.setMaximumHeight(90)
        self.user_instructions_input.setPlaceholderText("Extra instructions to layer on top of the selected style.")

        self.full_custom_prompt_input = QPlainTextEdit()
        self.full_custom_prompt_input.setMaximumHeight(140)
        self.full_custom_prompt_input.setPlaceholderText("Replace the style section with your own custom instructions.")

        self.project_notes_input = QPlainTextEdit()
        self.project_notes_input.setMaximumHeight(80)
        self.project_notes_input.setPlaceholderText("Names, terminology, tone notes, or continuity reminders for this project.")

        self.prompt_preview_input = QPlainTextEdit()
        self.prompt_preview_input.setReadOnly(True)
        self.prompt_preview_input.setMaximumHeight(220)

        self.output_contract_input = QPlainTextEdit()
        self.output_contract_input.setReadOnly(True)
        self.output_contract_input.setMaximumHeight(140)
        self.output_contract_input.setPlainText(default_output_contract())

        self.prompt_provider_note_label = QLabel("")
        self.prompt_provider_note_label.setWordWrap(True)
        self.prompt_provider_note_label.setProperty("role", "muted")
        self.prompt_provider_note_label.setVisible(False)

        prompt_form.addRow("Style:", self.style_input)
        prompt_form.addRow("Description:", self.style_description_label)
        prompt_form.addRow("Prompt Mode:", self.prompt_mode_input)
        prompt_form.addRow("My Instructions:", self.user_instructions_input)
        prompt_form.addRow("Full Custom Prompt:", self.full_custom_prompt_input)
        prompt_form.addRow("Project Notes:", self.project_notes_input)
        prompt_form.addRow("Final Prompt Preview:", self.prompt_preview_input)
        prompt_form.addRow("Output Contract:", self.output_contract_input)
        self.user_instructions_label = prompt_form.labelForField(self.user_instructions_input)
        self.full_custom_prompt_label = prompt_form.labelForField(self.full_custom_prompt_input)
        prompt_card.content_layout.addLayout(prompt_form)
        prompt_card.content_layout.addWidget(self.prompt_provider_note_label)

        prompt_actions_layout = QGridLayout()
        prompt_actions_layout.setContentsMargins(0, 0, 0, 0)
        prompt_actions_layout.setHorizontalSpacing(8)
        prompt_actions_layout.setVerticalSpacing(8)

        self.refresh_prompt_preview_button = QPushButton("Refresh Preview")
        style_button(self.refresh_prompt_preview_button, "secondary")
        self.refresh_prompt_preview_button.clicked.connect(self._refresh_prompt_preview)
        prompt_actions_layout.addWidget(self.refresh_prompt_preview_button, 0, 0)

        self.reset_prompt_button = QPushButton("Reset to Built-in")
        style_button(self.reset_prompt_button, "secondary")
        self.reset_prompt_button.clicked.connect(self._reset_prompt_to_built_in)
        prompt_actions_layout.addWidget(self.reset_prompt_button, 0, 1)

        self.save_custom_style_button = QPushButton("Save as Custom Style")
        style_button(self.save_custom_style_button, "primary")
        self.save_custom_style_button.clicked.connect(self._save_current_prompt_as_custom_style)
        prompt_actions_layout.addWidget(self.save_custom_style_button, 1, 0)

        self.delete_custom_style_button = QPushButton("Delete Custom Style")
        style_button(self.delete_custom_style_button, "danger")
        self.delete_custom_style_button.clicked.connect(self._delete_selected_custom_style)
        prompt_actions_layout.addWidget(self.delete_custom_style_button, 1, 1)
        prompt_card.content_layout.addLayout(prompt_actions_layout)
        self.content_layout.addWidget(prompt_card)

        provider_card = CollapsibleSection("Provider Credentials & Advanced", expanded=False)
        self.provider_hint_label = QLabel("")
        self.provider_hint_label.setWordWrap(True)
        self.provider_hint_label.setProperty("role", "muted")
        provider_card.content_layout.addWidget(self.provider_hint_label)

        self.gemini_provider_widget = QWidget()
        gemini_form = QFormLayout(self.gemini_provider_widget)
        gemini_form.setContentsMargins(0, 0, 0, 0)
        gemini_form.setSpacing(8)
        gemini_form.addRow("Gemini API Key:", self.gemini_api_key_input)
        provider_card.content_layout.addWidget(self.gemini_provider_widget)

        self.local_llm_provider_widget = QWidget()
        local_llm_form = QFormLayout(self.local_llm_provider_widget)
        local_llm_form.setContentsMargins(0, 0, 0, 0)
        local_llm_form.setSpacing(8)
        local_llm_form.addRow("Local LLM Server URL:", self.local_llm_server_url_input)
        local_llm_form.addRow("Local LLM Model:", self.local_llm_model_input)
        provider_card.content_layout.addWidget(self.local_llm_provider_widget)

        self.deepseek_provider_widget = QWidget()
        deepseek_form = QFormLayout(self.deepseek_provider_widget)
        deepseek_form.setContentsMargins(0, 0, 0, 0)
        deepseek_form.setSpacing(8)
        deepseek_form.addRow("DeepSeek API Key:", self.deepseek_api_key_input)
        deepseek_form.addRow("DeepSeek Model:", self.deepseek_model_input)
        deepseek_form.addRow("DeepSeek Options:", self.deepseek_thinking_checkbox)
        provider_card.content_layout.addWidget(self.deepseek_provider_widget)

        self.openai_provider_widget = QWidget()
        openai_form = QFormLayout(self.openai_provider_widget)
        openai_form.setContentsMargins(0, 0, 0, 0)
        openai_form.setSpacing(8)
        openai_form.addRow("Provider Preset:", self.openai_provider_preset_input)
        openai_form.addRow("Base URL:", self.openai_base_url_input)
        openai_form.addRow("API Key:", self.openai_api_key_input)
        openai_form.addRow("Model:", self.openai_model_input)
        openai_form.addRow("Temperature:", self.openai_temperature_input)
        openai_form.addRow("Max Tokens:", self.openai_max_tokens_input)
        openai_form.addRow("Timeout (s):", self.openai_timeout_input)
        openai_form.addRow("JSON Mode:", self.openai_json_mode_checkbox)
        provider_card.content_layout.addWidget(self.openai_provider_widget)
        self.content_layout.addWidget(provider_card)

        self.style_input.currentIndexChanged.connect(self._on_prompt_inputs_changed)
        self.prompt_mode_input.currentIndexChanged.connect(self._on_prompt_inputs_changed)
        self.user_instructions_input.textChanged.connect(self._on_prompt_inputs_changed)
        self.full_custom_prompt_input.textChanged.connect(self._on_prompt_inputs_changed)
        self.project_notes_input.textChanged.connect(self._on_prompt_inputs_changed)
        self.translator_input.currentTextChanged.connect(self._on_translator_changed)
        self.source_language_input.currentTextChanged.connect(self._on_prompt_inputs_changed)
        self.target_language_input.currentTextChanged.connect(self._on_prompt_inputs_changed)
        self.openai_provider_preset_input.currentTextChanged.connect(self._on_openai_preset_changed)

        self._update_provider_visibility()
        self._update_prompt_mode_visibility()
        self._refresh_prompt_preview()

        actions_card = CollapsibleSection("Translation Actions", expanded=True)
        actions_layout = QGridLayout()
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setHorizontalSpacing(8)
        actions_layout.setVerticalSpacing(8)

        init_current_label = QLabel("Initialize Current")
        init_current_label.setProperty("role", "muted")
        actions_layout.addWidget(init_current_label, 0, 0)

        self.initialize_selected_button = QPushButton("Initialize")
        style_button(self.initialize_selected_button, "primary")
        self.initialize_selected_button.clicked.connect(self.initialize_selected_requested.emit)
        self.initialize_selected_button.setToolTip("Initialize translation for the current page and reuse existing cache when available.")
        actions_layout.addWidget(self.initialize_selected_button, 0, 1)

        self.reinitialize_selected_button = QPushButton("Re-initialize")
        style_button(self.reinitialize_selected_button, "rerun")
        self.reinitialize_selected_button.clicked.connect(self.reinitialize_selected_requested.emit)
        self.reinitialize_selected_button.setToolTip("Force the current page to rebuild translation items from OCR.")
        actions_layout.addWidget(self.reinitialize_selected_button, 0, 2)

        init_all_label = QLabel("Initialize All")
        init_all_label.setProperty("role", "muted")
        actions_layout.addWidget(init_all_label, 1, 0)

        self.initialize_all_button = QPushButton("Initialize All")
        style_button(self.initialize_all_button, "primary")
        self.initialize_all_button.clicked.connect(self.initialize_all_requested.emit)
        self.initialize_all_button.setToolTip("Initialize translation for every page and reuse existing cache when available.")
        actions_layout.addWidget(self.initialize_all_button, 1, 1)

        self.reinitialize_all_button = QPushButton("Re-initialize All")
        style_button(self.reinitialize_all_button, "rerun")
        self.reinitialize_all_button.clicked.connect(self.reinitialize_all_requested.emit)
        self.reinitialize_all_button.setToolTip("Force every page to rebuild translation items from OCR.")
        actions_layout.addWidget(self.reinitialize_all_button, 1, 2)

        translate_current_label = QLabel("Current Page")
        translate_current_label.setProperty("role", "muted")
        actions_layout.addWidget(translate_current_label, 2, 0)

        self.run_selected_button = QPushButton("Translate")
        style_button(self.run_selected_button, "primary")
        self.run_selected_button.clicked.connect(self.run_selected_requested.emit)
        self.run_selected_button.setToolTip("Translate the current page and reuse existing results when available.")
        actions_layout.addWidget(self.run_selected_button, 2, 1)

        self.rerun_selected_button = QPushButton("Re-translate")
        style_button(self.rerun_selected_button, "rerun")
        self.rerun_selected_button.clicked.connect(self.rerun_selected_requested.emit)
        self.rerun_selected_button.setToolTip("Force the current page to regenerate translation output.")
        actions_layout.addWidget(self.rerun_selected_button, 2, 2)

        translate_all_label = QLabel("All Pages")
        translate_all_label.setProperty("role", "muted")
        actions_layout.addWidget(translate_all_label, 3, 0)

        self.run_all_button = QPushButton("Translate All")
        style_button(self.run_all_button, "primary")
        self.run_all_button.clicked.connect(self.run_all_requested.emit)
        self.run_all_button.setToolTip("Translate every page and reuse existing results when available.")
        actions_layout.addWidget(self.run_all_button, 3, 1)

        self.rerun_all_button = QPushButton("Re-translate All")
        style_button(self.rerun_all_button, "rerun")
        self.rerun_all_button.clicked.connect(self.rerun_all_requested.emit)
        self.rerun_all_button.setToolTip("Force every page to regenerate translation output.")
        actions_layout.addWidget(self.rerun_all_button, 3, 2)

        selected_items_label = QLabel("Selected Items")
        selected_items_label.setProperty("role", "muted")
        actions_layout.addWidget(selected_items_label, 4, 0)

        self.run_selected_items_button = QPushButton("Translate Selected")
        style_button(self.run_selected_items_button, "primary")
        self.run_selected_items_button.clicked.connect(self.run_selected_items_requested.emit)
        self.run_selected_items_button.setToolTip("Translate only the selected table rows and reuse results when available.")
        actions_layout.addWidget(self.run_selected_items_button, 4, 1)

        self.rerun_selected_items_button = QPushButton("Re-translate Selected")
        style_button(self.rerun_selected_items_button, "rerun")
        self.rerun_selected_items_button.clicked.connect(self.rerun_selected_items_requested.emit)
        self.rerun_selected_items_button.setToolTip("Force the selected translation rows to regenerate output.")
        actions_layout.addWidget(self.rerun_selected_items_button, 4, 2)

        cache_label = QLabel("Cache")
        cache_label.setProperty("role", "muted")
        actions_layout.addWidget(cache_label, 5, 0)

        self.reload_button = QPushButton("Reload Cache")
        style_button(self.reload_button, "secondary")
        self.reload_button.clicked.connect(self.reload_requested.emit)
        self.reload_button.setToolTip("Reload translation data from disk.")
        actions_layout.addWidget(self.reload_button, 5, 1)

        self.save_text_button = QPushButton("Save Edited Translation Text")
        style_button(self.save_text_button, "secondary")
        self.save_text_button.clicked.connect(self.save_text_requested.emit)
        self.save_text_button.setToolTip("Save the currently selected translation editor values.")
        actions_layout.addWidget(self.save_text_button, 5, 2)
        actions_card.content_layout.addLayout(actions_layout)
        self.content_layout.addWidget(actions_card)

        details_card = CollapsibleSection("Translation Summary", expanded=True)
        details_form = QFormLayout()
        details_form.setContentsMargins(0, 0, 0, 0)
        details_form.setSpacing(8)
        self.total_items_value = QLabel("0")
        self.pending_items_value = QLabel("0")
        self.done_items_value = QLabel("0")
        self.error_items_value = QLabel("0")
        self.cache_path_value = QLabel("-")
        self.cache_path_value.setWordWrap(True)
        self.cache_path_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        details_form.addRow("Total Items:", self.total_items_value)
        details_form.addRow("Pending:", self.pending_items_value)
        details_form.addRow("Done:", self.done_items_value)
        details_form.addRow("Error:", self.error_items_value)
        details_form.addRow("Translation JSON:", self.cache_path_value)
        details_card.content_layout.addLayout(details_form)
        self.content_layout.addWidget(details_card)

        items_card = CollapsibleSection("Translation Items", expanded=True)
        self.items_table = QTableWidget(0, 6)
        self.items_table.setProperty("stageTable", True)
        self.items_table.setHorizontalHeaderLabels(
            ["id", "ocr_item_id", "kind", "status", "source_text", "translated_text"]
        )
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
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.items_table.setMinimumHeight(240)
        items_card.content_layout.addWidget(self.items_table)
        self.content_layout.addWidget(items_card)

        self.editor_section = CollapsibleSection("Translation Item Editor", expanded=False)
        editor_info_layout = QVBoxLayout()
        editor_info_layout.setContentsMargins(0, 0, 0, 0)
        editor_info_layout.setSpacing(6)
        self.editor_details_label = QLabel("Select a translation item to edit its source and translated text.")
        self.editor_details_label.setWordWrap(True)
        self.editor_details_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.editor_dirty_label = QLabel("Saved")
        self.editor_dirty_label.setProperty("role", "muted")
        self.needs_retranslate_label = QLabel("")
        self.needs_retranslate_label.setProperty("role", "muted")
        self.needs_retranslate_label.setWordWrap(True)
        self.needs_retranslate_label.setVisible(False)
        editor_info_layout.addWidget(self.editor_details_label)
        editor_info_layout.addWidget(self.editor_dirty_label)
        editor_info_layout.addWidget(self.needs_retranslate_label)
        self.editor_section.content_layout.addLayout(editor_info_layout)

        editor_body = QWidget()
        editor_body_layout = QGridLayout(editor_body)
        editor_body_layout.setContentsMargins(0, 0, 0, 0)
        editor_body_layout.setHorizontalSpacing(12)
        editor_body_layout.setVerticalSpacing(10)

        self.crop_preview_panel = CropPreviewPanel()
        editor_body_layout.addWidget(self.crop_preview_panel, 0, 0, 2, 1)

        self.source_text_editor = TextItemEditorWidget(
            "Source OCR Text",
            placeholder="Select a translation item to edit the source OCR text.",
        )
        self.source_text_editor.dirty_changed.connect(self._update_editor_button_state)
        editor_body_layout.addWidget(self.source_text_editor, 0, 1)

        self.translated_text_editor = TextItemEditorWidget(
            "Translated Text",
            placeholder="Select a translation item to edit the translated text.",
        )
        self.translated_text_editor.dirty_changed.connect(self._update_editor_button_state)
        editor_body_layout.addWidget(self.translated_text_editor, 1, 1)
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

        self.editor_save_source_button = QPushButton("Save Source OCR")
        style_button(self.editor_save_source_button, "primary")
        self.editor_save_source_button.clicked.connect(self.save_source_text)
        editor_actions_layout.addWidget(self.editor_save_source_button, 1, 0)

        self.editor_save_translation_button = QPushButton("Save Translation")
        style_button(self.editor_save_translation_button, "primary")
        self.editor_save_translation_button.clicked.connect(self.save_translation_text)
        editor_actions_layout.addWidget(self.editor_save_translation_button, 1, 1)

        self.editor_save_both_button = QPushButton("Save Both")
        style_button(self.editor_save_both_button, "primary")
        self.editor_save_both_button.clicked.connect(self.save_both)
        editor_actions_layout.addWidget(self.editor_save_both_button, 1, 2)
        self.editor_section.content_layout.addLayout(editor_actions_layout)
        self.content_layout.addWidget(self.editor_section)

        self._set_editor_enabled(False)

    def config(self) -> TranslationConfig:
        current_style = self._current_prompt_style()
        prompt_mode = self.current_prompt_mode()
        legacy_custom_prompt = (
            self.full_custom_prompt_input.toPlainText().strip()
            if prompt_mode != PROMPT_MODE_BUILT_IN_PLUS_USER
            else self.user_instructions_input.toPlainText().strip()
        )
        return TranslationConfig(
            source_language=self.source_language_input.currentText().strip() or "ja",
            target_language=self.target_language_input.currentText().strip() or "en",
            translator=self.translator_input.currentText().strip() or "Google",
            style=current_style.get("display_name", "Default Manga"),
            custom_prompt=legacy_custom_prompt,
            batch_size_pages=self.batch_size_input.value(),
            use_context_memory=self.use_context_memory_checkbox.isChecked(),
            local_llm_server_url=self.local_llm_server_url_input.text().strip() or "http://127.0.0.1:8080",
            local_llm_model=self.local_llm_model_input.text().strip() or "gpt-4o",
            gemini_api_key=self.gemini_api_key_input.text().strip(),
            deepseek_api_key=self.deepseek_api_key_input.text().strip(),
            deepseek_model=self.deepseek_model_input.text().strip() or "deepseek-v4-flash",
            deepseek_thinking=self.deepseek_thinking_checkbox.isChecked(),
            prompt_style_id=str(current_style.get("id", DEFAULT_PROMPT_STYLE_ID) or DEFAULT_PROMPT_STYLE_ID),
            prompt_mode=prompt_mode,
            user_instructions=self.user_instructions_input.toPlainText().strip(),
            full_custom_prompt=self.full_custom_prompt_input.toPlainText().strip(),
            project_translation_notes=self.project_notes_input.toPlainText().strip(),
            output_contract=self.output_contract_input.toPlainText().strip() or default_output_contract(),
            custom_styles=list(self._custom_styles),
            openai_compatible_provider_preset=self.openai_provider_preset_input.currentText().strip() or "OpenAI",
            openai_compatible_base_url=self.openai_base_url_input.text().strip(),
            openai_compatible_api_key=self.openai_api_key_input.text().strip(),
            openai_compatible_model=self.openai_model_input.text().strip(),
            openai_compatible_temperature=float(self.openai_temperature_input.value()),
            openai_compatible_max_tokens=int(self.openai_max_tokens_input.value()),
            openai_compatible_timeout=int(self.openai_timeout_input.value()),
            openai_compatible_json_mode=self.openai_json_mode_checkbox.isChecked(),
        )

    def set_ocr_context(
        self,
        project_root: Path | None,
        ocr_data: dict[str, Any] | None,
        ocr_cache_path: Path | None,
    ) -> None:
        self._project_root = project_root.resolve() if isinstance(project_root, Path) else None
        self.crop_preview_panel.set_project_root(self._project_root)
        self._ocr_cache_path = ocr_cache_path.resolve() if isinstance(ocr_cache_path, Path) else None
        self._ocr_items_by_id = {}
        if isinstance(ocr_data, dict):
            items = ocr_data.get("items", [])
            if isinstance(items, list):
                self._ocr_items_by_id = {
                    int(item.get("id", index)): item
                    for index, item in enumerate(items)
                    if isinstance(item, dict)
                }
        self._refresh_editor_view()

    def set_items(self, items: list[dict[str, Any]], cache_path: Any) -> None:
        previous_item_id = self.current_editor_item_id()
        self._items = list(items)
        self._cache_path = cache_path.resolve() if isinstance(cache_path, Path) else (Path(str(cache_path)).resolve() if cache_path else None)

        self.items_table.blockSignals(True)
        self.items_table.setRowCount(len(self._items))

        for row_index, item in enumerate(self._items):
            source_text = str(item.get("source_text", "") or "")
            translated_text = str(item.get("translated_text", "") or "")
            row_values = [
                str(item.get("id", "")),
                str(item.get("ocr_item_id", "")),
                str(item.get("kind", "")),
                str(item.get("status", "")),
                self._display_text(source_text),
                self._display_text(translated_text),
            ]
            error_text = str(item.get("error", "") or "").strip()
            needs_retranslate = bool(item.get("needs_retranslate", False))
            for column_index, value in enumerate(row_values):
                table_item = QTableWidgetItem(value)
                table_item.setFlags(table_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if column_index == 0:
                    table_item.setData(Qt.ItemDataRole.UserRole, int(item.get("id", row_index)))
                tooltip_lines = []
                if column_index == 4 and source_text:
                    tooltip_lines.append(source_text)
                if column_index == 5 and translated_text:
                    tooltip_lines.append(translated_text)
                if needs_retranslate and column_index in {3, 4, 5}:
                    tooltip_lines.append("Source edited; re-translate recommended.")
                if error_text:
                    tooltip_lines.append(f"Error: {error_text}")
                if tooltip_lines:
                    table_item.setToolTip("\n\n".join(tooltip_lines))
                self.items_table.setItem(row_index, column_index, table_item)

        self.items_table.blockSignals(False)
        summary = summarize_translation_json({"items": self._items})
        self.total_items_value.setText(str(summary.get("total", 0)))
        self.pending_items_value.setText(str(summary.get("pending", 0)))
        self.done_items_value.setText(str(summary.get("done", 0)))
        self.error_items_value.setText(str(summary.get("error", 0)))
        self.cache_path_value.setText(str(self._cache_path) if self._cache_path is not None else "-")
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
            self.crop_preview_panel.clear("Select a translation item to preview the matching OCR crop.")

    def clear_view(self) -> None:
        self._items = []
        self._cache_path = None
        self._editor_row = None
        self.items_table.blockSignals(True)
        self.items_table.setRowCount(0)
        self.items_table.blockSignals(False)
        self.total_items_value.setText("0")
        self.pending_items_value.setText("0")
        self.done_items_value.setText("0")
        self.error_items_value.setText("0")
        self.cache_path_value.setText("-")
        self.editor_section.set_expanded(False)
        self._set_editor_enabled(False)
        self.crop_preview_panel.clear("Select a translation item to preview the matching OCR crop.")

    def selected_item_ids(self) -> list[int]:
        selection_model = self.items_table.selectionModel()
        if selection_model is None:
            return []
        selected_ids: list[int] = []
        for model_index in selection_model.selectedRows():
            row_index = model_index.row()
            if row_index < 0 or row_index >= len(self._items):
                continue
            selected_ids.append(int(self._items[row_index].get("id", row_index)))
        return selected_ids

    def current_editor_item_id(self) -> int | None:
        if self._editor_row is None or self._editor_row < 0 or self._editor_row >= len(self._items):
            return None
        return int(self._items[self._editor_row].get("id", self._editor_row))

    def has_unsaved_changes(self) -> bool:
        return self.source_text_editor.is_dirty() or self.translated_text_editor.is_dirty()

    def ensure_pending_changes_resolved(self, parent: QWidget | None = None) -> bool:
        if not self.has_unsaved_changes():
            return True

        message_box = QMessageBox(parent or self)
        message_box.setIcon(QMessageBox.Icon.Question)
        message_box.setWindowTitle("Unsaved translation edits")
        message_box.setText("Save translation editor changes before leaving this item?")
        save_button = message_box.addButton("Save changes", QMessageBox.ButtonRole.AcceptRole)
        discard_button = message_box.addButton("Discard", QMessageBox.ButtonRole.DestructiveRole)
        cancel_button = message_box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        message_box.setDefaultButton(save_button)
        message_box.exec()
        clicked = message_box.clickedButton()
        if clicked is save_button:
            return self.save_both()
        if clicked is discard_button:
            self.revert_current_item()
            return True
        if clicked is cancel_button:
            self.warning_emitted.emit("Translation row selection change canceled because edits are still unsaved.")
        return False

    def save_source_text(self) -> bool:
        return self._save_changes(save_source=True, save_translation=False)

    def save_translation_text(self) -> bool:
        return self._save_changes(save_source=False, save_translation=True)

    def save_both(self) -> bool:
        return self._save_changes(save_source=True, save_translation=True)

    def revert_current_item(self) -> None:
        self._refresh_editor_view()
        self.message_emitted.emit("Reverted translation editor changes.")

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
            self.initialize_selected_button,
            self.reinitialize_selected_button,
            self.initialize_all_button,
            self.reinitialize_all_button,
            self.run_selected_button,
            self.rerun_selected_button,
            self.run_all_button,
            self.rerun_all_button,
            self.run_selected_items_button,
            self.rerun_selected_items_button,
            self.reload_button,
            self.save_text_button,
            self.items_table,
            self.editor_previous_button,
            self.editor_next_button,
            self.editor_revert_button,
            self.editor_save_source_button,
            self.editor_save_translation_button,
            self.editor_save_both_button,
            self.source_text_editor.editor,
            self.translated_text_editor.editor,
        ):
            widget.setEnabled(enabled)
        self._update_editor_button_state(False)

    def settings_snapshot(self) -> dict[str, Any]:
        config = self.config()
        return {
            "source_language": self.source_language_input.currentText().strip(),
            "target_language": self.target_language_input.currentText().strip(),
            "translator": self.translator_input.currentText().strip(),
            "style": config.resolved_style_name(),
            "custom_prompt": config.custom_prompt,
            "prompt_style_id": config.prompt_style_id,
            "prompt_mode": config.prompt_mode,
            "user_instructions": config.user_instructions,
            "full_custom_prompt": config.full_custom_prompt,
            "project_translation_notes": config.project_translation_notes,
            "output_contract": config.output_contract,
            "custom_styles": list(self._custom_styles),
            "batch_size_pages": self.batch_size_input.value(),
            "use_context_memory": self.use_context_memory_checkbox.isChecked(),
            "gemini_api_key": self.gemini_api_key_input.text(),
            "local_llm_server_url": self.local_llm_server_url_input.text(),
            "local_llm_model": self.local_llm_model_input.text(),
            "deepseek_api_key": self.deepseek_api_key_input.text(),
            "deepseek_model": self.deepseek_model_input.text(),
            "deepseek_thinking": self.deepseek_thinking_checkbox.isChecked(),
            "openai_compatible_provider_preset": self.openai_provider_preset_input.currentText().strip(),
            "openai_compatible_base_url": self.openai_base_url_input.text(),
            "openai_compatible_api_key": self.openai_api_key_input.text(),
            "openai_compatible_model": self.openai_model_input.text(),
            "openai_compatible_temperature": float(self.openai_temperature_input.value()),
            "openai_compatible_max_tokens": int(self.openai_max_tokens_input.value()),
            "openai_compatible_timeout": int(self.openai_timeout_input.value()),
            "openai_compatible_json_mode": self.openai_json_mode_checkbox.isChecked(),
        }

    def apply_settings(self, settings: dict[str, Any]) -> None:
        if not isinstance(settings, dict):
            return
        config = TranslationConfig.from_value(settings)
        self._ui_refresh_guard = True
        try:
            self._custom_styles = normalize_custom_styles(settings.get("custom_styles", config.custom_styles))
            self._rebuild_style_dropdown(config.prompt_style_id)
            self.source_language_input.setCurrentText(config.source_language)
            self.target_language_input.setCurrentText(config.target_language)
            self.translator_input.setCurrentText(config.translator)
            self.prompt_mode_input.setCurrentIndex(
                max(self.prompt_mode_input.findData(config.prompt_mode), 0)
            )
            self.user_instructions_input.setPlainText(config.user_instructions)
            self.full_custom_prompt_input.setPlainText(config.full_custom_prompt)
            self.project_notes_input.setPlainText(config.project_translation_notes)
            self.output_contract_input.setPlainText(config.output_contract or default_output_contract())
            self.gemini_api_key_input.setText(config.gemini_api_key)
            self.local_llm_server_url_input.setText(config.local_llm_server_url)
            self.local_llm_model_input.setText(config.local_llm_model)
            self.deepseek_api_key_input.setText(config.deepseek_api_key)
            self.deepseek_model_input.setText(config.deepseek_model)
            self.deepseek_thinking_checkbox.setChecked(config.deepseek_thinking)
            self.openai_provider_preset_input.setCurrentText(config.openai_compatible_provider_preset)
            self.openai_base_url_input.setText(config.openai_compatible_base_url)
            self.openai_api_key_input.setText(config.openai_compatible_api_key)
            self.openai_model_input.setText(config.openai_compatible_model)
            self.openai_temperature_input.setValue(float(config.openai_compatible_temperature))
            self.openai_max_tokens_input.setValue(int(config.openai_compatible_max_tokens))
            self.openai_timeout_input.setValue(int(config.openai_compatible_timeout))
            self.openai_json_mode_checkbox.setChecked(config.openai_compatible_json_mode)
            try:
                self.batch_size_input.setValue(int(config.batch_size_pages))
            except Exception:
                pass
            self.use_context_memory_checkbox.setChecked(bool(config.use_context_memory))
        finally:
            self._ui_refresh_guard = False
        self._update_provider_visibility()
        self._update_prompt_mode_visibility()
        self._refresh_prompt_preview()

    def current_prompt_mode(self) -> str:
        value = self.prompt_mode_input.currentData()
        if isinstance(value, str) and value.strip():
            return value
        return PROMPT_MODE_BUILT_IN_PLUS_USER

    def current_prompt_style_id(self) -> str:
        value = self.style_input.currentData()
        if isinstance(value, str) and value.strip():
            return value
        return DEFAULT_PROMPT_STYLE_ID

    def _current_prompt_style(self) -> dict[str, Any]:
        target_id = self.current_prompt_style_id()
        for style in list_builtin_prompt_styles():
            if style.id == target_id:
                return style.to_dict()
        for style in self._custom_styles:
            if str(style.get("id", "") or "") == target_id:
                return dict(style)
        return list_builtin_prompt_styles()[0].to_dict()

    def _rebuild_style_dropdown(self, selected_style_id: str | None = None) -> None:
        built_in_styles = list_builtin_prompt_styles()
        self.style_input.blockSignals(True)
        self.style_input.clear()
        for style in built_in_styles:
            self.style_input.addItem(style.display_name, style.id)
        for style in self._custom_styles:
            display_name = str(style.get("display_name", "") or "").strip()
            style_id = str(style.get("id", "") or "").strip()
            if display_name and style_id:
                self.style_input.addItem(display_name, style_id)
        target_id = str(selected_style_id or "").strip() or DEFAULT_PROMPT_STYLE_ID
        target_index = self.style_input.findData(target_id)
        if target_index < 0:
            target_index = self.style_input.findData(DEFAULT_PROMPT_STYLE_ID)
        if target_index < 0:
            target_index = 0
        self.style_input.setCurrentIndex(target_index)
        self.style_input.blockSignals(False)

    def _is_selected_style_custom(self) -> bool:
        current_id = self.current_prompt_style_id()
        return any(str(style.get("id", "") or "") == current_id for style in self._custom_styles)

    def _on_prompt_inputs_changed(self) -> None:
        if self._ui_refresh_guard:
            return
        current_style = self._current_prompt_style()
        if not self._is_selected_style_custom():
            self._last_built_in_style_id = str(current_style.get("id", DEFAULT_PROMPT_STYLE_ID) or DEFAULT_PROMPT_STYLE_ID)
        self._update_prompt_mode_visibility()
        self._refresh_prompt_preview()

    def _on_translator_changed(self) -> None:
        if self._ui_refresh_guard:
            return
        self._update_provider_visibility()
        self._refresh_prompt_preview()

    def _on_openai_preset_changed(self, preset_name: str) -> None:
        if self._ui_refresh_guard:
            return
        default_base_url = OPENAI_COMPATIBLE_PRESET_BASE_URLS.get(str(preset_name or "").strip(), "")
        if default_base_url:
            self.openai_base_url_input.setText(default_base_url)

    def _update_prompt_mode_visibility(self) -> None:
        prompt_mode = self.current_prompt_mode()
        built_in_mode = prompt_mode == PROMPT_MODE_BUILT_IN_PLUS_USER
        self.user_instructions_input.setVisible(built_in_mode)
        self.full_custom_prompt_input.setVisible(not built_in_mode)
        if self.user_instructions_label is not None:
            self.user_instructions_label.setVisible(built_in_mode)
        if self.full_custom_prompt_label is not None:
            self.full_custom_prompt_label.setVisible(not built_in_mode)
        self.delete_custom_style_button.setEnabled(self._is_selected_style_custom())

    def _update_provider_visibility(self) -> None:
        translator_key = normalize_translator_name(self.translator_input.currentText())
        self.gemini_provider_widget.setVisible(translator_key == "gemini")
        self.local_llm_provider_widget.setVisible(translator_key == "local_llm")
        self.deepseek_provider_widget.setVisible(translator_key == "deepseek")
        self.openai_provider_widget.setVisible(translator_key == "openai_compatible")

        if translator_key == "openai_compatible":
            self.provider_hint_label.setText(
                "OpenAI Compatible uses a /chat/completions endpoint and can connect to OpenAI, OpenRouter, LM Studio, LocalAI, or another compatible server."
            )
        elif translator_key in {"google", "nllb", "baidu", "bing"}:
            self.provider_hint_label.setText("This provider may ignore Prompt Studio style instructions.")
        else:
            self.provider_hint_label.setText("")

    def _refresh_prompt_preview(self) -> None:
        config = self.config()
        prompt_result = config.prompt_build_result
        self.style_description_label.setText(prompt_result.style_description or "No style description available.")
        self.prompt_preview_input.setPlainText(prompt_result.prompt_preview)
        self.output_contract_input.setPlainText(prompt_result.output_contract)
        translator_key = config.translator_key
        if translator_key in {"google", "nllb", "baidu", "bing"}:
            self.prompt_provider_note_label.setText("This provider may ignore prompt/style instructions.")
            self.prompt_provider_note_label.setVisible(True)
        else:
            self.prompt_provider_note_label.setText("")
            self.prompt_provider_note_label.setVisible(False)

    def _reset_prompt_to_built_in(self) -> None:
        self.prompt_mode_input.setCurrentIndex(max(self.prompt_mode_input.findData(PROMPT_MODE_BUILT_IN_PLUS_USER), 0))
        self._rebuild_style_dropdown(self._last_built_in_style_id or DEFAULT_PROMPT_STYLE_ID)
        self.full_custom_prompt_input.clear()
        self._update_prompt_mode_visibility()
        self._refresh_prompt_preview()
        self.message_emitted.emit("Prompt Studio reset to the selected built-in style.")

    def _save_current_prompt_as_custom_style(self) -> None:
        prompt_mode = self.current_prompt_mode()
        if prompt_mode == PROMPT_MODE_BUILT_IN_PLUS_USER:
            current_style = self._current_prompt_style()
            base_instructions = str(current_style.get("instructions", "") or "").strip()
            extra_instructions = self.user_instructions_input.toPlainText().strip()
            instructions = base_instructions
            if extra_instructions:
                instructions = (
                    f"{instructions}\n\nUser extra instructions:\n{extra_instructions}".strip()
                    if instructions
                    else extra_instructions
                )
        else:
            instructions = self.full_custom_prompt_input.toPlainText().strip()

        if not instructions:
            self.error_emitted.emit("Prompt Studio", "There is no prompt text to save as a custom style.")
            return

        style_name, accepted = QInputDialog.getText(
            self,
            "Save Custom Style",
            "Custom style name:",
            text="My Custom Style",
        )
        if not accepted:
            return

        try:
            new_style = create_custom_style(
                style_name,
                instructions,
                description=str(self.style_description_label.text() or "").strip(),
                existing_styles=self._custom_styles,
            )
        except Exception as exc:
            self.error_emitted.emit("Custom style error", str(exc))
            return

        self._custom_styles = delete_custom_style(self._custom_styles, str(new_style.get("id", "") or ""))
        self._custom_styles.append(new_style)
        self._rebuild_style_dropdown(str(new_style.get("id", "") or DEFAULT_PROMPT_STYLE_ID))
        self._refresh_prompt_preview()
        self.message_emitted.emit(f"Saved custom style: {new_style.get('display_name', 'Custom Style')}")

    def _delete_selected_custom_style(self) -> None:
        if not self._is_selected_style_custom():
            return
        style_id = self.current_prompt_style_id()
        self._custom_styles = delete_custom_style(self._custom_styles, style_id)
        self._rebuild_style_dropdown(self._last_built_in_style_id or DEFAULT_PROMPT_STYLE_ID)
        self._refresh_prompt_preview()
        self.message_emitted.emit("Deleted the selected custom style.")

    def _on_current_cell_changed(self, current_row: int, _current_column: int, previous_row: int, _previous_column: int) -> None:
        if self._selection_guard:
            return
        if current_row < 0:
            return
        if self._editor_row == current_row:
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

    def _load_editor_for_row(self, row_index: int | None) -> None:
        self._editor_row = row_index
        self._refresh_editor_view()

    def _refresh_editor_view(self) -> None:
        if self._editor_row is None or self._editor_row < 0 or self._editor_row >= len(self._items):
            self._set_editor_enabled(False)
            self.crop_preview_panel.clear("Select a translation item to preview the matching OCR crop.")
            return

        item = self._items[self._editor_row]
        item_id = int(item.get("id", self._editor_row))
        ocr_item_id = int(item.get("ocr_item_id", item_id))
        kind = str(item.get("kind", "") or "-")
        status = str(item.get("status", "") or "-")
        bbox = self._format_bbox(item.get("bbox"))
        ocr_bbox = self._format_bbox(item.get("ocr_bbox"))
        error_text = str(item.get("error", "") or "").strip()
        needs_retranslate = bool(item.get("needs_retranslate", False))

        details = [
            f"Item {item_id}",
            f"OCR Item: {ocr_item_id}",
            f"Kind: {kind}",
            f"Status: {status}",
            f"BBox: {bbox}",
            f"OCR BBox: {ocr_bbox}",
        ]
        if error_text:
            details.append(f"Error: {error_text}")
        detail_text = "\n".join(details)

        ocr_item = self._ocr_items_by_id.get(ocr_item_id)
        crop_path = ocr_item.get("crop_path") if isinstance(ocr_item, dict) else None

        self._set_editor_enabled(True)
        self.editor_details_label.setText(detail_text)
        self.source_text_editor.set_loaded_text(
            str(item.get("source_text", "") or ""),
            status_text="Edit the source OCR text below.",
        )
        self.translated_text_editor.set_loaded_text(
            str(item.get("translated_text", "") or ""),
            status_text="Edit the translated text below.",
        )
        self.crop_preview_panel.set_crop(crop_path, details=detail_text)
        self.needs_retranslate_label.setText(
            "Source edited; re-translate recommended."
            if needs_retranslate and str(item.get("translated_text", "") or "").strip()
            else ""
        )
        self.needs_retranslate_label.setVisible(bool(self.needs_retranslate_label.text()))
        self._update_editor_button_state(False)

    def _set_editor_enabled(self, enabled: bool) -> None:
        active = bool(enabled and self._actions_enabled)
        if enabled:
            self.source_text_editor.set_enabled_for_item(active, message=self.source_text_editor.status_label.text())
            self.translated_text_editor.set_enabled_for_item(active, message=self.translated_text_editor.status_label.text())
        else:
            self.editor_details_label.setText("Select a translation item to edit its source and translated text.")
            self.editor_dirty_label.setText("Saved")
            self.needs_retranslate_label.setText("")
            self.needs_retranslate_label.setVisible(False)
            self.source_text_editor.set_enabled_for_item(False, message="No item selected.")
            self.translated_text_editor.set_enabled_for_item(False, message="No item selected.")
            self.source_text_editor.set_loaded_text("", status_text="No item selected.")
            self.translated_text_editor.set_loaded_text("", status_text="No item selected.")
        self.editor_previous_button.setEnabled(bool(active and self._editor_row not in (None, 0)))
        self.editor_next_button.setEnabled(
            bool(active and self._editor_row is not None and self._editor_row < len(self._items) - 1)
        )
        self.editor_revert_button.setEnabled(False)
        self.editor_save_source_button.setEnabled(False)
        self.editor_save_translation_button.setEnabled(False)
        self.editor_save_both_button.setEnabled(False)

    def _update_editor_button_state(self, _dirty: bool) -> None:
        has_item = self._actions_enabled and self._editor_row is not None and 0 <= self._editor_row < len(self._items)
        source_dirty = self.source_text_editor.is_dirty()
        translation_dirty = self.translated_text_editor.is_dirty()
        any_dirty = source_dirty or translation_dirty

        if source_dirty and translation_dirty:
            dirty_label = "Unsaved source and translation changes"
        elif source_dirty:
            dirty_label = "Unsaved source OCR changes"
        elif translation_dirty:
            dirty_label = "Unsaved translation changes"
        else:
            dirty_label = "Saved"

        self.editor_dirty_label.setText(dirty_label)
        self.editor_previous_button.setEnabled(bool(has_item and self._editor_row not in (None, 0)))
        self.editor_next_button.setEnabled(
            bool(has_item and self._editor_row is not None and self._editor_row < len(self._items) - 1)
        )
        self.editor_revert_button.setEnabled(bool(has_item and any_dirty))
        self.editor_save_source_button.setEnabled(bool(has_item and source_dirty))
        self.editor_save_translation_button.setEnabled(bool(has_item and translation_dirty))
        self.editor_save_both_button.setEnabled(bool(has_item and any_dirty))

    def _save_changes(self, *, save_source: bool, save_translation: bool) -> bool:
        if self._cache_path is None:
            self.error_emitted.emit("Translation JSON missing", "Reload translation items before saving.")
            return False
        if self._editor_row is None or self._editor_row < 0 or self._editor_row >= len(self._items):
            self.error_emitted.emit("No translation item selected", "Select a translation item before saving.")
            return False

        item = self._items[self._editor_row]
        item_id = int(item.get("id", self._editor_row))
        ocr_item_id = int(item.get("ocr_item_id", item_id))
        source_dirty = save_source and self.source_text_editor.is_dirty()
        translation_dirty = save_translation and self.translated_text_editor.is_dirty()
        if not source_dirty and not translation_dirty:
            return True

        warning_messages: list[str] = []
        try:
            translation_payload = None
            if source_dirty:
                translation_payload = update_translation_item_source_text(
                    self._cache_path,
                    text=self.source_text_editor.text(),
                    ocr_item_id=ocr_item_id,
                    item_id=item_id,
                    manually_edited=True,
                )
            if translation_dirty:
                translation_payload = update_translation_item_translated_text(
                    self._cache_path,
                    item_id,
                    self.translated_text_editor.text(),
                    manually_edited=True,
                )
            if translation_payload is None:
                return True
        except Exception as exc:
            self.error_emitted.emit("Failed to save translation text", str(exc))
            return False

        ocr_payload: dict[str, Any] | None = None
        if source_dirty:
            if self._ocr_cache_path is not None and self._ocr_cache_path.exists():
                try:
                    ocr_payload = update_ocr_item_text(
                        self._ocr_cache_path,
                        ocr_item_id,
                        self.source_text_editor.text(),
                        manually_edited=True,
                    )
                except Exception as exc:
                    warning_messages.append(
                        f"OCR cache update skipped for item {ocr_item_id}: {exc}"
                    )
            else:
                warning_messages.append(
                    "OCR cache not found; updated translation source text only."
                )

        self.set_items(translation_payload.get("items", []), self._cache_path)
        if ocr_payload is not None:
            self.set_ocr_context(self._project_root, ocr_payload, self._ocr_cache_path)
        target_row = self._row_for_item_id(item_id)
        if target_row is not None:
            self._load_editor_for_row(target_row)

        message_parts: list[str] = []
        if source_dirty and translation_dirty:
            message_parts.append(f"Saved source OCR and translation for item {item_id}.")
        elif source_dirty:
            message_parts.append(f"Saved source OCR text for item {item_id}.")
        elif translation_dirty:
            message_parts.append(f"Saved translated text for item {item_id}.")
        message_text = " ".join(message_parts).strip()
        if message_text:
            self.message_emitted.emit(message_text)
        for warning in warning_messages:
            self.warning_emitted.emit(warning)

        self.cache_updated.emit(
            {
                "stage": "translation",
                "translation_cache_path": str(self._cache_path),
                "translation_data": translation_payload,
                "ocr_cache_path": str(self._ocr_cache_path) if self._ocr_cache_path is not None else "",
                "ocr_data": ocr_payload,
                "message": message_text,
                "warnings": warning_messages,
            }
        )
        return True

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

    @staticmethod
    def _display_text(value: Any, limit: int = 72) -> str:
        text = " ".join(str(value or "").split())
        if len(text) <= limit:
            return text
        return f"{text[: limit - 1]}…"

    @staticmethod
    def _format_bbox(bbox: Any) -> str:
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return "-"
        return f"[{int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}]"


__all__ = ["TranslationPanel"]
