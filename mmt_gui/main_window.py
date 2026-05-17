"""Modern PyQt6 workbench window for the desktop app."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any

from PyQt6.QtCore import QByteArray, Qt, QThreadPool, QTimer
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QMainWindow,
    QMenu,
    QMessageBox,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from mmt_core import (
    DEFAULT_OCR_PROVIDER,
    ExportConfig,
    LlamaServerManager,
    OCRConfig,
    OCR_PROVIDER_CHROME_LENS,
    PROCESS_WORKFLOW_STAGE_BY_STEP,
    ProcessPipelineResult,
    RenderConfig,
    TranslationConfig,
    canon_item_bbox,
    ensure_canon_state,
    get_active_canon_items,
    load_detection_edit_items,
    detection_json_path,
    exclude_ocr_item,
    exclude_render_item,
    get_lama_model_manager,
    inpaint_image_path,
    inpaint_json_path,
    inpaint_preview_mask_path,
    load_detection_json,
    load_inpaint_json,
    load_ocr_edit_items,
    load_ocr_json,
    load_render_edit_items,
    load_render_json,
    save_detection_edit_items,
    save_ocr_edit_items,
    save_render_edit_items,
    load_translation_json,
    ocr_json_path,
    render_image_path,
    render_json_path,
    resolve_canon_item_for_stage_item,
    resolve_font_path,
    save_detection_json,
    summarize_ocr_edit_state,
    summarize_ocr_items,
    summarize_render_edit_state,
    summarize_render_json,
    summarize_translation_json,
    translation_json_path,
    validate_ocr_provider_config,
    validate_translation_config,
)
from mmt_core.crash_logging import set_project_log_dir, write_crash_breadcrumb

from . import APP_NAME
from .app_settings import AppSettings
from .page_status import get_page_workflow_status
from .project import MangaProject, PROJECT_FILENAME, remove_page_from_project
from .preview_controller import PreviewController, PreviewPreferences
from .services import ServiceCommandHandle, ServiceManager, ServiceStatusSnapshot
from .stages import (
    ConfigPanel,
    DetectionPanel,
    ExportPanel,
    InpaintPanel,
    OCRPanel,
    ProcessPanel,
    ProjectPanel,
    RenderPanel,
    TranslationPanel,
)
from .styles import DEFAULT_THEME, ThemeManager
from .window_layout import (
    DEFAULT_CONTENT_SPLITTER_SIZES,
    DEFAULT_MAIN_WINDOW_HEIGHT,
    DEFAULT_MAIN_WINDOW_MIN_HEIGHT,
    DEFAULT_MAIN_WINDOW_MIN_WIDTH,
    DEFAULT_MAIN_WINDOW_WIDTH,
    DEFAULT_WORKSPACE_SPLITTER_SIZES,
    WINDOW_LAYOUT_VERSION,
    clamp_window_geometry,
)
from .widgets import (
    STAGE_ORDER,
    AppHeader,
    ImagePreviewWidget,
    LeftToolBar,
    LogPanel,
    PageFilmstripWidget,
    StageSubToolBar,
    StageToolButton,
    StageToolGroup,
    StartupOverlay,
    WorkflowTabs,
)
from .workers import (
    DetectionTask,
    DetectionWorkerResult,
    ExportTask,
    ExportWorkerResult,
    InpaintMaskTask,
    InpaintMaskWorkerResult,
    InpaintTask,
    InpaintWorkerResult,
    LamaModelTask,
    LamaModelTaskResult,
    LlamaServerTask,
    LlamaServerTaskResult,
    OCRInferenceTask,
    OCRInferenceWorkerResult,
    OCRPreparationTask,
    OCRPreparationWorkerResult,
    ProcessTask,
    RenderPreparationTask,
    RenderPreparationWorkerResult,
    RenderTask,
    RenderWorkerResult,
    TaskWorker,
    TranslationInitializationTask,
    TranslationInitializationWorkerResult,
    TranslationTask,
    TranslationWorkerResult,
    create_llama_server_worker,
)

IMAGE_FILTER = "Images (*.jpg *.jpeg *.png *.webp)"
PROJECT_FILTER = f"Project Files ({PROJECT_FILENAME});;JSON Files (*.json)"

SERVER_STATE_UNKNOWN = "Unknown"
SERVER_STATE_STOPPED = "Stopped"
SERVER_STATE_STARTING = "Starting"
SERVER_STATE_READY = "Ready"
SERVER_STATE_ERROR = "Error"

PREVIEW_SOURCE = "Source"
PREVIEW_DETECTION = "Detection Overlay"
PREVIEW_MASK = "Mask Overlay"
PREVIEW_INPAINT = "Inpaint Result"
PREVIEW_RENDER = "Render Result"

GLOBAL_PREVIEW_MODES = [
    PREVIEW_SOURCE,
    PREVIEW_DETECTION,
    PREVIEW_MASK,
    PREVIEW_INPAINT,
    PREVIEW_RENDER,
]


class MainWindow(QMainWindow):
    """Orchestrates the modern GUI shell around the tested pipeline stages."""

    def __init__(self) -> None:
        super().__init__()
        self.workspace_root = Path(__file__).resolve().parents[1]
        self.app_settings = AppSettings()
        self.theme_manager = ThemeManager(self.app_settings.string_value("workspace/theme", DEFAULT_THEME))
        self.preview_controller = PreviewController(
            PreviewPreferences(
                auto_preview_result=self.app_settings.bool_value("workspace/auto_preview_result", True),
                follow_batch_progress=self.app_settings.bool_value("workspace/follow_batch_progress", False),
            )
        )
        self._theme_change_in_progress = False

        self.current_project: MangaProject | None = None
        self.current_stage_key = "project"
        self._stage_display_names = {stage_key: stage_label for stage_key, stage_label, _icon in STAGE_ORDER}
        self._stage_subtool_groups: dict[str, list[StageToolGroup]] = {}
        self._process_stage_status = "missing"
        self._process_active_stage_key: str | None = None
        self._process_restore_page_relative_path: str | None = None
        self.current_detection_data: dict[str, Any] | None = None
        self._ocr_edit_items: list[dict[str, Any]] = []
        self.current_ocr_data: dict[str, Any] | None = None
        self.current_ocr_cache_path: Path | None = None
        self.current_translation_data: dict[str, Any] | None = None
        self.current_translation_cache_path: Path | None = None
        self.current_inpaint_data: dict[str, Any] | None = None
        self.current_inpaint_cache_path: Path | None = None
        self._render_edit_items: list[dict[str, Any]] = []
        self.current_render_data: dict[str, Any] | None = None
        self.current_render_cache_path: Path | None = None
        self.last_export_result: dict[str, Any] | None = None
        self.preview_detection_overlay_enabled = True
        self._programmatic_page_selection = False
        self._active_batch_follow_stage: str | None = None
        self._follow_batch_paused = False
        self._content_splitter: QSplitter | None = None
        self._workspace_splitter: QSplitter | None = None
        self._pending_filmstrip_scroll_value = 0
        self._page_status_cache: dict[str, str] = {}
        self._processing_page_relative_path: str | None = None
        self._has_unread_log_alert = False
        self._has_unread_error_alert = False
        self.recent_projects_menu: QMenu | None = None
        self.new_project_action = None
        self.open_project_action = None
        self.save_project_action = None
        self.import_images_action = None
        self.remove_current_page_action = None
        self.toggle_developer_log_action = None
        self.open_crash_logs_action = None
        self.developer_log_dock: QDockWidget | None = None
        self._busy_stages: set[str] = set()
        self._busy_stage_counts: dict[str, int] = {}
        self._active_process_worker: TaskWorker | ServiceCommandHandle | None = None
        self._process_cancel_requested = False
        self._service_statuses: dict[str, ServiceStatusSnapshot] = {}
        self.service_manager: ServiceManager | None = None
        self.startup_overlay: StartupOverlay | None = None

        self.thread_pool = QThreadPool.globalInstance()
        self._active_workers: list[TaskWorker | ServiceCommandHandle] = []
        set_project_log_dir(None)
        write_crash_breadcrumb("MainWindow initializing", workspace_root=str(self.workspace_root))

        default_server_url = "http://127.0.0.1:8080"
        default_model_path = self.workspace_root / "model" / "paddleocr_vl" / "model.gguf"
        default_mmproj_path = self.workspace_root / "model" / "paddleocr_vl" / "mmproj.gguf"
        default_llama_cpp_dir = self.workspace_root / "tools" / "llama.cpp"
        self.llama_server_manager = LlamaServerManager(
            server_url=default_server_url,
            model_path=default_model_path,
            mmproj_path=default_mmproj_path,
            llama_cpp_dir=default_llama_cpp_dir,
            gpu_layers=99,
            ctx_size=8192,
        )

        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(DEFAULT_MAIN_WINDOW_MIN_WIDTH, DEFAULT_MAIN_WINDOW_MIN_HEIGHT)
        self.resize(DEFAULT_MAIN_WINDOW_WIDTH, DEFAULT_MAIN_WINDOW_HEIGHT)

        self._build_ui()
        self._build_menu()

        status_bar = QStatusBar(self)
        status_bar.setSizeGripEnabled(False)
        status_bar.messageChanged.connect(self.header.set_status_text)
        self.setStatusBar(status_bar)
        self.statusBar().hide()

        inpaint_panel_settings = self.app_settings.panel_settings("inpaint")
        startup_inpaint_device = str(
            inpaint_panel_settings.get(
                "device",
                self.app_settings.string_value("services/inpaint_startup_device", ""),
            )
            or ""
        )
        self.service_manager = ServiceManager(
            workspace_root=self.workspace_root,
            startup_options={
                "preload_detection": self.app_settings.bool_value("services/preload_detection_on_startup", True),
                "preload_inpaint": self.app_settings.bool_value("services/preload_inpaint_on_startup", True),
                "preload_render": self.app_settings.bool_value("services/preload_render_on_startup", True),
                "inpaint_device": startup_inpaint_device,
            },
            parent=self,
        )
        self.service_manager.service_status_changed.connect(self._on_service_status_changed)
        self.service_manager.diagnostic_message.connect(
            lambda message, level: self.log(message, level=str(level or "info"))
        )
        self.header.set_service_status_text("Services: starting...")
        self.service_manager.start_services()
        self._show_startup_overlay()

        self.header.theme_changed.connect(self._on_theme_changed)
        self.left_toolbar.auto_preview_changed.connect(self._on_auto_preview_changed)
        self.left_toolbar.follow_batch_progress_changed.connect(self._on_follow_batch_progress_changed)
        if self.developer_log_dock is not None:
            self.developer_log_dock.visibilityChanged.connect(self._on_developer_log_visibility_changed)

        initial_theme = self.theme_manager.current_theme
        self.header.set_theme_name(initial_theme)
        self._apply_theme_safely(initial_theme, show_error=False)
        self._restore_panel_preferences()
        self._restore_window_settings()
        self.log(f"Crash logs folder: {self.workspace_root / 'logs'}")
        self._restore_workspace_state()
        self.statusBar().showMessage("Ready")

    def _build_ui(self) -> None:
        self.header = AppHeader()

        self.workflow_tabs = WorkflowTabs()
        self.workflow_tabs.stage_selected.connect(self._on_stage_selected)
        self.stage_subtool_bar = StageSubToolBar()

        self.left_toolbar = LeftToolBar()
        self.left_toolbar.preview_mode_changed.connect(lambda _mode: self._refresh_preview_for_current_page())
        self.left_toolbar.previous_page_requested.connect(lambda: self._navigate_to_page_offset(-1))
        self.left_toolbar.next_page_requested.connect(lambda: self._navigate_to_page_offset(1))
        self.left_toolbar.first_page_requested.connect(lambda: self._navigate_to_page_index(0))
        self.left_toolbar.last_page_requested.connect(self._navigate_to_last_page)
        self.left_toolbar.developer_log_toggled.connect(self.toggle_developer_log)

        self.image_preview = ImagePreviewWidget()
        self.image_preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_preview.setMinimumHeight(0)
        self.left_toolbar.fit_requested.connect(self.image_preview.fit_to_view)
        self.left_toolbar.zoom_in_requested.connect(self.image_preview.zoom_in)
        self.left_toolbar.zoom_out_requested.connect(self.image_preview.zoom_out)
        self.left_toolbar.reset_zoom_requested.connect(self.image_preview.reset_zoom)

        preview_frame = QFrame()
        preview_frame.setObjectName("PreviewSurface")
        preview_frame.setMinimumHeight(0)
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(8, 8, 8, 8)
        preview_layout.setSpacing(0)
        preview_layout.addWidget(self.image_preview, 1)

        self.project_panel = ProjectPanel()
        self.process_panel = ProcessPanel()
        self.detection_panel = DetectionPanel()
        self.ocr_panel = OCRPanel()
        self.translation_panel = TranslationPanel()
        self.inpaint_panel = InpaintPanel()
        self.render_panel = RenderPanel(self.workspace_root)
        self.export_panel = ExportPanel()
        self.config_panel = ConfigPanel(
            detection_panel=self.detection_panel,
            ocr_panel=self.ocr_panel,
            translation_panel=self.translation_panel,
            inpaint_panel=self.inpaint_panel,
            render_panel=self.render_panel,
        )
        self.detection_panel.simplify_for_config_stage()
        self.ocr_panel.simplify_for_config_stage()
        self.translation_panel.simplify_for_config_stage()
        self.inpaint_panel.simplify_for_config_stage()
        self.render_panel.simplify_for_config_stage()
        self.export_panel.export_source_input.currentIndexChanged.connect(self._refresh_stage_statuses)
        self.export_panel.output_folder_input.textChanged.connect(self._refresh_stage_statuses)
        self.ocr_panel.items_table.itemSelectionChanged.connect(self._refresh_stage_statuses)
        self.translation_panel.items_table.itemSelectionChanged.connect(self._refresh_stage_statuses)

        self.stage_stack = QStackedWidget()
        self.stage_stack.setMinimumHeight(0)
        self.stage_panels: dict[str, QWidget] = {
            "process": self.process_panel,
            "project": self.project_panel,
            "config": self.config_panel,
            "detection": self.detection_panel,
            "ocr": self.ocr_panel,
            "translation": self.translation_panel,
            "inpaint": self.inpaint_panel,
            "render": self.render_panel,
            "export": self.export_panel,
        }
        self.stage_indices: dict[str, int] = {}
        for stage_key, panel in self.stage_panels.items():
            self.stage_indices[stage_key] = self.stage_stack.addWidget(panel)

        self._configure_stage_subtool_bar()
        self._connect_panel_signals()
        self.ocr_panel.set_server_values(self.llama_server_manager)

        self._content_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._content_splitter.setMinimumHeight(0)
        self._content_splitter.addWidget(preview_frame)
        self._content_splitter.addWidget(self.stage_stack)
        self._content_splitter.setStretchFactor(0, 1)
        self._content_splitter.setStretchFactor(1, 0)
        self._content_splitter.setSizes(list(DEFAULT_CONTENT_SPLITTER_SIZES))

        body_widget = QWidget()
        body_widget.setMinimumHeight(0)
        body_layout = QHBoxLayout(body_widget)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(10)
        self.left_toolbar.setMinimumHeight(0)
        body_layout.addWidget(self.left_toolbar)
        body_layout.addWidget(self._content_splitter, 1)

        self.page_filmstrip = PageFilmstripWidget()
        self.page_filmstrip.page_selected.connect(self._on_page_selected)
        self.page_filmstrip.page_order_changed.connect(self._on_page_order_changed)
        self.page_filmstrip.thumbnail_load_failed.connect(lambda message: self.log(message, level="warning"))
        self.page_filmstrip.set_reorder_enabled(True)

        self.log_panel = LogPanel()
        self.log_panel.set_dock_mode(True)
        self.developer_log_dock = QDockWidget("Developer Log", self)
        self.developer_log_dock.setObjectName("DeveloperLogDock")
        self.developer_log_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.developer_log_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.developer_log_dock.setWidget(self.log_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.developer_log_dock)
        self.developer_log_dock.hide()

        root_widget = QWidget()
        root_layout = QVBoxLayout(root_widget)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(10)
        self._workspace_splitter = QSplitter(Qt.Orientation.Vertical)
        self._workspace_splitter.addWidget(body_widget)
        self._workspace_splitter.addWidget(self.page_filmstrip)
        self._workspace_splitter.setStretchFactor(0, 1)
        self._workspace_splitter.setStretchFactor(1, 1)
        self._workspace_splitter.setCollapsible(0, False)
        self._workspace_splitter.setCollapsible(1, False)
        self._workspace_splitter.setSizes(list(DEFAULT_WORKSPACE_SPLITTER_SIZES))
        root_layout.addWidget(self.header)
        root_layout.addWidget(self.workflow_tabs)
        root_layout.addWidget(self.stage_subtool_bar)
        root_layout.addWidget(self._workspace_splitter, 1)
        self.setCentralWidget(root_widget)
        self.startup_overlay = StartupOverlay(root_widget)
        self.startup_overlay.hide()

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        assert isinstance(file_menu, QMenu)

        self.new_project_action = file_menu.addAction("New Project")
        self.new_project_action.triggered.connect(self.new_project)

        self.open_project_action = file_menu.addAction("Open Project")
        self.open_project_action.triggered.connect(self.open_project)

        self.recent_projects_menu = file_menu.addMenu("Recent Projects")
        self._refresh_recent_projects_menu()

        self.save_project_action = file_menu.addAction("Save Project")
        self.save_project_action.triggered.connect(self.save_project)

        file_menu.addSeparator()

        self.import_images_action = file_menu.addAction("Import Images")
        self.import_images_action.triggered.connect(self.import_images)

        self.remove_current_page_action = file_menu.addAction("Remove Current Page")
        self.remove_current_page_action.triggered.connect(self.remove_current_page_from_project)

        file_menu.addSeparator()

        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)

        view_menu = self.menuBar().addMenu("&View")
        assert isinstance(view_menu, QMenu)

        self.toggle_developer_log_action = view_menu.addAction("Developer Log")
        self.toggle_developer_log_action.setCheckable(True)
        self.toggle_developer_log_action.setShortcut("Ctrl+Shift+L")
        self.toggle_developer_log_action.triggered.connect(self.toggle_developer_log)

        help_menu = self.menuBar().addMenu("&Help")
        assert isinstance(help_menu, QMenu)

        self.open_crash_logs_action = help_menu.addAction("Open Crash Logs Folder")
        self.open_crash_logs_action.triggered.connect(self.open_crash_logs_folder)

    @staticmethod
    def _stage_tool_button(button: Any, label: str | None = None) -> StageToolButton:
        return StageToolButton(button=button, label=label)

    def _configure_stage_subtool_bar(self) -> None:
        self._stage_subtool_groups = {
            "process": [],
            "project": [
                StageToolGroup(
                    "Project",
                    [
                        self._stage_tool_button(self.project_panel.new_project_button),
                        self._stage_tool_button(self.project_panel.open_project_button),
                        self._stage_tool_button(self.project_panel.save_project_button),
                        self._stage_tool_button(self.project_panel.import_images_button),
                    ],
                ),
                StageToolGroup(
                    "Danger",
                    [self._stage_tool_button(self.project_panel.remove_current_page_button)],
                ),
            ],
            "config": [],
            "detection": [
                StageToolGroup(
                    "Normal",
                    [
                        self._stage_tool_button(self.detection_panel.run_selected_button, "Detect Page"),
                        self._stage_tool_button(self.detection_panel.reload_button),
                    ],
                ),
                StageToolGroup(
                    "Preview",
                    [self._stage_tool_button(self.detection_panel.clear_overlay_button)],
                ),
                StageToolGroup(
                    "Re-run / Force",
                    [
                        self._stage_tool_button(self.detection_panel.rerun_selected_button, "Re-detect Page"),
                    ],
                ),
            ],
            "ocr": [
                StageToolGroup(
                    "Normal",
                    [
                        self._stage_tool_button(self.ocr_panel.prepare_selected_button),
                        self._stage_tool_button(self.ocr_panel.run_selected_button),
                        self._stage_tool_button(self.ocr_panel.run_selected_items_button),
                        self._stage_tool_button(self.ocr_panel.save_text_button),
                        self._stage_tool_button(self.ocr_panel.reload_button),
                    ],
                ),
                StageToolGroup(
                    "Re-run / Force",
                    [
                        self._stage_tool_button(self.ocr_panel.reprepare_selected_button),
                        self._stage_tool_button(self.ocr_panel.rerun_selected_button),
                        self._stage_tool_button(self.ocr_panel.rerun_selected_items_button),
                    ],
                ),
            ],
            "translation": [
                StageToolGroup(
                    "Normal",
                    [
                        self._stage_tool_button(self.translation_panel.initialize_selected_button),
                        self._stage_tool_button(self.translation_panel.run_selected_button),
                        self._stage_tool_button(self.translation_panel.run_selected_items_button),
                        self._stage_tool_button(self.translation_panel.save_text_button, "Save Text"),
                        self._stage_tool_button(self.translation_panel.reload_button),
                    ],
                ),
                StageToolGroup(
                    "Re-run / Force",
                    [
                        self._stage_tool_button(self.translation_panel.reinitialize_selected_button),
                        self._stage_tool_button(self.translation_panel.rerun_selected_button),
                        self._stage_tool_button(self.translation_panel.rerun_selected_items_button),
                    ],
                ),
            ],
            "inpaint": [
                StageToolGroup(
                    "Normal",
                    [
                        self._stage_tool_button(self.inpaint_panel.prepare_selected_button),
                        self._stage_tool_button(self.inpaint_panel.run_selected_button),
                        self._stage_tool_button(self.inpaint_panel.reload_button),
                    ],
                ),
                StageToolGroup(
                    "Preview",
                    [self._stage_tool_button(self.inpaint_panel.clear_preview_button, "Clear Preview")],
                ),
                StageToolGroup(
                    "Re-run / Force",
                    [
                        self._stage_tool_button(self.inpaint_panel.reprepare_selected_button),
                        self._stage_tool_button(self.inpaint_panel.rerun_selected_button),
                    ],
                ),
            ],
            "render": [
                StageToolGroup(
                    "Normal",
                    [
                        self._stage_tool_button(self.render_panel.prepare_selected_button, "Prepare Render"),
                        self._stage_tool_button(self.render_panel.run_selected_button),
                        self._stage_tool_button(self.render_panel.reload_button),
                    ],
                ),
                StageToolGroup(
                    "Preview",
                    [self._stage_tool_button(self.render_panel.clear_preview_button, "Clear Preview")],
                ),
                StageToolGroup(
                    "Re-run / Force",
                    [
                        self._stage_tool_button(self.render_panel.reprepare_selected_button, "Re-prepare Render"),
                        self._stage_tool_button(self.render_panel.rerun_selected_button),
                    ],
                ),
            ],
            "export": [],
        }

        for panel in (
            self.project_panel,
            self.detection_panel,
            self.ocr_panel,
            self.translation_panel,
            self.inpaint_panel,
            self.render_panel,
        ):
            panel.detach_widget(getattr(panel, "actions_section", None))

    def _update_stage_subtool_bar(self, stage_key: str) -> None:
        self.stage_subtool_bar.set_stage_groups(
            self._stage_display_names.get(stage_key, stage_key.title()),
            self._stage_subtool_groups.get(stage_key, []),
        )

    def _connect_panel_signals(self) -> None:
        self.project_panel.new_project_requested.connect(self.new_project)
        self.project_panel.open_project_requested.connect(self.open_project)
        self.project_panel.save_project_requested.connect(self.save_project)
        self.project_panel.import_images_requested.connect(self.import_images)
        self.project_panel.remove_current_page_requested.connect(self.remove_current_page_from_project)

        self.process_panel.process_current_requested.connect(self.process_current_page)
        self.process_panel.reprocess_current_requested.connect(lambda: self.process_current_page(force=True))
        self.process_panel.process_chapter_requested.connect(self.process_chapter)
        self.process_panel.reprocess_chapter_requested.connect(lambda: self.process_chapter(force=True))
        self.process_panel.cancel_requested.connect(self.stop_process)
        self.process_panel.detect_all_requested.connect(self.run_detection_for_all_pages)
        self.process_panel.redetect_all_requested.connect(lambda: self.run_detection_for_all_pages(force=True))
        self.process_panel.prepare_ocr_all_requested.connect(self.prepare_ocr_items_for_all_pages)
        self.process_panel.reprepare_ocr_all_requested.connect(
            lambda: self.prepare_ocr_items_for_all_pages(force=True)
        )
        self.process_panel.run_ocr_all_requested.connect(self.run_ocr_for_all_pages)
        self.process_panel.rerun_ocr_all_requested.connect(lambda: self.run_ocr_for_all_pages(force=True))
        self.process_panel.initialize_translation_all_requested.connect(self.initialize_translation_for_all_pages)
        self.process_panel.reinitialize_translation_all_requested.connect(
            lambda: self.initialize_translation_for_all_pages(force=True)
        )
        self.process_panel.translate_all_requested.connect(self.run_translation_for_all_pages)
        self.process_panel.retranslate_all_requested.connect(lambda: self.run_translation_for_all_pages(force=True))
        self.process_panel.prepare_mask_all_requested.connect(self.prepare_inpaint_mask_for_all_pages)
        self.process_panel.reprepare_mask_all_requested.connect(
            lambda: self.prepare_inpaint_mask_for_all_pages(force=True)
        )
        self.process_panel.inpaint_all_requested.connect(self.run_inpaint_for_all_pages)
        self.process_panel.reinpaint_all_requested.connect(lambda: self.run_inpaint_for_all_pages(force=True))
        self.process_panel.prepare_render_all_requested.connect(self.prepare_render_for_all_pages)
        self.process_panel.reprepare_render_all_requested.connect(
            lambda: self.prepare_render_for_all_pages(force=True)
        )
        self.process_panel.render_all_requested.connect(self.run_render_for_all_pages)
        self.process_panel.rerender_all_requested.connect(lambda: self.run_render_for_all_pages(force=True))

        self.detection_panel.run_selected_requested.connect(self.run_detection_for_selected_page)
        self.detection_panel.rerun_selected_requested.connect(lambda: self.run_detection_for_selected_page(force=True))
        self.detection_panel.run_all_requested.connect(self.run_detection_for_all_pages)
        self.detection_panel.rerun_all_requested.connect(lambda: self.run_detection_for_all_pages(force=True))
        self.detection_panel.reload_requested.connect(self.reload_cached_detection)
        self.detection_panel.clear_overlay_requested.connect(self.clear_detection_overlay)
        self.detection_panel.edit_mode_toggled.connect(self._on_detection_edit_mode_toggled)
        self.detection_panel.box_type_changed.connect(self._on_detection_box_type_changed)
        self.detection_panel.create_box_toggled.connect(self._on_detection_create_box_toggled)
        self.detection_panel.save_box_edits_requested.connect(self.save_detection_box_edits)
        self.detection_panel.cancel_box_edits_requested.connect(self.cancel_detection_box_edits)
        self.detection_panel.exclude_selected_requested.connect(self.exclude_selected_detection_box)
        self.detection_panel.restore_selected_requested.connect(self.restore_selected_detection_box)
        self.detection_panel.show_excluded_toggled.connect(self._on_detection_show_excluded_toggled)
        self.detection_panel.reload_boxes_requested.connect(self.reload_detection_boxes_from_cache)

        self.image_preview.editable_box_selection_changed.connect(self._on_editable_box_selection_changed)
        self.image_preview.editable_box_changed.connect(self._on_editable_box_changed)
        self.image_preview.editable_box_dirty_changed.connect(self._on_editable_box_dirty_changed)
        self.image_preview.editable_box_error.connect(self.log)

        self.ocr_panel.start_server_requested.connect(self.start_llama_server)
        self.ocr_panel.check_server_requested.connect(self.check_llama_server)
        self.ocr_panel.stop_server_requested.connect(self.stop_llama_server)
        self.ocr_panel.prepare_selected_requested.connect(self.prepare_ocr_items_for_selected_page)
        self.ocr_panel.reprepare_selected_requested.connect(lambda: self.prepare_ocr_items_for_selected_page(force=True))
        self.ocr_panel.prepare_all_requested.connect(self.prepare_ocr_items_for_all_pages)
        self.ocr_panel.reprepare_all_requested.connect(lambda: self.prepare_ocr_items_for_all_pages(force=True))
        self.ocr_panel.run_selected_requested.connect(self.run_ocr_for_selected_page)
        self.ocr_panel.rerun_selected_requested.connect(lambda: self.run_ocr_for_selected_page(force=True))
        self.ocr_panel.run_all_requested.connect(self.run_ocr_for_all_pages)
        self.ocr_panel.rerun_all_requested.connect(lambda: self.run_ocr_for_all_pages(force=True))
        self.ocr_panel.run_selected_items_requested.connect(self.run_ocr_for_selected_items)
        self.ocr_panel.rerun_selected_items_requested.connect(lambda: self.run_ocr_for_selected_items(force=True))
        self.ocr_panel.reload_requested.connect(self.reload_cached_ocr_items)
        self.ocr_panel.save_text_requested.connect(self.save_edited_ocr_text)
        self.ocr_panel.box_edit_mode_toggled.connect(self._on_ocr_box_edit_mode_toggled)
        self.ocr_panel.box_field_changed.connect(self._on_ocr_box_field_changed)
        self.ocr_panel.save_box_edits_requested.connect(self.save_ocr_box_edits)
        self.ocr_panel.cancel_box_edits_requested.connect(self.cancel_ocr_box_edits)
        self.ocr_panel.exclude_selected_box_requested.connect(self.exclude_selected_ocr_item)
        self.ocr_panel.restore_selected_box_requested.connect(self.restore_selected_ocr_item)
        self.ocr_panel.show_excluded_items_toggled.connect(self._on_ocr_show_excluded_toggled)
        self.ocr_panel.reload_box_cache_requested.connect(self.reload_ocr_boxes_from_cache)
        self.ocr_panel.current_item_changed.connect(self._on_ocr_table_item_changed)
        self.ocr_panel.ocr_provider_changed.connect(self._on_ocr_provider_changed)
        self.ocr_panel.cache_updated.connect(self._on_ocr_editor_cache_updated)
        self.ocr_panel.error_emitted.connect(self.show_error)
        self.ocr_panel.warning_emitted.connect(self.log)
        self.ocr_panel.message_emitted.connect(self.log)

        self.translation_panel.initialize_selected_requested.connect(
            self.initialize_translation_for_selected_page
        )
        self.translation_panel.reinitialize_selected_requested.connect(
            lambda: self.initialize_translation_for_selected_page(force=True)
        )
        self.translation_panel.initialize_all_requested.connect(self.initialize_translation_for_all_pages)
        self.translation_panel.reinitialize_all_requested.connect(
            lambda: self.initialize_translation_for_all_pages(force=True)
        )
        self.translation_panel.run_selected_requested.connect(self.run_translation_for_selected_page)
        self.translation_panel.rerun_selected_requested.connect(
            lambda: self.run_translation_for_selected_page(force=True)
        )
        self.translation_panel.run_all_requested.connect(self.run_translation_for_all_pages)
        self.translation_panel.rerun_all_requested.connect(lambda: self.run_translation_for_all_pages(force=True))
        self.translation_panel.run_selected_items_requested.connect(self.run_translation_for_selected_items)
        self.translation_panel.rerun_selected_items_requested.connect(
            lambda: self.run_translation_for_selected_items(force=True)
        )
        self.translation_panel.reload_requested.connect(self.reload_cached_translation)
        self.translation_panel.save_text_requested.connect(self.save_edited_translation_text)
        self.translation_panel.cache_updated.connect(self._on_translation_editor_cache_updated)
        self.translation_panel.error_emitted.connect(self.show_error)
        self.translation_panel.warning_emitted.connect(self.log)
        self.translation_panel.message_emitted.connect(self.log)

        self.inpaint_panel.prepare_selected_requested.connect(self.prepare_inpaint_mask_for_selected_page)
        self.inpaint_panel.reprepare_selected_requested.connect(
            lambda: self.prepare_inpaint_mask_for_selected_page(force=True)
        )
        self.inpaint_panel.prepare_all_requested.connect(self.prepare_inpaint_mask_for_all_pages)
        self.inpaint_panel.reprepare_all_requested.connect(
            lambda: self.prepare_inpaint_mask_for_all_pages(force=True)
        )
        self.inpaint_panel.run_selected_requested.connect(self.run_inpaint_for_selected_page)
        self.inpaint_panel.rerun_selected_requested.connect(lambda: self.run_inpaint_for_selected_page(force=True))
        self.inpaint_panel.run_all_requested.connect(self.run_inpaint_for_all_pages)
        self.inpaint_panel.rerun_all_requested.connect(lambda: self.run_inpaint_for_all_pages(force=True))
        self.inpaint_panel.reload_requested.connect(self.reload_cached_inpaint)
        self.inpaint_panel.clear_preview_requested.connect(self.clear_inpaint_preview)
        self.inpaint_panel.load_model_requested.connect(self.load_lama_model)
        self.inpaint_panel.unload_model_requested.connect(self.unload_lama_model)

        self.render_panel.prepare_selected_requested.connect(self.prepare_render_for_selected_page)
        self.render_panel.reprepare_selected_requested.connect(lambda: self.prepare_render_for_selected_page(force=True))
        self.render_panel.prepare_all_requested.connect(self.prepare_render_for_all_pages)
        self.render_panel.reprepare_all_requested.connect(lambda: self.prepare_render_for_all_pages(force=True))
        self.render_panel.run_selected_requested.connect(self.run_render_for_selected_page)
        self.render_panel.rerun_selected_requested.connect(lambda: self.run_render_for_selected_page(force=True))
        self.render_panel.run_all_requested.connect(self.run_render_for_all_pages)
        self.render_panel.rerun_all_requested.connect(lambda: self.run_render_for_all_pages(force=True))
        self.render_panel.reload_requested.connect(self.reload_cached_render)
        self.render_panel.clear_preview_requested.connect(self.clear_render_preview)
        self.render_panel.box_edit_mode_toggled.connect(self._on_render_box_edit_mode_toggled)
        self.render_panel.save_box_edits_requested.connect(self.save_render_box_edits)
        self.render_panel.cancel_box_edits_requested.connect(self.cancel_render_box_edits)
        self.render_panel.exclude_selected_box_requested.connect(self.exclude_selected_render_item)
        self.render_panel.restore_selected_box_requested.connect(self.restore_selected_render_item)
        self.render_panel.show_excluded_items_toggled.connect(self._on_render_show_excluded_toggled)
        self.render_panel.reload_box_cache_requested.connect(self.reload_render_boxes_from_cache)
        self.render_panel.current_item_changed.connect(self._on_render_table_item_changed)

        self.export_panel.browse_output_requested.connect(self.browse_export_output_folder)
        self.export_panel.export_current_requested.connect(self.export_current_page)
        self.export_panel.export_all_requested.connect(self.export_all_pages)
        self.export_panel.export_selected_requested.connect(self.export_selected_pages)
        self.export_panel.open_output_folder_requested.connect(self.open_export_output_folder)
        self.export_panel.reload_summary_requested.connect(self.reload_last_export_summary)

    def _refresh_recent_projects_menu(self) -> None:
        if self.recent_projects_menu is None:
            return

        self.recent_projects_menu.clear()
        recent_projects = self.app_settings.recent_projects()
        if not recent_projects:
            empty_action = self.recent_projects_menu.addAction("No Recent Projects")
            empty_action.setEnabled(False)
        else:
            for project_path in recent_projects:
                action = self.recent_projects_menu.addAction(project_path)
                action.triggered.connect(
                    lambda checked=False, path=project_path: self._open_recent_project(path)
                )
            self.recent_projects_menu.addSeparator()

        clear_action = self.recent_projects_menu.addAction("Clear Recent Projects")
        clear_action.triggered.connect(self._clear_recent_projects)

    def _remember_project_path(self, project_file: Path) -> None:
        normalized = str(project_file.resolve())
        self.app_settings.set_last_project_path(normalized)
        self.app_settings.push_recent_project(normalized)
        self._refresh_recent_projects_menu()

    def _clear_recent_projects(self) -> None:
        self.app_settings.clear_recent_projects()
        self._refresh_recent_projects_menu()
        self.log("Cleared recent projects.")

    def _open_recent_project(self, project_path: str) -> None:
        if not self._ensure_current_editor_changes_resolved():
            return
        if not project_path:
            return
        project_file = Path(project_path)
        if not project_file.exists():
            self.show_error("Recent project missing", f"Project file not found:\n{project_file}")
            cleaned = [path for path in self.app_settings.recent_projects() if path != project_path]
            self.app_settings.set_recent_projects(cleaned)
            self._refresh_recent_projects_menu()
            return
        self._open_project_path(project_file, show_error_dialog=True)

    def _open_project_path(self, project_file: Path, *, show_error_dialog: bool) -> bool:
        try:
            project = MangaProject.load(project_file)
        except Exception as exc:
            self.log(f"Failed to open project {project_file}: {exc}")
            if show_error_dialog:
                self.show_error("Failed to open project", str(exc))
            return False

        self.current_project = project
        set_project_log_dir(self.current_project.root_dir)
        self._process_stage_status = "missing"
        self._process_active_stage_key = None
        self.process_panel.reset_process_state(scope_text="Current page")
        self._remember_project_path(project.project_file)
        self._refresh_project_view()
        self._restore_current_project_selection()
        self.log(f"Opened project from {self.current_project.project_file}")
        self.statusBar().showMessage("Project loaded")
        return True

    def _restore_panel_preferences(self) -> None:
        try:
            self.left_toolbar.set_auto_preview_enabled(
                self.app_settings.bool_value("workspace/auto_preview_result", True)
            )
            self.left_toolbar.set_follow_batch_progress_enabled(
                self.app_settings.bool_value("workspace/follow_batch_progress", False)
            )
            self.preview_controller.set_preferences(
                auto_preview_result=self.left_toolbar.auto_preview_enabled(),
                follow_batch_progress=self.left_toolbar.follow_batch_progress_enabled(),
            )
            self.config_panel.apply_ocr_settings(self.app_settings.panel_settings("ocr"))
            self.config_panel.apply_translation_settings(self.app_settings.panel_settings("translation"))
            inpaint_settings = self.app_settings.panel_settings("inpaint")
            self.config_panel.apply_inpaint_settings(inpaint_settings)
            migration_message = self.inpaint_panel.consume_settings_migration_message()
            if migration_message:
                migrated_settings = self.config_panel.inpaint_settings_snapshot()
                self.app_settings.set_panel_settings("inpaint", migrated_settings)
                self.log(migration_message)
            self.config_panel.apply_render_settings(self.app_settings.panel_settings("render"))
            self.export_panel.apply_settings(self.app_settings.panel_settings("export"))
        except Exception as exc:
            self.log(f"Settings load failure: {exc}")

    def _restore_window_settings(self) -> None:
        try:
            layout_version = self.app_settings.int_value("window/layout_version", 0)
            layout_is_compatible = layout_version == WINDOW_LAYOUT_VERSION
            geometry = self.app_settings.bytes_value("window/geometry")
            state = self.app_settings.bytes_value("window/state")
            was_maximized = self.app_settings.bool_value("window/was_maximized", False)
            if not layout_is_compatible and (geometry or state):
                self.log("Saved window layout was incompatible with the current studio layout. Using defaults.")

            if layout_is_compatible and geometry and not was_maximized:
                restored_geometry = self.restoreGeometry(QByteArray(geometry))
                if restored_geometry is False:
                    self.log("Saved window geometry was incompatible. Using defaults.")
            if layout_is_compatible and state:
                restored_state = self.restoreState(QByteArray(state))
                if restored_state is False:
                    self.log("Saved dock/window state was incompatible. Using defaults.")
            if self._content_splitter is not None:
                splitter_sizes = (
                    self.app_settings.json_value(
                        "window/content_splitter_sizes",
                        list(DEFAULT_CONTENT_SPLITTER_SIZES),
                    )
                    if layout_is_compatible
                    else list(DEFAULT_CONTENT_SPLITTER_SIZES)
                )
                if isinstance(splitter_sizes, list) and len(splitter_sizes) == 2:
                    try:
                        self._content_splitter.setSizes([int(value) for value in splitter_sizes])
                    except Exception:
                        self.log("Old layout splitter settings were incompatible. Using defaults.")
                        self._content_splitter.setSizes(list(DEFAULT_CONTENT_SPLITTER_SIZES))
                elif splitter_sizes:
                    self.log("Old layout splitter settings were incompatible. Using defaults.")
                    self._content_splitter.setSizes(list(DEFAULT_CONTENT_SPLITTER_SIZES))
            if self._workspace_splitter is not None:
                workspace_sizes = (
                    self.app_settings.json_value(
                        "window/workspace_splitter_sizes",
                        list(DEFAULT_WORKSPACE_SPLITTER_SIZES),
                    )
                    if layout_is_compatible
                    else list(DEFAULT_WORKSPACE_SPLITTER_SIZES)
                )
                if isinstance(workspace_sizes, list) and len(workspace_sizes) == 2:
                    try:
                        self._workspace_splitter.setSizes([int(value) for value in workspace_sizes])
                    except Exception:
                        self.log("Saved filmstrip layout sizes were incompatible. Using defaults.")
                        self._workspace_splitter.setSizes(list(DEFAULT_WORKSPACE_SPLITTER_SIZES))
                elif workspace_sizes:
                    self.log("Saved filmstrip layout sizes were incompatible. Using defaults.")
                    self._workspace_splitter.setSizes(list(DEFAULT_WORKSPACE_SPLITTER_SIZES))
            self._pending_filmstrip_scroll_value = self.app_settings.int_value("workspace/filmstrip_scroll_value", 0)
            self._set_developer_log_visible(
                self.app_settings.bool_value("workspace/log_dock_visible", False) if layout_is_compatible else False
            )
            self._apply_safe_window_geometry(was_maximized=was_maximized)
        except Exception as exc:
            self.log(f"Settings load failure: {exc}")

    def _apply_safe_window_geometry(self, *, was_maximized: bool) -> None:
        primary_screen = QApplication.primaryScreen()
        available = primary_screen.availableGeometry() if primary_screen is not None else None
        if available is None or available.width() <= 0 or available.height() <= 0:
            return

        safe_min_width = min(DEFAULT_MAIN_WINDOW_MIN_WIDTH, available.width())
        safe_min_height = min(DEFAULT_MAIN_WINDOW_MIN_HEIGHT, available.height())
        self.setMinimumSize(safe_min_width, safe_min_height)

        if self.isMaximized() or self.isFullScreen():
            self.showNormal()

        current_geometry = self.geometry()
        current_width = current_geometry.width() if current_geometry.width() > 0 else DEFAULT_MAIN_WINDOW_WIDTH
        current_height = current_geometry.height() if current_geometry.height() > 0 else DEFAULT_MAIN_WINDOW_HEIGHT
        safe_x, safe_y, safe_width, safe_height = clamp_window_geometry(
            x=current_geometry.x(),
            y=current_geometry.y(),
            width=current_width,
            height=current_height,
            available_x=available.x(),
            available_y=available.y(),
            available_width=available.width(),
            available_height=available.height(),
            minimum_width=safe_min_width,
            minimum_height=safe_min_height,
        )
        self.setGeometry(safe_x, safe_y, safe_width, safe_height)

        if was_maximized:
            self.showMaximized()

    def _restore_workspace_state(self) -> None:
        last_project_path = self.app_settings.last_project_path().strip()
        if last_project_path:
            last_project_file = Path(last_project_path)
            if last_project_file.exists():
                if not self._open_project_path(last_project_file, show_error_dialog=False):
                    self.app_settings.set_last_project_path("")
            else:
                self.log(f"Last project could not be reopened: {last_project_file}")
                self.app_settings.set_last_project_path("")
                cleaned = [path for path in self.app_settings.recent_projects() if path != last_project_path]
                self.app_settings.set_recent_projects(cleaned)
                self._refresh_recent_projects_menu()

        if self.current_project is None:
            self._refresh_project_view()

        saved_stage = self.app_settings.string_value("workspace/selected_stage", "project").strip().lower()
        if saved_stage not in self.stage_panels:
            self.log(f"Invalid persisted stage name '{saved_stage}'. Falling back to project.")
            saved_stage = "project"
        self._select_stage(saved_stage)

        saved_preview_mode = self.app_settings.string_value("workspace/preview_mode", PREVIEW_SOURCE).strip()
        if saved_preview_mode:
            self.set_preview_mode(saved_preview_mode)

        preview_zoom_mode = self.app_settings.string_value("workspace/preview_zoom_mode", "fit").strip().lower()
        if preview_zoom_mode == "fit":
            self.image_preview.fit_to_view()
        elif preview_zoom_mode == "manual":
            self.image_preview.reset_zoom()

    def _restore_current_project_selection(self) -> None:
        if self.current_project is None or self.current_project.page_count == 0:
            return

        saved_relative_path = self.app_settings.string_value("workspace/selected_page_relative_path", "").strip()
        target_index = 0
        if saved_relative_path:
            for index in range(self.current_project.page_count):
                if self.current_project.page_relative_path_for_index(index) == saved_relative_path:
                    target_index = index
                    break
        self._select_page_row(target_index, user_initiated=False)
        self._load_page_state_for_index(target_index, user_initiated=False)
        self.page_filmstrip.set_horizontal_scroll_value(self._pending_filmstrip_scroll_value)

    def _persist_panel_preferences(self) -> None:
        try:
            self.app_settings.set_panel_settings("ocr", self.config_panel.ocr_settings_snapshot())
            self.app_settings.set_panel_settings("translation", self.config_panel.translation_settings_snapshot())
            self.app_settings.set_panel_settings("inpaint", self.config_panel.inpaint_settings_snapshot())
            self.app_settings.set_panel_settings("render", self.config_panel.render_settings_snapshot())
            self.app_settings.set_panel_settings("export", self.export_panel.settings_snapshot())
        except Exception as exc:
            self.log(f"Settings save failure: {exc}")

    def _persist_workspace_state(self) -> None:
        try:
            self._persist_panel_preferences()
            self.app_settings.set_value("workspace/theme", self.theme_manager.current_theme)
            self.app_settings.set_value("workspace/selected_stage", self.current_stage_key)
            self.app_settings.set_value("workspace/preview_mode", self.left_toolbar.current_mode())
            self.app_settings.set_value(
                "workspace/preview_zoom_mode",
                "fit" if self.image_preview.auto_fit_enabled() else "manual",
            )
            self.app_settings.set_value(
                "workspace/auto_preview_result",
                self.left_toolbar.auto_preview_enabled(),
            )
            self.app_settings.set_value(
                "workspace/follow_batch_progress",
                self.left_toolbar.follow_batch_progress_enabled(),
            )
            self.app_settings.set_value("workspace/log_dock_visible", self.developer_log_dock.isVisible() if self.developer_log_dock is not None else False)
            self.app_settings.set_value("workspace/log_expanded", self.developer_log_dock.isVisible() if self.developer_log_dock is not None else False)
            self.app_settings.set_value(
                "workspace/filmstrip_scroll_value",
                self.page_filmstrip.horizontal_scroll_value(),
            )
            if self.current_project is not None:
                self._remember_project_path(self.current_project.project_file)
                self.app_settings.set_value(
                    "workspace/selected_page_relative_path",
                    self.current_page() or "",
                )
            else:
                self.app_settings.remove("workspace/selected_page_relative_path")
            self.app_settings.set_value("window/layout_version", WINDOW_LAYOUT_VERSION)
            self.app_settings.set_value("window/was_maximized", self.isMaximized())
            self.app_settings.set_value("window/geometry", bytes(self.saveGeometry()))
            self.app_settings.set_value("window/state", bytes(self.saveState()))
            if self._content_splitter is not None:
                self.app_settings.set_json_value(
                    "window/content_splitter_sizes",
                    list(self._content_splitter.sizes()),
                )
            if self._workspace_splitter is not None:
                self.app_settings.set_json_value(
                    "window/workspace_splitter_sizes",
                    list(self._workspace_splitter.sizes()),
                )
            self.app_settings.sync()
        except Exception as exc:
            self.log(f"Settings save failure: {exc}")

    def _on_auto_preview_changed(self, enabled: bool) -> None:
        self.preview_controller.set_preferences(
            auto_preview_result=enabled,
            follow_batch_progress=self.left_toolbar.follow_batch_progress_enabled(),
        )
        self.app_settings.set_value("workspace/auto_preview_result", enabled)

    def _on_follow_batch_progress_changed(self, enabled: bool) -> None:
        self.preview_controller.set_preferences(
            auto_preview_result=self.left_toolbar.auto_preview_enabled(),
            follow_batch_progress=enabled,
        )
        self.app_settings.set_value("workspace/follow_batch_progress", enabled)

    def _on_developer_log_visibility_changed(self, visible: bool) -> None:
        self.left_toolbar.set_log_button_checked(visible)
        if self.toggle_developer_log_action is not None:
            self.toggle_developer_log_action.blockSignals(True)
            self.toggle_developer_log_action.setChecked(bool(visible))
            self.toggle_developer_log_action.blockSignals(False)
        self.app_settings.set_value("workspace/log_dock_visible", bool(visible))
        self.app_settings.set_value("workspace/log_expanded", bool(visible))
        if visible:
            self._set_log_alert_state(False)
        if visible or not self._active_workers:
            return
        self.statusBar().showMessage("Developer log hidden")

    def _on_theme_changed(self, theme_name: str) -> None:
        self._apply_theme_safely(theme_name, show_error=True)

    def _apply_theme_safely(self, theme_name: str, *, show_error: bool) -> bool:
        if self._theme_change_in_progress:
            return False

        requested_theme = self.theme_manager.normalize_theme_name(theme_name)
        previous_theme = self.theme_manager.current_theme
        self._theme_change_in_progress = True

        try:
            applied_theme = self.theme_manager.set_theme(
                requested_theme,
                application=QApplication.instance(),
                root_widget=self,
            )
            self.header.set_theme_name(applied_theme)
            self._refresh_data_views_after_theme_change()
        except Exception as exc:
            self.log(f"Theme application failed for {requested_theme}: {exc}")
            try:
                self.theme_manager.set_theme(
                    previous_theme,
                    application=QApplication.instance(),
                    root_widget=self,
                )
            except Exception as restore_exc:
                self.log(f"Theme rollback also failed: {restore_exc}")
            self.header.set_theme_name(previous_theme)
            self.statusBar().showMessage("Theme change failed")
            if show_error:
                self.show_error("Theme change failed", str(exc))
            return False
        finally:
            self._theme_change_in_progress = False

        self.log(f"Applied {applied_theme} theme.")
        self.app_settings.set_value("workspace/theme", applied_theme)
        self.statusBar().showMessage(f"{applied_theme} theme applied")
        return True

    def _refresh_data_views_after_theme_change(self) -> None:
        try:
            self.workflow_tabs.refresh_theme_state()
            self.page_filmstrip.refresh_thumbnails()
            self.page_filmstrip.list_widget.viewport().update()
            self.ocr_panel.items_table.viewport().update()
            self.translation_panel.items_table.viewport().update()
            self.render_panel.items_table.viewport().update()
            self.export_panel.summary_table.viewport().update()
            self.ocr_panel.crop_preview_panel.preview.viewport().update()
            self.translation_panel.crop_preview_panel.preview.viewport().update()
            self.image_preview.viewport().update()
        except Exception as exc:
            self.log(f"Table refresh after theme change failed: {exc}")

    def _on_stage_selected(self, stage_key: str) -> None:
        if stage_key != self.current_stage_key and not self._ensure_current_editor_changes_resolved():
            self.workflow_tabs.set_current_stage(self.current_stage_key)
            return
        self._select_stage(stage_key)
        self._refresh_preview_for_current_page()

    def _select_stage(self, stage_key: str) -> None:
        if stage_key not in self.stage_panels:
            stage_key = "project"

        if stage_key != "detection" and self.detection_panel.edit_mode_enabled():
            self.detection_panel.set_edit_mode_checked(False)
            self.detection_panel.set_create_box_checked(False)
            self.image_preview.set_box_edit_mode(False)
            self._refresh_preview_for_current_page()
        if stage_key != "ocr" and self.ocr_panel.box_edit_mode_enabled():
            self.ocr_panel.set_box_edit_mode_checked(False)
            self.image_preview.set_box_edit_mode(False)
            self._refresh_preview_for_current_page()
        if stage_key != "render" and self.render_panel.box_edit_mode_enabled():
            self.render_panel.set_box_edit_mode_checked(False)
            self.image_preview.set_box_edit_mode(False)
            self._refresh_preview_for_current_page()

        self.current_stage_key = stage_key
        self.workflow_tabs.set_current_stage(stage_key)
        self.stage_stack.setCurrentIndex(self.stage_indices[stage_key])
        self._update_stage_subtool_bar(stage_key)
        self.left_toolbar.set_modes(
            GLOBAL_PREVIEW_MODES,
            self.left_toolbar.current_mode(),
        )
        self._refresh_stage_statuses()
        self.app_settings.set_value("workspace/selected_stage", stage_key)

    def new_project(self) -> None:
        if not self._ensure_current_editor_changes_resolved():
            return
        if self._workflow_busy():
            self.show_error(
                "Workflow already running",
                "Stop or wait for the current workflow task before creating a new project.",
            )
            return
        self._persist_workspace_state()
        project_dir = QFileDialog.getExistingDirectory(self, "Select Project Folder")
        if not project_dir:
            return

        project_root = Path(project_dir)
        project_file = project_root / PROJECT_FILENAME
        if project_file.exists():
            response = QMessageBox.question(
                self,
                "Project Already Exists",
                f"{project_file} already exists.\n\nOverwrite the existing project file?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if response != QMessageBox.StandardButton.Yes:
                return

        try:
            self.current_project = MangaProject.create(project_root)
        except Exception as exc:  # pragma: no cover - GUI error path.
            self.show_error("Failed to create project", str(exc))
            return

        set_project_log_dir(self.current_project.root_dir)
        self._process_stage_status = "missing"
        self._process_active_stage_key = None
        self.process_panel.reset_process_state(scope_text="Current page")
        self._remember_project_path(self.current_project.project_file)
        self._select_stage("project")
        self._refresh_project_view()
        self._restore_current_project_selection()
        self.log(f"Created project at {self.current_project.root_dir}")
        self.statusBar().showMessage("Project created")

    def open_project(self) -> None:
        if not self._ensure_current_editor_changes_resolved():
            return
        if self._workflow_busy():
            self.show_error(
                "Workflow already running",
                "Stop or wait for the current workflow task before opening a different project.",
            )
            return
        self._persist_workspace_state()
        project_file, _ = QFileDialog.getOpenFileName(self, "Open Project", "", PROJECT_FILTER)
        if not project_file:
            return

        self._open_project_path(Path(project_file), show_error_dialog=True)

    def save_project(self) -> None:
        if self.current_project is None:
            self.show_error("No project open", "Create or open a project before saving.")
            return
        if self._workflow_busy():
            self.show_error(
                "Workflow already running",
                "Wait for the current workflow task to finish before saving the project.",
            )
            return

        try:
            self.current_project.save()
        except Exception as exc:  # pragma: no cover - GUI error path.
            self.show_error("Failed to save project", str(exc))
            return

        self._remember_project_path(self.current_project.project_file)
        self.log(f"Saved project to {self.current_project.project_file}")
        self.statusBar().showMessage("Project saved")

    def import_images(self) -> None:
        if self.current_project is None:
            self.show_error("No project open", "Create or open a project before importing images.")
            return
        if self._workflow_busy():
            self.show_error(
                "Workflow already running",
                "Stop or wait for the current workflow task before importing images.",
            )
            return

        file_names, _ = QFileDialog.getOpenFileNames(self, "Import Images", "", IMAGE_FILTER)
        if not file_names:
            return

        try:
            imported_images = self.current_project.import_images([Path(file_name) for file_name in file_names])
            self.current_project.save()
        except Exception as exc:  # pragma: no cover - GUI error path.
            self.show_error("Failed to import images", str(exc))
            return

        if not imported_images:
            self.show_error("No images imported", "No supported image files were selected.")
            return

        self._refresh_project_view()
        self._persist_workspace_state()
        self.log(f"Imported {len(imported_images)} image(s) into {self.current_project.source_dir}")
        self.statusBar().showMessage(f"Imported {len(imported_images)} image(s)")

    def remove_current_page_from_project(self) -> None:
        if not self._ensure_current_editor_changes_resolved():
            return
        if self.current_project is None:
            self.show_error("No project open", "Create or open a project before removing a page.")
            return
        if self._active_workers or self._busy_stages:
            self.show_error(
                "Project is busy",
                "Wait for the current workflow tasks to finish before removing a page from the project.",
            )
            return

        current_index = self._current_page_index()
        if current_index is None:
            self.show_error("No page selected", "Select a page first.")
            return

        image_relative_path = self.current_project.page_relative_path_for_index(current_index)
        if not image_relative_path:
            self.show_error("Page not found", "The selected page could not be found in the current project.")
            return

        source_path = self.current_project.root_dir / image_relative_path
        page_name = Path(image_relative_path).name
        if source_path.exists():
            if not self._confirm_remove_page(page_name):
                return
            allow_missing_source = False
        else:
            if not self._confirm_remove_missing_source_page(page_name):
                return
            allow_missing_source = True

        result = remove_page_from_project(
            self.current_project,
            image_relative_path,
            move_to_trash=True,
            allow_missing_source=allow_missing_source,
        )

        removed_pages = result.get("removed_pages", [])
        errors = result.get("errors", [])
        skipped_paths = result.get("skipped_paths", [])
        if not removed_pages:
            message = str(errors[0]) if errors else f"Could not remove {page_name} from the project."
            self.show_error("Failed to remove page", message)
            return

        self._refresh_project_view()
        self._persist_workspace_state()

        for error_message in errors:
            self.log(error_message)
        for skipped_path in skipped_paths:
            self.log(f"Skipped missing path during page removal: {skipped_path}")

        if errors:
            self.statusBar().showMessage(
                f"Removed {page_name} from the project. Some cache files could not be moved to .trash; see the log."
            )
        else:
            self.statusBar().showMessage(f"Removed {page_name} from the project")
        self.log(f"Removed page from project: {image_relative_path}")

    def _confirm_remove_page(self, page_name: str) -> bool:
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setWindowTitle("Remove page from project?")
        dialog.setText(
            "This will remove the page from the project and move its source/cache files to the project .trash folder."
        )
        dialog.setInformativeText(page_name)
        remove_button = dialog.addButton("Remove", QMessageBox.ButtonRole.AcceptRole)
        dialog.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        dialog.setDefaultButton(remove_button)
        dialog.exec()
        return dialog.clickedButton() == remove_button

    def _confirm_remove_missing_source_page(self, page_name: str) -> bool:
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setWindowTitle("Source image missing")
        dialog.setText("The source image is missing. Remove this page entry from the project anyway?")
        dialog.setInformativeText(page_name)
        remove_button = dialog.addButton("Remove", QMessageBox.ButtonRole.AcceptRole)
        dialog.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        dialog.setDefaultButton(remove_button)
        dialog.exec()
        return dialog.clickedButton() == remove_button

    def run_detection_for_selected_page(self, force: bool = False) -> None:
        if not self._ensure_current_editor_changes_resolved():
            return
        selected_context = self._selected_page_context(show_error=True)
        if selected_context is None:
            return

        _, _, image_path = selected_context
        self._select_stage("detection")
        self._start_detection_task(
            [image_path],
            task_name=f"{'Re-run' if force else 'Run'} Detection: {image_path.name}",
            force=force,
        )

    def run_detection_for_all_pages(self, force: bool = False) -> None:
        if not self._ensure_current_editor_changes_resolved():
            return
        if self.current_project is None:
            self.show_error("No project open", "Create or open a project before running detection.")
            return

        all_image_paths = self.current_project.all_image_paths()
        missing_image_paths = [path for path in all_image_paths if not path.exists()]
        image_paths = [path for path in all_image_paths if path.exists()]
        if not image_paths:
            self.show_error("No source images", "Import images before running detection.")
            return

        for missing_image_path in missing_image_paths:
            self.log(f"Skipping missing source image during batch detection: {missing_image_path}")

        self._select_stage("detection")
        self._start_detection_task(
            image_paths,
            task_name=f"{'Re-run' if force else 'Run'} Detection: {len(image_paths)} page(s)",
            force=force,
        )

    def reload_cached_detection(self) -> None:
        if not self._ensure_current_editor_changes_resolved():
            return
        selected_context = self._selected_page_context(show_error=True)
        if selected_context is None:
            return

        index, _, _ = selected_context
        self._select_stage("detection")
        if self._load_cached_detection_for_index(index, show_errors=True, persist_stage_status=True):
            self.statusBar().showMessage("Detection cache reloaded")
            self._refresh_preview_for_current_page()

    def clear_detection_overlay(self) -> None:
        self.preview_detection_overlay_enabled = False
        self._refresh_preview_for_current_page()
        self.statusBar().showMessage("Detection overlay cleared")
        self.log("Cleared detection overlay preview.")

    def save_detection_box_edits(self) -> None:
        if self.current_project is None:
            self.show_error("No project open", "Open a project before saving detection box edits.")
            return

        image_relative_path = self.current_page()
        if not image_relative_path:
            self.show_error("No page selected", "Select a page before saving detection box edits.")
            return

        cache_path = detection_json_path(self.current_project, image_relative_path)
        if not cache_path.exists():
            self.show_error("Detection cache missing", "Run Detection first before editing boxes.")
            return

        box_items = self.image_preview.editable_boxes_snapshot()
        try:
            detection_data = save_detection_edit_items(cache_path, box_items, mark_edited=True)
            editable_items = load_detection_edit_items(cache_path)
        except Exception as exc:
            self.show_error("Failed to save detection edits", str(exc))
            return

        self.current_detection_data = detection_data
        self.preview_detection_overlay_enabled = True
        self.image_preview.load_editable_boxes(editable_items)
        self.image_preview.set_box_edit_mode(self.detection_panel.edit_mode_enabled())
        self.image_preview.set_editable_box_category_filter(self.detection_panel.selected_box_category())
        self.image_preview.set_show_excluded_boxes(self.detection_panel.show_excluded_enabled())

        self.current_project.update_stage_status(
            image_relative_path,
            "detection",
            status="done",
            cache_path=self._relative_project_path(cache_path),
        )
        self._mark_detection_downstream_stale(image_relative_path)
        self._persist_project(show_errors=False)
        self.detection_panel.set_detection_data(detection_data, str(cache_path))
        self._update_detection_stale_warning(detection_data)
        self._reload_current_page_cached_views()
        self._refresh_stage_statuses()
        self._refresh_preview_for_current_page()
        self.statusBar().showMessage("Detection box edits saved. Canon state updated.")
        self.log(f"Saved detection box edits for {Path(image_relative_path).name}")
        self.log("Canon state updated. Downstream stages will use the edited boxes.")

    def cancel_detection_box_edits(self) -> None:
        if self.current_project is None or not self.detection_panel.edit_mode_enabled():
            return
        if not self._load_detection_edit_session_for_current_page(show_errors=True):
            self.image_preview.discard_editable_box_changes()
            self.image_preview.set_box_edit_mode(False)
            self.detection_panel.set_edit_mode_checked(False)
            return
        self.statusBar().showMessage("Discarded unsaved detection box edits")
        self.log("Discarded unsaved detection box edits.")

    def reload_detection_boxes_from_cache(self) -> None:
        if not self._ensure_current_editor_changes_resolved():
            return
        if self._load_detection_edit_session_for_current_page(show_errors=True):
            self.statusBar().showMessage("Reloaded editable detection boxes")

    def exclude_selected_detection_box(self) -> None:
        if not self.image_preview.exclude_selected_box(True):
            self.show_error("No box selected", "Select a detection box before excluding it.")
            return
        self.statusBar().showMessage("Selected box marked excluded")

    def restore_selected_detection_box(self) -> None:
        if not self.image_preview.exclude_selected_box(False):
            self.show_error("No excluded box selected", "Select an excluded detection box before restoring it.")
            return
        self.statusBar().showMessage("Selected box restored")

    def start_llama_server(self) -> None:
        if not self._apply_server_inputs_to_manager():
            return
        self._select_stage("ocr")
        self.ocr_panel.set_server_status(SERVER_STATE_STARTING)
        self._start_llama_server_action("start", timeout_seconds=60.0)

    def check_llama_server(self) -> None:
        if not self._apply_server_inputs_to_manager():
            return
        self._select_stage("ocr")
        self._start_llama_server_action("check", timeout_seconds=5.0)

    def stop_llama_server(self) -> None:
        if not self._apply_server_inputs_to_manager():
            return
        self._select_stage("ocr")
        self._start_llama_server_action("stop", timeout_seconds=10.0)

    def _on_ocr_provider_changed(self, provider_name: str) -> None:
        normalized = str(provider_name or DEFAULT_OCR_PROVIDER).strip()
        self.statusBar().showMessage(f"OCR provider set to {self.current_ocr_config().provider_label}")
        self.log(f"OCR provider selected: {normalized}")
        self._refresh_stage_statuses()

    def current_ocr_config(self) -> OCRConfig:
        return self.config_panel.ocr_config()

    def current_translation_config(self) -> TranslationConfig:
        return self.config_panel.translation_config()

    def current_inpaint_settings(self, *, force: bool = False) -> dict[str, Any]:
        return self.config_panel.inpaint_settings(force_override=force)

    def current_render_config(self, *, force: bool) -> RenderConfig:
        render_config = self.config_panel.render_config(force_override=force)
        manual_font_path = render_config.font_path.strip()
        if manual_font_path:
            resolve_font_path(
                self.workspace_root,
                font_name=render_config.font_name,
                font_path=manual_font_path,
            )
        return render_config

    def _ocr_config_from_panel(self) -> OCRConfig:
        return self.current_ocr_config()

    def _validate_ocr_provider_for_run(self) -> OCRConfig | None:
        ocr_config = self._ocr_config_from_panel()
        if ocr_config.ocr_provider == OCR_PROVIDER_CHROME_LENS:
            try:
                return validate_ocr_provider_config(ocr_config)
            except Exception as exc:
                self.show_error("Chrome Lens OCR is unavailable", str(exc))
                return None

        if not self._apply_server_inputs_to_manager():
            return None
        ocr_config.server_url = self.llama_server_manager.server_url
        return ocr_config

    def prepare_ocr_items_for_selected_page(self, force: bool = False) -> None:
        if not self._ensure_current_editor_changes_resolved():
            return
        selected_context = self._selected_page_context(show_error=True)
        if selected_context is None or self.current_project is None:
            return

        _, image_relative_path, _ = selected_context
        self._select_stage("ocr")
        self._start_ocr_preparation_task(
            [image_relative_path],
            task_name=f"{'Re-prepare' if force else 'Prepare'} OCR: {Path(image_relative_path).name}",
            force=force,
        )

    def prepare_ocr_items_for_all_pages(self, force: bool = False) -> None:
        if not self._ensure_current_editor_changes_resolved():
            return
        if self.current_project is None:
            self.show_error("No project open", "Create or open a project before preparing OCR items.")
            return

        image_relative_paths = self._all_page_relative_paths()
        if not image_relative_paths:
            self.show_error("No source images", "Import images before preparing OCR items.")
            return

        self._select_stage("ocr")
        self._start_ocr_preparation_task(
            image_relative_paths,
            task_name=f"{'Re-prepare' if force else 'Prepare'} OCR: {len(image_relative_paths)} page(s)",
            force=force,
        )

    def reload_cached_ocr_items(self) -> None:
        if not self._ensure_current_editor_changes_resolved():
            return
        selected_context = self._selected_page_context(show_error=True)
        if selected_context is None:
            return

        index, _, _ = selected_context
        self._select_stage("ocr")
        if self._load_cached_ocr_for_index(index, show_errors=True, persist_stage_status=True):
            self.statusBar().showMessage("OCR cache reloaded")

    def run_ocr_for_selected_page(self, force: bool = False) -> None:
        if not self._ensure_current_editor_changes_resolved():
            return
        selected_context = self._selected_page_context(show_error=True)
        if selected_context is None or self.current_project is None:
            return
        ocr_config = self._validate_ocr_provider_for_run()
        if ocr_config is None:
            return

        _, image_relative_path, _ = selected_context
        self._select_stage("ocr")
        self._start_ocr_inference_task(
            [image_relative_path],
            task_name=f"{'Re-run' if force else 'Run'} OCR: {Path(image_relative_path).name}",
            selected_item_ids_by_page=None,
            force=force,
            ocr_config=ocr_config,
        )

    def run_ocr_for_all_pages(self, force: bool = False) -> None:
        if not self._ensure_current_editor_changes_resolved():
            return
        if self.current_project is None:
            self.show_error("No project open", "Create or open a project before running OCR.")
            return
        ocr_config = self._validate_ocr_provider_for_run()
        if ocr_config is None:
            return

        image_relative_paths = self._all_page_relative_paths()
        if not image_relative_paths:
            self.show_error("No source images", "Import images and prepare OCR items before running OCR.")
            return

        self._select_stage("ocr")
        self._start_ocr_inference_task(
            image_relative_paths,
            task_name=f"{'Re-run' if force else 'Run'} OCR: {len(image_relative_paths)} page(s)",
            selected_item_ids_by_page=None,
            force=force,
            ocr_config=ocr_config,
        )

    def run_ocr_for_selected_items(self, force: bool = False) -> None:
        if not self._ensure_current_editor_changes_resolved():
            return
        selected_context = self._selected_page_context(show_error=True)
        if selected_context is None or self.current_project is None:
            return
        if self.current_ocr_data is None:
            self.show_error("OCR items not prepared", "Prepare OCR items before running OCR.")
            return
        ocr_config = self._validate_ocr_provider_for_run()
        if ocr_config is None:
            return

        selected_item_ids = self.ocr_panel.selected_item_ids()
        if not selected_item_ids:
            self.show_error("No OCR items selected", "Select one or more OCR rows first.")
            return

        _, image_relative_path, _ = selected_context
        self._select_stage("ocr")
        self._start_ocr_inference_task(
            [image_relative_path],
            task_name=f"{'Re-run' if force else 'Run'} OCR Items: {Path(image_relative_path).name}",
            selected_item_ids_by_page={image_relative_path: selected_item_ids},
            force=force,
            ocr_config=ocr_config,
        )

    def save_edited_ocr_text(self) -> None:
        self.ocr_panel.save_current_item()

    def initialize_translation_for_selected_page(self, force: bool = False) -> None:
        if not self._ensure_current_editor_changes_resolved():
            return
        selected_context = self._selected_page_context(show_error=True)
        if selected_context is None or self.current_project is None:
            return

        _, image_relative_path, _ = selected_context
        self._select_stage("translation")
        self._start_translation_initialization_task(
            [image_relative_path],
            task_name=f"{'Re-initialize' if force else 'Initialize'} Translation: {Path(image_relative_path).name}",
            force=force,
        )

    def initialize_translation_for_all_pages(self, force: bool = False) -> None:
        if not self._ensure_current_editor_changes_resolved():
            return
        if self.current_project is None:
            self.show_error("No project open", "Create or open a project before initializing translation.")
            return

        image_relative_paths = self._all_page_relative_paths()
        if not image_relative_paths:
            self.show_error("No source images", "Import images and prepare OCR before initializing translation.")
            return

        self._select_stage("translation")
        self._start_translation_initialization_task(
            image_relative_paths,
            task_name=f"{'Re-initialize' if force else 'Initialize'} Translation: {len(image_relative_paths)} page(s)",
            force=force,
        )

    def run_translation_for_selected_page(self, force: bool = False) -> None:
        if not self._ensure_current_editor_changes_resolved():
            return
        selected_context = self._selected_page_context(show_error=True)
        if selected_context is None or self.current_project is None:
            return

        _, image_relative_path, _ = selected_context
        self._select_stage("translation")
        self._start_translation_task(
            [image_relative_path],
            task_name=f"{'Re-translate' if force else 'Translate'}: {Path(image_relative_path).name}",
            selected_item_ids_by_page=None,
            force=force,
        )

    def run_translation_for_all_pages(self, force: bool = False) -> None:
        if not self._ensure_current_editor_changes_resolved():
            return
        if self.current_project is None:
            self.show_error("No project open", "Create or open a project before running translation.")
            return

        image_relative_paths = self._all_page_relative_paths()
        if not image_relative_paths:
            self.show_error("No source images", "Import images and prepare OCR before running translation.")
            return

        self._select_stage("translation")
        self._start_translation_task(
            image_relative_paths,
            task_name=f"{'Re-translate' if force else 'Translate'}: {len(image_relative_paths)} page(s)",
            selected_item_ids_by_page=None,
            force=force,
        )

    def run_translation_for_selected_items(self, force: bool = False) -> None:
        if not self._ensure_current_editor_changes_resolved():
            return
        selected_context = self._selected_page_context(show_error=True)
        if selected_context is None or self.current_project is None:
            return
        if self.current_translation_data is None:
            self.show_error("Translation not initialized", "Initialize translation for this page first.")
            return

        selected_item_ids = self.translation_panel.selected_item_ids()
        if not selected_item_ids:
            self.show_error("No translation items selected", "Select one or more translation rows first.")
            return

        _, image_relative_path, _ = selected_context
        self._select_stage("translation")
        self._start_translation_task(
            [image_relative_path],
            task_name=f"{'Re-translate' if force else 'Translate'} Items: {Path(image_relative_path).name}",
            selected_item_ids_by_page={image_relative_path: selected_item_ids},
            force=force,
        )

    def reload_cached_translation(self) -> None:
        if not self._ensure_current_editor_changes_resolved():
            return
        selected_context = self._selected_page_context(show_error=True)
        if selected_context is None:
            return

        index, _, _ = selected_context
        self._select_stage("translation")
        if self._load_cached_translation_for_index(index, show_errors=True, persist_stage_status=True):
            self.statusBar().showMessage("Translation cache reloaded")

    def save_edited_translation_text(self) -> None:
        self.translation_panel.save_both()

    def prepare_inpaint_mask_for_selected_page(self, force: bool = False) -> None:
        selected_context = self._selected_page_context(show_error=True)
        if selected_context is None or self.current_project is None:
            return

        _, image_relative_path, _ = selected_context
        self._select_stage("inpaint")
        self._start_inpaint_mask_task(
            [image_relative_path],
            task_name=f"{'Re-prepare' if force else 'Prepare'} Mask: {Path(image_relative_path).name}",
            force=force,
        )

    def prepare_inpaint_mask_for_all_pages(self, force: bool = False) -> None:
        if self.current_project is None:
            self.show_error("No project open", "Create or open a project before preparing inpaint masks.")
            return

        image_relative_paths = self._all_page_relative_paths()
        if not image_relative_paths:
            self.show_error("No source images", "Import images before preparing inpaint masks.")
            return

        self._select_stage("inpaint")
        self._start_inpaint_mask_task(
            image_relative_paths,
            task_name=f"{'Re-prepare' if force else 'Prepare'} Mask: {len(image_relative_paths)} page(s)",
            force=force,
        )

    def run_inpaint_for_selected_page(self, force: bool = False) -> None:
        selected_context = self._selected_page_context(show_error=True)
        if selected_context is None or self.current_project is None:
            return

        _, image_relative_path, _ = selected_context
        self._select_stage("inpaint")
        self._start_inpaint_task(
            [image_relative_path],
            task_name=f"{'Re-inpaint' if force else 'Inpaint'}: {Path(image_relative_path).name}",
            force=force,
        )

    def run_inpaint_for_all_pages(self, force: bool = False) -> None:
        if self.current_project is None:
            self.show_error("No project open", "Create or open a project before running inpaint.")
            return

        image_relative_paths = self._all_page_relative_paths()
        if not image_relative_paths:
            self.show_error("No source images", "Import images before running inpaint.")
            return

        self._select_stage("inpaint")
        self._start_inpaint_task(
            image_relative_paths,
            task_name=f"{'Re-inpaint' if force else 'Inpaint'}: {len(image_relative_paths)} page(s)",
            force=force,
        )

    def reload_cached_inpaint(self) -> None:
        selected_context = self._selected_page_context(show_error=True)
        if selected_context is None:
            return

        index, _, _ = selected_context
        self._select_stage("inpaint")
        self._load_cached_inpaint_for_index(index, show_errors=True, persist_stage_status=True)
        self._refresh_preview_for_current_page()
        self.statusBar().showMessage("Inpaint cache reloaded")

    def clear_inpaint_preview(self) -> None:
        self.set_preview_mode(PREVIEW_SOURCE)
        self.statusBar().showMessage("Inpaint preview cleared")
        self.log("Cleared inpaint preview overlay.")

    def load_lama_model(self) -> None:
        self._select_stage("inpaint")
        self._start_lama_model_task("load")

    def unload_lama_model(self) -> None:
        self._select_stage("inpaint")
        self._start_lama_model_task("unload")

    def prepare_render_for_selected_page(self, force: bool = False) -> None:
        selected_context = self._selected_page_context(show_error=True)
        if selected_context is None or self.current_project is None:
            return

        _, image_relative_path, _ = selected_context
        self._select_stage("render")
        self._start_render_preparation_task(
            [image_relative_path],
            task_name=f"{'Re-prepare' if force else 'Prepare'} Render: {Path(image_relative_path).name}",
            force=force,
        )

    def prepare_render_for_all_pages(self, force: bool = False) -> None:
        if self.current_project is None:
            self.show_error("No project open", "Create or open a project before preparing render data.")
            return

        image_relative_paths = self._all_page_relative_paths()
        if not image_relative_paths:
            self.show_error("No source images", "Import images before preparing render data.")
            return

        self._select_stage("render")
        self._start_render_preparation_task(
            image_relative_paths,
            task_name=f"{'Re-prepare' if force else 'Prepare'} Render: {len(image_relative_paths)} page(s)",
            force=force,
        )

    def run_render_for_selected_page(self, force: bool = False) -> None:
        selected_context = self._selected_page_context(show_error=True)
        if selected_context is None or self.current_project is None:
            return

        _, image_relative_path, _ = selected_context
        self._select_stage("render")
        self._start_render_task(
            [image_relative_path],
            task_name=f"{'Re-render' if force else 'Render'}: {Path(image_relative_path).name}",
            force=force,
        )

    def run_render_for_all_pages(self, force: bool = False) -> None:
        if self.current_project is None:
            self.show_error("No project open", "Create or open a project before running render.")
            return

        image_relative_paths = self._all_page_relative_paths()
        if not image_relative_paths:
            self.show_error("No source images", "Import images before running render.")
            return

        self._select_stage("render")
        self._start_render_task(
            image_relative_paths,
            task_name=f"{'Re-render' if force else 'Render'}: {len(image_relative_paths)} page(s)",
            force=force,
        )

    def reload_cached_render(self) -> None:
        selected_context = self._selected_page_context(show_error=True)
        if selected_context is None:
            return

        index, _, _ = selected_context
        self._select_stage("render")
        self._load_cached_render_for_index(index, show_errors=True, persist_stage_status=True)
        self._refresh_preview_for_current_page()
        self.statusBar().showMessage("Render cache reloaded")

    def clear_render_preview(self) -> None:
        self.set_preview_mode(PREVIEW_SOURCE)
        self.statusBar().showMessage("Render preview cleared")
        self.log("Cleared render preview.")

    def _process_is_running(self) -> bool:
        return "process" in self._busy_stages or self._active_process_worker is not None

    def _workflow_busy(self) -> bool:
        manager_busy = bool(self.service_manager is not None and self.service_manager.has_active_commands())
        return bool(self._active_workers or self._busy_stages or manager_busy)

    def _submit_service_task(
        self,
        task: Any,
        *,
        progress_label: str | None,
        finished_handler: Any,
        failed_handler: Any,
        event_handler: Any | None = None,
    ) -> ServiceCommandHandle:
        if self.service_manager is None:
            raise RuntimeError("Resident service manager is not available.")
        handle = self.service_manager.dispatch_task(task)
        self._active_workers.append(handle)
        self._attach_worker_signals(
            handle,
            progress_label=progress_label,
            finished_handler=finished_handler,
            failed_handler=failed_handler,
            event_handler=event_handler,
        )
        return handle

    def open_crash_logs_folder(self) -> None:
        self._open_output_folder(self.workspace_root / "logs")

    def _on_service_status_changed(self, payload: object) -> None:
        if not isinstance(payload, ServiceStatusSnapshot):
            return
        self._service_statuses[payload.service_name] = payload
        if hasattr(self, "process_panel"):
            self.process_panel.set_service_status(payload.service_name, payload.state, payload.message)
        if self.startup_overlay is not None:
            self.startup_overlay.set_service_status(payload.service_name, payload.state, payload.message)
            self.startup_overlay.set_overall_message(
                f"{payload.service_name.title()}: {payload.message or payload.state.title()}"
            )
        self._refresh_service_status_summary()
        self._refresh_startup_overlay_visibility()
        if payload.state == "error":
            self.statusBar().showMessage(f"{payload.service_name.title()} service error: {payload.message}")

    def _refresh_service_status_summary(self) -> None:
        if not self._service_statuses:
            self.header.set_service_status_text("Services: starting...")
            return
        states = [snapshot.state for snapshot in self._service_statuses.values()]
        ready_count = len([state for state in states if state in {"ready", "idle"}])
        loading_count = len([state for state in states if state in {"starting", "loading"}])
        running_count = len([state for state in states if state in {"busy", "running"}])
        error_count = len([state for state in states if state == "error"])
        summary_parts = [f"Services: {ready_count}/{len(states)} ready"]
        if loading_count:
            summary_parts.append(f"{loading_count} loading")
        if running_count:
            summary_parts.append(f"{running_count} busy")
        if error_count:
            summary_parts.append(f"{error_count} error")
        self.header.set_service_status_text(" | ".join(summary_parts))

    def _show_startup_overlay(self) -> None:
        if self.startup_overlay is None:
            return
        self._position_startup_overlay()
        self.startup_overlay.show()
        self.startup_overlay.raise_()
        self.startup_overlay.set_overall_message("Preloading resident workers and models...")

    def _position_startup_overlay(self) -> None:
        if self.startup_overlay is None or self.centralWidget() is None:
            return
        self.startup_overlay.setGeometry(self.centralWidget().rect())

    def _refresh_startup_overlay_visibility(self) -> None:
        if self.startup_overlay is None:
            return
        if not self._service_statuses:
            self._show_startup_overlay()
            return
        unresolved_states = {
            snapshot.state
            for snapshot in self._service_statuses.values()
            if snapshot.state in {"starting", "loading"}
        }
        if unresolved_states:
            self._show_startup_overlay()
            return
        self.startup_overlay.hide()

    def process_current_page(self, force: bool = False) -> None:
        if not self._ensure_current_editor_changes_resolved():
            return
        if self._process_is_running():
            self.show_error("Process already running", "A process is already running.")
            return
        if self._active_workers or self._busy_stages:
            self.show_error(
                "Workflow already running",
                "Wait for the current workflow task to finish before starting one-click processing.",
            )
            return

        selected_context = self._selected_page_context(show_error=True)
        if selected_context is None or self.current_project is None:
            return

        _, image_relative_path, _ = selected_context
        process_payload = self._build_process_payload(
            [image_relative_path],
            scope="current",
            force=force,
        )
        if process_payload is None:
            return

        self._select_stage("process")
        self._start_process_task(
            [image_relative_path],
            scope="current",
            force=force,
            **process_payload,
        )

    def process_chapter(self, force: bool = False) -> None:
        if not self._ensure_current_editor_changes_resolved():
            return
        if self._process_is_running():
            self.show_error("Process already running", "A process is already running.")
            return
        if self._active_workers or self._busy_stages:
            self.show_error(
                "Workflow already running",
                "Wait for the current workflow task to finish before starting one-click processing.",
            )
            return
        if self.current_project is None:
            self.show_error("No project open", "Create or open a project before starting one-click processing.")
            return

        image_relative_paths = self._all_page_relative_paths()
        if not image_relative_paths:
            self.show_error("No pages in project", "Import images before starting one-click processing.")
            return

        process_payload = self._build_process_payload(
            image_relative_paths,
            scope="chapter",
            force=force,
        )
        if process_payload is None:
            return

        self._select_stage("process")
        self._start_process_task(
            image_relative_paths,
            scope="chapter",
            force=force,
            **process_payload,
        )

    def _build_process_payload(
        self,
        image_relative_paths: list[str],
        *,
        scope: str,
        force: bool,
    ) -> dict[str, Any] | None:
        ocr_config = self._validate_ocr_provider_for_run()
        if ocr_config is None:
            return None

        translation_config = self.current_translation_config()
        try:
            render_config = self._render_config_from_panel(force=force)
        except Exception as exc:
            self.show_error("Invalid render settings", str(exc))
            return None

        inpaint_settings = self.current_inpaint_settings(force=force)
        return {
            "ocr_config": ocr_config,
            "translation_config": translation_config,
            "inpaint_settings": inpaint_settings,
            "render_config": render_config,
        }

    def stop_process(self) -> None:
        worker = self._active_process_worker
        if worker is None or not self._process_is_running():
            self.statusBar().showMessage("No process is currently running.")
            return
        if self._process_cancel_requested or worker.cancel_requested():
            self.process_panel.set_action_message("Stopping after current safe point...")
            return

        self._process_cancel_requested = True
        worker.request_cancel()
        self.process_panel.set_stage_status_text("Stopping")
        if (self._process_active_stage_key or "").strip().lower() == "inpaint":
            self.process_panel.set_action_message("Stopping after current inpaint page...")
        else:
            self.process_panel.set_action_message("Stopping after current safe point...")
        self.process_panel.set_cancel_stopping(True)
        self.process_panel.set_cancel_enabled(False)
        self.statusBar().showMessage("Stopping one-click process...")
        self.log("Stop requested for one-click process.")
        self._refresh_stage_statuses()

    def browse_export_output_folder(self) -> None:
        if self.current_project is None:
            self.show_error("No project open", "Create or open a project before choosing an export folder.")
            return

        start_dir = self.export_panel.output_dir().strip() or str(self.current_project.root_dir / "exports")
        output_dir = QFileDialog.getExistingDirectory(self, "Select Export Output Folder", start_dir)
        if not output_dir:
            return

        self.export_panel.set_output_dir(output_dir)
        self._store_export_preferences(output_dir=output_dir)
        self.statusBar().showMessage("Export output folder updated")
        self.log(f"Export output folder set to {output_dir}")

    def export_current_page(self) -> None:
        if self.current_project is None:
            self.show_error("No project open", "Create or open a project before exporting pages.")
            return

        image_relative_path = self.current_page()
        if not image_relative_path:
            self.show_error("No page selected", "Select a page before exporting the current page.")
            return

        self._select_stage("export")
        try:
            export_config = self.export_panel.build_config(page_scope_override="current")
        except Exception as exc:
            self.show_error("Invalid export settings", str(exc))
            return

        self._start_export_task(
            export_config,
            task_name=f"Export: {Path(image_relative_path).name}",
            current_page=image_relative_path,
            selected_pages=None,
        )

    def export_all_pages(self) -> None:
        if self.current_project is None:
            self.show_error("No project open", "Create or open a project before exporting pages.")
            return

        image_relative_paths = self._all_page_relative_paths()
        if not image_relative_paths:
            self.show_error("No pages in project", "Import images before exporting pages.")
            return

        self._select_stage("export")
        try:
            export_config = self.export_panel.build_config(page_scope_override="all")
        except Exception as exc:
            self.show_error("Invalid export settings", str(exc))
            return

        self._start_export_task(
            export_config,
            task_name=f"Export: {len(image_relative_paths)} page(s)",
            current_page=self.current_page(),
            selected_pages=image_relative_paths,
        )

    def export_selected_pages(self) -> None:
        self.show_error(
            "Selected pages not available yet",
            "Selected-pages export will be enabled after multi-select page picking is added.",
        )

    def open_export_output_folder(self) -> None:
        output_dir = self.export_panel.output_dir().strip()
        if not output_dir:
            self.show_error("No output folder", "Choose an export output folder first.")
            return
        self._open_output_folder(output_dir)

    def reload_last_export_summary(self) -> None:
        if self.current_project is None:
            self.show_error("No project open", "Create or open a project before reloading export summaries.")
            return

        if self._load_last_export_summary(show_errors=True):
            self.statusBar().showMessage("Export summary reloaded")
            return

        self.show_error(
            "No export summary available",
            "Run an export first, or make sure the saved export manifest still exists.",
        )

    def refresh_current_page(self) -> None:
        current_index = self._current_page_index()
        if current_index is None:
            self._refresh_project_view()
            return
        self._load_page_state_for_index(current_index, user_initiated=False)

    def refresh_stage_status(self) -> None:
        self._refresh_stage_statuses()

    def set_preview_mode(self, mode: str) -> None:
        self.left_toolbar.set_current_mode(mode)
        self.app_settings.set_value("workspace/preview_mode", self.left_toolbar.current_mode())
        self._refresh_preview_for_current_page()

    def _ensure_current_editor_changes_resolved(self) -> bool:
        if not self._ensure_pending_detection_box_changes_resolved():
            return False
        if not self._ensure_pending_ocr_box_changes_resolved():
            return False
        if not self._ensure_pending_render_box_changes_resolved():
            return False
        if self.current_stage_key == "ocr":
            return self.ocr_panel.ensure_pending_changes_resolved(self)
        if self.current_stage_key == "translation":
            return self.translation_panel.ensure_pending_changes_resolved(self)
        return True

    def _ensure_pending_detection_box_changes_resolved(self) -> bool:
        if not self.detection_panel.has_unsaved_box_edits():
            return True

        box_result = QMessageBox.question(
            self,
            "Unsaved Detection Box Edits",
            "Save your detection box edits before continuing?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if box_result == QMessageBox.StandardButton.Save:
            self.save_detection_box_edits()
            return not self.detection_panel.has_unsaved_box_edits()
        if box_result == QMessageBox.StandardButton.Discard:
            self.cancel_detection_box_edits()
            return True
        return False

    def _ensure_pending_ocr_box_changes_resolved(self) -> bool:
        if not self.ocr_panel.has_unsaved_box_edits():
            return True

        box_result = QMessageBox.question(
            self,
            "Unsaved OCR Box Edits",
            "Save your OCR box edits before continuing?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if box_result == QMessageBox.StandardButton.Save:
            self.save_ocr_box_edits()
            return not self.ocr_panel.has_unsaved_box_edits()
        if box_result == QMessageBox.StandardButton.Discard:
            self.cancel_ocr_box_edits()
            return True
        return False

    def _ensure_pending_render_box_changes_resolved(self) -> bool:
        if not self.render_panel.has_unsaved_box_edits():
            return True

        box_result = QMessageBox.question(
            self,
            "Unsaved Render Box Edits",
            "Save your render box edits before continuing?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if box_result == QMessageBox.StandardButton.Save:
            self.save_render_box_edits()
            return not self.render_panel.has_unsaved_box_edits()
        if box_result == QMessageBox.StandardButton.Discard:
            self.cancel_render_box_edits()
            return True
        return False

    def _on_detection_edit_mode_toggled(self, enabled: bool) -> None:
        if enabled:
            if not self._load_detection_edit_session_for_current_page(show_errors=True):
                self.detection_panel.set_edit_mode_checked(False)
                return
            self._select_stage("detection")
            self.set_preview_mode(PREVIEW_DETECTION)
            self.image_preview.set_box_edit_mode(True)
            self.image_preview.set_editable_box_category_filter(self.detection_panel.selected_box_category())
            self.image_preview.set_show_excluded_boxes(self.detection_panel.show_excluded_enabled())
            self._refresh_stage_statuses()
            self.statusBar().showMessage("Detection box editing enabled")
            return

        if not self._ensure_pending_detection_box_changes_resolved():
            self.detection_panel.set_edit_mode_checked(True)
            return

        self.image_preview.set_box_edit_mode(False)
        self.detection_panel.set_create_box_checked(False)
        self.detection_panel.set_dirty(False)
        self.detection_panel.set_selected_box(None)
        self._refresh_stage_statuses()
        self._refresh_preview_for_current_page()
        self.statusBar().showMessage("Detection box editing disabled")

    def _on_detection_box_type_changed(self, category: str) -> None:
        self.image_preview.set_editable_box_category_filter(category)
        self.detection_panel.set_selected_box(self.image_preview.selected_editable_box())

    def _on_detection_create_box_toggled(self, enabled: bool) -> None:
        self.image_preview.set_create_box_mode(enabled)
        if enabled and not self.detection_panel.edit_mode_enabled():
            self.detection_panel.set_create_box_checked(False)

    def _on_detection_show_excluded_toggled(self, enabled: bool) -> None:
        self.image_preview.set_show_excluded_boxes(enabled)
        self.detection_panel.set_selected_box(self.image_preview.selected_editable_box())

    def _on_editable_box_selection_changed(self, box_data: object) -> None:
        if self.current_stage_key == "detection" and self.detection_panel.edit_mode_enabled():
            self._on_detection_box_selection_changed(box_data)
            return
        if self.current_stage_key == "ocr" and self.ocr_panel.box_edit_mode_enabled():
            self._on_ocr_box_selection_changed(box_data)
            return
        if self.current_stage_key == "render" and self.render_panel.box_edit_mode_enabled():
            self._on_render_box_selection_changed(box_data)

    def _on_editable_box_changed(self, box_data: object) -> None:
        if self.current_stage_key == "detection" and self.detection_panel.edit_mode_enabled():
            self._on_detection_box_changed(box_data)
            return
        if self.current_stage_key == "ocr" and self.ocr_panel.box_edit_mode_enabled():
            self._on_ocr_box_changed(box_data)
            return
        if self.current_stage_key == "render" and self.render_panel.box_edit_mode_enabled():
            self._on_render_box_changed(box_data)

    def _on_editable_box_dirty_changed(self, dirty: object) -> None:
        if self.current_stage_key == "detection" and self.detection_panel.edit_mode_enabled():
            self._on_detection_box_dirty_changed(dirty)
            return
        if self.current_stage_key == "ocr" and self.ocr_panel.box_edit_mode_enabled():
            self._on_ocr_box_dirty_changed(dirty)
            return
        if self.current_stage_key == "render" and self.render_panel.box_edit_mode_enabled():
            self._on_render_box_dirty_changed(dirty)

    def _on_detection_box_selection_changed(self, box_data: object) -> None:
        self.detection_panel.set_selected_box(box_data if isinstance(box_data, dict) else None)

    def _on_detection_box_changed(self, box_data: object) -> None:
        if isinstance(box_data, dict):
            self.detection_panel.set_selected_box(box_data)
        self.detection_panel.set_dirty(self.image_preview.has_editable_box_changes())

    def _on_detection_box_dirty_changed(self, dirty: object) -> None:
        self.detection_panel.set_dirty(bool(dirty))

    def _load_detection_edit_session_for_current_page(self, *, show_errors: bool) -> bool:
        if self.current_project is None:
            if show_errors:
                self.show_error("No project open", "Open a project before editing detection boxes.")
            return False

        current_page = self.current_page()
        if not current_page:
            if show_errors:
                self.show_error("No page selected", "Select a page before editing detection boxes.")
            return False

        cache_path = detection_json_path(self.current_project, current_page)
        if not cache_path.exists():
            self.image_preview.set_box_edit_mode(False)
            if show_errors:
                self.show_error("Detection cache missing", "Run Detection first before editing boxes.")
            return False

        try:
            detection_data = self._ensure_detection_canon_state(cache_path)
            editable_items = load_detection_edit_items(cache_path)
        except Exception as exc:
            self.image_preview.set_box_edit_mode(False)
            if show_errors:
                self.show_error("Invalid detection cache", str(exc))
            else:
                self.log(f"Failed to load editable detection boxes from {cache_path}: {exc}")
            return False

        self.current_detection_data = detection_data
        self.image_preview.load_editable_boxes(editable_items)
        self.image_preview.set_box_edit_mode(True)
        self.image_preview.set_editable_box_category_filter(self.detection_panel.selected_box_category())
        self.image_preview.set_show_excluded_boxes(self.detection_panel.show_excluded_enabled())
        self.detection_panel.set_edit_mode_checked(True)
        self.detection_panel.set_create_box_checked(False)
        self.detection_panel.set_dirty(False)
        self.detection_panel.set_detection_data(detection_data, str(cache_path))
        self.detection_panel.set_selected_box(None)
        self._update_detection_stale_warning(detection_data)
        self._refresh_preview_for_current_page()
        return True

    def _mark_detection_downstream_stale(self, image_relative_path: str) -> None:
        if self.current_project is None:
            return
        for stage_name in ("ocr", "translation", "inpaint", "render", "export"):
            self.current_project.set_stage_stale(image_relative_path, stage_name, True)

    def _update_detection_stale_warning(self, detection_data: dict[str, Any] | None) -> None:
        if not isinstance(detection_data, dict):
            self.detection_panel.set_stale_warning(None)
            return
        if bool(detection_data.get("edited", False)):
            self.detection_panel.set_stale_warning(
                "Detection was edited. Re-prepare OCR is recommended before continuing downstream."
            )
        else:
            self.detection_panel.set_stale_warning(None)

    def _on_ocr_box_edit_mode_toggled(self, enabled: bool) -> None:
        if enabled:
            if not self._load_ocr_edit_session_for_current_page(show_errors=True):
                self.ocr_panel.set_box_edit_mode_checked(False)
                return
            self._select_stage("ocr")
            self.set_preview_mode(PREVIEW_SOURCE)
            self.image_preview.set_box_edit_mode(True)
            self._apply_ocr_edit_session_to_preview(preserve_dirty=False)
            self._refresh_stage_statuses()
            self.statusBar().showMessage("OCR box editing enabled")
            return

        if not self._ensure_pending_ocr_box_changes_resolved():
            self.ocr_panel.set_box_edit_mode_checked(True)
            return

        self.image_preview.set_box_edit_mode(False)
        self.ocr_panel.set_box_dirty(False)
        self.ocr_panel.set_selected_box(None)
        self._refresh_stage_statuses()
        self._refresh_preview_for_current_page()
        self.statusBar().showMessage("OCR box editing disabled")

    def _on_ocr_box_field_changed(self, _field: str) -> None:
        if not self.ocr_panel.box_edit_mode_enabled():
            return
        self._apply_ocr_edit_session_to_preview(preserve_dirty=self.ocr_panel.has_unsaved_box_edits())

    def _on_ocr_show_excluded_toggled(self, _enabled: bool) -> None:
        if not self.ocr_panel.box_edit_mode_enabled():
            return
        self._apply_ocr_edit_session_to_preview(preserve_dirty=self.ocr_panel.has_unsaved_box_edits())

    def _on_ocr_table_item_changed(self, item_id: int) -> None:
        if not self.ocr_panel.box_edit_mode_enabled():
            return
        self.image_preview.select_editable_box(self._ocr_overlay_category(), item_id)

    def _on_ocr_box_selection_changed(self, box_data: object) -> None:
        if not isinstance(box_data, dict):
            self.ocr_panel.set_selected_box(None)
            return
        full_item = self._ocr_overlay_item_to_full_item(box_data)
        self.ocr_panel.set_selected_box(full_item)
        if not self.ocr_panel.select_item_by_id(int(full_item.get("id", 0))):
            current_item_id = self.ocr_panel.current_table_item_id()
            if current_item_id is not None:
                self.image_preview.select_editable_box(self._ocr_overlay_category(), current_item_id)

    def _on_ocr_box_changed(self, box_data: object) -> None:
        self._ocr_edit_items = self._ocr_items_from_overlay_snapshot(self.image_preview.editable_boxes_snapshot())
        if isinstance(box_data, dict):
            self.ocr_panel.set_selected_box(self._ocr_overlay_item_to_full_item(box_data))
        self.ocr_panel.set_box_dirty(self.image_preview.has_editable_box_changes())

    def _on_ocr_box_dirty_changed(self, dirty: object) -> None:
        self.ocr_panel.set_box_dirty(bool(dirty))

    def _load_ocr_edit_session_for_current_page(self, *, show_errors: bool) -> bool:
        if self.current_project is None:
            if show_errors:
                self.show_error("No project open", "Open a project before editing OCR boxes.")
            return False

        current_page = self.current_page()
        if not current_page:
            if show_errors:
                self.show_error("No page selected", "Select a page before editing OCR boxes.")
            return False

        cache_path = ocr_json_path(self.current_project, current_page)
        if not cache_path.exists():
            self.image_preview.set_box_edit_mode(False)
            if show_errors:
                self.show_error("OCR JSON missing", "Prepare OCR items first before editing OCR boxes.")
            return False

        try:
            ocr_data = load_ocr_json(cache_path)
            self._ocr_edit_items = load_ocr_edit_items(cache_path)
        except Exception as exc:
            self.image_preview.set_box_edit_mode(False)
            if show_errors:
                self.show_error("Invalid OCR cache", str(exc))
            else:
                self.log(f"Failed to load editable OCR boxes from {cache_path}: {exc}")
            return False

        self.current_ocr_data = dict(ocr_data)
        self.current_ocr_data["items"] = [dict(item) for item in self._ocr_edit_items]
        self.current_ocr_cache_path = cache_path
        self.ocr_panel.set_items(self.current_ocr_data.get("items", []), cache_path)
        self.ocr_panel.set_box_edit_mode_checked(True)
        self.ocr_panel.set_box_dirty(False)
        self.ocr_panel.set_selected_box(None)
        self._update_ocr_box_stale_warning(self.current_ocr_data)
        self._apply_ocr_edit_session_to_preview(preserve_dirty=False)
        self._refresh_preview_for_current_page()
        return True

    def _apply_ocr_edit_session_to_preview(self, *, preserve_dirty: bool) -> None:
        if not self.ocr_panel.box_edit_mode_enabled():
            return
        overlay_items = self._ocr_overlay_items_for_preview(self._ocr_edit_items)
        selected_item_id = self.ocr_panel.current_table_item_id()
        was_dirty = preserve_dirty and self.ocr_panel.has_unsaved_box_edits()
        self.image_preview.load_editable_boxes(overlay_items)
        self.image_preview.set_box_edit_mode(True)
        self.image_preview.set_editable_box_category_filter(self._ocr_overlay_category())
        self.image_preview.set_show_excluded_boxes(self.ocr_panel.show_excluded_items_enabled())
        if selected_item_id is not None:
            self.image_preview.select_editable_box(self._ocr_overlay_category(), selected_item_id)
        if was_dirty:
            self.image_preview.set_editable_boxes_dirty(True)
            self.ocr_panel.set_box_dirty(True)

    def save_ocr_box_edits(self) -> None:
        if self.current_project is None:
            self.show_error("No project open", "Open a project before saving OCR box edits.")
            return
        image_relative_path = self.current_page()
        if not image_relative_path:
            self.show_error("No page selected", "Select a page before saving OCR box edits.")
            return
        cache_path = ocr_json_path(self.current_project, image_relative_path)
        if not cache_path.exists():
            self.show_error("OCR JSON missing", "Prepare OCR items first before editing OCR boxes.")
            return

        self._ocr_edit_items = self._ocr_items_from_overlay_snapshot(self.image_preview.editable_boxes_snapshot())
        try:
            ocr_data = save_ocr_edit_items(cache_path, self._ocr_edit_items, mark_edited=True)
            self._ocr_edit_items = load_ocr_edit_items(cache_path)
        except Exception as exc:
            self.show_error("Failed to save OCR box edits", str(exc))
            return

        self.current_ocr_data = dict(ocr_data)
        self.current_ocr_data["items"] = [dict(item) for item in self._ocr_edit_items]
        self.current_ocr_cache_path = cache_path
        self.ocr_panel.set_items(self.current_ocr_data.get("items", []), cache_path)
        self.translation_panel.set_ocr_context(self.current_project.root_dir, self.current_ocr_data, cache_path)
        self._update_project_ocr_stage_status(image_relative_path, self.current_ocr_data)
        self._mark_ocr_downstream_stale(image_relative_path)
        self._persist_project(show_errors=False)
        self._update_ocr_box_stale_warning(self.current_ocr_data)
        self._reload_current_page_cached_views()
        self._apply_ocr_edit_session_to_preview(preserve_dirty=False)
        self._refresh_stage_statuses()
        self._refresh_preview_for_current_page()
        self.statusBar().showMessage("OCR box edits saved. Canon state updated.")
        self.log(f"Saved OCR box edits for {Path(image_relative_path).name}")
        self.log("Canon state updated. Downstream stages will use the edited boxes.")

    def cancel_ocr_box_edits(self) -> None:
        if self.current_project is None or not self.ocr_panel.box_edit_mode_enabled():
            return
        if not self._load_ocr_edit_session_for_current_page(show_errors=True):
            self.image_preview.set_box_edit_mode(False)
            self.ocr_panel.set_box_edit_mode_checked(False)
            return
        self.statusBar().showMessage("Discarded unsaved OCR box edits")
        self.log("Discarded unsaved OCR box edits.")

    def reload_ocr_boxes_from_cache(self) -> None:
        if not self._ensure_current_editor_changes_resolved():
            return
        if self._load_ocr_edit_session_for_current_page(show_errors=True):
            self.statusBar().showMessage("Reloaded editable OCR boxes")

    def exclude_selected_ocr_item(self) -> None:
        if not self.image_preview.exclude_selected_box(True):
            self.show_error("No OCR item selected", "Select an OCR item box before excluding it.")
            return
        self.statusBar().showMessage("Selected OCR item marked excluded")

    def restore_selected_ocr_item(self) -> None:
        if not self.image_preview.exclude_selected_box(False):
            self.show_error("No excluded OCR item selected", "Select an excluded OCR item before restoring it.")
            return
        self.statusBar().showMessage("Selected OCR item restored")

    def _mark_ocr_downstream_stale(self, image_relative_path: str) -> None:
        if self.current_project is None:
            return
        for stage_name in ("ocr", "translation", "inpaint", "render", "export"):
            self.current_project.set_stage_stale(image_relative_path, stage_name, True)

    def _update_ocr_box_stale_warning(self, ocr_data: dict[str, Any] | None) -> None:
        if not isinstance(ocr_data, dict):
            self.ocr_panel.set_box_warning(None)
            return
        summary = summarize_ocr_edit_state(ocr_data)
        if bool(summary.get("needs_ocr_items", 0)):
            self.ocr_panel.set_box_warning(
                "OCR boxes changed. Re-run OCR for affected items is recommended."
            )
        else:
            self.ocr_panel.set_box_warning(None)

    def _ocr_overlay_category(self) -> str:
        return "ocr_bbox" if self.ocr_panel.selected_box_field() == "ocr_bbox" else "ocr_item"

    def _ocr_overlay_items_for_preview(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        category = self._ocr_overlay_category()
        overlay_items: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            overlay_item = dict(item)
            overlay_item["category"] = category
            overlay_item["_original_bbox"] = item.get("bbox")
            overlay_item["_original_ocr_bbox"] = item.get("ocr_bbox")
            overlay_item["bbox"] = item.get("ocr_bbox") if category == "ocr_bbox" else item.get("bbox")
            overlay_items.append(overlay_item)
        return overlay_items

    def _ocr_overlay_item_to_full_item(self, overlay_item: dict[str, Any]) -> dict[str, Any]:
        item = dict(overlay_item)
        category = str(item.get("category", "") or "")
        if category == "ocr_bbox":
            item["ocr_bbox"] = item.get("bbox")
            item["bbox"] = item.get("_original_bbox", item.get("bbox"))
        else:
            item["bbox"] = item.get("bbox")
            item["ocr_bbox"] = item.get("_original_ocr_bbox", item.get("ocr_bbox"))
        item.pop("category", None)
        item.pop("_original_bbox", None)
        item.pop("_original_ocr_bbox", None)
        return item

    def _ocr_items_from_overlay_snapshot(self, snapshot: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self._ocr_overlay_item_to_full_item(item) for item in snapshot if isinstance(item, dict)]

    def _on_render_box_edit_mode_toggled(self, enabled: bool) -> None:
        if enabled:
            if not self._load_render_edit_session_for_current_page(show_errors=True):
                self.render_panel.set_box_edit_mode_checked(False)
                return
            self._select_stage("render")
            self.set_preview_mode(PREVIEW_RENDER)
            self.image_preview.set_box_edit_mode(True)
            self._apply_render_edit_session_to_preview(preserve_dirty=False)
            self._refresh_stage_statuses()
            self.statusBar().showMessage("Render box editing enabled")
            return

        if not self._ensure_pending_render_box_changes_resolved():
            self.render_panel.set_box_edit_mode_checked(True)
            return

        self.image_preview.set_box_edit_mode(False)
        self.render_panel.set_box_dirty(False)
        self.render_panel.set_selected_box(None)
        self._refresh_stage_statuses()
        self._refresh_preview_for_current_page()
        self.statusBar().showMessage("Render box editing disabled")

    def _on_render_show_excluded_toggled(self, _enabled: bool) -> None:
        if not self.render_panel.box_edit_mode_enabled():
            return
        self._apply_render_edit_session_to_preview(preserve_dirty=self.render_panel.has_unsaved_box_edits())

    def _on_render_table_item_changed(self, item_id: int) -> None:
        if not self.render_panel.box_edit_mode_enabled():
            return
        self.image_preview.select_editable_box("render_bbox", item_id)

    def _on_render_box_selection_changed(self, box_data: object) -> None:
        if not isinstance(box_data, dict):
            self.render_panel.set_selected_box(None)
            return
        full_item = self._render_overlay_item_to_full_item(box_data)
        self.render_panel.set_selected_box(full_item)
        if not self.render_panel.select_item_by_id(int(full_item.get("id", 0))):
            current_item_id = self.render_panel.current_table_item_id()
            if current_item_id is not None:
                self.image_preview.select_editable_box("render_bbox", current_item_id)

    def _on_render_box_changed(self, box_data: object) -> None:
        self._render_edit_items = self._render_items_from_overlay_snapshot(self.image_preview.editable_boxes_snapshot())
        if isinstance(box_data, dict):
            self.render_panel.set_selected_box(self._render_overlay_item_to_full_item(box_data))
        self.render_panel.set_box_dirty(self.image_preview.has_editable_box_changes())

    def _on_render_box_dirty_changed(self, dirty: object) -> None:
        self.render_panel.set_box_dirty(bool(dirty))

    def _load_render_edit_session_for_current_page(self, *, show_errors: bool) -> bool:
        if self.current_project is None:
            if show_errors:
                self.show_error("No project open", "Open a project before editing render boxes.")
            return False
        current_page = self.current_page()
        if not current_page:
            if show_errors:
                self.show_error("No page selected", "Select a page before editing render boxes.")
            return False

        cache_path = render_json_path(self.current_project, current_page)
        if not cache_path.exists():
            self.image_preview.set_box_edit_mode(False)
            if show_errors:
                self.show_error("Render JSON missing", "Prepare or run Render first before editing render boxes.")
            return False

        try:
            render_data = load_render_json(cache_path)
            self._render_edit_items = load_render_edit_items(cache_path)
        except Exception as exc:
            self.image_preview.set_box_edit_mode(False)
            if show_errors:
                self.show_error("Invalid render cache", str(exc))
            else:
                self.log(f"Failed to load editable render boxes from {cache_path}: {exc}")
            return False

        self.current_render_data = render_data
        self.current_render_cache_path = cache_path
        self.render_panel.set_data(render_data)
        self.render_panel.set_box_edit_mode_checked(True)
        self.render_panel.set_box_dirty(False)
        self.render_panel.set_selected_box(None)
        self._update_render_box_stale_warning(render_data)
        self._apply_render_edit_session_to_preview(preserve_dirty=False)
        self._refresh_preview_for_current_page()
        return True

    def _apply_render_edit_session_to_preview(self, *, preserve_dirty: bool) -> None:
        if not self.render_panel.box_edit_mode_enabled():
            return
        overlay_items = self._render_overlay_items_for_preview(self._render_edit_items)
        selected_item_id = self.render_panel.current_table_item_id()
        was_dirty = preserve_dirty and self.render_panel.has_unsaved_box_edits()
        self.image_preview.load_editable_boxes(overlay_items)
        self.image_preview.set_box_edit_mode(True)
        self.image_preview.set_editable_box_category_filter("render_bbox")
        self.image_preview.set_show_excluded_boxes(self.render_panel.show_excluded_items_enabled())
        if selected_item_id is not None:
            self.image_preview.select_editable_box("render_bbox", selected_item_id)
        if was_dirty:
            self.image_preview.set_editable_boxes_dirty(True)
            self.render_panel.set_box_dirty(True)

    def save_render_box_edits(self) -> None:
        if self.current_project is None:
            self.show_error("No project open", "Open a project before saving render box edits.")
            return
        image_relative_path = self.current_page()
        if not image_relative_path:
            self.show_error("No page selected", "Select a page before saving render box edits.")
            return
        cache_path = render_json_path(self.current_project, image_relative_path)
        if not cache_path.exists():
            self.show_error("Render JSON missing", "Prepare or run Render first before editing render boxes.")
            return

        self._render_edit_items = self._render_items_from_overlay_snapshot(self.image_preview.editable_boxes_snapshot())
        try:
            render_data = save_render_edit_items(cache_path, self._render_edit_items, mark_edited=True)
            self._render_edit_items = load_render_edit_items(cache_path)
        except Exception as exc:
            self.show_error("Failed to save render box edits", str(exc))
            return

        self.current_render_data = render_data
        self.current_render_cache_path = cache_path
        self.render_panel.set_data(render_data)
        self._update_project_render_stage_status(image_relative_path, render_data)
        self._mark_render_downstream_stale(image_relative_path)
        self._persist_project(show_errors=False)
        self._update_render_box_stale_warning(render_data)
        self._reload_current_page_cached_views()
        self._apply_render_edit_session_to_preview(preserve_dirty=False)
        self._refresh_stage_statuses()
        self._refresh_preview_for_current_page()
        self.statusBar().showMessage("Render box edits saved. Canon state updated.")
        self.log(f"Saved render box edits for {Path(image_relative_path).name}")
        self.log("Canon state updated. Downstream stages will use the edited boxes.")

    def cancel_render_box_edits(self) -> None:
        if self.current_project is None or not self.render_panel.box_edit_mode_enabled():
            return
        if not self._load_render_edit_session_for_current_page(show_errors=True):
            self.image_preview.set_box_edit_mode(False)
            self.render_panel.set_box_edit_mode_checked(False)
            return
        self.statusBar().showMessage("Discarded unsaved render box edits")
        self.log("Discarded unsaved render box edits.")

    def reload_render_boxes_from_cache(self) -> None:
        if not self._ensure_current_editor_changes_resolved():
            return
        if self._load_render_edit_session_for_current_page(show_errors=True):
            self.statusBar().showMessage("Reloaded editable render boxes")

    def exclude_selected_render_item(self) -> None:
        if not self.image_preview.exclude_selected_box(True):
            self.show_error("No render item selected", "Select a render item box before excluding it.")
            return
        self.statusBar().showMessage("Selected render item marked excluded")

    def restore_selected_render_item(self) -> None:
        if not self.image_preview.exclude_selected_box(False):
            self.show_error("No excluded render item selected", "Select an excluded render item before restoring it.")
            return
        self.statusBar().showMessage("Selected render item restored")

    def _mark_render_downstream_stale(self, image_relative_path: str) -> None:
        if self.current_project is None:
            return
        for stage_name in ("render", "export"):
            self.current_project.set_stage_stale(image_relative_path, stage_name, True)

    def _update_render_box_stale_warning(self, render_data: dict[str, Any] | None) -> None:
        if not isinstance(render_data, dict):
            self.render_panel.set_box_warning(None)
            return
        if bool(render_data.get("needs_render", False)):
            self.render_panel.set_box_warning("Render boxes changed. Re-render is recommended.")
        else:
            self.render_panel.set_box_warning(None)

    def _render_overlay_items_for_preview(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        overlay_items: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            overlay_item = dict(item)
            overlay_item["category"] = "render_bbox"
            overlay_item["_original_render_bbox"] = item.get("render_bbox")
            overlay_item["bbox"] = item.get("render_bbox")
            overlay_items.append(overlay_item)
        return overlay_items

    def _render_overlay_item_to_full_item(self, overlay_item: dict[str, Any]) -> dict[str, Any]:
        item = dict(overlay_item)
        item["render_bbox"] = item.get("bbox")
        item.pop("category", None)
        item.pop("_original_render_bbox", None)
        return item

    def _render_items_from_overlay_snapshot(self, snapshot: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self._render_overlay_item_to_full_item(item) for item in snapshot if isinstance(item, dict)]

    def _on_ocr_editor_cache_updated(self, payload: object) -> None:
        if not isinstance(payload, dict) or self.current_project is None:
            return
        current_page = self.current_page()
        if not current_page:
            return

        ocr_data = payload.get("ocr_data")
        cache_path_value = str(payload.get("cache_path", "") or "").strip()
        if isinstance(ocr_data, dict):
            self.current_ocr_data = ocr_data
        if cache_path_value:
            self.current_ocr_cache_path = Path(cache_path_value)

        if self.current_ocr_data is not None:
            self._update_project_ocr_stage_status(current_page, self.current_ocr_data)
            self.translation_panel.set_ocr_context(
                self.current_project.root_dir,
                self.current_ocr_data,
                self.current_ocr_cache_path,
            )
        self._persist_project(show_errors=False)
        self._refresh_stage_statuses()
        self.statusBar().showMessage(str(payload.get("message", "Saved OCR text")))

    def _on_translation_editor_cache_updated(self, payload: object) -> None:
        if not isinstance(payload, dict) or self.current_project is None:
            return
        current_page = self.current_page()
        if not current_page:
            return

        translation_data = payload.get("translation_data")
        translation_cache_path_value = str(payload.get("translation_cache_path", "") or "").strip()
        if isinstance(translation_data, dict):
            self.current_translation_data = translation_data
        if translation_cache_path_value:
            self.current_translation_cache_path = Path(translation_cache_path_value)

        ocr_data = payload.get("ocr_data")
        ocr_cache_path_value = str(payload.get("ocr_cache_path", "") or "").strip()
        if isinstance(ocr_data, dict):
            self.current_ocr_data = ocr_data
            if ocr_cache_path_value:
                self.current_ocr_cache_path = Path(ocr_cache_path_value)
            self.ocr_panel.set_items(
                self.current_ocr_data.get("items", []),
                self.current_ocr_cache_path,
            )
            self.translation_panel.set_ocr_context(
                self.current_project.root_dir,
                self.current_ocr_data,
                self.current_ocr_cache_path,
            )

        if self.current_translation_data is not None:
            self._update_project_translation_stage_status(current_page, self.current_translation_data)
        if self.current_ocr_data is not None:
            self._update_project_ocr_stage_status(current_page, self.current_ocr_data)
        self._persist_project(show_errors=False)
        self._refresh_stage_statuses()
        self.statusBar().showMessage(str(payload.get("message", "Saved translation text")))

    def closeEvent(self, event: QCloseEvent) -> None:  # pragma: no cover - GUI shutdown path.
        if not self._ensure_current_editor_changes_resolved():
            event.ignore()
            return
        if self._workflow_busy():
            self.show_error(
                "Workflow still running",
                "Stop or wait for the current workflow task before closing the app.",
            )
            event.ignore()
            return
        try:
            self._persist_workspace_state()
        except Exception as exc:
            self.log(f"Settings save failure during close: {exc}")
        try:
            if self.service_manager is not None:
                self.service_manager.shutdown()
        except Exception as exc:
            self.log(f"Resident service shutdown failure: {exc}", level="warning")
        super().closeEvent(event)

    def resizeEvent(self, event) -> None:  # pragma: no cover - GUI geometry path.
        super().resizeEvent(event)
        self._position_startup_overlay()

    def current_page(self) -> str | None:
        if self.current_project is None:
            return None
        index = self._current_page_index()
        if index is None:
            return None
        return self.current_project.page_relative_path_for_index(index)

    def current_project_value(self) -> MangaProject | None:
        return self.current_project

    def toggle_developer_log(self, checked: bool | None = None) -> None:
        if checked is None:
            visible = bool(self.developer_log_dock is not None and self.developer_log_dock.isVisible())
            self._set_developer_log_visible(not visible)
            return
        self._set_developer_log_visible(bool(checked))

    def _set_developer_log_visible(self, visible: bool) -> None:
        if self.developer_log_dock is None:
            self.log("Developer log could not be opened because the dock was not created.")
            return
        self.developer_log_dock.setVisible(bool(visible))
        self.left_toolbar.set_log_button_checked(bool(visible))
        if visible:
            self._set_log_alert_state(False)
        if self.toggle_developer_log_action is not None:
            self.toggle_developer_log_action.blockSignals(True)
            self.toggle_developer_log_action.setChecked(bool(visible))
            self.toggle_developer_log_action.blockSignals(False)

    def _navigate_to_page_offset(self, offset: int) -> None:
        if self.current_project is None or self.current_project.page_count == 0:
            return
        current_index = self._current_page_index()
        if current_index is None:
            current_index = 0
        target_index = max(0, min(current_index + int(offset), self.current_project.page_count - 1))
        self._navigate_to_page_index(target_index)

    def _navigate_to_page_index(self, index: int) -> None:
        if self.current_project is None or self.current_project.page_count == 0:
            return
        if index < 0 or index >= self.current_project.page_count:
            return
        current_index = self._current_page_index()
        self._select_page_row(index, user_initiated=True)
        if current_index == index:
            self._load_page_state_for_index(index, user_initiated=True)

    def _navigate_to_last_page(self) -> None:
        if self.current_project is None or self.current_project.page_count == 0:
            return
        self._navigate_to_page_index(self.current_project.page_count - 1)

    def _set_log_alert_state(self, has_alert: bool, *, error: bool = False) -> None:
        self._has_unread_log_alert = bool(has_alert)
        self._has_unread_error_alert = bool(has_alert and error)
        self.left_toolbar.set_log_alert(self._has_unread_log_alert, error=self._has_unread_error_alert)

    def _invalidate_page_statuses(self, image_relative_paths: list[str] | tuple[str, ...] | set[str] | None = None) -> None:
        if image_relative_paths is None:
            self._page_status_cache.clear()
            return

        for image_relative_path in image_relative_paths:
            normalized = str(image_relative_path or "").strip()
            if normalized:
                self._page_status_cache.pop(normalized, None)

    def refresh_page_statuses(
        self,
        image_relative_paths: list[str] | tuple[str, ...] | set[str] | None = None,
        *,
        update_filmstrip: bool = True,
    ) -> dict[str, str]:
        if self.current_project is None:
            self._page_status_cache.clear()
            return {}

        if image_relative_paths is None:
            target_paths = self._all_page_relative_paths()
        else:
            seen: set[str] = set()
            target_paths = []
            for image_relative_path in image_relative_paths:
                normalized = str(image_relative_path or "").strip()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                target_paths.append(normalized)

        for image_relative_path in target_paths:
            current_detection_data = None
            if image_relative_path == self.current_page() and isinstance(self.current_detection_data, dict):
                current_detection_data = self.current_detection_data
            status_payload = get_page_workflow_status(
                self.current_project,
                image_relative_path,
                current_detection_data=current_detection_data,
                processing_page=self._processing_page_relative_path,
            )
            self._page_status_cache[image_relative_path] = status_payload.status

        status_map = {
            page_relative_path: self._page_status_cache.get(page_relative_path, "missing")
            for page_relative_path in self._all_page_relative_paths()
        }
        if update_filmstrip:
            self.page_filmstrip.set_page_statuses(status_map)
        return status_map

    def _page_status_map(self) -> dict[str, str]:
        if self.current_project is None:
            return {}

        page_paths = self._all_page_relative_paths()
        missing_paths = [image_relative_path for image_relative_path in page_paths if image_relative_path not in self._page_status_cache]
        if missing_paths:
            self.refresh_page_statuses(missing_paths, update_filmstrip=False)
        return {
            image_relative_path: self._page_status_cache.get(image_relative_path, "missing")
            for image_relative_path in page_paths
        }

    def _select_page_by_relative_path(self, image_relative_path: str, *, user_initiated: bool) -> bool:
        if self.current_project is None:
            return False
        for index in range(self.current_project.page_count):
            if self.current_project.page_relative_path_for_index(index) != image_relative_path:
                continue
            current_index = self._current_page_index()
            self._select_page_row(index, user_initiated=user_initiated)
            if current_index == index:
                self._load_page_state_for_index(index, user_initiated=user_initiated)
            return True
        return False

    def _on_page_order_changed(self, ordered_paths: list[str]) -> None:
        if self.current_project is None:
            return
        if self._active_workers or self._busy_stages:
            self.log("Page reorder ignored because workflow tasks are still running.")
            self._refresh_project_view()
            return
        previous_order = list(self.current_project.data.source_images)
        scroll_value = self.page_filmstrip.horizontal_scroll_value()
        try:
            self.current_project.reorder_pages(ordered_paths)
            self.current_project.save()
        except Exception as exc:
            try:
                self.current_project.reorder_pages(previous_order)
            except Exception as revert_exc:
                self.log(f"Page reorder revert failed: {revert_exc}")
            self.log(f"Page reorder failed: {exc}")
            self.show_error("Failed to reorder pages", str(exc))
            self._refresh_project_view()
            return

        self._invalidate_page_statuses()
        self._refresh_project_view()
        self.page_filmstrip.set_horizontal_scroll_value(scroll_value)
        self._persist_workspace_state()
        self.log("Page order updated.")
        self.statusBar().showMessage("Page order updated.")

    def _refresh_project_view(self) -> None:
        if self.current_project is None:
            self._process_stage_status = "missing"
            self._process_active_stage_key = None
            self.process_panel.reset_process_state(scope_text="Current page")
            self._invalidate_page_statuses()
            self._processing_page_relative_path = None
            self.page_filmstrip.set_pages(None, [])
            self._clear_loaded_page_state()
            self.last_export_result = None
            self.export_panel.clear_summary()
            self.export_panel.set_output_dir("")
            self.app_settings.remove("workspace/selected_page_relative_path")
            self.header.set_project_name("No Project Open")
            self.header.set_page_name("No page selected")
            self.project_panel.set_project_details(
                project_name=None,
                project_root=None,
                source_dir=None,
                page_count=0,
                current_page_name=None,
            )
            self._update_window_title()
            self._refresh_stage_statuses()
            return

        self._load_export_preferences()
        self._invalidate_page_statuses()
        selected_index = self.current_project.data.current_page_index
        self.page_filmstrip.set_pages(
            self.current_project.root_dir,
            list(self.current_project.data.source_images),
            selected_index=selected_index,
            status_map=self._page_status_map(),
        )
        self.page_filmstrip.set_horizontal_scroll_value(self._pending_filmstrip_scroll_value)

        if self.current_project.page_count == 0:
            self._clear_loaded_page_state()
            self.last_export_result = None
            self.export_panel.clear_summary()
            self.app_settings.remove("workspace/selected_page_relative_path")
            self.header.set_project_name(self.current_project.data.name)
            self.header.set_page_name("No page selected")
            self.project_panel.set_project_details(
                project_name=self.current_project.data.name,
                project_root=str(self.current_project.root_dir),
                source_dir=str(self.current_project.source_dir),
                page_count=0,
                current_page_name=None,
            )
            self._update_window_title()
            self.statusBar().showMessage("Project ready")
            self._refresh_stage_statuses()
            return

        self._update_window_title()
        self._load_page_state_for_index(selected_index, user_initiated=False)

    def _on_page_selected(self, index: int) -> None:
        if not self._programmatic_page_selection and not self._ensure_current_editor_changes_resolved():
            previous_index = self.current_project.data.current_page_index if self.current_project is not None else -1
            if previous_index >= 0:
                self._select_page_row(previous_index, user_initiated=False)
            return
        if self._active_batch_follow_stage is not None and not self._programmatic_page_selection:
            self._follow_batch_paused = True
            self.log("Follow batch progress paused because you selected a different page manually.")
        self._load_page_state_for_index(index, user_initiated=not self._programmatic_page_selection)

    def _select_page_row(self, index: int, *, user_initiated: bool) -> None:
        if self.current_project is None:
            return
        if index < 0 or index >= self.current_project.page_count:
            return
        self._programmatic_page_selection = not user_initiated
        try:
            self.page_filmstrip.set_current_row(index)
        finally:
            self._programmatic_page_selection = False

    def _load_page_state_for_index(self, index: int, *, user_initiated: bool) -> None:
        if self.current_project is None:
            return

        self.current_project.set_current_page(index)
        self.app_settings.set_value(
            "workspace/selected_page_relative_path",
            self.current_project.page_relative_path_for_index(index) or "",
        )
        page_path = self.current_project.image_path_for_index(index)
        page_name = self.current_project.page_display_names()[index]
        image_relative_path = self.current_project.page_relative_path_for_index(index)

        self.header.set_project_name(self.current_project.data.name)
        self.header.set_page_name(page_name)
        self.project_panel.set_project_details(
            project_name=self.current_project.data.name,
            project_root=str(self.current_project.root_dir),
            source_dir=str(self.current_project.source_dir),
            page_count=self.current_project.page_count,
            current_page_name=page_name,
        )

        if page_path is None or image_relative_path is None or not page_path.exists():
            self._clear_loaded_page_state(source_image=image_relative_path)
            if user_initiated:
                self.log(f"Missing source image: {page_name}")
            self.statusBar().showMessage("Source image is missing")
            self._refresh_stage_statuses()
            return

        self.ocr_panel.set_project_root(self.current_project.root_dir)
        self.preview_detection_overlay_enabled = True
        has_overlay = self._load_cached_detection_for_index(index, show_errors=False)
        if self.current_stage_key == "detection" and self.detection_panel.edit_mode_enabled() and not has_overlay:
            self.image_preview.set_box_edit_mode(False)
            self.detection_panel.set_edit_mode_checked(False)
            self.detection_panel.set_create_box_checked(False)
            self.detection_panel.set_stale_warning("Run Detection first before editing boxes on this page.")
        has_ocr = self._load_cached_ocr_for_index(index, show_errors=False)
        if self.current_stage_key == "ocr" and self.ocr_panel.box_edit_mode_enabled() and not has_ocr:
            self.image_preview.set_box_edit_mode(False)
            self.ocr_panel.set_box_edit_mode_checked(False)
            self.ocr_panel.set_box_warning("Prepare OCR items first before editing OCR boxes on this page.")
        has_translation = self._load_cached_translation_for_index(index, show_errors=False)
        has_inpaint = self._load_cached_inpaint_for_index(index, show_errors=False)
        has_render = self._load_cached_render_for_index(index, show_errors=False)
        if self.current_stage_key == "render" and self.render_panel.box_edit_mode_enabled() and not has_render:
            self.image_preview.set_box_edit_mode(False)
            self.render_panel.set_box_edit_mode_checked(False)
            self.render_panel.set_box_warning("Prepare or run Render first before editing render boxes on this page.")

        if not self._refresh_preview_for_current_page():
            self._clear_loaded_page_state(source_image=image_relative_path)
            if user_initiated:
                self.log(f"Unable to preview page: {page_name}")
            self.statusBar().showMessage("Preview unavailable")
            self._refresh_stage_statuses()
            return

        self._refresh_stage_statuses()
        if has_render:
            self.statusBar().showMessage(
                f"Showing page {index + 1} of {self.current_project.page_count} with cached render output"
            )
        elif has_inpaint:
            self.statusBar().showMessage(
                f"Showing page {index + 1} of {self.current_project.page_count} with cached inpaint output"
            )
        elif has_translation:
            self.statusBar().showMessage(
                f"Showing page {index + 1} of {self.current_project.page_count} with cached translation"
            )
        elif has_ocr:
            self.statusBar().showMessage(
                f"Showing page {index + 1} of {self.current_project.page_count} with cached OCR items"
            )
        elif has_overlay:
            self.statusBar().showMessage(
                f"Showing page {index + 1} of {self.current_project.page_count} with detection overlay"
            )
        else:
            self.statusBar().showMessage(f"Showing page {index + 1} of {self.current_project.page_count}")

        if user_initiated:
            self.log(f"Selected page: {page_name}")

    def _clear_loaded_page_state(self, *, source_image: str | None = None) -> None:
        self.current_detection_data = None
        self._ocr_edit_items = []
        self.current_ocr_data = None
        self.current_ocr_cache_path = None
        self.current_translation_data = None
        self.current_translation_cache_path = None
        self.current_inpaint_data = None
        self.current_inpaint_cache_path = None
        self._render_edit_items = []
        self.current_render_data = None
        self.current_render_cache_path = None
        self.image_preview.clear_image()
        self.detection_panel.clear_detection_data()
        self.detection_panel.set_edit_mode_checked(False)
        self.detection_panel.set_create_box_checked(False)
        self.ocr_panel.set_project_root(None)
        self.ocr_panel.clear_view()
        self.ocr_panel.set_box_edit_mode_checked(False)
        self.translation_panel.set_ocr_context(None, None, None)
        self.translation_panel.clear_view()
        self.inpaint_panel.clear_view(source_image=source_image)
        self.render_panel.clear_view()
        self.render_panel.set_box_edit_mode_checked(False)

    def _ensure_detection_canon_state(self, cache_path: Path) -> dict[str, Any]:
        detection_data = load_detection_json(cache_path)
        had_canon_state = isinstance(detection_data.get("canon_state"), dict)
        ensure_canon_state(detection_data)
        if not had_canon_state:
            save_detection_json(cache_path, detection_data)
        return detection_data

    def _sync_translation_items_with_canon(
        self,
        image_relative_path: str,
        translation_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if self.current_project is None:
            return list(translation_data.get("items", []))

        detection_cache_path = detection_json_path(self.current_project, image_relative_path)
        if not detection_cache_path.exists():
            return list(translation_data.get("items", []))

        try:
            detection_data = self._ensure_detection_canon_state(detection_cache_path)
        except Exception as exc:
            self.log(f"Failed to sync translation items with canon_state from {detection_cache_path}: {exc}")
            return list(translation_data.get("items", []))

        canon_state = detection_data.get("canon_state", {})
        ocr_items_by_id: dict[int, dict[str, Any]] = {}
        ocr_cache_path = ocr_json_path(self.current_project, image_relative_path)
        if ocr_cache_path.exists():
            try:
                ocr_payload = load_ocr_json(ocr_cache_path)
            except Exception as exc:
                self.log(f"Failed to load OCR cache for translation canon sync {ocr_cache_path}: {exc}")
            else:
                for index, item in enumerate(ocr_payload.get("items", [])):
                    if not isinstance(item, dict):
                        continue
                    try:
                        item_id = int(item.get("id", index))
                    except Exception:
                        item_id = index
                    ocr_items_by_id[item_id] = dict(item)

        synced_items: list[dict[str, Any]] = []
        for index, item in enumerate(translation_data.get("items", [])):
            if not isinstance(item, dict):
                continue
            synced_item = dict(item)
            ocr_item_id = int(synced_item.get("ocr_item_id", index) or index)
            ocr_item = ocr_items_by_id.get(ocr_item_id, {})
            canon_item = resolve_canon_item_for_stage_item(canon_state, synced_item, active_only=False)
            if canon_item is None and ocr_item:
                canon_item = resolve_canon_item_for_stage_item(canon_state, ocr_item, active_only=False)
            if canon_item is None:
                synced_items.append(synced_item)
                continue
            if not bool(canon_item.get("enabled", True)):
                continue
            synced_item["canon_id"] = str(canon_item.get("canon_id", "") or "")
            synced_item["bbox"] = canon_item_bbox(canon_item, "bbox")
            synced_item["ocr_bbox"] = canon_item_bbox(canon_item, "ocr_bbox")
            synced_item["excluded"] = False
            synced_items.append(synced_item)
        return synced_items

    def _reload_current_page_cached_views(self) -> None:
        index = self._current_page_index()
        if index is None:
            return
        self._load_cached_ocr_for_index(index, show_errors=False)
        self._load_cached_translation_for_index(index, show_errors=False)
        self._load_cached_inpaint_for_index(index, show_errors=False)
        self._load_cached_render_for_index(index, show_errors=False)

    def _load_cached_detection_for_index(
        self,
        index: int,
        *,
        show_errors: bool,
        persist_stage_status: bool = False,
    ) -> bool:
        if self.current_project is None:
            return False

        image_relative_path = self.current_project.page_relative_path_for_index(index)
        if image_relative_path is None:
            self.current_detection_data = None
            self.detection_panel.clear_detection_data()
            return False

        cache_path = detection_json_path(self.current_project, image_relative_path)
        if not cache_path.exists():
            self.current_detection_data = None
            self.detection_panel.clear_detection_data(str(cache_path))
            return False

        try:
            detection_data = self._ensure_detection_canon_state(cache_path)
        except Exception as exc:
            self.current_detection_data = None
            self.detection_panel.clear_detection_data(str(cache_path))
            self.log(f"Failed to load detection cache {cache_path}: {exc}")
            if show_errors:
                self.show_error("Invalid detection cache", str(exc))
            return False

        self.current_detection_data = detection_data
        self.preview_detection_overlay_enabled = True
        self.detection_panel.set_detection_data(detection_data, str(cache_path))
        self._update_detection_stale_warning(detection_data)

        if persist_stage_status:
            self.current_project.update_stage_status(
                image_relative_path,
                "detection",
                status="done",
                cache_path=self._relative_project_path(cache_path),
            )
            self._persist_project(show_errors=show_errors)

        if self.current_stage_key == "detection" and self.detection_panel.edit_mode_enabled():
            try:
                self.image_preview.load_editable_boxes(load_detection_edit_items(cache_path))
                self.image_preview.set_box_edit_mode(True)
                self.image_preview.set_editable_box_category_filter(self.detection_panel.selected_box_category())
                self.image_preview.set_show_excluded_boxes(self.detection_panel.show_excluded_enabled())
            except Exception as exc:
                self.log(f"Failed to reload detection edit session from {cache_path}: {exc}")
                self.image_preview.set_box_edit_mode(False)
                self.detection_panel.set_edit_mode_checked(False)

        return True

    def _load_cached_ocr_for_index(
        self,
        index: int,
        *,
        show_errors: bool,
        persist_stage_status: bool = False,
    ) -> bool:
        if self.current_project is None:
            return False

        image_relative_path = self.current_project.page_relative_path_for_index(index)
        if image_relative_path is None:
            self._ocr_edit_items = []
            self.current_ocr_data = None
            self.current_ocr_cache_path = None
            self.ocr_panel.clear_view()
            self.translation_panel.set_ocr_context(self.current_project.root_dir, None, None)
            return False

        cache_path = ocr_json_path(self.current_project, image_relative_path)
        if not cache_path.exists():
            self._ocr_edit_items = []
            self.current_ocr_data = None
            self.current_ocr_cache_path = None
            self.ocr_panel.clear_view()
            self.translation_panel.set_ocr_context(self.current_project.root_dir, None, None)
            return False

        try:
            ocr_data = load_ocr_json(cache_path)
            try:
                canon_synced_items = load_ocr_edit_items(cache_path)
            except Exception as sync_exc:
                self.log(f"Failed to sync OCR items with canon_state from {cache_path}: {sync_exc}")
                canon_synced_items = list(ocr_data.get("items", []))
        except Exception as exc:
            self._ocr_edit_items = []
            self.current_ocr_data = None
            self.current_ocr_cache_path = None
            self.ocr_panel.clear_view()
            self.translation_panel.set_ocr_context(self.current_project.root_dir, None, None)
            self.log(f"Failed to load OCR cache {cache_path}: {exc}")
            if show_errors:
                self.show_error("Invalid OCR cache", str(exc))
            return False

        self.current_ocr_data = dict(ocr_data)
        self.current_ocr_data["items"] = canon_synced_items
        self.current_ocr_cache_path = cache_path
        self.ocr_panel.set_items(canon_synced_items, cache_path)
        self.translation_panel.set_ocr_context(self.current_project.root_dir, self.current_ocr_data, cache_path)
        self._update_ocr_box_stale_warning(self.current_ocr_data)

        if persist_stage_status:
            self._update_project_ocr_stage_status(image_relative_path, self.current_ocr_data)
            self._persist_project(show_errors=show_errors)
        if self.current_stage_key == "ocr" and self.ocr_panel.box_edit_mode_enabled():
            try:
                self._ocr_edit_items = load_ocr_edit_items(cache_path)
                self._apply_ocr_edit_session_to_preview(preserve_dirty=False)
            except Exception as exc:
                self.log(f"Failed to reload OCR edit session from {cache_path}: {exc}")
                self.image_preview.set_box_edit_mode(False)
                self.ocr_panel.set_box_edit_mode_checked(False)
        return True

    def _load_cached_translation_for_index(
        self,
        index: int,
        *,
        show_errors: bool,
        persist_stage_status: bool = False,
    ) -> bool:
        if self.current_project is None:
            return False

        image_relative_path = self.current_project.page_relative_path_for_index(index)
        if image_relative_path is None:
            self.current_translation_data = None
            self.current_translation_cache_path = None
            self.translation_panel.clear_view()
            return False

        cache_path = translation_json_path(self.current_project, image_relative_path)
        if not cache_path.exists():
            self.current_translation_data = None
            self.current_translation_cache_path = None
            self.translation_panel.clear_view()
            return False

        try:
            translation_data = load_translation_json(cache_path)
        except Exception as exc:
            self.current_translation_data = None
            self.current_translation_cache_path = None
            self.translation_panel.clear_view()
            self.log(f"Failed to load translation cache {cache_path}: {exc}")
            if show_errors:
                self.show_error("Invalid translation cache", str(exc))
            return False

        synced_items = self._sync_translation_items_with_canon(image_relative_path, translation_data)
        self.current_translation_data = dict(translation_data)
        self.current_translation_data["items"] = synced_items
        self.current_translation_cache_path = cache_path
        self.translation_panel.set_items(synced_items, cache_path)

        if persist_stage_status:
            self._update_project_translation_stage_status(image_relative_path, self.current_translation_data)
            self._persist_project(show_errors=show_errors)
        return True

    def _load_cached_inpaint_for_index(
        self,
        index: int,
        *,
        show_errors: bool,
        persist_stage_status: bool = False,
    ) -> bool:
        if self.current_project is None:
            return False

        image_relative_path = self.current_project.page_relative_path_for_index(index)
        if image_relative_path is None:
            self.current_inpaint_data = None
            self.current_inpaint_cache_path = None
            self.inpaint_panel.clear_view()
            return False

        cache_path = inpaint_json_path(self.current_project, image_relative_path)
        expected_output_path = inpaint_image_path(self.current_project, image_relative_path)
        if not cache_path.exists():
            self.current_inpaint_data = None
            self.current_inpaint_cache_path = None
            self.inpaint_panel.clear_view(
                source_image=image_relative_path,
                output_image=str(expected_output_path) if expected_output_path.exists() else None,
            )
            self._sync_lama_status_label()
            return expected_output_path.exists()

        try:
            inpaint_data = load_inpaint_json(cache_path)
        except Exception as exc:
            self.current_inpaint_data = None
            self.current_inpaint_cache_path = None
            self.inpaint_panel.clear_view(
                source_image=image_relative_path,
                output_image=str(expected_output_path) if expected_output_path.exists() else None,
            )
            self._sync_lama_status_label()
            self.log(f"Failed to load inpaint cache {cache_path}: {exc}")
            if show_errors:
                self.show_error("Invalid inpaint cache", str(exc))
            return expected_output_path.exists()

        self.current_inpaint_data = inpaint_data
        self.current_inpaint_cache_path = cache_path
        self.inpaint_panel.set_metadata(inpaint_data)
        self._sync_lama_status_label()

        if persist_stage_status:
            self._update_project_inpaint_stage_status(image_relative_path, inpaint_data)
            self._persist_project(show_errors=show_errors)
        return True

    def _load_cached_render_for_index(
        self,
        index: int,
        *,
        show_errors: bool,
        persist_stage_status: bool = False,
    ) -> bool:
        if self.current_project is None:
            return False

        image_relative_path = self.current_project.page_relative_path_for_index(index)
        if image_relative_path is None:
            self._render_edit_items = []
            self.current_render_data = None
            self.current_render_cache_path = None
            self.render_panel.clear_view()
            return False

        cache_path = render_json_path(self.current_project, image_relative_path)
        expected_output_path = render_image_path(self.current_project, image_relative_path)
        if not cache_path.exists():
            self._render_edit_items = []
            self.current_render_data = None
            self.current_render_cache_path = None
            self.render_panel.clear_view(
                output_image=str(expected_output_path) if expected_output_path.exists() else None
            )
            return expected_output_path.exists()

        try:
            render_data = load_render_json(cache_path)
            try:
                canon_synced_items = load_render_edit_items(cache_path)
            except Exception as sync_exc:
                self.log(f"Failed to sync render items with canon_state from {cache_path}: {sync_exc}")
                canon_synced_items = list(render_data.get("items", []))
        except Exception as exc:
            self._render_edit_items = []
            self.current_render_data = None
            self.current_render_cache_path = None
            self.render_panel.clear_view(
                output_image=str(expected_output_path) if expected_output_path.exists() else None
            )
            self.log(f"Failed to load render cache {cache_path}: {exc}")
            if show_errors:
                self.show_error("Invalid render cache", str(exc))
            return expected_output_path.exists()

        self.current_render_data = dict(render_data)
        self.current_render_data["items"] = canon_synced_items
        self.current_render_cache_path = cache_path
        self.render_panel.set_data(self.current_render_data)
        self._update_render_box_stale_warning(self.current_render_data)

        if persist_stage_status:
            self._update_project_render_stage_status(image_relative_path, self.current_render_data)
            self._persist_project(show_errors=show_errors)
        if self.current_stage_key == "render" and self.render_panel.box_edit_mode_enabled():
            try:
                self._render_edit_items = load_render_edit_items(cache_path)
                self._apply_render_edit_session_to_preview(preserve_dirty=False)
            except Exception as exc:
                self.log(f"Failed to reload render edit session from {cache_path}: {exc}")
                self.image_preview.set_box_edit_mode(False)
                self.render_panel.set_box_edit_mode_checked(False)
        return True

    def _load_export_preferences(self) -> None:
        if self.current_project is None:
            self.last_export_result = None
            self.export_panel.clear_summary()
            self.export_panel.set_output_dir("")
            return

        export_settings = self._project_export_settings(create=False)
        output_dir = str(
            export_settings.get("output_dir", "") if isinstance(export_settings, dict) else ""
        ).strip()
        if not output_dir:
            output_dir = str(self.current_project.root_dir / "exports")
        self.export_panel.set_output_dir(output_dir)

        if not self._load_last_export_summary(show_errors=False):
            self.last_export_result = None
            self.export_panel.clear_summary()

    def _project_export_settings(self, *, create: bool) -> dict[str, Any]:
        if self.current_project is None:
            return {}

        settings = self.current_project.data.settings
        export_settings = settings.get("export")
        if isinstance(export_settings, dict):
            return export_settings
        if not create:
            return {}

        export_settings = {}
        settings["export"] = export_settings
        return export_settings

    def _store_export_preferences(
        self,
        *,
        output_dir: str | None = None,
        manifest_path: str | None = None,
        zip_path: str | None = None,
    ) -> None:
        if self.current_project is None:
            return

        export_settings = self._project_export_settings(create=True)
        if output_dir is not None:
            export_settings["output_dir"] = str(output_dir)
        if manifest_path is not None:
            export_settings["last_manifest_path"] = str(manifest_path)
        if zip_path is not None:
            export_settings["last_zip_path"] = str(zip_path)
        self._persist_project(show_errors=False)

    def _load_last_export_summary(self, *, show_errors: bool) -> bool:
        if self.current_project is None:
            return False

        export_settings = self._project_export_settings(create=False)
        manifest_path_value = str(export_settings.get("last_manifest_path", "") or "").strip()
        if not manifest_path_value:
            if self.last_export_result is not None:
                self._set_export_summary(self.last_export_result, persist=False)
                return True
            return False

        manifest_path = Path(manifest_path_value)
        if not manifest_path.is_absolute():
            manifest_path = (self.current_project.root_dir / manifest_path).resolve()

        if not manifest_path.exists():
            self.last_export_result = None
            if show_errors:
                self.log(f"Export manifest not found: {manifest_path}")
            return False

        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.last_export_result = None
            self.log(f"Failed to load export summary {manifest_path}: {exc}")
            if show_errors:
                self.show_error("Invalid export manifest", str(exc))
            return False

        if not isinstance(payload, dict):
            self.last_export_result = None
            if show_errors:
                self.show_error("Invalid export manifest", "The saved export manifest is not a JSON object.")
            return False

        if not payload.get("manifest_path"):
            payload["manifest_path"] = str(manifest_path)
        self._set_export_summary(payload, persist=False)
        return True

    def _set_export_summary(self, manifest: dict[str, Any] | None, *, persist: bool) -> None:
        self.last_export_result = dict(manifest) if isinstance(manifest, dict) else None
        self.export_panel.set_summary(self.last_export_result)
        if persist and self.last_export_result is not None:
            self._store_export_preferences(
                output_dir=str(self.last_export_result.get("output_dir", "") or self.export_panel.output_dir()),
                manifest_path=str(self.last_export_result.get("manifest_path", "") or ""),
                zip_path=str(self.last_export_result.get("zip_path", "") or ""),
            )
        self._refresh_stage_statuses()

    def _refresh_process_panel_summary(self, *, scope_text: str | None = None) -> None:
        render_style = self.config_panel.render_style_label()
        inpaint_device = self.config_panel.inpaint_device_label()
        target_language = self.config_panel.translation_target_language()
        translator_name = self.config_panel.translation_provider_name()
        ocr_provider = self.config_panel.ocr_provider_label()
        self.process_panel.set_settings_summary(
            ocr_provider=ocr_provider,
            translator=translator_name,
            target_language=target_language,
            inpaint_device=inpaint_device,
            render_style=render_style,
        )
        if scope_text is not None:
            self.process_panel.set_scope_summary(scope_text)

    def _set_process_ui_running(self, running: bool) -> None:
        enabled = not bool(running)
        self.project_panel.set_actions_enabled(enabled)
        self.process_panel.set_actions_enabled(enabled)
        self.process_panel.set_cancel_enabled(bool(running) and not self._process_cancel_requested)
        self.process_panel.set_cancel_stopping(bool(running and self._process_cancel_requested))
        self.detection_panel.set_actions_enabled(enabled)
        self.ocr_panel.set_actions_enabled(enabled)
        self.ocr_panel.set_server_actions_enabled(enabled)
        self.translation_panel.set_actions_enabled(enabled)
        self.inpaint_panel.set_actions_enabled(enabled)
        self.render_panel.set_actions_enabled(enabled)
        self.export_panel.set_actions_enabled(enabled)
        for action in (
            self.new_project_action,
            self.open_project_action,
            self.save_project_action,
            self.import_images_action,
            self.remove_current_page_action,
        ):
            if action is not None:
                action.setEnabled(enabled)

    def _start_process_task(
        self,
        image_relative_paths: list[str],
        *,
        scope: str,
        force: bool,
        ocr_config: OCRConfig,
        translation_config: TranslationConfig,
        inpaint_settings: dict[str, Any],
        render_config: RenderConfig,
    ) -> None:
        if self.current_project is None:
            self.show_error("No project open", "Create or open a project before starting one-click processing.")
            return
        if self._active_process_worker is not None or self._process_is_running():
            self.show_error("Process already running", "A process is already running.")
            return

        self._persist_panel_preferences()
        self._process_stage_status = "running"
        self._process_active_stage_key = None
        self._process_cancel_requested = False
        self._process_restore_page_relative_path = self.current_page()
        scope_text = "Current page" if scope == "current" else "Chapter"
        self._refresh_process_panel_summary(scope_text=scope_text)
        self.process_panel.reset_process_state(scope_text=scope_text)
        self.process_panel.set_stage_status_text("Running")
        self.process_panel.set_action_message("Preparing one-click process...")
        self.process_panel.set_last_error("")

        task = ProcessTask(
            name=f"{'Re-process' if force else 'Process'} {scope_text}",
            stage="process",
            project=self.current_project,
            image_relative_paths=list(image_relative_paths),
            scope=scope,
            force=force,
            ocr_config=ocr_config,
            translation_config=translation_config,
            inpaint_settings=dict(inpaint_settings),
            render_config=render_config,
        )
        try:
            worker = self._submit_service_task(
                task,
                progress_label="Process progress",
                finished_handler=self._on_process_worker_finished,
                failed_handler=lambda message, active_worker: self._on_process_worker_failed(
                    message,
                    active_worker,
                    task,
                ),
                event_handler=lambda payload, process_scope=scope: self._on_process_worker_event(
                    payload,
                    scope=process_scope,
                ),
            )
        except Exception as exc:
            self._process_stage_status = "error"
            self._set_process_ui_running(False)
            self.show_error("Process service unavailable", str(exc))
            return
        self._active_process_worker = worker
        self._set_process_ui_running(True)
        self.header.set_progress_value(0)
        self.statusBar().showMessage("Running one-click process...")

    def _start_export_task(
        self,
        config: ExportConfig,
        *,
        task_name: str,
        current_page: str | None,
        selected_pages: list[str] | None,
    ) -> None:
        if self.current_project is None:
            self.show_error("No project open", "Create or open a project before exporting pages.")
            return

        self._persist_panel_preferences()
        self._store_export_preferences(output_dir=str(config.output_dir))
        task = ExportTask(
            name=task_name,
            stage="export",
            project=self.current_project,
            config=config.to_dict(),
            current_page=current_page,
            selected_pages=list(selected_pages or []),
        )
        try:
            self._submit_service_task(
                task,
                progress_label="Export progress",
                finished_handler=lambda result, active_worker: self._on_export_worker_finished(
                    result,
                    active_worker,
                    task,
                ),
                failed_handler=lambda message, active_worker: self._on_export_worker_failed(
                    message,
                    active_worker,
                    task,
                ),
                event_handler=lambda payload: self._on_worker_event("export", payload, is_batch=config.page_scope != "current"),
            )
        except Exception as exc:
            self.export_panel.set_actions_enabled(True)
            self.show_error("Export service unavailable", str(exc))
            return
        self.export_panel.set_actions_enabled(False)
        self.header.set_progress_value(0)
        self.statusBar().showMessage("Exporting pages...")

    def _open_output_folder(self, output_dir: str | Path) -> bool:
        target_dir = Path(output_dir).expanduser().resolve()
        if not target_dir.exists():
            self.show_error("Output folder missing", f"Output folder not found:\n{target_dir}")
            return False

        try:
            if hasattr(os, "startfile"):
                os.startfile(str(target_dir))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(target_dir)])
            else:
                subprocess.Popen(["xdg-open", str(target_dir)])
        except Exception as exc:
            self.show_error("Failed to open output folder", str(exc))
            return False

        self.log(f"Opened export output folder: {target_dir}")
        return True

    def _start_detection_task(self, image_paths: list[Path], *, task_name: str, force: bool) -> None:
        if self.current_project is None:
            self.show_error("No project open", "Create or open a project before running detection.")
            return
        self._persist_panel_preferences()
        if self.detection_panel.edit_mode_enabled():
            self.detection_panel.set_edit_mode_checked(False)
            self.detection_panel.set_create_box_checked(False)
            self.image_preview.set_box_edit_mode(False)
            self._refresh_preview_for_current_page()

        write_crash_breadcrumb(
            "MainWindow before creating DetectionTask",
            task_name=task_name,
            page_total=len(image_paths),
            force=force,
        )
        task = DetectionTask(
            name=task_name,
            stage="detection",
            image_paths=image_paths,
            detection_cache_dir=self.current_project.cache_dir / "detection",
            masks_cache_dir=self.current_project.cache_dir / "masks",
            force=force,
        )
        write_crash_breadcrumb(
            "MainWindow after creating DetectionTask",
            task_name=task.name,
            page_total=len(task.image_paths),
            force=task.force,
        )
        try:
            write_crash_breadcrumb(
                "MainWindow before _submit_service_task",
                task_name=task.name,
                page_total=len(task.image_paths),
            )
            handle = self._submit_service_task(
                task,
                progress_label="Detection progress",
                finished_handler=self._on_detection_worker_finished,
                failed_handler=lambda message, active_worker: self._on_detection_worker_failed(
                    message,
                    active_worker,
                    task,
                ),
                event_handler=lambda payload: self._on_worker_event("detection", payload, is_batch=len(image_paths) > 1),
            )
            write_crash_breadcrumb(
                "MainWindow after _submit_service_task returns",
                task_name=task.name,
                page_total=len(task.image_paths),
                command_id=getattr(handle, "command_id", ""),
            )
        except Exception as exc:
            self.detection_panel.set_actions_enabled(True)
            self.show_error("Detection service unavailable", str(exc))
            return
        self.detection_panel.set_actions_enabled(False)
        self.header.set_progress_value(0)
        self.statusBar().showMessage("Detection is running...")

    def _start_ocr_preparation_task(
        self,
        image_relative_paths: list[str],
        *,
        task_name: str,
        force: bool,
    ) -> None:
        if self.current_project is None:
            self.show_error("No project open", "Create or open a project before preparing OCR items.")
            return
        self._persist_panel_preferences()

        task = OCRPreparationTask(
            name=task_name,
            stage="ocr",
            project=self.current_project,
            image_relative_paths=image_relative_paths,
            force=force,
            save_crops=True,
        )
        try:
            self._submit_service_task(
                task,
                progress_label="OCR preparation progress",
                finished_handler=self._on_ocr_worker_finished,
                failed_handler=lambda message, active_worker: self._on_ocr_worker_failed(
                    message,
                    active_worker,
                    task,
                ),
                event_handler=lambda payload: self._on_worker_event("ocr_prepare", payload, is_batch=len(image_relative_paths) > 1),
            )
        except Exception as exc:
            self.ocr_panel.set_actions_enabled(True)
            self.show_error("OCR service unavailable", str(exc))
            return
        self.ocr_panel.set_actions_enabled(False)
        self.header.set_progress_value(0)
        self.statusBar().showMessage("Preparing OCR items...")

    def _start_ocr_inference_task(
        self,
        image_relative_paths: list[str],
        *,
        task_name: str,
        selected_item_ids_by_page: dict[str, list[int]] | None,
        force: bool,
        ocr_config: OCRConfig,
    ) -> None:
        if self.current_project is None:
            self.show_error("No project open", "Create or open a project before running OCR.")
            return
        self._persist_panel_preferences()

        task = OCRInferenceTask(
            name=task_name,
            stage="ocr",
            project=self.current_project,
            image_relative_paths=image_relative_paths,
            config=ocr_config.to_metadata(),
            server_url=ocr_config.server_url,
            force=force,
            selected_item_ids_by_page=selected_item_ids_by_page or {},
            timeout=float(ocr_config.timeout),
        )
        try:
            self._submit_service_task(
                task,
                progress_label="OCR progress",
                finished_handler=self._on_ocr_inference_worker_finished,
                failed_handler=lambda message, active_worker: self._on_ocr_inference_worker_failed(
                    message,
                    active_worker,
                    task,
                ),
                event_handler=lambda payload: self._on_worker_event("ocr", payload, is_batch=len(image_relative_paths) > 1),
            )
        except Exception as exc:
            self.ocr_panel.set_actions_enabled(True)
            self.show_error("OCR service unavailable", str(exc))
            return
        self.ocr_panel.set_actions_enabled(False)
        self.header.set_progress_value(0)
        self.statusBar().showMessage(f"Running OCR with {ocr_config.provider_label}...")

    def _start_translation_initialization_task(
        self,
        image_relative_paths: list[str],
        *,
        task_name: str,
        force: bool,
    ) -> None:
        if self.current_project is None:
            self.show_error("No project open", "Create or open a project before initializing translation.")
            return
        self._persist_panel_preferences()

        task = TranslationInitializationTask(
            name=task_name,
            stage="translation",
            project=self.current_project,
            image_relative_paths=image_relative_paths,
            config=self.current_translation_config(),
            force=force,
        )
        try:
            self._submit_service_task(
                task,
                progress_label="Translation initialization progress",
                finished_handler=self._on_translation_init_worker_finished,
                failed_handler=lambda message, active_worker: self._on_translation_init_worker_failed(
                    message,
                    active_worker,
                    task,
                ),
                event_handler=lambda payload: self._on_worker_event("translation_init", payload, is_batch=len(image_relative_paths) > 1),
            )
        except Exception as exc:
            self.translation_panel.set_actions_enabled(True)
            self.show_error("Translation service unavailable", str(exc))
            return
        self.translation_panel.set_actions_enabled(False)
        self.header.set_progress_value(0)
        self.statusBar().showMessage("Initializing translation...")

    def _start_translation_task(
        self,
        image_relative_paths: list[str],
        *,
        task_name: str,
        selected_item_ids_by_page: dict[str, list[int]] | None,
        force: bool,
    ) -> None:
        if self.current_project is None:
            self.show_error("No project open", "Create or open a project before running translation.")
            return
        self._persist_panel_preferences()
        translation_config = self.current_translation_config()

        task = TranslationTask(
            name=task_name,
            stage="translation",
            project=self.current_project,
            image_relative_paths=image_relative_paths,
            config=translation_config,
            force=force,
            selected_item_ids_by_page=selected_item_ids_by_page or {},
        )
        try:
            self._submit_service_task(
                task,
                progress_label="Translation progress",
                finished_handler=self._on_translation_worker_finished,
                failed_handler=lambda message, active_worker: self._on_translation_worker_failed(
                    message,
                    active_worker,
                    task,
                ),
                event_handler=lambda payload: self._on_worker_event("translation", payload, is_batch=len(image_relative_paths) > 1),
            )
        except Exception as exc:
            self.translation_panel.set_actions_enabled(True)
            self.show_error("Translation service unavailable", str(exc))
            return
        self.translation_panel.set_actions_enabled(False)
        self.header.set_progress_value(0)
        self.statusBar().showMessage(f"Running translation with {translation_config.translator}...")

    def _start_inpaint_mask_task(
        self,
        image_relative_paths: list[str],
        *,
        task_name: str,
        force: bool,
    ) -> None:
        if self.current_project is None:
            self.show_error("No project open", "Create or open a project before preparing inpaint masks.")
            return
        self._persist_panel_preferences()

        settings = self.current_inpaint_settings(force=force)
        task = InpaintMaskTask(
            name=task_name,
            stage="inpaint",
            project=self.current_project,
            image_relative_paths=image_relative_paths,
            force=bool(settings["force"]),
            mask_padding=int(settings["mask_padding"]),
            use_bubble_mask=bool(settings["use_bubble_mask"]),
        )
        try:
            self._submit_service_task(
                task,
                progress_label="Inpaint mask preparation progress",
                finished_handler=self._on_inpaint_mask_worker_finished,
                failed_handler=lambda message, active_worker: self._on_inpaint_mask_worker_failed(
                    message,
                    active_worker,
                    task,
                ),
                event_handler=lambda payload: self._on_worker_event("inpaint_mask", payload, is_batch=len(image_relative_paths) > 1),
            )
        except Exception as exc:
            self.inpaint_panel.set_actions_enabled(True)
            self.show_error("Inpaint service unavailable", str(exc))
            return
        self.inpaint_panel.set_actions_enabled(False)
        self.header.set_progress_value(0)
        self.statusBar().showMessage("Preparing inpaint masks...")

    def _start_inpaint_task(
        self,
        image_relative_paths: list[str],
        *,
        task_name: str,
        force: bool,
    ) -> None:
        if self.current_project is None:
            self.show_error("No project open", "Create or open a project before running inpaint.")
            return
        self._persist_panel_preferences()

        settings = self.current_inpaint_settings(force=force)
        task = InpaintTask(
            name=task_name,
            stage="inpaint",
            project=self.current_project,
            image_relative_paths=image_relative_paths,
            force=bool(settings["force"]),
            mask_padding=int(settings["mask_padding"]),
            use_bubble_mask=bool(settings["use_bubble_mask"]),
            use_crop_windows=bool(settings["use_crop_windows"]),
            device=settings["device"],
        )
        try:
            self._submit_service_task(
                task,
                progress_label="Inpaint progress",
                finished_handler=self._on_inpaint_worker_finished,
                failed_handler=lambda message, active_worker: self._on_inpaint_worker_failed(
                    message,
                    active_worker,
                    task,
                ),
                event_handler=lambda payload: self._on_worker_event("inpaint", payload, is_batch=len(image_relative_paths) > 1),
            )
        except Exception as exc:
            self.inpaint_panel.set_actions_enabled(True)
            self.show_error("Inpaint service unavailable", str(exc))
            return
        self.inpaint_panel.set_actions_enabled(False)
        self.header.set_progress_value(0)
        self.statusBar().showMessage("Running inpaint...")

    def _start_lama_model_task(self, action: str) -> None:
        task = LamaModelTask(
            name=f"LaMa model: {action}",
            stage="inpaint",
            action=action,
            device=self.current_inpaint_settings(force=False).get("device"),
        )
        try:
            self._submit_service_task(
                task,
                progress_label=None,
                finished_handler=lambda result, active_worker: self._on_lama_model_worker_finished(
                    result,
                    active_worker,
                    action,
                ),
                failed_handler=lambda message, active_worker: self._on_lama_model_worker_failed(
                    message,
                    active_worker,
                    action,
                ),
            )
        except Exception as exc:
            self.inpaint_panel.set_actions_enabled(True)
            self.show_error("Inpaint service unavailable", str(exc))
            return
        self.inpaint_panel.set_actions_enabled(False)
        self.header.set_progress_value(0)

    def _start_render_preparation_task(
        self,
        image_relative_paths: list[str],
        *,
        task_name: str,
        force: bool,
    ) -> None:
        if self.current_project is None:
            self.show_error("No project open", "Create or open a project before preparing render data.")
            return
        self._persist_panel_preferences()

        task = RenderPreparationTask(
            name=task_name,
            stage="render",
            project=self.current_project,
            image_relative_paths=image_relative_paths,
            force=force,
        )
        try:
            self._submit_service_task(
                task,
                progress_label="Render preparation progress",
                finished_handler=self._on_render_prep_worker_finished,
                failed_handler=lambda message, active_worker: self._on_render_prep_worker_failed(
                    message,
                    active_worker,
                    task,
                ),
                event_handler=lambda payload: self._on_worker_event("render_prepare", payload, is_batch=len(image_relative_paths) > 1),
            )
        except Exception as exc:
            self.render_panel.set_actions_enabled(True)
            self.show_error("Render service unavailable", str(exc))
            return
        self.render_panel.set_actions_enabled(False)
        self.header.set_progress_value(0)
        self.statusBar().showMessage("Preparing render data...")

    def _start_render_task(
        self,
        image_relative_paths: list[str],
        *,
        task_name: str,
        force: bool,
    ) -> None:
        if self.current_project is None:
            self.show_error("No project open", "Create or open a project before running render.")
            return
        self._persist_panel_preferences()

        try:
            render_config = self._render_config_from_panel(force=force)
        except Exception as exc:
            self.show_error("Invalid render settings", str(exc))
            return

        task = RenderTask(
            name=task_name,
            stage="render",
            project=self.current_project,
            image_relative_paths=image_relative_paths,
            config=render_config.to_metadata(),
            force=render_config.force,
        )
        try:
            self._submit_service_task(
                task,
                progress_label="Render progress",
                finished_handler=self._on_render_worker_finished,
                failed_handler=lambda message, active_worker: self._on_render_worker_failed(
                    message,
                    active_worker,
                    task,
                ),
                event_handler=lambda payload: self._on_worker_event("render", payload, is_batch=len(image_relative_paths) > 1),
            )
        except Exception as exc:
            self.render_panel.set_actions_enabled(True)
            self.show_error("Render service unavailable", str(exc))
            return
        self.render_panel.set_actions_enabled(False)
        self.header.set_progress_value(0)
        self.statusBar().showMessage("Rendering translated pages...")

    def _start_llama_server_action(self, action: str, *, timeout_seconds: float) -> None:
        task = LlamaServerTask(
            name=f"llama.cpp server: {action}",
            stage="ocr",
            manager=self.llama_server_manager,
            action=action,
            timeout_seconds=timeout_seconds,
        )
        worker = create_llama_server_worker(task)
        self._attach_worker_signals(
            worker,
            progress_label=None,
            finished_handler=lambda result, active_worker: self._on_llama_worker_finished(
                result,
                active_worker,
                action,
            ),
            failed_handler=lambda message, active_worker: self._on_llama_worker_failed(
                message,
                active_worker,
                action,
            ),
        )
        self._active_workers.append(worker)
        self.ocr_panel.set_server_actions_enabled(False)
        self.header.set_progress_value(0)
        self.thread_pool.start(worker)

    def _attach_worker_signals(
        self,
        worker: TaskWorker | ServiceCommandHandle,
        *,
        progress_label: str | None,
        finished_handler: Any,
        failed_handler: Any,
        event_handler: Any | None = None,
    ) -> None:
        self._mark_stage_busy(getattr(worker.task, "stage", ""), True)
        worker.signals.started.connect(self._on_background_task_started)
        if progress_label:
            worker.signals.progress.connect(
                lambda value, label=progress_label: self._on_background_task_progress(label, value)
            )
        worker.signals.message.connect(self.log)
        if event_handler is not None:
            worker.signals.event.connect(event_handler)
        worker.signals.finished.connect(
            lambda result, active_worker=worker: finished_handler(result, active_worker)
        )
        worker.signals.failed.connect(
            lambda message, active_worker=worker: failed_handler(message, active_worker)
        )

    def _on_background_task_started(self, task_name: str) -> None:
        self.log(f"Started background task: {task_name}")

    def _mark_stage_busy(self, stage_name: str, busy: bool) -> None:
        normalized = str(stage_name or "").strip().lower()
        if not normalized:
            return
        if busy:
            updated_count = int(self._busy_stage_counts.get(normalized, 0) or 0) + 1
            self._busy_stage_counts[normalized] = updated_count
            self._busy_stages.add(normalized)
        else:
            updated_count = max(0, int(self._busy_stage_counts.get(normalized, 0) or 0) - 1)
            if updated_count <= 0:
                self._busy_stage_counts.pop(normalized, None)
                self._busy_stages.discard(normalized)
            else:
                self._busy_stage_counts[normalized] = updated_count
        self.page_filmstrip.set_reorder_enabled(not bool(self._busy_stages))
        if hasattr(self, "workflow_tabs"):
            self._refresh_stage_statuses()

    def _on_background_task_progress(self, label: str, progress_value: int) -> None:
        self.header.set_progress_value(progress_value)
        self.statusBar().showMessage(f"{label}: {progress_value}%")

    def _on_worker_event(self, stage_name: str, payload: object, *, is_batch: bool) -> None:
        try:
            if not isinstance(payload, dict):
                return

            normalized_stage = str(stage_name or "").strip().lower()
            image_relative_path, event_name = self._update_processing_page_from_event(payload, stage_name=normalized_stage)
            if is_batch and self.preview_controller.should_follow_batch():
                self._active_batch_follow_stage = normalized_stage

            if self._active_batch_follow_stage != normalized_stage or self._follow_batch_paused:
                return

            if not image_relative_path or self.current_project is None:
                return

            self._follow_worker_event(normalized_stage, image_relative_path, event_name)
        except Exception as exc:
            self.log(f"Worker event handling failed for {stage_name}: {exc}", level="error")
            self.log(traceback.format_exc(), level="error")

    def _update_processing_page_from_event(
        self,
        payload: dict[str, Any],
        *,
        stage_name: str = "",
    ) -> tuple[str | None, str]:
        image_relative_path = self._image_relative_path_from_event(payload)
        if not image_relative_path or self.current_project is None:
            return image_relative_path, str(payload.get("event", "") or "").strip().lower()

        event_name = str(payload.get("event", "") or "").strip().lower()
        normalized_stage = str(stage_name or "").strip().lower()
        refresh_paths: set[str] = set()
        previous_processing_page = self._processing_page_relative_path
        if event_name in {"page_start", "batch_page_start"}:
            self._processing_page_relative_path = image_relative_path
            if normalized_stage != "detection":
                if previous_processing_page:
                    refresh_paths.add(previous_processing_page)
                refresh_paths.add(image_relative_path)
        elif event_name == "mask_ready":
            refresh_paths.add(image_relative_path)
        elif event_name in {"page_done", "page_error"}:
            refresh_paths.add(image_relative_path)
            if previous_processing_page == image_relative_path:
                self._processing_page_relative_path = None
                refresh_paths.add(image_relative_path)
        if refresh_paths:
            self._invalidate_page_statuses(refresh_paths)
            self.refresh_page_statuses(refresh_paths)
        return image_relative_path, event_name

    def _follow_worker_event(self, stage_name: str, image_relative_path: str, event_name: str) -> None:
        if not image_relative_path or self.current_project is None:
            return

        if event_name in {"page_start", "batch_page_start"}:
            if stage_name == "detection":
                return
            self._select_page_for_event(image_relative_path)
            return

        if event_name in {"page_done", "mask_ready"}:
            self._select_page_for_event(image_relative_path)
            if stage_name in {"inpaint", "inpaint_mask"}:
                self._schedule_inpaint_preview_refresh(stage_name, image_relative_path=image_relative_path)
            else:
                self._apply_preview_after_stage(stage_name, image_relative_path=image_relative_path)

    def _schedule_inpaint_preview_refresh(self, stage_name: str, *, image_relative_path: str) -> None:
        self.log(
            f"Inpaint preview refresh queued for {Path(image_relative_path).name} ({stage_name}).",
            level="info",
        )

        def _refresh() -> None:
            try:
                if self.current_project is None:
                    return
                if image_relative_path not in self._all_page_relative_paths():
                    return
                self.log(
                    f"Before preview refresh from {stage_name} event for {Path(image_relative_path).name}.",
                    level="info",
                )
                self._apply_preview_after_stage(stage_name, image_relative_path=image_relative_path)
                self.log(
                    f"After preview refresh from {stage_name} event for {Path(image_relative_path).name}.",
                    level="info",
                )
            except Exception as exc:
                self.log(f"Inpaint preview refresh failed: {exc}", level="error")
                self.log(traceback.format_exc(), level="error")

        QTimer.singleShot(50, _refresh)

    def _select_page_for_event(self, image_relative_path: str) -> None:
        if self.current_project is None:
            return
        if self.detection_panel.has_unsaved_box_edits():
            self._follow_batch_paused = True
            self.log("Follow batch progress paused because the Detection editor has unsaved changes.")
            return
        if self.ocr_panel.has_unsaved_box_edits():
            self._follow_batch_paused = True
            self.log("Follow batch progress paused because the OCR box editor has unsaved changes.")
            return
        if self.current_stage_key == "ocr" and self.ocr_panel.has_unsaved_changes():
            self._follow_batch_paused = True
            self.log("Follow batch progress paused because the OCR editor has unsaved changes.")
            return
        if self.current_stage_key == "translation" and self.translation_panel.has_unsaved_changes():
            self._follow_batch_paused = True
            self.log("Follow batch progress paused because the Translation editor has unsaved changes.")
            return
        if self.render_panel.has_unsaved_box_edits():
            self._follow_batch_paused = True
            self.log("Follow batch progress paused because the Render box editor has unsaved changes.")
            return
        for index in range(self.current_project.page_count):
            if self.current_project.page_relative_path_for_index(index) == image_relative_path:
                self._select_page_row(index, user_initiated=False)
                break

    def _image_relative_path_from_event(self, payload: dict[str, Any]) -> str | None:
        relative_path = str(payload.get("image_relative_path", "") or "").strip()
        if relative_path:
            return relative_path
        image_path = str(payload.get("image_path", "") or "").strip()
        if image_path and self.current_project is not None:
            return self.current_project.relative_source_path(Path(image_path))
        return None

    def _apply_preview_after_stage(
        self,
        stage_name: str,
        *,
        image_relative_path: str | None = None,
        export_source: str | None = None,
    ) -> None:
        target_mode = self.preview_controller.result_preview_mode(
            stage_name,
            export_source=export_source,
            current_mode=self.left_toolbar.current_mode(),
        )
        if not target_mode:
            return
        if target_mode == PREVIEW_RENDER and not self._preview_mode_artifact_exists(PREVIEW_RENDER, image_relative_path):
            self.log(
                f"Preview requested Render Result for {Path(image_relative_path or '').name or 'current page'}, "
                "but no render output is available yet.",
                level="warning",
            )
            return
        if stage_name == "detection":
            self.preview_detection_overlay_enabled = True
        self.set_preview_mode(target_mode)

    def _clear_batch_follow_state(self) -> None:
        self._active_batch_follow_stage = None
        self._follow_batch_paused = False
        previous_processing_page = self._processing_page_relative_path
        self._processing_page_relative_path = None
        if previous_processing_page:
            self._invalidate_page_statuses([previous_processing_page])

    def _finish_worker(self, worker: TaskWorker | ServiceCommandHandle) -> None:
        self._release_worker(worker)
        self._mark_stage_busy(getattr(worker.task, "stage", ""), False)
        self.header.set_progress_value(None)
        self._invalidate_page_statuses()
        self._clear_batch_follow_state()
        self._refresh_stage_statuses()

    def _on_process_worker_event(self, payload: object, *, scope: str) -> None:
        try:
            if not isinstance(payload, dict):
                return

            process_stage = str(payload.get("process_stage", "") or "").strip().lower()
            workflow_stage = str(
                payload.get("workflow_stage", "") or PROCESS_WORKFLOW_STAGE_BY_STEP.get(process_stage, "")
            ).strip().lower()
            event_name = str(payload.get("event", "") or "").strip().lower()
            message = str(payload.get("message", "") or "").strip()
            image_relative_path, updated_event_name = self._update_processing_page_from_event(payload)
            if updated_event_name:
                event_name = updated_event_name

            if process_stage:
                if event_name == "process_stage_started":
                    self.process_panel.set_step_status(process_stage, "running")
                    self.process_panel.set_current_stage(str(payload.get("display_name", process_stage.title())))
                    self._process_active_stage_key = workflow_stage or PROCESS_WORKFLOW_STAGE_BY_STEP.get(process_stage)
                    if self._process_active_stage_key:
                        self._select_stage(self._process_active_stage_key)
                    self._apply_process_preview(
                        process_stage,
                        event_name=event_name,
                        image_relative_path=image_relative_path or self.current_page(),
                    )
                elif event_name == "process_stage_completed":
                    self.process_panel.set_step_status(process_stage, str(payload.get("status", "done") or "done"))
                    if workflow_stage:
                        self._apply_process_preview(
                            process_stage,
                            event_name=event_name,
                            image_relative_path=image_relative_path or self.current_page(),
                        )
                elif event_name == "page_error":
                    self.process_panel.set_step_status(process_stage, "error")
                elif event_name == "process_canceled":
                    self.process_panel.set_step_status(process_stage, "canceled")
                    self.process_panel.set_stage_status_text("Canceled")

            if image_relative_path:
                self.process_panel.set_current_page(image_relative_path)
                if event_name in {"page_done", "page_error"}:
                    self._update_project_stage_status_from_process_event(
                        process_stage,
                        image_relative_path,
                        succeeded=(event_name == "page_done"),
                        error_message=str(payload.get("error", "") or message),
                    )
                should_follow = scope == "current" or (
                    self.preview_controller.should_follow_batch() and not self._follow_batch_paused
                )
                if should_follow and process_stage:
                    self._follow_process_worker_event(process_stage, image_relative_path, event_name)

            if "overall_progress" in payload:
                try:
                    self.process_panel.set_overall_progress(int(payload.get("overall_progress", 0) or 0))
                except Exception:
                    self.process_panel.set_overall_progress(0)
            if message:
                self.process_panel.set_action_message(message)
            if event_name == "page_error":
                self.process_panel.set_last_error(str(payload.get("error", "") or message))

            if event_name in {
                "process_stage_started",
                "process_stage_completed",
                "page_start",
                "page_done",
                "page_error",
                "batch_page_start",
                "mask_ready",
                "process_canceled",
            }:
                self._refresh_stage_statuses()
        except Exception as exc:
            self.log(f"Process event handling failed: {exc}", level="error")
            self.log(traceback.format_exc(), level="error")

    def _follow_process_worker_event(self, process_stage: str, image_relative_path: str, event_name: str) -> None:
        if not image_relative_path or self.current_project is None:
            return

        if event_name in {"page_start", "batch_page_start", "mask_ready", "page_done", "page_error"}:
            self._select_page_for_event(image_relative_path)

        if event_name in {"page_start", "batch_page_start", "mask_ready", "page_done"}:
            self._apply_process_preview(
                process_stage,
                event_name=event_name,
                image_relative_path=image_relative_path,
            )

    def _apply_process_preview(
        self,
        process_stage: str,
        *,
        event_name: str,
        image_relative_path: str | None,
    ) -> None:
        target_mode = self.preview_controller.process_preview_mode(
            process_stage,
            event_name=event_name,
            current_mode=self.left_toolbar.current_mode(),
        )
        if not target_mode:
            return

        preview_page = image_relative_path or self.current_page()
        if target_mode == PREVIEW_DETECTION:
            self.preview_detection_overlay_enabled = True
        elif target_mode == PREVIEW_INPAINT and not self._preview_mode_artifact_exists(PREVIEW_INPAINT, preview_page):
            self.log(
                f"Process preview requested Inpaint Result for {Path(preview_page or '').name or 'current page'}, "
                "but no inpaint output is available yet.",
                level="warning",
            )
            target_mode = PREVIEW_MASK
        elif target_mode == PREVIEW_RENDER and not self._preview_mode_artifact_exists(PREVIEW_RENDER, preview_page):
            self.log(
                f"Process preview requested Render Result for {Path(preview_page or '').name or 'current page'}, "
                "but no render output is available yet.",
                level="warning",
            )
            target_mode = PREVIEW_INPAINT if self._preview_mode_artifact_exists(PREVIEW_INPAINT, preview_page) else PREVIEW_SOURCE

        self.set_preview_mode(target_mode)

    def _update_project_stage_status_from_process_event(
        self,
        process_stage: str,
        image_relative_path: str,
        *,
        succeeded: bool,
        error_message: str | None = None,
    ) -> None:
        if self.current_project is None or not process_stage or not image_relative_path:
            return

        workflow_stage = PROCESS_WORKFLOW_STAGE_BY_STEP.get(process_stage, process_stage)
        if not succeeded:
            self.current_project.update_stage_status(
                image_relative_path,
                workflow_stage,
                status="failed",
                error=(str(error_message or "").strip() or "Process step failed."),
            )
            return

        try:
            if process_stage == "detection":
                cache_path = detection_json_path(self.current_project, image_relative_path)
                self.current_project.update_stage_status(
                    image_relative_path,
                    "detection",
                    status="done",
                    cache_path=self._relative_project_path(cache_path),
                )
            elif process_stage in {"ocr_prepare", "ocr"}:
                ocr_data = load_ocr_json(ocr_json_path(self.current_project, image_relative_path))
                self._update_project_ocr_stage_status(image_relative_path, ocr_data)
            elif process_stage in {"translation_init", "translation"}:
                translation_data = load_translation_json(
                    translation_json_path(self.current_project, image_relative_path)
                )
                self._update_project_translation_stage_status(image_relative_path, translation_data)
            elif process_stage in {"inpaint_mask", "inpaint"}:
                inpaint_data = load_inpaint_json(inpaint_json_path(self.current_project, image_relative_path))
                self._update_project_inpaint_stage_status(image_relative_path, inpaint_data)
            elif process_stage in {"render_prepare", "render"}:
                render_data = load_render_json(render_json_path(self.current_project, image_relative_path))
                self._update_project_render_stage_status(image_relative_path, render_data)
        except Exception as exc:
            self.log(
                f"Deferred process stage status update for {process_stage} / {Path(image_relative_path).name}: {exc}",
                level="warning",
            )

    def _apply_process_completion_view(self, *, scope: str, show_render_preview: bool) -> None:
        restore_path = self._process_restore_page_relative_path
        self._process_restore_page_relative_path = None
        if scope == "chapter" and restore_path:
            self._select_page_by_relative_path(restore_path, user_initiated=False)
        self._select_stage("process")
        if show_render_preview:
            self._apply_process_preview(
                "render",
                event_name="process_finished",
                image_relative_path=self.current_page(),
            )

    def _on_process_worker_finished(self, result: object, worker: TaskWorker) -> None:
        self._finish_worker(worker)
        self._set_process_ui_running(False)
        self._process_active_stage_key = None
        self._process_cancel_requested = False

        if not isinstance(result, ProcessPipelineResult):
            self._process_stage_status = "done"
            self.process_panel.set_stage_status_text("Done")
            self.process_panel.set_action_message("Process completed.")
            self._refresh_stage_statuses()
            return

        if result.canceled:
            self._process_stage_status = "ready"
            self.process_panel.set_stage_status_text("Canceled")
        elif result.final_state == "failed":
            self._process_stage_status = "error"
            self.process_panel.set_stage_status_text("Failed")
        elif result.completed_with_errors:
            self._process_stage_status = "error"
            self.process_panel.set_stage_status_text("Completed with errors")
        else:
            self._process_stage_status = "done"
            self.process_panel.set_stage_status_text("Done")
        for step_key, step_status in result.step_statuses.items():
            self.process_panel.set_step_status(step_key, step_status)

        if result.canceled:
            summary_text = (
                "Canceled by user\n"
                f"Last completed stage: {result.last_completed_stage or '-'}\n"
                f"Last completed page: {Path(result.last_completed_page).name if result.last_completed_page else '-'}\n"
                f"Unfinished stage: {result.unfinished_stage or '-'}"
            )
        else:
            summary_text = (
                f"Pages processed: {result.pages_processed}\n"
                f"Pages succeeded: {result.succeeded_pages}\n"
                f"Pages failed: {result.failed_pages}\n"
                f"Stages completed: {result.stages_completed}"
            )
        self.process_panel.set_done_summary(summary_text)
        if result.canceled:
            self.process_panel.set_action_message(result.cancel_message or "Process canceled by user.")
        elif result.final_state == "failed":
            self.process_panel.set_overall_progress(100)
            self.process_panel.set_action_message("Process failed.")
        else:
            self.process_panel.set_overall_progress(100)
            self.process_panel.set_action_message(
                "Process completed with errors." if result.completed_with_errors else "Process completed."
            )
        self.process_panel.set_last_error("" if result.canceled else result.last_error)
        self.process_panel.set_cancel_enabled(False)
        self.process_panel.set_cancel_stopping(False)

        if self.current_project is not None:
            self._persist_project(show_errors=False)

        self._apply_process_completion_view(
            scope=result.scope,
            show_render_preview=not result.canceled and not (result.scope == "current" and result.final_state == "failed"),
        )
        self._refresh_stage_statuses()

        if result.canceled:
            self.statusBar().showMessage("One-click process canceled")
        elif result.final_state == "failed" and result.scope == "current" and result.last_error:
            self.statusBar().showMessage("One-click process failed for the current page.")
            self.show_error("Process failed", result.last_error)
        elif result.completed_with_errors:
            self.statusBar().showMessage(
                f"One-click process completed with {result.failed_pages} failed page(s)."
            )
        else:
            self.statusBar().showMessage(
                f"One-click process completed for {result.succeeded_pages} page(s)."
            )

    def _on_process_worker_failed(
        self,
        message: str,
        worker: TaskWorker,
        task: ProcessTask,
    ) -> None:
        self._finish_worker(worker)
        self._set_process_ui_running(False)
        self._process_stage_status = "error"
        self._process_active_stage_key = None
        self._process_cancel_requested = False
        self.process_panel.set_stage_status_text("Error")
        self.process_panel.set_last_error(message)
        self.process_panel.set_action_message("Process failed.")
        self.process_panel.set_cancel_enabled(False)
        self.process_panel.set_cancel_stopping(False)
        self.process_panel.set_done_summary(
            "Pages processed: 0\nPages succeeded: 0\nPages failed: 0\nStages completed: 0"
        )
        self._apply_process_completion_view(scope=task.scope, show_render_preview=False)
        self._refresh_stage_statuses()
        self.statusBar().showMessage("One-click process failed")
        self.show_error("One-click process failed", message)

    def _on_detection_worker_finished(self, result: object, worker: TaskWorker) -> None:
        self._finish_worker(worker)
        self.detection_panel.set_actions_enabled(True)
        self._refresh_stage_statuses()

        if not isinstance(result, DetectionWorkerResult):
            self.statusBar().showMessage("Detection finished")
            return

        if self.current_project is not None:
            for page_result in result.page_results:
                relative_source_path = self.current_project.relative_source_path(page_result.image_path)
                if relative_source_path is None:
                    continue
                if page_result.json_path is not None:
                    self.current_project.update_stage_status(
                        relative_source_path,
                        "detection",
                        status="done",
                        cache_path=self._relative_project_path(page_result.json_path),
                    )
                else:
                    self.current_project.update_stage_status(
                        relative_source_path,
                        "detection",
                        status="failed",
                        error=page_result.error or "Unknown detection failure.",
                    )
            self._persist_project(show_errors=True)

        current_index = self._current_page_index()
        if current_index is not None:
            self._load_cached_detection_for_index(current_index, show_errors=False)
            self._refresh_preview_for_current_page()
            current_page = self.current_page()
            if current_page is not None:
                self._apply_preview_after_stage("detection", image_relative_path=current_page)

        success_count = len(result.json_paths)
        failure_count = len(result.failures)
        if failure_count:
            self.statusBar().showMessage(
                f"Detection finished with {success_count} success(es) and {failure_count} failure(s)."
            )
        else:
            self.statusBar().showMessage(f"Detection finished for {success_count} page(s).")

    def _on_detection_worker_failed(
        self,
        message: str,
        worker: TaskWorker,
        task: DetectionTask,
    ) -> None:
        self._finish_worker(worker)
        self.detection_panel.set_actions_enabled(True)
        self._refresh_stage_statuses()

        if self.current_project is not None:
            for image_path in task.image_paths:
                relative_source_path = self.current_project.relative_source_path(image_path)
                if relative_source_path is None:
                    continue
                self.current_project.update_stage_status(
                    relative_source_path,
                    "detection",
                    status="failed",
                    error=message,
                )
            self._persist_project(show_errors=False)

        self.statusBar().showMessage("Detection failed")
        self.show_error("Detection failed", message)

    def _on_ocr_worker_finished(self, result: object, worker: TaskWorker) -> None:
        self._finish_worker(worker)
        self.ocr_panel.set_actions_enabled(True)

        if not isinstance(result, OCRPreparationWorkerResult):
            self.statusBar().showMessage("OCR preparation finished")
            return

        if self.current_project is not None:
            for page_result in result.page_results:
                if page_result.json_path is not None:
                    try:
                        ocr_data = load_ocr_json(page_result.json_path)
                    except Exception as exc:
                        self.current_project.update_stage_status(
                            page_result.image_relative_path,
                            "ocr",
                            status="failed",
                            error=str(exc),
                        )
                    else:
                        self._update_project_ocr_stage_status(page_result.image_relative_path, ocr_data)
                else:
                    self.current_project.update_stage_status(
                        page_result.image_relative_path,
                        "ocr",
                        status="failed",
                        error=page_result.error or "Unknown OCR preparation failure.",
                    )
            self._persist_project(show_errors=True)

        current_index = self._current_page_index()
        if current_index is not None:
            self._load_cached_ocr_for_index(current_index, show_errors=False)
            current_page = self.current_page()
            if current_page is not None:
                self._apply_preview_after_stage("ocr_prepare", image_relative_path=current_page)

        success_count = len(result.json_paths)
        failure_count = len(result.failures)
        if failure_count:
            self.statusBar().showMessage(
                f"OCR preparation finished with {success_count} success(es) and {failure_count} failure(s)."
            )
        else:
            self.statusBar().showMessage(f"OCR preparation finished for {success_count} page(s).")

    def _on_ocr_worker_failed(
        self,
        message: str,
        worker: TaskWorker,
        task: OCRPreparationTask,
    ) -> None:
        self._finish_worker(worker)
        self.ocr_panel.set_actions_enabled(True)

        if self.current_project is not None:
            for image_relative_path in task.image_relative_paths:
                self.current_project.update_stage_status(
                    image_relative_path,
                    "ocr",
                    status="failed",
                    error=message,
                )
            self._persist_project(show_errors=False)

        self.statusBar().showMessage("OCR preparation failed")
        self.show_error("OCR preparation failed", message)

    def _on_ocr_inference_worker_finished(self, result: object, worker: TaskWorker) -> None:
        self._finish_worker(worker)
        self.ocr_panel.set_actions_enabled(True)

        if not isinstance(result, OCRInferenceWorkerResult):
            self.statusBar().showMessage("OCR finished")
            return

        if self.current_project is not None:
            for page_result in result.page_results:
                if page_result.json_path is not None:
                    try:
                        ocr_data = load_ocr_json(page_result.json_path)
                    except Exception as exc:
                        self.current_project.update_stage_status(
                            page_result.image_relative_path,
                            "ocr",
                            status="failed",
                            error=str(exc),
                        )
                    else:
                        self._update_project_ocr_stage_status(page_result.image_relative_path, ocr_data)
                else:
                    self.current_project.update_stage_status(
                        page_result.image_relative_path,
                        "ocr",
                        status="failed",
                        error=page_result.error or "Unknown OCR failure.",
                    )
            self._persist_project(show_errors=True)

        current_index = self._current_page_index()
        if current_index is not None:
            self._load_cached_ocr_for_index(current_index, show_errors=False)
            current_page = self.current_page()
            if current_page is not None:
                self._apply_preview_after_stage("ocr", image_relative_path=current_page)

        success_count = len(result.json_paths)
        failure_count = len(result.failures)
        if failure_count:
            self.statusBar().showMessage(
                f"OCR finished with {success_count} success(es) and {failure_count} failure(s)."
            )
        else:
            self.statusBar().showMessage(f"OCR finished for {success_count} page(s).")

    def _on_ocr_inference_worker_failed(
        self,
        message: str,
        worker: TaskWorker,
        task: OCRInferenceTask,
    ) -> None:
        self._finish_worker(worker)
        self.ocr_panel.set_actions_enabled(True)

        if self.current_project is not None:
            for image_relative_path in task.image_relative_paths:
                self.current_project.update_stage_status(
                    image_relative_path,
                    "ocr",
                    status="failed",
                    error=message,
                )
            self._persist_project(show_errors=False)

        self.statusBar().showMessage("OCR failed")
        self.show_error("OCR failed", message)

    def _on_translation_init_worker_finished(self, result: object, worker: TaskWorker) -> None:
        self._finish_worker(worker)
        self.translation_panel.set_actions_enabled(True)

        if not isinstance(result, TranslationInitializationWorkerResult):
            self.statusBar().showMessage("Translation initialization finished")
            return

        if self.current_project is not None:
            for page_result in result.page_results:
                if page_result.json_path is not None:
                    try:
                        translation_data = load_translation_json(page_result.json_path)
                    except Exception as exc:
                        self.current_project.update_stage_status(
                            page_result.image_relative_path,
                            "translation",
                            status="failed",
                            error=str(exc),
                        )
                    else:
                        self._update_project_translation_stage_status(
                            page_result.image_relative_path,
                            translation_data,
                        )
                else:
                    self.current_project.update_stage_status(
                        page_result.image_relative_path,
                        "translation",
                        status="failed",
                        error=page_result.error or "Unknown translation initialization failure.",
                    )
            self._persist_project(show_errors=True)

        current_index = self._current_page_index()
        if current_index is not None:
            self._load_cached_translation_for_index(current_index, show_errors=False)

        success_count = len(result.json_paths)
        failure_count = len(result.failures)
        if failure_count:
            self.statusBar().showMessage(
                f"Translation initialization finished with {success_count} success(es) and {failure_count} failure(s)."
            )
        else:
            self.statusBar().showMessage(
                f"Translation initialization finished for {success_count} page(s)."
            )

    def _on_translation_init_worker_failed(
        self,
        message: str,
        worker: TaskWorker,
        task: TranslationInitializationTask,
    ) -> None:
        self._finish_worker(worker)
        self.translation_panel.set_actions_enabled(True)

        if self.current_project is not None:
            for image_relative_path in task.image_relative_paths:
                self.current_project.update_stage_status(
                    image_relative_path,
                    "translation",
                    status="failed",
                    error=message,
                )
            self._persist_project(show_errors=False)

        self.statusBar().showMessage("Translation initialization failed")
        self.show_error("Translation initialization failed", message)

    def _on_translation_worker_finished(self, result: object, worker: TaskWorker) -> None:
        self._finish_worker(worker)
        self.translation_panel.set_actions_enabled(True)

        if not isinstance(result, TranslationWorkerResult):
            self.statusBar().showMessage("Translation finished")
            return

        if self.current_project is not None:
            for page_result in result.page_results:
                if page_result.json_path is not None:
                    try:
                        translation_data = load_translation_json(page_result.json_path)
                    except Exception as exc:
                        self.current_project.update_stage_status(
                            page_result.image_relative_path,
                            "translation",
                            status="failed",
                            error=str(exc),
                        )
                    else:
                        self._update_project_translation_stage_status(
                            page_result.image_relative_path,
                            translation_data,
                        )
                else:
                    self.current_project.update_stage_status(
                        page_result.image_relative_path,
                        "translation",
                        status="failed",
                        error=page_result.error or "Unknown translation failure.",
                    )
            self._persist_project(show_errors=True)

        current_index = self._current_page_index()
        if current_index is not None:
            self._load_cached_translation_for_index(current_index, show_errors=False)

        success_count = len(result.json_paths)
        failure_count = len(result.failures)
        if failure_count:
            self.statusBar().showMessage(
                f"Translation finished with {success_count} success(es) and {failure_count} failure(s)."
            )
        else:
            self.statusBar().showMessage(f"Translation finished for {success_count} page(s).")

    def _on_translation_worker_failed(
        self,
        message: str,
        worker: TaskWorker,
        task: TranslationTask,
    ) -> None:
        self._finish_worker(worker)
        self.translation_panel.set_actions_enabled(True)

        if self.current_project is not None:
            for image_relative_path in task.image_relative_paths:
                self.current_project.update_stage_status(
                    image_relative_path,
                    "translation",
                    status="failed",
                    error=message,
                )
            self._persist_project(show_errors=False)

        self.statusBar().showMessage("Translation failed")
        self.show_error("Translation failed", message)

    def _on_inpaint_mask_worker_finished(self, result: object, worker: TaskWorker) -> None:
        self._finish_worker(worker)
        self.inpaint_panel.set_actions_enabled(True)

        if not isinstance(result, InpaintMaskWorkerResult):
            self.statusBar().showMessage("Inpaint mask preparation finished")
            return

        if self.current_project is not None:
            for page_result in result.page_results:
                if page_result.mask_path is not None:
                    metadata_path = inpaint_json_path(self.current_project, page_result.image_relative_path)
                    try:
                        inpaint_data = load_inpaint_json(metadata_path)
                    except Exception as exc:
                        self.current_project.update_stage_status(
                            page_result.image_relative_path,
                            "inpaint",
                            status="failed",
                            error=str(exc),
                        )
                    else:
                        self._update_project_inpaint_stage_status(
                            page_result.image_relative_path,
                            inpaint_data,
                        )
                else:
                    self.current_project.update_stage_status(
                        page_result.image_relative_path,
                        "inpaint",
                        status="failed",
                        error=page_result.error or "Unknown inpaint mask preparation failure.",
                    )
            self._persist_project(show_errors=True)

        current_index = self._current_page_index()
        if current_index is not None:
            self._load_cached_inpaint_for_index(current_index, show_errors=False)
            self._refresh_preview_for_current_page()
            current_page = self.current_page()
            if current_page is not None:
                self._apply_preview_after_stage("inpaint_mask", image_relative_path=current_page)

        success_count = len(result.mask_paths)
        failure_count = len(result.failures)
        if failure_count:
            self.statusBar().showMessage(
                f"Inpaint mask preparation finished with {success_count} success(es) and {failure_count} failure(s)."
            )
        else:
            self.statusBar().showMessage(
                f"Inpaint mask preparation finished for {success_count} page(s)."
            )

    def _on_inpaint_mask_worker_failed(
        self,
        message: str,
        worker: TaskWorker,
        task: InpaintMaskTask,
    ) -> None:
        self._finish_worker(worker)
        self.inpaint_panel.set_actions_enabled(True)

        if self.current_project is not None:
            for image_relative_path in task.image_relative_paths:
                self.current_project.update_stage_status(
                    image_relative_path,
                    "inpaint",
                    status="failed",
                    error=message,
                )
            self._persist_project(show_errors=False)

        self.statusBar().showMessage("Inpaint mask preparation failed")
        self.show_error("Inpaint mask preparation failed", message)

    def _on_inpaint_worker_finished(self, result: object, worker: TaskWorker) -> None:
        self._finish_worker(worker)
        self.inpaint_panel.set_actions_enabled(True)

        if not isinstance(result, InpaintWorkerResult):
            self.statusBar().showMessage("Inpaint finished")
            return

        if self.current_project is not None:
            for page_result in result.page_results:
                if page_result.image_path is not None:
                    metadata_path = inpaint_json_path(self.current_project, page_result.image_relative_path)
                    try:
                        inpaint_data = load_inpaint_json(metadata_path)
                    except Exception as exc:
                        self.current_project.update_stage_status(
                            page_result.image_relative_path,
                            "inpaint",
                            status="failed",
                            error=str(exc),
                        )
                    else:
                        self._update_project_inpaint_stage_status(
                            page_result.image_relative_path,
                            inpaint_data,
                        )
                else:
                    self.current_project.update_stage_status(
                        page_result.image_relative_path,
                        "inpaint",
                        status="failed",
                        error=page_result.error or "Unknown inpaint failure.",
                    )
            self._persist_project(show_errors=True)

        current_index = self._current_page_index()
        if current_index is not None:
            self._load_cached_inpaint_for_index(current_index, show_errors=False)
            self._refresh_preview_for_current_page()
            current_page = self.current_page()
            if current_page is not None:
                self._apply_preview_after_stage("inpaint", image_relative_path=current_page)

        success_count = len(result.image_paths)
        failure_count = len(result.failures)
        if failure_count:
            self.statusBar().showMessage(
                f"Inpaint finished with {success_count} success(es) and {failure_count} failure(s)."
            )
        else:
            self.statusBar().showMessage(f"Inpaint finished for {success_count} page(s).")

    def _on_inpaint_worker_failed(
        self,
        message: str,
        worker: TaskWorker,
        task: InpaintTask,
    ) -> None:
        self._finish_worker(worker)
        self.inpaint_panel.set_actions_enabled(True)

        if self.current_project is not None:
            for image_relative_path in task.image_relative_paths:
                self.current_project.update_stage_status(
                    image_relative_path,
                    "inpaint",
                    status="failed",
                    error=message,
                )
            self._persist_project(show_errors=False)

        self.statusBar().showMessage("Inpaint failed")
        self.show_error("Inpaint failed", message)

    def _on_lama_model_worker_finished(
        self,
        result: object,
        worker: TaskWorker,
        action: str,
    ) -> None:
        self._finish_worker(worker)
        self.inpaint_panel.set_actions_enabled(True)

        if not isinstance(result, LamaModelTaskResult):
            self.statusBar().showMessage("LaMa model action finished")
            return

        status_text = f"Loaded ({result.device or 'auto'})" if result.loaded else "Not loaded"
        self.inpaint_panel.set_model_status(status_text)
        self.statusBar().showMessage(result.message)
        self.log(result.message)

    def _on_lama_model_worker_failed(
        self,
        message: str,
        worker: TaskWorker,
        action: str,
    ) -> None:
        self._finish_worker(worker)
        self.inpaint_panel.set_actions_enabled(True)
        self.inpaint_panel.set_model_status("Error")
        self.statusBar().showMessage("LaMa model action failed")
        self.show_error("LaMa model error", message)

    def _on_render_prep_worker_finished(self, result: object, worker: TaskWorker) -> None:
        self._finish_worker(worker)
        self.render_panel.set_actions_enabled(True)

        if not isinstance(result, RenderPreparationWorkerResult):
            self.statusBar().showMessage("Render preparation finished")
            return

        if self.current_project is not None:
            for page_result in result.page_results:
                if page_result.json_path is not None:
                    try:
                        render_data = load_render_json(page_result.json_path)
                    except Exception as exc:
                        self.current_project.update_stage_status(
                            page_result.image_relative_path,
                            "render",
                            status="failed",
                            error=str(exc),
                        )
                    else:
                        self._update_project_render_stage_status(page_result.image_relative_path, render_data)
                else:
                    self.current_project.update_stage_status(
                        page_result.image_relative_path,
                        "render",
                        status="failed",
                        error=page_result.error or "Unknown render preparation failure.",
                    )
            self._persist_project(show_errors=True)

        current_index = self._current_page_index()
        if current_index is not None:
            self._load_cached_render_for_index(current_index, show_errors=False)
            self._refresh_preview_for_current_page()

        success_count = len(result.json_paths)
        failure_count = len(result.failures)
        if failure_count:
            self.statusBar().showMessage(
                f"Render preparation finished with {success_count} success(es) and {failure_count} failure(s)."
            )
        else:
            self.statusBar().showMessage(f"Render preparation finished for {success_count} page(s).")

    def _on_render_prep_worker_failed(
        self,
        message: str,
        worker: TaskWorker,
        task: RenderPreparationTask,
    ) -> None:
        self._finish_worker(worker)
        self.render_panel.set_actions_enabled(True)

        if self.current_project is not None:
            for image_relative_path in task.image_relative_paths:
                self.current_project.update_stage_status(
                    image_relative_path,
                    "render",
                    status="failed",
                    error=message,
                )
            self._persist_project(show_errors=False)

        self.statusBar().showMessage("Render preparation failed")
        self.show_error("Render preparation failed", message)

    def _on_render_worker_finished(self, result: object, worker: TaskWorker) -> None:
        self._finish_worker(worker)
        self.render_panel.set_actions_enabled(True)

        if not isinstance(result, RenderWorkerResult):
            self.statusBar().showMessage("Render finished")
            return

        if self.current_project is not None:
            for page_result in result.page_results:
                if page_result.image_path is not None:
                    metadata_path = render_json_path(self.current_project, page_result.image_relative_path)
                    try:
                        render_data = load_render_json(metadata_path)
                    except Exception as exc:
                        self.current_project.update_stage_status(
                            page_result.image_relative_path,
                            "render",
                            status="failed",
                            error=str(exc),
                        )
                    else:
                        self._update_project_render_stage_status(page_result.image_relative_path, render_data)
                else:
                    self.current_project.update_stage_status(
                        page_result.image_relative_path,
                        "render",
                        status="failed",
                        error=page_result.error or "Unknown render failure.",
                    )
            self._persist_project(show_errors=True)

        current_index = self._current_page_index()
        if current_index is not None:
            self._load_cached_render_for_index(current_index, show_errors=False)
            self._refresh_preview_for_current_page()
            current_page = self.current_page()
            if current_page is not None:
                self._apply_preview_after_stage("render", image_relative_path=current_page)

        success_count = len(result.image_paths)
        failure_count = len(result.failures)
        if failure_count:
            self.statusBar().showMessage(
                f"Render finished with {success_count} success(es) and {failure_count} failure(s)."
            )
        else:
            self.statusBar().showMessage(f"Render finished for {success_count} page(s).")

    def _on_render_worker_failed(
        self,
        message: str,
        worker: TaskWorker,
        task: RenderTask,
    ) -> None:
        self._finish_worker(worker)
        self.render_panel.set_actions_enabled(True)

        if self.current_project is not None:
            for image_relative_path in task.image_relative_paths:
                self.current_project.update_stage_status(
                    image_relative_path,
                    "render",
                    status="failed",
                    error=message,
                )
            self._persist_project(show_errors=False)

        self.statusBar().showMessage("Render failed")
        self.show_error("Render failed", message)

    def _on_export_worker_finished(
        self,
        result: object,
        worker: TaskWorker,
        task: ExportTask,
    ) -> None:
        self._finish_worker(worker)
        self.export_panel.set_actions_enabled(True)

        if not isinstance(result, ExportWorkerResult):
            self.statusBar().showMessage("Export finished")
            return

        manifest = dict(result.manifest)
        self._set_export_summary(manifest, persist=True)
        exported_count = int(manifest.get("exported_count", 0) or 0)
        skipped_count = int(manifest.get("skipped_count", 0) or 0)
        error_count = int(manifest.get("error_count", 0) or 0)
        output_dir = str(manifest.get("output_dir", "") or "")
        manifest_path = str(manifest.get("manifest_path", "") or "")
        zip_path = str(manifest.get("zip_path", "") or "")

        if output_dir:
            self.log(f"Export output folder: {output_dir}")
        if manifest_path:
            self.log(f"Export manifest: {manifest_path}")
        if zip_path:
            self.log(f"Export archive: {zip_path}")

        if error_count:
            self.statusBar().showMessage(
                f"Export finished with {exported_count} exported, {skipped_count} skipped, and {error_count} error(s)."
            )
        else:
            self.statusBar().showMessage(
                f"Export finished with {exported_count} exported and {skipped_count} skipped."
            )

        export_config = ExportConfig.from_value(task.config)
        self._apply_preview_after_stage(
            "export",
            image_relative_path=self.current_page(),
            export_source=export_config.export_source,
        )
        if export_config.open_output_folder and exported_count > 0 and output_dir:
            self._open_output_folder(output_dir)

    def _on_export_worker_failed(
        self,
        message: str,
        worker: TaskWorker,
        task: ExportTask,
    ) -> None:
        self._finish_worker(worker)
        self.export_panel.set_actions_enabled(True)
        self.statusBar().showMessage("Export failed")
        self.show_error("Export failed", message)

    def _on_llama_worker_finished(
        self,
        result: object,
        worker: TaskWorker,
        action: str,
    ) -> None:
        self._finish_worker(worker)
        self.ocr_panel.set_server_actions_enabled(True)

        if not isinstance(result, LlamaServerTaskResult):
            self.statusBar().showMessage("llama.cpp server action finished")
            return

        self.ocr_panel.set_server_status(result.state)
        self.statusBar().showMessage(result.message)
        self.log(result.message)

    def _on_llama_worker_failed(
        self,
        message: str,
        worker: TaskWorker,
        action: str,
    ) -> None:
        self._finish_worker(worker)
        self.ocr_panel.set_server_actions_enabled(True)
        self.ocr_panel.set_server_status(SERVER_STATE_ERROR)
        self.statusBar().showMessage("llama.cpp server action failed")
        self.show_error("llama.cpp server error", message)

    def _apply_server_inputs_to_manager(self) -> bool:
        try:
            self._persist_panel_preferences()
            self.llama_server_manager.update_config(**self.config_panel.ocr_server_values())
        except Exception as exc:
            self.ocr_panel.set_server_status(SERVER_STATE_ERROR)
            self.show_error("Invalid server settings", str(exc))
            return False
        return True

    def _render_config_from_panel(self, *, force: bool) -> RenderConfig:
        return self.current_render_config(force=force)

    def _selected_page_context(self, *, show_error: bool) -> tuple[int, str, Path] | None:
        if self.current_project is None:
            if show_error:
                self.show_error("No project open", "Create or open a project before working with this stage.")
            return None

        if self.current_project.page_count == 0:
            if show_error:
                self.show_error("No page selected", "Import images before using this stage.")
            return None

        index = self._current_page_index()
        if index is None:
            if show_error:
                self.show_error("No page selected", "Select a page first.")
            return None

        image_relative_path = self.current_project.page_relative_path_for_index(index)
        image_path = self.current_project.image_path_for_index(index)
        if image_relative_path is None or image_path is None:
            if show_error:
                self.show_error("No page selected", "Select a valid page first.")
            return None

        if not image_path.exists():
            if show_error:
                self.show_error("Missing source image", f"Source image not found:\n{image_path}")
            return None

        return index, image_relative_path, image_path

    def _current_page_index(self) -> int | None:
        if self.current_project is None or self.current_project.page_count == 0:
            return None

        index = self.page_filmstrip.current_row()
        if index < 0:
            index = self.current_project.data.current_page_index
        if index < 0 or index >= self.current_project.page_count:
            return None
        return index

    def _all_page_relative_paths(self) -> list[str]:
        if self.current_project is None:
            return []
        return [
            relative_path
            for relative_path in (
                self.current_project.page_relative_path_for_index(index)
                for index in range(self.current_project.page_count)
            )
            if relative_path is not None
        ]

    def _persist_project(self, *, show_errors: bool) -> None:
        if self.current_project is None:
            return
        try:
            self.current_project.save()
        except Exception as exc:  # pragma: no cover - GUI error path.
            if show_errors:
                self.show_error("Failed to save project", str(exc))
            else:
                self.log(f"Failed to save project state: {exc}")

    def _refresh_stage_statuses(self) -> None:
        statuses = {
            "process": self._process_stage_status if self.current_project is not None else "missing",
            "project": "done" if self.current_project is not None else "missing",
            "config": "done",
            "detection": "missing",
            "ocr": "missing",
            "translation": "missing",
            "inpaint": "missing",
            "render": "missing",
            "export": "missing",
        }

        current_page = self.current_page()
        refresh_paths: set[str] = set()
        if current_page is not None:
            refresh_paths.add(current_page)
        if self._processing_page_relative_path is not None:
            refresh_paths.add(self._processing_page_relative_path)
        if refresh_paths:
            self._invalidate_page_statuses(refresh_paths)
            self.refresh_page_statuses(refresh_paths, update_filmstrip=False)

        if self.current_project is not None and current_page is not None:
            statuses["detection"] = self._sidebar_status_from_raw(
                self._stage_status_for_current_page("detection", current_page)
            )
            statuses["ocr"] = self._sidebar_status_from_raw(
                self._stage_status_for_current_page("ocr", current_page)
            )
            statuses["translation"] = self._sidebar_status_from_raw(
                self._stage_status_for_current_page("translation", current_page)
            )
            statuses["inpaint"] = self._sidebar_status_from_raw(
                self._stage_status_for_current_page("inpaint", current_page)
            )
            statuses["render"] = self._sidebar_status_from_raw(
                self._stage_status_for_current_page("render", current_page)
            )
            statuses["export"] = self._sidebar_status_from_raw(self._export_stage_status_for_page(current_page))

        for busy_stage_name in self._busy_stages:
            stage_key = self._workflow_stage_key_for_busy_stage(busy_stage_name)
            if stage_key in statuses and statuses[stage_key] != "error":
                statuses[stage_key] = "ready"
        if self._process_is_running() and self._process_active_stage_key in statuses:
            if statuses[self._process_active_stage_key] != "error":
                statuses[self._process_active_stage_key] = "ready"

        self.workflow_tabs.set_stage_statuses(statuses)
        if self.current_project is not None:
            self.page_filmstrip.set_page_statuses(self._page_status_map())
        self._refresh_process_panel_summary()
        self.process_panel.set_stage_status_text(
            {
                "done": "Done",
                "error": "Error",
                "ready": "Running" if self._process_is_running() else "Idle",
            }.get(statuses["process"], "Idle")
        )
        self.project_panel.set_stage_status_text("Ready" if self.current_project is not None else "No project")
        self.config_panel.set_stage_status_text("Ready")
        self.detection_panel.set_stage_status_text(statuses["detection"].title())
        self.ocr_panel.set_stage_status_text(statuses["ocr"].title())
        self.translation_panel.set_stage_status_text(statuses["translation"].title())
        self.inpaint_panel.set_stage_status_text(statuses["inpaint"].title())
        self.render_panel.set_stage_status_text(statuses["render"].title())
        self.export_panel.set_stage_status_text(self._export_stage_label(statuses["export"]))
        self._refresh_stage_notes_and_actions()

    @staticmethod
    def _workflow_stage_key_for_busy_stage(stage_name: str) -> str | None:
        normalized = str(stage_name or "").strip().lower()
        return {
            "process": "process",
            "detection": "detection",
            "ocr_prepare": "ocr",
            "ocr": "ocr",
            "translation_init": "translation",
            "translation": "translation",
            "inpaint_mask": "inpaint",
            "inpaint": "inpaint",
            "render_prepare": "render",
            "render": "render",
            "export": "export",
        }.get(normalized)

    def _metadata_status(self, image_relative_path: str, stage_name: str) -> str:
        if self.current_project is None:
            return "missing"
        stage_metadata = self.current_project.stage_metadata(image_relative_path, stage_name)
        if not isinstance(stage_metadata, dict):
            return "missing"
        return str(stage_metadata.get("status", "missing") or "missing").strip().lower()

    def _stage_is_stale(self, image_relative_path: str, stage_name: str) -> bool:
        if self.current_project is None:
            return False
        stage_metadata = self.current_project.stage_metadata(image_relative_path, stage_name)
        if not isinstance(stage_metadata, dict):
            return False
        return bool(stage_metadata.get("stale", False))

    @staticmethod
    def _sidebar_status_from_raw(raw_status: str) -> str:
        normalized = str(raw_status or "missing").strip().lower()
        if normalized in {"done"}:
            return "done"
        if normalized in {"failed", "error"}:
            return "error"
        if normalized in {"prepared", "partial", "initialized", "running", "ready", "canceled"}:
            return "ready"
        return "missing"

    def _stage_status_for_current_page(self, stage_name: str, image_relative_path: str) -> str:
        if self.current_project is None:
            return "missing"

        if self._stage_is_stale(image_relative_path, stage_name):
            return "ready"

        if stage_name == "detection":
            cache_path = detection_json_path(self.current_project, image_relative_path)
            if cache_path.exists():
                return "done"
            return self._metadata_status(image_relative_path, "detection")

        if stage_name == "ocr":
            cache_path = ocr_json_path(self.current_project, image_relative_path)
            if cache_path.exists():
                try:
                    payload = load_ocr_json(cache_path)
                    summary = summarize_ocr_items(payload.get("items", []))
                except Exception:
                    return "failed"
                if summary.get("total", 0) > 0 or summary.get("prepared", 0) > 0 or summary.get("done", 0) > 0:
                    return "done"
                return "ready"
            return self._metadata_status(image_relative_path, "ocr")

        if stage_name == "translation":
            cache_path = translation_json_path(self.current_project, image_relative_path)
            if cache_path.exists():
                try:
                    payload = load_translation_json(cache_path)
                except Exception:
                    return "failed"
                return self._translation_stage_status_from_data(payload)
            return self._metadata_status(image_relative_path, "translation")

        if stage_name == "inpaint":
            output_path = inpaint_image_path(self.current_project, image_relative_path)
            if output_path.exists():
                return "done"
            cache_path = inpaint_json_path(self.current_project, image_relative_path)
            if cache_path.exists():
                try:
                    payload = load_inpaint_json(cache_path)
                except Exception:
                    return "failed"
                return self._inpaint_stage_status_from_data(payload)
            return self._metadata_status(image_relative_path, "inpaint")

        if stage_name == "render":
            output_path = render_image_path(self.current_project, image_relative_path)
            if output_path.exists():
                return "done"
            cache_path = render_json_path(self.current_project, image_relative_path)
            if cache_path.exists():
                try:
                    payload = load_render_json(cache_path)
                except Exception:
                    return "failed"
                return self._render_stage_status_from_data(payload)
            return self._metadata_status(image_relative_path, "render")

        return self._metadata_status(image_relative_path, stage_name)

    def _preview_mode_artifact_exists(self, preview_mode: str, image_relative_path: str | None) -> bool:
        if self.current_project is None or not image_relative_path:
            return False

        normalized_mode = str(preview_mode or "").strip()
        if normalized_mode == PREVIEW_INPAINT:
            return inpaint_image_path(self.current_project, image_relative_path).exists()
        if normalized_mode == PREVIEW_RENDER:
            return render_image_path(self.current_project, image_relative_path).exists()
        if normalized_mode == PREVIEW_MASK:
            return inpaint_preview_mask_path(self.current_project, image_relative_path).exists()
        if normalized_mode == PREVIEW_DETECTION:
            return detection_json_path(self.current_project, image_relative_path).exists()
        if normalized_mode == PREVIEW_SOURCE:
            image_path = self.current_project.root_dir / image_relative_path
            return image_path.exists()
        return False

    def _export_stage_status_for_page(self, image_relative_path: str) -> str:
        if not isinstance(self.last_export_result, dict):
            return "missing"

        items = self.last_export_result.get("items", [])
        if not isinstance(items, list):
            return "missing"

        matched_item: dict[str, Any] | None = None
        for item in items:
            if not isinstance(item, dict):
                continue
            if str(item.get("image_relative_path", "") or "") == image_relative_path:
                matched_item = item
                break

        if matched_item is None:
            return "missing"

        item_status = str(matched_item.get("status", "") or "").strip().lower()
        if item_status == "exported":
            return "done"
        if item_status == "error":
            return "failed"
        if item_status == "skipped":
            return "ready"
        return "missing"

    @staticmethod
    def _export_stage_label(status: str) -> str:
        normalized = str(status or "missing").strip().lower()
        if normalized == "done":
            return "Exported"
        if normalized == "error":
            return "Error"
        if normalized == "ready":
            return "Partial"
        return "Not Exported"

    def _refresh_stage_notes_and_actions(self) -> None:
        has_project = self.current_project is not None
        has_pages = bool(self.current_project and self.current_project.page_count > 0)
        current_page = self.current_page()
        has_page = has_project and current_page is not None
        process_running = self._process_is_running()
        workflow_busy = self._workflow_busy()
        remove_page_enabled = bool(has_page and not self._active_workers and not self._busy_stages)
        has_detection_cache = bool(
            has_page and self.current_project and detection_json_path(self.current_project, current_page).exists()
        )
        detection_ready = bool(has_page and self._stage_status_for_current_page("detection", current_page) == "done")
        ocr_ready = bool(has_page and self._stage_status_for_current_page("ocr", current_page) in {"done", "ready"})
        translation_ready = bool(
            has_page and self._stage_status_for_current_page("translation", current_page) in {"done", "ready", "initialized", "partial"}
        )
        inpaint_ready = bool(
            has_page and self._stage_status_for_current_page("inpaint", current_page) in {"done", "ready", "prepared", "partial"}
        )
        render_ready = bool(has_page and self._stage_status_for_current_page("render", current_page) == "done")
        ocr_stale = bool(has_page and current_page and self._stage_is_stale(current_page, "ocr"))
        translation_stale = bool(has_page and current_page and self._stage_is_stale(current_page, "translation"))
        inpaint_stale = bool(has_page and current_page and self._stage_is_stale(current_page, "inpaint"))
        render_stale = bool(has_page and current_page and self._stage_is_stale(current_page, "render"))
        export_stale = bool(has_page and current_page and self._stage_is_stale(current_page, "export"))
        ocr_boxes_need_rerun = bool(has_page and isinstance(self.current_ocr_data, dict) and summarize_ocr_edit_state(self.current_ocr_data).get("needs_ocr_items", 0))
        inpaint_needs_rerun = bool(
            has_page and isinstance(self.current_inpaint_data, dict) and self.current_inpaint_data.get("needs_inpaint", False)
        )
        render_boxes_need_rerun = bool(has_page and isinstance(self.current_render_data, dict) and self.current_render_data.get("needs_render", False))
        render_no_text_done = bool(
            has_page
            and isinstance(self.current_render_data, dict)
            and bool(self.current_render_data.get("no_text_page", False))
            and str(self.current_render_data.get("status", "") or "").strip().lower() == "done"
        )
        render_no_text_candidate = bool(
            has_page
            and has_detection_cache
            and current_page
            and self._current_page_has_no_active_render_items(current_page)
        )

        detection_note = None if has_page else "Open a project and select a page to run detection."
        if has_page and isinstance(self.current_detection_data, dict) and bool(self.current_detection_data.get("edited", False)):
            detection_note = "Detection was edited manually. Re-prepare OCR is recommended."
        self.detection_panel.set_stage_note(detection_note)
        self.process_panel.set_stage_note(
                "One-click processing is running. The inspector will follow each active stage."
                if process_running
                else (
                "Process runs Detection through Render using the current Config settings. Export is not included."
                if has_project
                else "Open a project before using one-click processing."
            )
        )
        self.ocr_panel.set_stage_note(
            "OCR boxes changed. Re-run OCR for affected items is recommended." if has_page and ocr_boxes_need_rerun else (
                "Detection was edited. Re-prepare OCR is recommended." if has_page and ocr_stale else (
                "Run Detection first before preparing OCR items." if has_page and not detection_ready else (
                "Open a project and select a page to work with OCR." if not has_page else None
                )
                )
            )
        )
        self.translation_panel.set_stage_note(
            "Upstream OCR or detection edits exist. Re-run translation when ready." if has_page and translation_stale else (
                "Prepare OCR items and run OCR first." if has_page and not ocr_ready else (
                "Open a project and select a page to work with translation." if not has_page else None
                )
            )
        )
        self.inpaint_panel.set_stage_note(
            "Mask changed. Inpaint will run again." if has_page and inpaint_needs_rerun else (
                "Upstream OCR or detection edits exist. Re-run inpaint when ready." if has_page and inpaint_stale else (
                "Prepare OCR items first before masking or inpainting." if has_page and not ocr_ready else (
                "Open a project and select a page to work with inpaint." if not has_page else None
                )
                )
            )
        )
        render_note = None
        if has_page and render_boxes_need_rerun:
            render_note = "Render boxes changed. Re-render is recommended."
        elif has_page and render_stale:
            render_note = "Upstream detection/OCR/translation/inpaint edits exist. Re-run render when ready."
        elif render_no_text_done:
            render_note = "No text items found. Render output was created from base image."
        elif render_no_text_candidate:
            render_note = "No active text items found. Render will create output from the inpaint image or source page."
        elif has_page and not (translation_ready and inpaint_ready):
            render_note = "Run Translation and Inpaint first before preparing or rendering text."
        elif not has_page:
            render_note = "Open a project and select a page to work with render."
        self.render_panel.set_stage_note(render_note)

        export_source = str(self.export_panel.export_source_input.currentData() or "render")
        export_source_ready = self._export_source_available_for_current_page(export_source)
        export_note = None
        if not has_project:
            export_note = "Open a project before exporting."
        elif export_source == "render" and has_page and not export_source_ready:
            export_note = "Render result missing. Run Render first or choose a different export source."
        elif export_source == "inpaint" and has_page and not export_source_ready:
            export_note = "Inpaint result missing. Run Inpaint first or choose a different export source."
        elif export_source == "source" and has_page and not export_source_ready:
            export_note = "Source image missing. Re-import the page or choose another export source."
        elif has_page and export_stale:
            export_note = "Upstream edits exist. Re-export when your latest render or inpaint output is ready."
        self.export_panel.set_stage_note(export_note)

        current_index = self._current_page_index() if has_pages else None
        self.left_toolbar.first_page_button.setEnabled(bool(has_pages and current_index not in {None, 0}))
        self.left_toolbar.previous_page_button.setEnabled(bool(has_pages and current_index not in {None, 0}))
        self.left_toolbar.next_page_button.setEnabled(
            bool(has_pages and current_index is not None and current_index < self.current_project.page_count - 1)
        )
        self.left_toolbar.last_page_button.setEnabled(
            bool(has_pages and current_index is not None and current_index < self.current_project.page_count - 1)
        )
        self.left_toolbar.mode_combo.setEnabled(bool(has_page))
        self.left_toolbar.fit_button.setEnabled(bool(has_page))
        self.left_toolbar.reset_button.setEnabled(bool(has_page))
        self.left_toolbar.zoom_out_button.setEnabled(bool(has_page))
        self.left_toolbar.zoom_in_button.setEnabled(bool(has_page))
        self.left_toolbar.auto_preview_checkbox.setEnabled(True)
        self.left_toolbar.follow_batch_checkbox.setEnabled(True)

        self.project_panel.remove_current_page_button.setEnabled(remove_page_enabled)
        self.project_panel.new_project_button.setEnabled(not workflow_busy)
        self.project_panel.open_project_button.setEnabled(not workflow_busy)
        self.project_panel.save_project_button.setEnabled(bool(has_project and not workflow_busy))
        self.project_panel.import_images_button.setEnabled(bool(has_project and not workflow_busy))
        if self.remove_current_page_action is not None:
            self.remove_current_page_action.setEnabled(remove_page_enabled)
        if self.new_project_action is not None:
            self.new_project_action.setEnabled(not workflow_busy)
        if self.open_project_action is not None:
            self.open_project_action.setEnabled(not workflow_busy)
        if self.save_project_action is not None:
            self.save_project_action.setEnabled(bool(has_project and not workflow_busy))
        if self.import_images_action is not None:
            self.import_images_action.setEnabled(bool(has_project and not workflow_busy))
        self.process_panel.process_current_button.setEnabled(bool(has_page and not process_running and not self._active_workers))
        self.process_panel.reprocess_current_button.setEnabled(bool(has_page and not process_running and not self._active_workers))
        self.process_panel.process_chapter_button.setEnabled(bool(has_pages and not process_running and not self._active_workers))
        self.process_panel.reprocess_chapter_button.setEnabled(bool(has_pages and not process_running and not self._active_workers))
        self.process_panel.set_cancel_enabled(bool(process_running and not self._process_cancel_requested))
        self.process_panel.set_cancel_stopping(bool(process_running and self._process_cancel_requested))

        if not process_running and "detection" not in self._busy_stages:
            self.detection_panel.run_selected_button.setEnabled(bool(has_page))
            self.detection_panel.rerun_selected_button.setEnabled(bool(has_page))
            self.detection_panel.run_all_button.setEnabled(bool(has_pages))
            self.detection_panel.rerun_all_button.setEnabled(bool(has_pages))
            self.detection_panel.reload_button.setEnabled(bool(has_page))
            self.detection_panel.clear_overlay_button.setEnabled(bool(has_page and not self.detection_panel.edit_mode_enabled()))
            self.detection_panel.enable_edit_checkbox.setEnabled(bool(has_page and has_detection_cache))
            if not has_detection_cache:
                self.detection_panel.set_create_box_checked(False)

        if not process_running and "ocr" not in self._busy_stages:
            self.ocr_panel.prepare_selected_button.setEnabled(bool(has_page and detection_ready))
            self.ocr_panel.reprepare_selected_button.setEnabled(bool(has_page and detection_ready))
            self.ocr_panel.prepare_all_button.setEnabled(bool(has_pages))
            self.ocr_panel.reprepare_all_button.setEnabled(bool(has_pages))
            self.ocr_panel.run_selected_button.setEnabled(bool(has_page and ocr_ready))
            self.ocr_panel.rerun_selected_button.setEnabled(bool(has_page and ocr_ready))
            self.ocr_panel.run_all_button.setEnabled(bool(has_pages))
            self.ocr_panel.rerun_all_button.setEnabled(bool(has_pages))
            has_selected_ocr_items = bool(self.current_ocr_data and self.ocr_panel.selected_item_ids())
            self.ocr_panel.run_selected_items_button.setEnabled(bool(has_page and has_selected_ocr_items))
            self.ocr_panel.rerun_selected_items_button.setEnabled(bool(has_page and has_selected_ocr_items))
            self.ocr_panel.reload_button.setEnabled(bool(has_page))
            self.ocr_panel.save_text_button.setEnabled(bool(has_page and self.current_ocr_data is not None))

        if not process_running and "translation" not in self._busy_stages:
            self.translation_panel.initialize_selected_button.setEnabled(bool(has_page and ocr_ready))
            self.translation_panel.reinitialize_selected_button.setEnabled(bool(has_page and ocr_ready))
            self.translation_panel.initialize_all_button.setEnabled(bool(has_pages))
            self.translation_panel.reinitialize_all_button.setEnabled(bool(has_pages))
            self.translation_panel.run_selected_button.setEnabled(bool(has_page and translation_ready))
            self.translation_panel.rerun_selected_button.setEnabled(bool(has_page and translation_ready))
            self.translation_panel.run_all_button.setEnabled(bool(has_pages))
            self.translation_panel.rerun_all_button.setEnabled(bool(has_pages))
            has_selected_translation_items = bool(
                self.current_translation_data and self.translation_panel.selected_item_ids()
            )
            self.translation_panel.run_selected_items_button.setEnabled(
                bool(has_page and has_selected_translation_items)
            )
            self.translation_panel.rerun_selected_items_button.setEnabled(
                bool(has_page and has_selected_translation_items)
            )
            self.translation_panel.reload_button.setEnabled(bool(has_page))
            self.translation_panel.save_text_button.setEnabled(
                bool(has_page and self.current_translation_data is not None)
            )

        if not process_running and "inpaint" not in self._busy_stages:
            self.inpaint_panel.prepare_selected_button.setEnabled(bool(has_page and ocr_ready))
            self.inpaint_panel.reprepare_selected_button.setEnabled(bool(has_page and ocr_ready))
            self.inpaint_panel.prepare_all_button.setEnabled(bool(has_pages))
            self.inpaint_panel.reprepare_all_button.setEnabled(bool(has_pages))
            self.inpaint_panel.run_selected_button.setEnabled(bool(has_page and ocr_ready))
            self.inpaint_panel.rerun_selected_button.setEnabled(bool(has_page and ocr_ready))
            self.inpaint_panel.run_all_button.setEnabled(bool(has_pages))
            self.inpaint_panel.rerun_all_button.setEnabled(bool(has_pages))
            self.inpaint_panel.reload_button.setEnabled(bool(has_page))
            self.inpaint_panel.clear_preview_button.setEnabled(bool(has_page))

        if not process_running and "render" not in self._busy_stages:
            self.render_panel.prepare_selected_button.setEnabled(bool(has_page and translation_ready))
            self.render_panel.reprepare_selected_button.setEnabled(bool(has_page and translation_ready))
            self.render_panel.prepare_all_button.setEnabled(bool(has_pages))
            self.render_panel.reprepare_all_button.setEnabled(bool(has_pages))
            self.render_panel.run_selected_button.setEnabled(bool(has_page and translation_ready and inpaint_ready))
            self.render_panel.rerun_selected_button.setEnabled(bool(has_page and translation_ready and inpaint_ready))
            self.render_panel.run_all_button.setEnabled(bool(has_pages))
            self.render_panel.rerun_all_button.setEnabled(bool(has_pages))
            self.render_panel.reload_button.setEnabled(bool(has_page))
            self.render_panel.clear_preview_button.setEnabled(bool(has_page))

        if not process_running and "export" not in self._busy_stages:
            self.export_panel.export_current_button.setEnabled(bool(has_page and export_source_ready))
            self.export_panel.export_all_button.setEnabled(bool(has_pages))
            self.export_panel.open_output_folder_button.setEnabled(bool(self.export_panel.output_dir().strip()))

    def _export_source_available_for_current_page(self, export_source: str) -> bool:
        if self.current_project is None:
            return False
        current_page = self.current_page()
        if not current_page:
            return False
        normalized_source = str(export_source or "render").strip().lower()
        if normalized_source == "source":
            page_index = self._current_page_index()
            if page_index is None:
                return False
            image_path = self.current_project.image_path_for_index(page_index)
            return bool(image_path is not None and image_path.exists())
        if normalized_source == "inpaint":
            return inpaint_image_path(self.current_project, current_page).exists()
        return render_image_path(self.current_project, current_page).exists()

    @staticmethod
    def _ocr_stage_status_from_data(ocr_data: dict[str, Any] | None) -> str:
        if not isinstance(ocr_data, dict):
            return "missing"
        summary = summarize_ocr_items(ocr_data.get("items", []))
        total_items = summary.get("total", 0)
        done_items = summary.get("done", 0)
        error_items = summary.get("error", 0)
        if total_items == 0:
            return "prepared"
        if done_items == total_items and error_items == 0:
            return "done"
        if done_items > 0 or error_items > 0:
            return "partial"
        return "prepared"

    def _update_project_ocr_stage_status(self, image_relative_path: str, ocr_data: dict[str, Any]) -> None:
        if self.current_project is None:
            return
        cache_path = ocr_json_path(self.current_project, image_relative_path)
        self.current_project.update_stage_status(
            image_relative_path,
            "ocr",
            status=self._ocr_stage_status_from_data(ocr_data),
            cache_path=self._relative_project_path(cache_path),
        )

    @staticmethod
    def _translation_stage_status_from_data(translation_data: dict[str, Any] | None) -> str:
        if not isinstance(translation_data, dict):
            return "missing"
        summary = summarize_translation_json(translation_data)
        total_items = summary.get("total", 0)
        done_items = summary.get("done", 0)
        error_items = summary.get("error", 0)
        pending_items = summary.get("pending", 0)
        manual_items = summary.get("manually_edited", 0)
        skipped_items = summary.get("skipped", 0)
        if total_items == 0:
            return "initialized"
        if done_items + manual_items + skipped_items == total_items and error_items == 0 and pending_items == 0:
            return "done"
        if error_items == total_items:
            return "failed"
        if done_items > 0 or manual_items > 0 or error_items > 0:
            return "partial"
        return "initialized"

    def _update_project_translation_stage_status(
        self,
        image_relative_path: str,
        translation_data: dict[str, Any],
    ) -> None:
        if self.current_project is None:
            return
        cache_path = translation_json_path(self.current_project, image_relative_path)
        self.current_project.update_stage_status(
            image_relative_path,
            "translation",
            status=self._translation_stage_status_from_data(translation_data),
            cache_path=self._relative_project_path(cache_path),
        )

    @staticmethod
    def _inpaint_stage_status_from_data(inpaint_data: dict[str, Any] | None) -> str:
        if not isinstance(inpaint_data, dict):
            return "missing"
        if bool(inpaint_data.get("needs_inpaint", False)):
            return "prepared"
        status = str(inpaint_data.get("status", "pending") or "pending").strip().lower()
        if status == "done":
            return "done"
        if status == "error":
            return "failed"
        if status == "running":
            return "running"
        return "prepared"

    def _update_project_inpaint_stage_status(
        self,
        image_relative_path: str,
        inpaint_data: dict[str, Any],
    ) -> None:
        if self.current_project is None:
            return
        cache_path = inpaint_json_path(self.current_project, image_relative_path)
        self.current_project.update_stage_status(
            image_relative_path,
            "inpaint",
            status=self._inpaint_stage_status_from_data(inpaint_data),
            cache_path=self._relative_project_path(cache_path),
        )

    @staticmethod
    def _render_stage_status_from_data(render_data: dict[str, Any] | None) -> str:
        if not isinstance(render_data, dict):
            return "missing"
        summary = summarize_render_json(render_data)
        total_items = int(summary.get("total", 0) or 0)
        rendered_items = int(summary.get("rendered", 0) or 0)
        error_items = int(summary.get("error", 0) or 0)
        status = str(render_data.get("status", "pending") or "pending").lower()
        if status == "done" and (rendered_items > 0 or bool(render_data.get("no_text_page", False))):
            return "done"
        if status == "error" and rendered_items == 0:
            return "failed"
        if rendered_items > 0 and rendered_items < max(total_items, 1):
            return "partial"
        if rendered_items > 0:
            return "done"
        if error_items > 0:
            return "failed"
        return "prepared"

    def _update_project_render_stage_status(
        self,
        image_relative_path: str,
        render_data: dict[str, Any],
    ) -> None:
        if self.current_project is None:
            return
        cache_path = render_json_path(self.current_project, image_relative_path)
        self.current_project.update_stage_status(
            image_relative_path,
            "render",
            status=self._render_stage_status_from_data(render_data),
            cache_path=self._relative_project_path(cache_path),
        )

    def _current_page_has_no_active_render_items(self, image_relative_path: str) -> bool:
        if self.current_project is None:
            return False

        normalized_relative_path = str(Path(image_relative_path).as_posix())
        detection_data: dict[str, Any] | None = None
        if isinstance(self.current_detection_data, dict):
            current_detection_source = str(
                Path(str(self.current_detection_data.get("source_image", "") or "")).as_posix()
            )
            if current_detection_source == normalized_relative_path:
                detection_data = self.current_detection_data

        if detection_data is None:
            detection_path = detection_json_path(self.current_project, normalized_relative_path)
            if not detection_path.exists():
                return False
            try:
                detection_data = load_detection_json(detection_path)
            except Exception:
                return False

        try:
            ensure_canon_state(detection_data)
        except Exception:
            return False

        return len(get_active_canon_items(detection_data)) == 0

    def _refresh_preview_for_current_page(self) -> bool:
        if self.current_project is None:
            self.image_preview.clear_image()
            return False

        index = self._current_page_index()
        if index is None:
            self.image_preview.clear_image()
            return False

        image_relative_path = self.current_project.page_relative_path_for_index(index)
        page_path = self.current_project.image_path_for_index(index)
        if image_relative_path is None or page_path is None or not page_path.exists():
            self.image_preview.clear_image()
            return False

        preview_mode = self.left_toolbar.current_mode()
        preview_image_path = page_path
        mask_overlay_path: Path | None = None

        if self.current_stage_key == "render" and self.render_panel.box_edit_mode_enabled():
            cached_render_path = render_image_path(self.current_project, image_relative_path)
            cached_inpaint_path = inpaint_image_path(self.current_project, image_relative_path)
            if cached_render_path.exists():
                preview_image_path = cached_render_path
            elif cached_inpaint_path.exists():
                preview_image_path = cached_inpaint_path
        elif self.current_stage_key == "ocr" and self.ocr_panel.box_edit_mode_enabled():
            preview_image_path = page_path

        if preview_mode == PREVIEW_INPAINT:
            cached_inpaint_path = inpaint_image_path(self.current_project, image_relative_path)
            if cached_inpaint_path.exists():
                preview_image_path = cached_inpaint_path
        elif preview_mode == PREVIEW_RENDER:
            cached_render_path = render_image_path(self.current_project, image_relative_path)
            if cached_render_path.exists():
                preview_image_path = cached_render_path
        elif preview_mode == PREVIEW_MASK:
            cached_mask_path = inpaint_preview_mask_path(self.current_project, image_relative_path)
            if cached_mask_path.exists():
                mask_overlay_path = cached_mask_path

        if not self.image_preview.set_image(preview_image_path):
            return False

        if mask_overlay_path is not None and mask_overlay_path.exists():
            self.image_preview.set_mask_overlay(mask_overlay_path)
        else:
            self.image_preview.clear_mask_overlay()

        if self.current_stage_key == "detection" and self.detection_panel.edit_mode_enabled():
            self.image_preview.set_box_edit_mode(True)
            self.image_preview.set_editable_box_category_filter(self.detection_panel.selected_box_category())
            self.image_preview.set_show_excluded_boxes(self.detection_panel.show_excluded_enabled())
            self.image_preview.refresh_editable_boxes()
        elif self.current_stage_key == "ocr" and self.ocr_panel.box_edit_mode_enabled():
            self.image_preview.set_box_edit_mode(True)
            self.image_preview.set_editable_box_category_filter(self._ocr_overlay_category())
            self.image_preview.set_show_excluded_boxes(self.ocr_panel.show_excluded_items_enabled())
            self.image_preview.refresh_editable_boxes()
        elif self.current_stage_key == "render" and self.render_panel.box_edit_mode_enabled():
            self.image_preview.set_box_edit_mode(True)
            self.image_preview.set_editable_box_category_filter("render_bbox")
            self.image_preview.set_show_excluded_boxes(self.render_panel.show_excluded_items_enabled())
            self.image_preview.refresh_editable_boxes()
        elif (
            preview_mode == PREVIEW_DETECTION
            and self.preview_detection_overlay_enabled
            and self.current_detection_data is not None
        ):
            self.image_preview.set_detection_overlay(self.current_detection_data)
        else:
            self.image_preview.set_box_edit_mode(False)
            self.image_preview.clear_overlays()

        return True

    def _sync_lama_status_label(self) -> None:
        model_status = get_lama_model_manager().status()
        if model_status.get("busy"):
            self.inpaint_panel.set_model_status(
                f"Busy ({model_status.get('device', 'auto') or 'auto'})"
            )
        elif model_status.get("loaded"):
            self.inpaint_panel.set_model_status(
                f"Loaded ({model_status.get('device', 'auto') or 'auto'})"
            )
        else:
            self.inpaint_panel.set_model_status("Not loaded")

    def _relative_project_path(self, path: Path) -> str:
        if self.current_project is None:
            return str(path)
        try:
            return path.resolve().relative_to(self.current_project.root_dir).as_posix()
        except ValueError:
            return str(path)

    def _release_worker(self, worker: TaskWorker | ServiceCommandHandle) -> None:
        if worker in self._active_workers:
            self._active_workers.remove(worker)
        if self._active_process_worker is worker:
            self._active_process_worker = None

    def _update_window_title(self) -> None:
        if self.current_project is None:
            self.setWindowTitle(APP_NAME)
            return
        self.setWindowTitle(f"{APP_NAME} - {self.current_project.data.name}")

    def log(self, message: str, *, level: str = "info") -> None:
        self.log_panel.append_message(message)
        normalized_level = str(level or "info").strip().lower()
        if normalized_level in {"warning", "error"}:
            dock_visible = bool(self.developer_log_dock is not None and self.developer_log_dock.isVisible())
            if not dock_visible:
                self._set_log_alert_state(True, error=(normalized_level == "error") or self._has_unread_error_alert)

    def show_error(self, title: str, message: str) -> None:
        self._set_developer_log_visible(True)
        QMessageBox.critical(self, title, message)
        self.log(f"{title}: {message}", level="error")


__all__ = ["MainWindow"]
