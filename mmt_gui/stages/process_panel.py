"""Inspector panel for one-click chapter/current-page processing."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFormLayout,
    QGridLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mmt_core import PROCESS_PIPELINE_STEPS
from mmt_gui.widgets import CollapsibleSection
from mmt_gui.widgets.settings_card import style_button
from mmt_gui.widgets.stage_status import StatusLabel

from .base_panel import StagePanel


class ProcessPanel(StagePanel):
    """Shows one-click process actions, current progress, and step state."""

    process_current_requested = pyqtSignal()
    reprocess_current_requested = pyqtSignal()
    process_chapter_requested = pyqtSignal()
    reprocess_chapter_requested = pyqtSignal()
    cancel_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Process", parent)
        self._actions_enabled = True
        self._step_labels: dict[str, StatusLabel] = {}

        actions_card = CollapsibleSection("One-click Process", expanded=True)
        actions_layout = QGridLayout()
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setHorizontalSpacing(8)
        actions_layout.setVerticalSpacing(8)

        current_label = QLabel("Current Page")
        current_label.setProperty("role", "muted")
        actions_layout.addWidget(current_label, 0, 0)

        self.process_current_button = QPushButton("Process Current Page")
        style_button(self.process_current_button, "primary")
        self.process_current_button.clicked.connect(self.process_current_requested.emit)
        actions_layout.addWidget(self.process_current_button, 0, 1)

        self.reprocess_current_button = QPushButton("Re-process Current Page")
        style_button(self.reprocess_current_button, "rerun")
        self.reprocess_current_button.clicked.connect(self.reprocess_current_requested.emit)
        actions_layout.addWidget(self.reprocess_current_button, 0, 2)

        chapter_label = QLabel("Chapter")
        chapter_label.setProperty("role", "muted")
        actions_layout.addWidget(chapter_label, 1, 0)

        self.process_chapter_button = QPushButton("Process Chapter")
        style_button(self.process_chapter_button, "primary")
        self.process_chapter_button.clicked.connect(self.process_chapter_requested.emit)
        actions_layout.addWidget(self.process_chapter_button, 1, 1)

        self.reprocess_chapter_button = QPushButton("Re-process Chapter")
        style_button(self.reprocess_chapter_button, "rerun")
        self.reprocess_chapter_button.clicked.connect(self.reprocess_chapter_requested.emit)
        actions_layout.addWidget(self.reprocess_chapter_button, 1, 2)

        self.stop_process_button = QPushButton("Stop Process")
        style_button(self.stop_process_button, "danger")
        self.stop_process_button.clicked.connect(self.cancel_requested.emit)
        self.stop_process_button.setEnabled(False)
        actions_layout.addWidget(self.stop_process_button, 2, 1)

        self.cancel_note_label = QLabel("Process uses cooperative cancellation and stops at the next safe point.")
        self.cancel_note_label.setWordWrap(True)
        self.cancel_note_label.setProperty("role", "muted")
        actions_layout.addWidget(self.cancel_note_label, 3, 0, 1, 3)

        actions_card.content_layout.addLayout(actions_layout)
        self.content_layout.addWidget(actions_card)

        summary_card = CollapsibleSection("Settings Summary", expanded=True)
        self.settings_note_label = QLabel("Process uses the current settings from each stage.")
        self.settings_note_label.setWordWrap(True)
        self.settings_note_label.setProperty("role", "muted")
        summary_card.content_layout.addWidget(self.settings_note_label)

        summary_form = QFormLayout()
        summary_form.setContentsMargins(0, 0, 0, 0)
        summary_form.setSpacing(8)
        self.scope_value = QLabel("Current page")
        self.ocr_provider_value = QLabel("-")
        self.translator_value = QLabel("-")
        self.target_language_value = QLabel("-")
        self.inpaint_device_value = QLabel("-")
        self.render_style_value = QLabel("-")
        summary_form.addRow("Scope:", self.scope_value)
        summary_form.addRow("OCR Provider:", self.ocr_provider_value)
        summary_form.addRow("Translator:", self.translator_value)
        summary_form.addRow("Target Language:", self.target_language_value)
        summary_form.addRow("Inpaint Device:", self.inpaint_device_value)
        summary_form.addRow("Render Font/Style:", self.render_style_value)
        summary_card.content_layout.addLayout(summary_form)
        self.content_layout.addWidget(summary_card)

        pipeline_card = CollapsibleSection("Pipeline Steps", expanded=True)
        pipeline_form = QFormLayout()
        pipeline_form.setContentsMargins(0, 0, 0, 0)
        pipeline_form.setSpacing(8)
        for step in PROCESS_PIPELINE_STEPS:
            status_label = StatusLabel()
            status_label.set_status_text("Pending")
            self._step_labels[step.key] = status_label
            pipeline_form.addRow(f"{step.display_name}:", status_label)
        pipeline_card.content_layout.addLayout(pipeline_form)
        self.content_layout.addWidget(pipeline_card)

        progress_card = CollapsibleSection("Progress", expanded=True)
        progress_form = QFormLayout()
        progress_form.setContentsMargins(0, 0, 0, 0)
        progress_form.setSpacing(8)

        self.current_scope_value = QLabel("Idle")
        self.current_scope_value.setWordWrap(True)
        self.current_page_value = QLabel("-")
        self.current_page_value.setWordWrap(True)
        self.current_stage_value = QLabel("-")
        self.current_stage_value.setWordWrap(True)
        self.overall_progress_bar = QProgressBar()
        self.overall_progress_bar.setRange(0, 100)
        self.overall_progress_bar.setValue(0)
        self.action_message_value = QLabel("Ready")
        self.action_message_value.setWordWrap(True)
        self.last_error_value = QLabel("-")
        self.last_error_value.setWordWrap(True)
        self.last_error_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.done_summary_value = QLabel("-")
        self.done_summary_value.setWordWrap(True)
        self.done_summary_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        progress_form.addRow("Process Type:", self.current_scope_value)
        progress_form.addRow("Current Page:", self.current_page_value)
        progress_form.addRow("Current Stage:", self.current_stage_value)
        progress_form.addRow("Overall Progress:", self.overall_progress_bar)
        progress_form.addRow("Current Action:", self.action_message_value)
        progress_form.addRow("Last Error:", self.last_error_value)
        progress_form.addRow("Summary:", self.done_summary_value)
        progress_card.content_layout.addLayout(progress_form)
        self.content_layout.addWidget(progress_card)

        self.set_stage_note("Process runs Detection through Render using the current stage settings. Export is not included.")

    def set_actions_enabled(self, enabled: bool) -> None:
        self._actions_enabled = bool(enabled)
        for widget in (
            self.process_current_button,
            self.reprocess_current_button,
            self.process_chapter_button,
            self.reprocess_chapter_button,
        ):
            widget.setEnabled(bool(enabled))

    def set_cancel_enabled(self, enabled: bool) -> None:
        self.stop_process_button.setEnabled(bool(enabled))

    def set_cancel_stopping(self, stopping: bool) -> None:
        if stopping:
            self.stop_process_button.setText("Stopping...")
            self.stop_process_button.setEnabled(False)
        else:
            self.stop_process_button.setText("Stop Process")

    def set_scope_summary(self, scope_text: str) -> None:
        self.scope_value.setText(str(scope_text or "-"))
        self.current_scope_value.setText(str(scope_text or "Idle"))

    def set_settings_summary(
        self,
        *,
        ocr_provider: str,
        translator: str,
        target_language: str,
        inpaint_device: str,
        render_style: str,
    ) -> None:
        self.ocr_provider_value.setText(str(ocr_provider or "-"))
        self.translator_value.setText(str(translator or "-"))
        self.target_language_value.setText(str(target_language or "-"))
        self.inpaint_device_value.setText(str(inpaint_device or "-"))
        self.render_style_value.setText(str(render_style or "-"))

    def reset_process_state(self, *, scope_text: str) -> None:
        self.set_scope_summary(scope_text)
        self.current_page_value.setText("-")
        self.current_stage_value.setText("-")
        self.action_message_value.setText("Ready")
        self.last_error_value.setText("-")
        self.done_summary_value.setText("-")
        self.overall_progress_bar.setValue(0)
        self.set_cancel_stopping(False)
        self.set_cancel_enabled(False)
        for status_label in self._step_labels.values():
            status_label.set_status_text("Pending")

    def set_step_status(self, step_key: str, status: str) -> None:
        label = self._step_labels.get(step_key)
        if label is None:
            return
        normalized = str(status or "pending").strip().lower()
        display = {
            "pending": "Pending",
            "running": "Running",
            "done": "Done",
            "error": "Error",
            "skipped": "Skipped",
            "canceled": "Canceled",
        }.get(normalized, normalized.title())
        label.set_status_text(display)

    def set_current_page(self, image_relative_path: str | None) -> None:
        if not image_relative_path:
            self.current_page_value.setText("-")
            return
        self.current_page_value.setText(Path(str(image_relative_path)).name)

    def set_current_stage(self, stage_text: str | None) -> None:
        self.current_stage_value.setText(str(stage_text or "-"))

    def set_overall_progress(self, value: int) -> None:
        self.overall_progress_bar.setValue(max(0, min(100, int(value))))

    def set_action_message(self, message: str | None) -> None:
        self.action_message_value.setText(str(message or "Ready"))

    def set_last_error(self, message: str | None) -> None:
        normalized = str(message or "").strip()
        self.last_error_value.setText(normalized or "-")

    def set_done_summary(self, message: str | None) -> None:
        normalized = str(message or "").strip()
        self.done_summary_value.setText(normalized or "-")


__all__ = ["ProcessPanel"]
