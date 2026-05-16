"""Reusable widgets for the desktop GUI shell."""

from __future__ import annotations

from .app_header import AppHeader
from .collapsible_section import CollapsibleSection, StaticSection
from .crop_preview_panel import CropPreviewPanel
from .image_preview import ImagePreviewWidget
from .left_tool_bar import LeftToolBar
from .log_panel import LogPanel
from .page_list import PageListWidget
from .page_filmstrip import PageFilmstripWidget
from .preview_toolbar import PreviewToolbar
from .settings_card import SettingsCard, style_button
from .startup_overlay import StartupOverlay
from .stage_status import StageStatusDot, StageStatusLine, StatusLabel
from .text_item_editor import TextItemEditorWidget
from .workflow_tabs import STAGE_ORDER, WorkflowTabs
from .workflow_sidebar import WorkflowSidebar

__all__ = [
    "AppHeader",
    "CollapsibleSection",
    "CropPreviewPanel",
    "ImagePreviewWidget",
    "LeftToolBar",
    "LogPanel",
    "PageFilmstripWidget",
    "PageListWidget",
    "PreviewToolbar",
    "SettingsCard",
    "StartupOverlay",
    "StaticSection",
    "StageStatusDot",
    "StageStatusLine",
    "StatusLabel",
    "STAGE_ORDER",
    "TextItemEditorWidget",
    "WorkflowTabs",
    "WorkflowSidebar",
    "style_button",
]
