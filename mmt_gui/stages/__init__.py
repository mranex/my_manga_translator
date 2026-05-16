"""Stage inspector panels for the modern workbench layout."""

from __future__ import annotations

from .config_panel import ConfigPanel
from .detection_panel import DetectionPanel
from .export_panel import ExportPanel
from .inpaint_panel import InpaintPanel
from .ocr_panel import OCRPanel
from .process_panel import ProcessPanel
from .project_panel import ProjectPanel
from .render_panel import RenderPanel
from .translation_panel import TranslationPanel

__all__ = [
    "DetectionPanel",
    "ConfigPanel",
    "ExportPanel",
    "InpaintPanel",
    "OCRPanel",
    "ProcessPanel",
    "ProjectPanel",
    "RenderPanel",
    "TranslationPanel",
]
