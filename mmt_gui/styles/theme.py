"""QSS theme loader for the desktop GUI shell."""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QSpinBox,
    QTableWidget,
    QTextEdit,
    QWidget,
)

AVAILABLE_THEMES = ("Light", "Dark")
DEFAULT_THEME = "Dark"
LOGGER = logging.getLogger(__name__)


THEME_COLORS: dict[str, dict[str, str]] = {
    "Dark": {
        "window": "#0a0a0c",         # Deep Metallic Black
        "panel": "#111116",          # Dark panel background
        "elevated": "#181820",       # Elevated surfaces
        "base": "#111116",           # Input backgrounds
        "alternate_base": "#181820", # Alternate list/table rows
        "text": "#e5e7eb",           # Off-white text
        "muted": "#8a8a9e",          # Muted gray text
        "border": "#2c2c36",         # Subtle borders
        "highlight": "#9d4edd",      # Neon Purple highlight
        "highlight_text": "#ffffff", # Text on highlight
        "button": "#181820",         # Button base
        "button_text": "#e5e7eb",
    },
    "Light": {
        "window": "#f4f7fb",
        "panel": "#ffffff",
        "elevated": "#fbfdff",
        "base": "#ffffff",
        "alternate_base": "#f2f6fb",
        "text": "#152033",
        "muted": "#5f6c84",
        "border": "#d6deea",
        "highlight": "#d8e6ff",
        "highlight_text": "#13203a",
        "button": "#ffffff",
        "button_text": "#152033",
    },
}


class ThemeManager:
    """Loads and applies packaged QSS themes."""

    def __init__(self, initial_theme: str = DEFAULT_THEME) -> None:
        self.current_theme = self.normalize_theme_name(initial_theme)

    @staticmethod
    def normalize_theme_name(name: str) -> str:
        normalized = str(name or "").strip().lower()
        if normalized == "light":
            return "Light"
        if normalized == "dark":
            return "Dark"
        if normalized:
            LOGGER.warning("Unknown theme name '%s'; falling back to %s.", name, DEFAULT_THEME)
        return "Dark"

    def apply_theme(self, application: QApplication | None = None, root_widget: QWidget | None = None) -> str:
        app = application or QApplication.instance()
        if app is None:
            raise RuntimeError("A QApplication instance is required before applying a theme.")

        theme_name = self.normalize_theme_name(self.current_theme)
        qss_path = Path(__file__).with_name(f"{theme_name.lower()}.qss")
        if not qss_path.exists():
            raise FileNotFoundError(f"Theme stylesheet is missing: {qss_path}")

        stylesheet = qss_path.read_text(encoding="utf-8")
        palette = self._build_palette(theme_name)
        app.setPalette(palette)
        app.setStyleSheet("")
        app.setStyleSheet(stylesheet)
        if root_widget is not None:
            root_widget.setProperty("theme", theme_name.lower())
            self._apply_widget_palettes(root_widget, palette)
            root_widget.update()
        self.current_theme = theme_name
        LOGGER.info("Applied %s theme from %s.", theme_name, qss_path)
        return theme_name

    def set_theme(
        self,
        theme_name: str,
        *,
        application: QApplication | None = None,
        root_widget: QWidget | None = None,
    ) -> str:
        self.current_theme = self.normalize_theme_name(theme_name)
        return self.apply_theme(application=application, root_widget=root_widget)

    def _build_palette(self, theme_name: str) -> QPalette:
        colors = THEME_COLORS[self.normalize_theme_name(theme_name)]
        palette = QPalette()

        window = QColor(colors["window"])
        panel = QColor(colors["panel"])
        elevated = QColor(colors["elevated"])
        base = QColor(colors["base"])
        alternate_base = QColor(colors["alternate_base"])
        text = QColor(colors["text"])
        muted = QColor(colors["muted"])
        border = QColor(colors["border"])
        highlight = QColor(colors["highlight"])
        highlight_text = QColor(colors["highlight_text"])
        button = QColor(colors["button"])
        button_text = QColor(colors["button_text"])

        palette.setColor(QPalette.ColorRole.Window, window)
        palette.setColor(QPalette.ColorRole.WindowText, text)
        palette.setColor(QPalette.ColorRole.Base, base)
        palette.setColor(QPalette.ColorRole.AlternateBase, alternate_base)
        palette.setColor(QPalette.ColorRole.Text, text)
        palette.setColor(QPalette.ColorRole.Button, button)
        palette.setColor(QPalette.ColorRole.ButtonText, button_text)
        palette.setColor(QPalette.ColorRole.Light, elevated)
        palette.setColor(QPalette.ColorRole.Midlight, border)
        palette.setColor(QPalette.ColorRole.Dark, border)
        palette.setColor(QPalette.ColorRole.Mid, border)
        palette.setColor(QPalette.ColorRole.Shadow, window)
        palette.setColor(QPalette.ColorRole.ToolTipBase, panel)
        palette.setColor(QPalette.ColorRole.ToolTipText, text)
        palette.setColor(QPalette.ColorRole.PlaceholderText, muted)
        palette.setColor(QPalette.ColorRole.Highlight, highlight)
        palette.setColor(QPalette.ColorRole.HighlightedText, highlight_text)
        palette.setColor(QPalette.ColorRole.BrightText, highlight_text)

        for group in (QPalette.ColorGroup.Disabled,):
            palette.setColor(group, QPalette.ColorRole.WindowText, muted)
            palette.setColor(group, QPalette.ColorRole.Text, muted)
            palette.setColor(group, QPalette.ColorRole.ButtonText, muted)
            palette.setColor(group, QPalette.ColorRole.PlaceholderText, muted)
            palette.setColor(group, QPalette.ColorRole.Base, base)
            palette.setColor(group, QPalette.ColorRole.AlternateBase, alternate_base)
            palette.setColor(group, QPalette.ColorRole.Button, button)
            palette.setColor(group, QPalette.ColorRole.Highlight, border)
            palette.setColor(group, QPalette.ColorRole.HighlightedText, text)

        return palette

    def _apply_widget_palettes(self, root_widget: QWidget, palette: QPalette) -> None:
        targets = (QListWidget, QTableWidget, QAbstractItemView, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit)
        for widget in root_widget.findChildren(QWidget):
            if not isinstance(widget, targets):
                continue
            try:
                widget.setPalette(palette)
                if isinstance(widget, QAbstractItemView):
                    widget.viewport().setPalette(palette)
                    widget.viewport().setAutoFillBackground(True)
                    widget.setAutoFillBackground(True)
            except Exception:
                LOGGER.exception("Failed to refresh palette for widget %r during theme application.", widget)


__all__ = ["AVAILABLE_THEMES", "DEFAULT_THEME", "ThemeManager"]
