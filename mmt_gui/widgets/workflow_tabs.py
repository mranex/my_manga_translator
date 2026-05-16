"""Top workflow tab strip with folder-like buttons and line status indicators."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QStyle,
    QVBoxLayout,
    QWidget,
)

STAGE_ORDER = (
    ("process", "Process", QStyle.StandardPixmap.SP_BrowserReload),
    ("project", "Project", QStyle.StandardPixmap.SP_DirIcon),
    ("config", "Config", QStyle.StandardPixmap.SP_FileDialogDetailedView),
    ("detection", "Detection", QStyle.StandardPixmap.SP_FileDialogContentsView),
    ("ocr", "OCR", QStyle.StandardPixmap.SP_FileIcon),
    ("translation", "Translation", QStyle.StandardPixmap.SP_MessageBoxInformation),
    ("inpaint", "Inpaint", QStyle.StandardPixmap.SP_ToolBarHorizontalExtensionButton),
    ("render", "Render", QStyle.StandardPixmap.SP_DesktopIcon),
    ("export", "Export", QStyle.StandardPixmap.SP_DialogSaveButton),
)

def get_colorized_icon(style: QStyle, icon_enum: QStyle.StandardPixmap, color_hex: str = "#8a8a9e") -> QIcon:
    icon = style.standardIcon(icon_enum)
    pixmap = icon.pixmap(24, 24)
    if pixmap.isNull():
        return icon
    painter = QPainter(pixmap)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    painter.fillRect(pixmap.rect(), QColor(color_hex))
    painter.end()
    return QIcon(pixmap)


class StageTabButton(QPushButton):
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        
        rect = self.rect()
        w, h = rect.width(), rect.height()
        slant = 14
        
        path = QPainterPath()
        path.moveTo(slant, 0)
        path.lineTo(w, 0)
        path.lineTo(w - slant, h)
        path.lineTo(0, h)
        path.closeSubpath()

        is_active = self.property("activeTab")
        is_hover = self.underMouse()
        is_dark = self.palette().window().color().lightness() < 128
        
        if is_active:
            bg_color = QColor("#1e1033") if is_dark else QColor("#f3e8ff")
            border_color = QColor("#9d4edd") if is_dark else QColor("#9333ea")
        elif is_hover:
            bg_color = QColor("#1f1f2e") if is_dark else QColor("#f1f5f9")
            border_color = QColor("#ffd60a") if is_dark else QColor("#fbbf24")
        else:
            bg_color = QColor("#111116") if is_dark else QColor("#ffffff")
            border_color = QColor("#2c2c36") if is_dark else QColor("#e2e8f0")
            
        painter.setBrush(bg_color)
        painter.setPen(QPen(border_color, 2 if is_active or is_hover else 1))
        painter.drawPath(path)
        
        status = self.property("stageStatus")
        if status and status != "missing":
            from .stage_status import normalize_stage_status, status_gradient_colors
            from PyQt6.QtGui import QLinearGradient
            norm_status = normalize_stage_status(status)
            start_color, end_color = status_gradient_colors(norm_status)
            grad = QLinearGradient(0, h - 4, w, h - 4)
            grad.setColorAt(0, QColor(start_color))
            grad.setColorAt(1, QColor(end_color))
            
            line_path = QPainterPath()
            line_path.moveTo(slant + 1, h - 4)
            line_path.lineTo(w - 1, h - 4)
            line_path.lineTo(w - slant - 1, h)
            line_path.lineTo(1, h)
            line_path.closeSubpath()
            painter.setBrush(grad)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPath(line_path)

        painter.end()
        super().paintEvent(event)


class WorkflowTabs(QFrame):
    """Folder-like workflow tabs shown above the studio workspace."""

    stage_selected = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("WorkflowTabs")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMaximumHeight(84)
        self._current_stage_key = "project"
        self._suppress_emits = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 4)
        layout.setSpacing(8)

        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True)
        self.stage_rows: dict[str, QPushButton] = {}
        self._glow_effects: dict[str, QGraphicsDropShadowEffect] = {}

        for stage_key, stage_label, icon_enum in STAGE_ORDER:
            tab_container = QWidget(self)
            tab_container.setObjectName("WorkflowTabContainer")
            tab_layout = QVBoxLayout(tab_container)
            tab_layout.setContentsMargins(0, 0, 0, 0)
            tab_layout.setSpacing(0)

            button = StageTabButton(stage_label, tab_container)
            colorized = get_colorized_icon(self.style(), icon_enum)
            button.setIcon(colorized)
            button.setCheckable(True)
            button.setProperty("folderTab", True)
            button.setToolTip(stage_label)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setAutoDefault(False)
            button.setDefault(False)
            button.setMinimumWidth(80)
            button.setFixedHeight(42)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            font = button.font()
            font.setFamily("Orbitron")
            font.setPointSize(11)
            button.setFont(font)
            button.clicked.connect(lambda checked=False, key=stage_key: self._emit_stage_selected(key))

            glow_effect = QGraphicsDropShadowEffect(button)
            glow_effect.setBlurRadius(28)
            glow_effect.setOffset(0, 4)
            glow_effect.setEnabled(False)
            button.setGraphicsEffect(glow_effect)

            self.button_group.addButton(button)
            tab_layout.addWidget(button)

            layout.addWidget(tab_container)
            self.stage_rows[stage_key] = button
            self._glow_effects[stage_key] = glow_effect
            button.setProperty("stageKey", stage_key)

        self.set_current_stage(self._current_stage_key)

    def set_current_stage(self, stage_key: str) -> None:
        if stage_key not in self.stage_rows:
            stage_key = "project"
        self._current_stage_key = stage_key
        self._suppress_emits = True
        try:
            for key, button in self.stage_rows.items():
                is_active = key == stage_key
                button.setChecked(is_active)
                button.setProperty("activeTab", is_active)
                button.style().unpolish(button)
                button.style().polish(button)
                self._update_button_glow(key, active=is_active)
        finally:
            self._suppress_emits = False

    def set_stage_status(self, stage_key: str, status: str) -> None:
        button = self.stage_rows.get(stage_key)
        if button is None:
            return
        button.setProperty("stageStatus", status)
        button.update()
        button.setToolTip(f"{button.text()}\nStatus: {status.title()}")

    def set_stage_statuses(self, statuses: dict[str, str]) -> None:
        for stage_key, _label, _icon in STAGE_ORDER:
            self.set_stage_status(stage_key, statuses.get(stage_key, "missing"))

    def _emit_stage_selected(self, stage_key: str) -> None:
        if self._suppress_emits:
            return
        self._current_stage_key = stage_key
        self.stage_selected.emit(stage_key)

    def refresh_theme_state(self) -> None:
        self.set_current_stage(self._current_stage_key)
        self.update()

    def _update_button_glow(self, stage_key: str, *, active: bool) -> None:
        effect = self._glow_effects.get(stage_key)
        if effect is None:
            return

        is_dark = self.palette().window().color().lightness() < 128
        glow_color = QColor("#9d4edd" if is_dark else "#b026ff")
        glow_color.setAlpha(95 if active and is_dark else (70 if active else 0))
        effect.setColor(glow_color)
        effect.setEnabled(active)
        effect.setBlurRadius(28 if active else 0)
        effect.setOffset(0, 5 if active else 0)


__all__ = ["STAGE_ORDER", "WorkflowTabs"]
