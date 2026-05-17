"""Small status badges used by the workflow sidebar and stage panels."""

from __future__ import annotations

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QLabel, QWidget

STATUS_COLORS = {
    "missing": "#3f3f5a",
    "ready": "#ffd60a",
    "done": "#00f2fe",
    "error": "#ff0844",
}
STATUS_LINE_GRADIENTS = {
    "missing": ("#2c2c36", "#3f3f5a"),
    "ready": ("#ffd60a", "#ff9f0a"),
    "done": ("#00f2fe", "#4facfe"),
    "error": ("#ff0844", "#ffb199"),
}
STATUS_ALIASES = {
    "untouched": "missing",
    "missing": "missing",
    "working": "ready",
    "partial": "ready",
    "stale": "ready",
    "processing": "ready",
    "running": "ready",
    "ready": "ready",
    "complete": "done",
    "done": "done",
    "error": "error",
    "failed": "error",
}


class StageStatusDot(QWidget):
    """A small colored dot used for compact stage status display."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._status = "missing"
        self.setFixedSize(12, 12)

    def set_status(self, status: str) -> None:
        normalized = str(status or "missing").strip().lower()
        if normalized not in STATUS_COLORS:
            normalized = "missing"
        self._status = normalized
        self.setToolTip(normalized.title())
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(STATUS_COLORS.get(self._status, STATUS_COLORS["missing"])))
        painter.drawEllipse(1, 1, 10, 10)


class StageStatusLine(QWidget):
    """A thin rounded status line with a gradient fill."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._status = "missing"
        self.setFixedHeight(6)
        self.setMinimumWidth(24)
        self.setToolTip("Missing")

    def set_status(self, status: str) -> None:
        normalized = normalize_stage_status(status)
        self._status = normalized
        self.setToolTip(normalized.title())
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect().adjusted(1, 1, -1, -1)
        if rect.width() <= 0 or rect.height() <= 0:
            return

        gradient = QLinearGradient(
            float(rect.left()),
            float(rect.top()),
            float(rect.right()),
            float(rect.top()),
        )
        start_color, end_color = status_gradient_colors(self._status)
        gradient.setColorAt(0.0, QColor(start_color))
        gradient.setColorAt(1.0, QColor(end_color))

        path = QPainterPath()
        radius = min(rect.height() / 2.0, 4.0)
        path.addRoundedRect(QRectF(rect), radius, radius)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255, 30))
        painter.drawPath(path)

        painter.setBrush(gradient)
        painter.drawPath(path)

        outline = QColor(255, 255, 255, 40)
        if self._status == "missing":
            outline = QColor(148, 163, 184, 110)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(outline, 1))
        painter.drawPath(path)


class StatusLabel(QLabel):
    """Compact text label for stage/page status."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Unknown", parent)
        self.setProperty("role", "muted")

    def set_status_text(self, text: str) -> None:
        self.setText(str(text or "Unknown"))

def normalize_stage_status(status: str) -> str:
    normalized = str(status or "missing").strip().lower()
    return STATUS_ALIASES.get(normalized, "missing")


def status_gradient_colors(status: str) -> tuple[str, str]:
    normalized = normalize_stage_status(status)
    return STATUS_LINE_GRADIENTS.get(normalized, STATUS_LINE_GRADIENTS["missing"])


__all__ = [
    "STATUS_COLORS",
    "STATUS_LINE_GRADIENTS",
    "STATUS_ALIASES",
    "StageStatusDot",
    "StageStatusLine",
    "StatusLabel",
    "normalize_stage_status",
    "status_gradient_colors",
]
