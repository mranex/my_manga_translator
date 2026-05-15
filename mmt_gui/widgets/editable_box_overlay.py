"""Editable detection box graphics helpers for the preview canvas."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QGraphicsRectItem

HANDLE_KEYS = ("nw", "n", "ne", "e", "se", "s", "sw", "w")
MIN_BOX_SIZE = 4

CATEGORY_COLORS = {
    "bubble": QColor(56, 189, 248),
    "text_region": QColor(251, 191, 36),
    "layout_region": QColor(168, 85, 247),
    "ocr_bbox": QColor(251, 191, 36),
    "ocr_item": QColor(56, 189, 248),
    "render_bbox": QColor(217, 70, 239),
}
SELECTED_COLOR = QColor(255, 255, 255)
EXCLUDED_COLOR = QColor(148, 163, 184)


class EditableBoxItem(QGraphicsRectItem):
    """Graphics item that renders one editable detection box."""

    def __init__(self, box_data: dict[str, Any]) -> None:
        super().__init__()
        self._box_data = deepcopy(box_data)
        self._selected_visual = False
        self.setZValue(25)
        self.setAcceptHoverEvents(False)
        self.set_box_data(box_data)

    @property
    def box_key(self) -> tuple[str, int]:
        return (str(self._box_data.get("category") or ""), int(self._box_data.get("id") or 0))

    @property
    def box_data(self) -> dict[str, Any]:
        return deepcopy(self._box_data)

    def set_box_data(self, box_data: dict[str, Any]) -> None:
        self._box_data = deepcopy(box_data)
        bbox = self._box_data.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
        self.setRect(QRectF(x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)))
        self.setToolTip(_tooltip_for_box(self._box_data))
        self.update()

    def set_selected_visual(self, selected: bool) -> None:
        self._selected_visual = bool(selected)
        self.update()

    def handle_hit(self, scene_pos: QPointF, handle_size: float) -> str | None:
        if not self._selected_visual:
            return None
        for handle_key, handle_rect in self.handle_rects(handle_size).items():
            if handle_rect.contains(scene_pos):
                return handle_key
        return None

    def handle_rects(self, handle_size: float) -> dict[str, QRectF]:
        rect = self.rect()
        half = handle_size / 2.0
        center_x = rect.center().x()
        center_y = rect.center().y()
        left = rect.left()
        right = rect.right()
        top = rect.top()
        bottom = rect.bottom()
        return {
            "nw": QRectF(left - half, top - half, handle_size, handle_size),
            "n": QRectF(center_x - half, top - half, handle_size, handle_size),
            "ne": QRectF(right - half, top - half, handle_size, handle_size),
            "e": QRectF(right - half, center_y - half, handle_size, handle_size),
            "se": QRectF(right - half, bottom - half, handle_size, handle_size),
            "s": QRectF(center_x - half, bottom - half, handle_size, handle_size),
            "sw": QRectF(left - half, bottom - half, handle_size, handle_size),
            "w": QRectF(left - half, center_y - half, handle_size, handle_size),
        }

    def paint(self, painter: QPainter, option, widget=None) -> None:  # type: ignore[override]
        base_color = _color_for_box(self._box_data)
        pen = QPen(SELECTED_COLOR if self._selected_visual else base_color)
        pen.setCosmetic(True)
        pen.setWidth(3 if self._selected_visual else 2)
        pen.setStyle(Qt.PenStyle.DashLine if bool(self._box_data.get("excluded", False)) else Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(self.rect())

        if not self._selected_visual:
            return

        scale = max(abs(painter.worldTransform().m11()), 0.001)
        handle_size = max(6.0 / scale, 3.0)
        painter.setPen(QPen(base_color))
        painter.setBrush(base_color)
        for handle_rect in self.handle_rects(handle_size).values():
            painter.drawRect(handle_rect)


def normalize_box_payload(box_data: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(box_data)
    normalized["category"] = normalize_category(normalized.get("category"))
    normalized["id"] = safe_int(normalized.get("id"))
    normalized["manual"] = bool(normalized.get("manual", False))
    normalized["excluded"] = bool(normalized.get("excluded", False))
    normalized.setdefault("source", normalized.get("detector") or "unknown")
    return normalized


def normalize_category(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace(" ", "_")
    if normalized == "text":
        return "text_region"
    if normalized == "layout":
        return "layout_region"
    if normalized in {"ocr_crop_box", "ocr_crop", "ocr"}:
        return "ocr_bbox"
    if normalized in {"item_box", "ocr_item_box"}:
        return "ocr_item"
    if normalized in {"render", "render_box"}:
        return "render_bbox"
    if normalized in CATEGORY_COLORS:
        return normalized
    return "bubble"


def sanitize_bbox(bbox: Any, *, image_width: int, image_height: int) -> list[int] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(value))) for value in bbox[:4]]
    except (TypeError, ValueError):
        return None

    x1 = max(0, min(x1, image_width))
    x2 = max(0, min(x2, image_width))
    y1 = max(0, min(y1, image_height))
    y2 = max(0, min(y2, image_height))

    left = min(x1, x2)
    top = min(y1, y2)
    right = max(x1, x2)
    bottom = max(y1, y2)
    if right - left < MIN_BOX_SIZE or bottom - top < MIN_BOX_SIZE:
        return None
    return [left, top, right, bottom]


def make_manual_box(
    category: str,
    bbox: list[int],
    *,
    next_id: int,
    next_reading_order: int | None,
) -> dict[str, Any]:
    normalized_category = normalize_category(category)
    item: dict[str, Any] = {
        "id": int(next_id),
        "category": normalized_category,
        "bbox": list(bbox),
        "confidence": 1.0,
        "detector": "manual",
        "source": "manual",
        "manual": True,
        "excluded": False,
    }
    if normalized_category == "bubble":
        item["mask_path"] = None
    elif normalized_category == "text_region":
        item["bubble_id"] = None
        item["reading_order"] = next_reading_order
        item["source_direction"] = infer_source_direction(bbox)
        item["rotation_deg"] = 0.0
    else:
        item["label"] = "text"
        item["reading_order"] = next_reading_order
        item["label_id"] = None
    return item


def infer_source_direction(bbox: list[int]) -> str:
    width = max(1, int(bbox[2]) - int(bbox[0]))
    height = max(1, int(bbox[3]) - int(bbox[1]))
    return "vertical" if height > int(width * 1.2) else "horizontal"


def next_box_id(boxes: list[dict[str, Any]], category: str) -> int:
    normalized_category = normalize_category(category)
    used_ids = [
        safe_int(box.get("id"))
        for box in boxes
        if normalize_category(box.get("category")) == normalized_category
    ]
    return (max(used_ids) + 1) if used_ids else 0


def next_reading_order(boxes: list[dict[str, Any]], category: str) -> int | None:
    normalized_category = normalize_category(category)
    if normalized_category not in {"text_region", "layout_region"}:
        return None
    orders = [
        safe_int(box.get("reading_order"))
        for box in boxes
        if normalize_category(box.get("category")) == normalized_category
        and box.get("reading_order") is not None
    ]
    return (max(orders) + 1) if orders else 0


def safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _color_for_box(box_data: dict[str, Any]) -> QColor:
    if bool(box_data.get("excluded", False)):
        return QColor(EXCLUDED_COLOR)
    return QColor(CATEGORY_COLORS.get(normalize_category(box_data.get("category")), CATEGORY_COLORS["bubble"]))


def _tooltip_for_box(box_data: dict[str, Any]) -> str:
    category = normalize_category(box_data.get("category"))
    category_label = category.replace("_", " ").title()
    detector = str(box_data.get("detector") or box_data.get("source") or "unknown")
    confidence = box_data.get("confidence")
    status_bits = []
    if bool(box_data.get("manual", False)):
        status_bits.append("manual")
    if bool(box_data.get("excluded", False)):
        status_bits.append("excluded")
    status_text = f"\nFlags: {', '.join(status_bits)}" if status_bits else ""
    if confidence is None:
        return f"{category_label} #{safe_int(box_data.get('id'))}\nSource: {detector}{status_text}"
    return (
        f"{category_label} #{safe_int(box_data.get('id'))}\n"
        f"Source: {detector}\nConfidence: {float(confidence):.3f}{status_text}"
    )


__all__ = [
    "CATEGORY_COLORS",
    "EditableBoxItem",
    "MIN_BOX_SIZE",
    "infer_source_direction",
    "make_manual_box",
    "next_box_id",
    "next_reading_order",
    "normalize_box_payload",
    "normalize_category",
    "sanitize_bbox",
]
