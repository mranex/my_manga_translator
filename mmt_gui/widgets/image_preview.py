"""Image preview widget with fit-to-window scaling, overlays, and box editing."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QMouseEvent, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)

from .editable_box_overlay import (
    EditableBoxItem,
    make_manual_box,
    next_box_id,
    next_reading_order,
    normalize_box_payload,
    normalize_category,
    sanitize_bbox,
)


class ImagePreviewWidget(QGraphicsView):
    """Displays a page image and optional detection overlays."""

    editable_box_selection_changed = pyqtSignal(object)
    editable_box_changed = pyqtSignal(object)
    editable_box_dirty_changed = pyqtSignal(bool)
    editable_box_error = pyqtSignal(str)

    def __init__(self, parent: QGraphicsView | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self._pixmap_item = QGraphicsPixmapItem()
        self._mask_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self._scene.addItem(self._mask_item)
        self._overlay_items: list[QGraphicsRectItem] = []
        self._editable_box_items: list[EditableBoxItem] = []
        self._editable_boxes: list[dict[str, Any]] = []
        self._editable_original_boxes: list[dict[str, Any]] = []
        self._editable_mode_enabled = False
        self._create_box_mode_enabled = False
        self._show_excluded_boxes = False
        self._editable_category_filter = "bubble"
        self._selected_box_key: tuple[str, int] | None = None
        self._edit_dirty = False
        self._drag_state: dict[str, Any] | None = None
        self._create_preview_item: QGraphicsRectItem | None = None
        self.setScene(self._scene)

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setBackgroundBrush(QBrush(QColor("#0a0a0c")))
        self._auto_fit_enabled = True

    def clear_image(self) -> None:
        self.clear_overlays()
        self.clear_mask_overlay()
        self.clear_editable_boxes()
        self._pixmap_item.setPixmap(QPixmap())
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self.resetTransform()
        self._auto_fit_enabled = True

    def clear_overlays(self) -> None:
        for item in self._overlay_items:
            self._scene.removeItem(item)
        self._overlay_items.clear()

    def clear_mask_overlay(self) -> None:
        self._mask_item.setPixmap(QPixmap())
        self._mask_item.setOpacity(0.0)
        self._mask_item.setVisible(False)

    def set_image(self, image_path: Path | str | None) -> bool:
        if image_path is None:
            self.clear_image()
            return False

        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self.clear_image()
            return False

        self.clear_overlays()
        self.clear_mask_overlay()
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self._refresh_editable_box_items()
        self.fit_to_view()
        return True

    def set_mask_overlay(self, mask_path: Path | str | None) -> bool:
        if mask_path is None:
            self.clear_mask_overlay()
            return False

        pixmap = QPixmap(str(mask_path))
        if pixmap.isNull():
            self.clear_mask_overlay()
            return False

        self._mask_item.setPixmap(pixmap)
        self._mask_item.setPos(0, 0)
        self._mask_item.setOpacity(0.35)
        self._mask_item.setZValue(5)
        self._mask_item.setVisible(True)
        return True

    def set_detection_overlay(self, detection_data: dict[str, Any] | None, *, show_excluded: bool = False) -> None:
        self.clear_overlays()

        if not detection_data:
            return

        self._add_rectangles(
            detection_data.get("layout_regions", []),
            color=QColor(168, 85, 247),
            line_style=Qt.PenStyle.DashLine,
            label_prefix="Layout",
            show_excluded=show_excluded,
        )
        self._add_rectangles(
            detection_data.get("bubbles", []),
            color=QColor(56, 189, 248),
            line_style=Qt.PenStyle.SolidLine,
            label_prefix="Bubble",
            show_excluded=show_excluded,
        )
        self._add_rectangles(
            detection_data.get("text_regions", []),
            color=QColor(251, 191, 36),
            line_style=Qt.PenStyle.DotLine,
            label_prefix="Text",
            show_excluded=show_excluded,
        )

    def fit_to_view(self) -> None:
        pixmap = self._pixmap_item.pixmap()
        if pixmap.isNull():
            return

        self._auto_fit_enabled = True
        self.resetTransform()
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def zoom_in(self) -> None:
        if self._pixmap_item.pixmap().isNull():
            return
        self._auto_fit_enabled = False
        self.scale(1.2, 1.2)

    def zoom_out(self) -> None:
        if self._pixmap_item.pixmap().isNull():
            return
        self._auto_fit_enabled = False
        self.scale(1 / 1.2, 1 / 1.2)

    def reset_zoom(self) -> None:
        if self._pixmap_item.pixmap().isNull():
            return
        self._auto_fit_enabled = False
        self.resetTransform()

    def auto_fit_enabled(self) -> bool:
        return self._auto_fit_enabled

    def set_box_edit_mode(self, enabled: bool) -> None:
        self._editable_mode_enabled = bool(enabled)
        self.setDragMode(
            QGraphicsView.DragMode.NoDrag if self._editable_mode_enabled else QGraphicsView.DragMode.ScrollHandDrag
        )
        if not self._editable_mode_enabled:
            self._create_box_mode_enabled = False
            self._drag_state = None
            self._remove_create_preview_item()
            self._clear_selected_box()
        self._refresh_editable_box_items()

    def box_edit_mode_enabled(self) -> bool:
        return self._editable_mode_enabled

    def set_create_box_mode(self, enabled: bool) -> None:
        self._create_box_mode_enabled = bool(enabled) and self._editable_mode_enabled
        if not self._create_box_mode_enabled:
            self._drag_state = None
            self._remove_create_preview_item()

    def create_box_mode_enabled(self) -> bool:
        return self._create_box_mode_enabled

    def set_editable_box_category_filter(self, category: str) -> None:
        self._editable_category_filter = normalize_category(category)
        self._clear_selected_box()
        self._refresh_editable_box_items()

    def editable_box_category_filter(self) -> str:
        return self._editable_category_filter

    def set_show_excluded_boxes(self, show_excluded: bool) -> None:
        self._show_excluded_boxes = bool(show_excluded)
        if not self._show_excluded_boxes:
            selected = self.selected_editable_box()
            if selected and bool(selected.get("excluded", False)):
                self._clear_selected_box()
        self._refresh_editable_box_items()

    def show_excluded_boxes(self) -> bool:
        return self._show_excluded_boxes

    def load_editable_boxes(self, boxes: list[dict[str, Any]]) -> None:
        self._editable_original_boxes = [normalize_box_payload(box) for box in deepcopy(boxes)]
        self._editable_boxes = [deepcopy(box) for box in self._editable_original_boxes]
        self._edit_dirty = False
        self._emit_dirty_changed()
        self._clear_selected_box()
        self._refresh_editable_box_items()

    def clear_editable_boxes(self) -> None:
        self._remove_create_preview_item()
        for item in self._editable_box_items:
            self._scene.removeItem(item)
        self._editable_box_items.clear()
        self._editable_boxes = []
        self._editable_original_boxes = []
        self._editable_mode_enabled = False
        self._create_box_mode_enabled = False
        self._drag_state = None
        self._selected_box_key = None
        self._edit_dirty = False
        self._emit_dirty_changed()
        self.editable_box_selection_changed.emit(None)

    def discard_editable_box_changes(self) -> None:
        self._editable_boxes = [deepcopy(box) for box in self._editable_original_boxes]
        self._edit_dirty = False
        self._emit_dirty_changed()
        self._clear_selected_box()
        self._refresh_editable_box_items()

    def mark_editable_boxes_clean(self) -> None:
        self._editable_original_boxes = [deepcopy(box) for box in self._editable_boxes]
        self._edit_dirty = False
        self._emit_dirty_changed()

    def set_editable_boxes_dirty(self, dirty: bool) -> None:
        self._edit_dirty = bool(dirty)
        self._emit_dirty_changed()

    def editable_boxes_snapshot(self) -> list[dict[str, Any]]:
        return deepcopy(self._editable_boxes)

    def has_editable_box_changes(self) -> bool:
        return self._edit_dirty

    def selected_editable_box(self) -> dict[str, Any] | None:
        if self._selected_box_key is None:
            return None
        for box in self._editable_boxes:
            if self._box_key(box) == self._selected_box_key:
                return deepcopy(box)
        return None

    def select_editable_box(self, category: str, item_id: int) -> bool:
        target_key = (normalize_category(category), int(item_id))
        for box in self._editable_boxes:
            if self._box_key(box) != target_key:
                continue
            if bool(box.get("excluded", False)) and not self._show_excluded_boxes:
                return False
            if normalize_category(box.get("category")) != self._editable_category_filter:
                return False
            self._set_selected_box(target_key)
            return True
        return False

    def exclude_selected_box(self, excluded: bool = True) -> bool:
        if self._selected_box_key is None:
            return False
        for box in self._editable_boxes:
            if self._box_key(box) != self._selected_box_key:
                continue
            if bool(box.get("excluded", False)) == bool(excluded):
                return True
            box["excluded"] = bool(excluded)
            self._mark_dirty()
            if box["excluded"] and not self._show_excluded_boxes:
                self._clear_selected_box()
            self._refresh_editable_box_items()
            self.editable_box_changed.emit(deepcopy(box))
            return True
        return False

    def refresh_editable_boxes(self) -> None:
        self._refresh_editable_box_items()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self._auto_fit_enabled:
            self.fit_to_view()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if not self._editable_mode_enabled or self._pixmap_item.pixmap().isNull():
            super().mousePressEvent(event)
            return

        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        scene_pos = self.mapToScene(event.position().toPoint())
        image_pos = self._clamp_point_to_image(scene_pos)
        if image_pos is None:
            self.editable_box_error.emit("Unable to map the click to image coordinates.")
            return

        if self._create_box_mode_enabled:
            self._drag_state = {
                "mode": "create",
                "origin": image_pos,
            }
            self._ensure_create_preview_item()
            self._update_create_preview_rect(image_pos, image_pos)
            event.accept()
            return

        clicked_item = self.itemAt(event.position().toPoint())
        if isinstance(clicked_item, EditableBoxItem):
            self._set_selected_box(clicked_item.box_key)
            handle_size = self._handle_size_scene()
            handle_key = clicked_item.handle_hit(image_pos, handle_size)
            current_bbox = self.selected_editable_box()
            if current_bbox is not None:
                self._drag_state = {
                    "mode": "resize" if handle_key else "move",
                    "handle": handle_key,
                    "origin": image_pos,
                    "bbox": list(current_bbox.get("bbox", [0, 0, 0, 0])),
                }
            event.accept()
            return

        self._clear_selected_box()
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if not self._editable_mode_enabled or self._drag_state is None:
            super().mouseMoveEvent(event)
            return

        scene_pos = self.mapToScene(event.position().toPoint())
        image_pos = self._clamp_point_to_image(scene_pos)
        if image_pos is None:
            return

        mode = str(self._drag_state.get("mode") or "")
        if mode == "create":
            origin = self._drag_state.get("origin", image_pos)
            self._update_create_preview_rect(origin, image_pos)
            event.accept()
            return

        current_box = self.selected_editable_box()
        if current_box is None:
            return

        original_bbox = self._drag_state.get("bbox", current_box.get("bbox", [0, 0, 0, 0]))
        if not isinstance(original_bbox, list) or len(original_bbox) < 4:
            return

        dx = image_pos.x() - self._drag_state["origin"].x()
        dy = image_pos.y() - self._drag_state["origin"].y()

        if mode == "move":
            updated_bbox = self._move_bbox(original_bbox, dx, dy)
        else:
            updated_bbox = self._resize_bbox(
                original_bbox,
                dx,
                dy,
                handle=str(self._drag_state.get("handle") or "se"),
            )

        if updated_bbox is None:
            return
        self._update_selected_bbox(updated_bbox)
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if not self._editable_mode_enabled or self._drag_state is None:
            super().mouseReleaseEvent(event)
            return

        scene_pos = self.mapToScene(event.position().toPoint())
        image_pos = self._clamp_point_to_image(scene_pos)
        drag_state = self._drag_state
        self._drag_state = None

        if image_pos is None:
            self._remove_create_preview_item()
            event.accept()
            return

        if str(drag_state.get("mode") or "") == "create":
            origin = drag_state.get("origin", image_pos)
            bbox = sanitize_bbox(
                [origin.x(), origin.y(), image_pos.x(), image_pos.y()],
                image_width=self._image_width(),
                image_height=self._image_height(),
            )
            self._remove_create_preview_item()
            if bbox is None:
                self.editable_box_error.emit("Create box ignored because it was too small or outside the image.")
                event.accept()
                return
            new_box = make_manual_box(
                self._editable_category_filter,
                bbox,
                next_id=next_box_id(self._editable_boxes, self._editable_category_filter),
                next_reading_order=next_reading_order(self._editable_boxes, self._editable_category_filter),
            )
            self._editable_boxes.append(new_box)
            self._mark_dirty()
            self._refresh_editable_box_items()
            self._set_selected_box(self._box_key(new_box))
            self.editable_box_changed.emit(deepcopy(new_box))
            event.accept()
            return

        selected_box = self.selected_editable_box()
        if selected_box is not None:
            self.editable_box_changed.emit(selected_box)
        event.accept()

    def _add_rectangles(
        self,
        regions: list[dict[str, Any]],
        *,
        color: QColor,
        line_style: Qt.PenStyle,
        label_prefix: str,
        show_excluded: bool,
    ) -> None:
        for region in regions:
            if bool(region.get("excluded", False)) and not show_excluded:
                continue
            bbox = region.get("bbox")
            rect = _rect_from_bbox(bbox)
            if rect is None:
                continue

            pen = QPen(QColor(148, 163, 184) if bool(region.get("excluded", False)) else color)
            pen.setStyle(Qt.PenStyle.DashLine if bool(region.get("excluded", False)) else line_style)
            pen.setWidth(2)
            pen.setCosmetic(True)

            rect_item = QGraphicsRectItem(rect)
            rect_item.setPen(pen)
            rect_item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            rect_item.setZValue(10)
            rect_item.setToolTip(_tooltip_for_region(region, label_prefix))
            self._scene.addItem(rect_item)
            self._overlay_items.append(rect_item)

    def _refresh_editable_box_items(self) -> None:
        for item in self._editable_box_items:
            self._scene.removeItem(item)
        self._editable_box_items.clear()

        if not self._editable_mode_enabled or self._pixmap_item.pixmap().isNull():
            return

        for box in self._editable_boxes:
            if normalize_category(box.get("category")) != self._editable_category_filter:
                continue
            if bool(box.get("excluded", False)) and not self._show_excluded_boxes:
                continue
            bbox = sanitize_bbox(
                box.get("bbox"),
                image_width=self._image_width(),
                image_height=self._image_height(),
            )
            if bbox is None:
                continue
            box["bbox"] = bbox
            item = EditableBoxItem(box)
            item.set_selected_visual(self._box_key(box) == self._selected_box_key)
            self._scene.addItem(item)
            self._editable_box_items.append(item)

        if self._selected_box_key is not None and self.selected_editable_box() is None:
            self._clear_selected_box()

    def _set_selected_box(self, box_key: tuple[str, int] | None) -> None:
        self._selected_box_key = box_key
        for item in self._editable_box_items:
            item.set_selected_visual(item.box_key == box_key)
        self.editable_box_selection_changed.emit(self.selected_editable_box())

    def _clear_selected_box(self) -> None:
        self._set_selected_box(None)

    def _update_selected_bbox(self, bbox: list[int]) -> None:
        if self._selected_box_key is None:
            return
        for index, box in enumerate(self._editable_boxes):
            if self._box_key(box) != self._selected_box_key:
                continue
            if list(box.get("bbox", [])) == list(bbox):
                return
            self._editable_boxes[index]["bbox"] = list(bbox)
            self._mark_dirty()
            self._refresh_editable_box_items()
            self._set_selected_box(self._selected_box_key)
            self.editable_box_changed.emit(deepcopy(self._editable_boxes[index]))
            return

    def _move_bbox(self, bbox: list[int], dx: float, dy: float) -> list[int] | None:
        x1, y1, x2, y2 = [int(value) for value in bbox[:4]]
        width = x2 - x1
        height = y2 - y1
        new_x1 = x1 + int(round(dx))
        new_y1 = y1 + int(round(dy))
        new_x2 = new_x1 + width
        new_y2 = new_y1 + height

        max_width = self._image_width()
        max_height = self._image_height()
        if new_x1 < 0:
            new_x2 -= new_x1
            new_x1 = 0
        if new_y1 < 0:
            new_y2 -= new_y1
            new_y1 = 0
        if new_x2 > max_width:
            delta = new_x2 - max_width
            new_x1 -= delta
            new_x2 = max_width
        if new_y2 > max_height:
            delta = new_y2 - max_height
            new_y1 -= delta
            new_y2 = max_height
        return sanitize_bbox(
            [new_x1, new_y1, new_x2, new_y2],
            image_width=max_width,
            image_height=max_height,
        )

    def _resize_bbox(self, bbox: list[int], dx: float, dy: float, *, handle: str) -> list[int] | None:
        x1, y1, x2, y2 = [int(value) for value in bbox[:4]]
        delta_x = int(round(dx))
        delta_y = int(round(dy))

        if "w" in handle:
            x1 += delta_x
        if "e" in handle:
            x2 += delta_x
        if "n" in handle:
            y1 += delta_y
        if "s" in handle:
            y2 += delta_y

        if x2 - x1 < 4:
            if "w" in handle:
                x1 = x2 - 4
            else:
                x2 = x1 + 4
        if y2 - y1 < 4:
            if "n" in handle:
                y1 = y2 - 4
            else:
                y2 = y1 + 4

        return sanitize_bbox(
            [x1, y1, x2, y2],
            image_width=self._image_width(),
            image_height=self._image_height(),
        )

    def _image_width(self) -> int:
        return int(round(self._pixmap_item.pixmap().width()))

    def _image_height(self) -> int:
        return int(round(self._pixmap_item.pixmap().height()))

    def _handle_size_scene(self) -> float:
        scale = max(abs(self.transform().m11()), 0.001)
        return max(10.0 / scale, 4.0)

    def _clamp_point_to_image(self, point: QPointF) -> QPointF | None:
        if self._pixmap_item.pixmap().isNull():
            return None
        width = self._image_width()
        height = self._image_height()
        if width <= 0 or height <= 0:
            return None
        return QPointF(
            min(max(point.x(), 0.0), float(width)),
            min(max(point.y(), 0.0), float(height)),
        )

    def _ensure_create_preview_item(self) -> None:
        if self._create_preview_item is not None:
            return
        self._create_preview_item = QGraphicsRectItem()
        pen = QPen(QColor(255, 255, 255))
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setWidth(2)
        pen.setCosmetic(True)
        self._create_preview_item.setPen(pen)
        self._create_preview_item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self._create_preview_item.setZValue(40)
        self._scene.addItem(self._create_preview_item)

    def _remove_create_preview_item(self) -> None:
        if self._create_preview_item is None:
            return
        self._scene.removeItem(self._create_preview_item)
        self._create_preview_item = None

    def _update_create_preview_rect(self, start: QPointF, end: QPointF) -> None:
        if self._create_preview_item is None:
            return
        left = min(start.x(), end.x())
        top = min(start.y(), end.y())
        width = abs(end.x() - start.x())
        height = abs(end.y() - start.y())
        self._create_preview_item.setRect(QRectF(left, top, width, height))

    def _mark_dirty(self) -> None:
        if self._edit_dirty:
            return
        self._edit_dirty = True
        self._emit_dirty_changed()

    def _emit_dirty_changed(self) -> None:
        self.editable_box_dirty_changed.emit(self._edit_dirty)

    @staticmethod
    def _box_key(box: dict[str, Any]) -> tuple[str, int]:
        return (normalize_category(box.get("category")), int(box.get("id") or 0))


def _rect_from_bbox(bbox: Any) -> QRectF | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None

    try:
        x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    except (TypeError, ValueError):
        return None

    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return None

    return QRectF(x1, y1, width, height)


def _tooltip_for_region(region: dict[str, Any], label_prefix: str) -> str:
    region_id = region.get("id", "?")
    detector = region.get("detector", "unknown")
    confidence = region.get("confidence")
    flags = []
    if bool(region.get("manual", False)):
        flags.append("manual")
    if bool(region.get("excluded", False)):
        flags.append("excluded")
    flag_text = f"\nFlags: {', '.join(flags)}" if flags else ""
    if confidence is None:
        return f"{label_prefix} #{region_id}\nDetector: {detector}{flag_text}"
    return (
        f"{label_prefix} #{region_id}\nDetector: {detector}\nConfidence: {float(confidence):.3f}{flag_text}"
    )


__all__ = ["ImagePreviewWidget"]
