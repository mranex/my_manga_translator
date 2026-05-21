"""Scrollable page filmstrip with thumbnails, status lines, and drag reorder."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QMargins, QByteArray, QMimeData, QPoint, QRect, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QDrag,
    QFont,
    QImageReader,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QListView,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)

from .stage_status import normalize_stage_status, status_gradient_colors

PAGE_ROLE = int(Qt.ItemDataRole.UserRole) + 1
STATUS_ROLE = int(Qt.ItemDataRole.UserRole) + 2
PIXMAP_ROLE = int(Qt.ItemDataRole.UserRole) + 3
DEFAULT_THUMBNAIL_SIZE = QSize(144, 177)
DEFAULT_ITEM_SIZE = QSize(176, 224)
THUMBNAIL_BATCH_SIZE = 10
FILMSTRIP_DRAG_MIME = "application/x-manga-translator-filmstrip-row"


class _FilmstripDelegate(QStyledItemDelegate):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._index_font = QFont()
        self._index_font.setPointSize(9)
        self._index_font.setBold(True)
        self.thumbnail_size = QSize(DEFAULT_THUMBNAIL_SIZE)
        self.item_size = QSize(DEFAULT_ITEM_SIZE)

    def set_metrics(self, thumbnail_size: QSize, item_size: QSize) -> None:
        self.thumbnail_size = QSize(thumbnail_size)
        self.item_size = QSize(item_size)

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:  # type: ignore[override]
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        is_selected = bool(option.state & QStyle.StateFlag.State_Selected)
        is_hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)
        lift = 5 if is_selected else (2 if is_hovered else 0)

        rect = option.rect.adjusted(8, 12 - lift, -8, -10 - lift)
        if rect.width() <= 0 or rect.height() <= 0:
            painter.restore()
            return

        is_dark = option.palette.window().color().lightness() < 128

        shadow_rect = QRectF(rect.adjusted(4, 7, -4, 0))
        shadow_path = QPainterPath()
        shadow_path.addRect(shadow_rect)
        shadow_color = QColor(0, 0, 0, 150 if is_selected else (100 if is_hovered else 60))
        if not is_dark:
            shadow_color = QColor(51, 65, 85, 28 if is_selected else (20 if is_hovered else 12))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(shadow_color)
        painter.drawPath(shadow_path)

        border_color = QColor("#9d4edd" if is_selected else ("#2c2c36" if is_dark else "#d8e0eb"))
        if not is_dark and is_selected:
            border_color = QColor("#3b82f6")

        frame_gradient = QLinearGradient(
            float(rect.left()),
            float(rect.top()),
            float(rect.right()),
            float(rect.bottom()),
        )
        if is_dark:
            frame_gradient.setColorAt(0.0, QColor("#1f1f2e" if is_selected else "#111116"))
            frame_gradient.setColorAt(0.55, QColor("#181824" if is_selected else "#0a0a0c"))
            frame_gradient.setColorAt(1.0, QColor("#050508"))
        else:
            frame_gradient.setColorAt(0.0, QColor("#ffffff"))
            frame_gradient.setColorAt(0.55, QColor("#f7fbff" if is_selected else "#f8fbff"))
            frame_gradient.setColorAt(1.0, QColor("#edf4ff" if is_selected else "#f4f8ff"))

        card_path = QPainterPath()
        card_path.addRect(QRectF(rect))
        painter.setPen(QPen(border_color, 2 if is_selected else 1))
        painter.setBrush(frame_gradient)
        painter.drawPath(card_path)

        if is_selected:
            inner_rect = QRectF(rect.adjusted(3, 3, -3, -3))
            inner_path = QPainterPath()
            inner_path.addRect(inner_rect)
            glow_color = QColor("#9d4edd" if is_dark else "#93c5fd")
            glow_color.setAlpha(60 if is_dark else 48)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(glow_color, 1))
            painter.drawPath(inner_path)

        thumb_rect = QRect(
            rect.left() + 10,
            rect.top() + 10,
            rect.width() - 20,
            rect.height() - 36,
        )
        thumb_path = QPainterPath()
        thumb_path.addRect(QRectF(thumb_rect))
        thumb_fill = QColor("#050508") if is_dark else QColor("#f8fafc")
        painter.setPen(QPen(QColor(255, 255, 255, 20) if is_dark else QColor(15, 23, 42, 18), 1))
        painter.setBrush(thumb_fill)
        painter.drawPath(thumb_path)

        pixmap = index.data(PIXMAP_ROLE)
        if isinstance(pixmap, QPixmap) and not pixmap.isNull():
            scaled = pixmap.scaled(
                thumb_rect.size() - QSize(8, 8),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            draw_x = thumb_rect.left() + (thumb_rect.width() - scaled.width()) // 2
            draw_y = thumb_rect.top() + (thumb_rect.height() - scaled.height()) // 2
            opacity = 1.0 if is_selected else (0.8 if is_hovered else 0.44)
            painter.setOpacity(opacity)
            painter.drawPixmap(draw_x, draw_y, scaled)
            painter.setOpacity(1.0)
        else:
            painter.setPen(QColor("#94a3b8"))
            painter.drawText(thumb_rect, Qt.AlignmentFlag.AlignCenter, "No Preview")

        badge_rect = QRect(thumb_rect.left() + 8, thumb_rect.top() + 8, 26, 20)
        badge_path = QPainterPath()
        badge_path.addRect(QRectF(badge_rect))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(15, 23, 42, 200) if is_dark else QColor(255, 255, 255, 224))
        painter.drawPath(badge_path)
        painter.setFont(self._index_font)
        painter.setPen(QColor("#f8fafc") if is_dark else QColor("#0f172a"))
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, str(index.row() + 1))

        status = normalize_stage_status(index.data(STATUS_ROLE) or "missing")
        line_rect = QRect(rect.left() + 14, thumb_rect.bottom() + 10, rect.width() - 28, 6)
        gradient = QLinearGradient(
            float(line_rect.left()),
            float(line_rect.top()),
            float(line_rect.right()),
            float(line_rect.top()),
        )
        start_color, end_color = status_gradient_colors(status)
        gradient.setColorAt(0.0, QColor(start_color))
        gradient.setColorAt(1.0, QColor(end_color))
        track_path = QPainterPath()
        track_path.addRect(QRectF(line_rect))
        painter.setBrush(QColor(255, 255, 255, 22) if is_dark else QColor(15, 23, 42, 12))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(track_path)
        line_path = QPainterPath()
        line_path.addRect(QRectF(line_rect))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(gradient)
        painter.drawPath(line_path)

        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index) -> QSize:  # type: ignore[override]
        del option, index
        return self.item_size


class _FilmstripListWidget(QListWidget):
    page_selected = pyqtSignal(int)
    page_order_changed = pyqtSignal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("PageFilmstripList")
        self.setViewMode(QListView.ViewMode.IconMode)
        self.setFlow(QListView.Flow.LeftToRight)
        self.setWrapping(False)
        self.setResizeMode(QListView.ResizeMode.Adjust)
        self.setMovement(QListView.Movement.Static)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDragDropOverwriteMode(False)
        self.setDropIndicatorShown(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setMouseTracking(True)
        self.setSpacing(4)
        self._delegate = _FilmstripDelegate(self)
        self.setGridSize(DEFAULT_ITEM_SIZE + QSize(16, 10))
        self.setItemDelegate(self._delegate)
        self._drag_row = -1
        self.currentRowChanged.connect(self._emit_page_selected)

    def set_metrics(self, thumbnail_size: QSize, item_size: QSize) -> None:
        self._delegate.set_metrics(thumbnail_size, item_size)
        self.setGridSize(item_size + QSize(16, 10))

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        hbar = self.horizontalScrollBar()
        pixel_delta = event.pixelDelta()
        angle_delta = event.angleDelta()

        delta = 0
        if not pixel_delta.isNull():
            delta = pixel_delta.x() if pixel_delta.x() else pixel_delta.y()
        elif not angle_delta.isNull():
            delta = angle_delta.x() if angle_delta.x() else angle_delta.y()

        if delta:
            hbar.setValue(hbar.value() - delta)
        event.accept()

    def _emit_page_selected(self, row: int) -> None:
        if row >= 0:
            self.page_selected.emit(row)

    def startDrag(self, supported_actions) -> None:  # type: ignore[override]
        source_row = self.currentRow()
        if source_row < 0:
            return

        self._drag_row = source_row
        item_rect = self.visualRect(self.model().index(source_row, 0))

        mime = QMimeData()
        mime.setData(FILMSTRIP_DRAG_MIME, QByteArray(str(source_row).encode("utf-8")))

        drag = QDrag(self)
        drag.setMimeData(mime)
        if item_rect.isValid():
            drag.setPixmap(self.viewport().grab(item_rect))
            drag.setHotSpot(item_rect.center() - item_rect.topLeft())
        try:
            drag.exec(Qt.DropAction.MoveAction)
        finally:
            self._drag_row = -1

    def dragEnterEvent(self, event) -> None:  # type: ignore[override]
        if self._accept_internal_drag(event):
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:  # type: ignore[override]
        if self._accept_internal_drag(event):
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:  # type: ignore[override]
        if not self._accept_internal_drag(event, accept=False):
            super().dropEvent(event)
            return

        source_row = self._source_row_from_event(event)
        target_row = self._target_row_for_position(event.position().toPoint(), source_row)
        if self.move_page_row(source_row, target_row):
            event.acceptProposedAction()
            self.page_order_changed.emit(self.page_order())
            return
        event.ignore()

    def move_page_row(self, source_row: int, target_row: int) -> bool:
        if source_row < 0 or source_row >= self.count():
            return False

        target_row = max(0, min(target_row, self.count() - 1))
        if source_row == target_row:
            return False

        item = self.takeItem(source_row)
        if item is None:
            return False

        self.insertItem(target_row, item)
        self.setCurrentRow(target_row)
        return True

    def _accept_internal_drag(self, event, *, accept: bool = True) -> bool:
        if self.dragDropMode() == QAbstractItemView.DragDropMode.NoDragDrop:
            return False
        if event.source() is not self:
            return False
        mime_data = event.mimeData()
        if mime_data is None or not mime_data.hasFormat(FILMSTRIP_DRAG_MIME):
            return False
        if accept:
            event.acceptProposedAction()
        return True

    def _source_row_from_event(self, event) -> int:
        mime_data = event.mimeData()
        if mime_data is not None and mime_data.hasFormat(FILMSTRIP_DRAG_MIME):
            try:
                return int(bytes(mime_data.data(FILMSTRIP_DRAG_MIME)).decode("utf-8"))
            except Exception:
                pass
        if self._drag_row >= 0:
            return self._drag_row
        return self.currentRow()

    def _target_row_for_position(self, position: QPoint, source_row: int) -> int:
        if self.count() <= 1:
            return source_row

        index = self.indexAt(position)
        if index.isValid():
            rect = self.visualRect(index)
            target_row = index.row()
            if position.x() >= rect.center().x():
                target_row += 1
            if source_row < target_row:
                target_row -= 1
            return max(0, min(target_row, self.count() - 1))

        first_rect = self.visualRect(self.model().index(0, 0))
        last_rect = self.visualRect(self.model().index(self.count() - 1, 0))
        if position.x() <= first_rect.center().x():
            return 0
        if position.x() >= last_rect.center().x():
            return self.count() - 1
        return source_row

    def page_order(self) -> list[str]:
        ordered: list[str] = []
        for row in range(self.count()):
            item = self.item(row)
            ordered.append(str(item.data(PAGE_ROLE) or ""))
        return ordered


class PageFilmstripWidget(QFrame):
    """Horizontal thumbnail strip with selection and drag reorder support."""

    page_selected = pyqtSignal(int)
    page_order_changed = pyqtSignal(list)
    thumbnail_load_failed = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("PageFilmstrip")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._project_root: Path | None = None
        self._thumbnail_cache: dict[str, tuple[float | None, QPixmap]] = {}
        self._page_rows: dict[str, int] = {}
        self._thumbnail_warning_tokens: set[str] = set()
        self._pending_thumbnail_paths: list[str] = []
        self._pending_thumbnail_set: set[str] = set()
        self._thumbnail_timer = QTimer(self)
        self._thumbnail_timer.setInterval(0)
        self._thumbnail_timer.timeout.connect(self._load_thumbnail_batch)
        self._reorder_enabled = True
        self._thumbnail_size = QSize(DEFAULT_THUMBNAIL_SIZE)
        self._item_size = QSize(DEFAULT_ITEM_SIZE)

        self.setMinimumHeight(96)
        self.setMaximumHeight(16777215)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(0)

        self.list_widget = _FilmstripListWidget(self)
        self.list_widget.set_metrics(self._thumbnail_size, self._item_size)
        self.list_widget.page_selected.connect(self.page_selected.emit)
        self.list_widget.page_order_changed.connect(self._on_page_order_changed)
        layout.addWidget(self.list_widget)
        self._apply_dynamic_metrics(force=True)

    def set_pages(
        self,
        project_root: Path | None,
        page_relative_paths: list[str],
        *,
        selected_index: int | None = None,
        status_map: dict[str, str] | None = None,
    ) -> None:
        self._project_root = project_root.resolve() if project_root is not None else None
        self._page_rows.clear()
        self._thumbnail_timer.stop()
        self._pending_thumbnail_paths.clear()
        self._pending_thumbnail_set.clear()
        self.list_widget.blockSignals(True)
        try:
            self.list_widget.clear()
            for row, page_relative_path in enumerate(page_relative_paths):
                item = QListWidgetItem()
                item.setData(PAGE_ROLE, page_relative_path)
                item_status = normalize_stage_status((status_map or {}).get(page_relative_path, "missing"))
                item.setData(STATUS_ROLE, item_status)
                item.setToolTip(self._thumbnail_tooltip(page_relative_path, item_status))
                item.setSizeHint(self._item_size)
                item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsSelectable
                    | Qt.ItemFlag.ItemIsDragEnabled
                    | Qt.ItemFlag.ItemIsDropEnabled
                )
                item.setData(PIXMAP_ROLE, self._display_thumbnail_for_page(page_relative_path))
                self.list_widget.addItem(item)
                self._page_rows[page_relative_path] = row
        finally:
            self.list_widget.blockSignals(False)

        self.set_reorder_enabled(self._reorder_enabled)

        if not page_relative_paths:
            return

        if selected_index is None or selected_index < 0 or selected_index >= len(page_relative_paths):
            selected_index = 0
        self.set_current_row(selected_index, emit_signal=False)
        self._queue_thumbnail_paths(self._thumbnail_load_order(page_relative_paths, selected_index))

    def set_page_statuses(self, status_map: dict[str, str]) -> None:
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            page_relative_path = str(item.data(PAGE_ROLE) or "")
            status = normalize_stage_status(status_map.get(page_relative_path, "missing"))
            item.setData(STATUS_ROLE, status)
            item.setToolTip(self._thumbnail_tooltip(page_relative_path, status))
        self.list_widget.viewport().update()

    def current_row(self) -> int:
        return self.list_widget.currentRow()

    def set_current_row(self, index: int, *, emit_signal: bool = True) -> None:
        if index < 0 or index >= self.list_widget.count():
            return
        signals_blocked = self.list_widget.blockSignals(not emit_signal)
        try:
            self.list_widget.setCurrentRow(index)
        finally:
            self.list_widget.blockSignals(signals_blocked)
        item = self.list_widget.item(index)
        if item is not None:
            self.list_widget.scrollToItem(item, QAbstractItemView.ScrollHint.PositionAtCenter)

    def horizontal_scroll_value(self) -> int:
        return self.list_widget.horizontalScrollBar().value()

    def set_horizontal_scroll_value(self, value: int) -> None:
        self.list_widget.horizontalScrollBar().setValue(max(0, int(value)))

    def refresh_thumbnails(self) -> None:
        self._thumbnail_timer.stop()
        self._pending_thumbnail_paths.clear()
        self._pending_thumbnail_set.clear()
        page_paths: list[str] = []
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            page_relative_path = str(item.data(PAGE_ROLE) or "")
            page_paths.append(page_relative_path)
            item.setData(PIXMAP_ROLE, self._display_thumbnail_for_page(page_relative_path))
        self._queue_thumbnail_paths(self._thumbnail_load_order(page_paths, self.current_row()))
        self.list_widget.viewport().update()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._apply_dynamic_metrics()

    def set_reorder_enabled(self, enabled: bool) -> None:
        self._reorder_enabled = bool(enabled)
        self.list_widget.setMovement(QListView.Movement.Static)
        self.list_widget.setDragDropMode(
            QAbstractItemView.DragDropMode.DragDrop
            if self._reorder_enabled
            else QAbstractItemView.DragDropMode.NoDragDrop
        )
        self.list_widget.setDragEnabled(self._reorder_enabled)
        self.list_widget.setAcceptDrops(self._reorder_enabled)
        self.list_widget.viewport().setAcceptDrops(self._reorder_enabled)
        tooltip = "Drag thumbnails to reorder project pages." if self._reorder_enabled else (
            "Page reordering is temporarily disabled while workflow tasks are running."
        )
        self.list_widget.setToolTip(tooltip)

    def _on_page_order_changed(self, ordered_paths: list[str]) -> None:
        self._rebuild_page_rows()
        self.page_order_changed.emit(ordered_paths)

    def _rebuild_page_rows(self) -> None:
        self._page_rows.clear()
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            if item is None:
                continue
            page_relative_path = str(item.data(PAGE_ROLE) or "")
            if page_relative_path:
                self._page_rows[page_relative_path] = row

    def _display_thumbnail_for_page(self, page_relative_path: str) -> QPixmap:
        cached = self._thumbnail_from_cache(page_relative_path)
        if cached is not None:
            return cached

        if self._project_root is None:
            return self._placeholder_thumbnail("No Project")

        image_path = self._project_root / page_relative_path
        if not image_path.exists():
            self._emit_thumbnail_warning_once(
                f"missing:{image_path.as_posix()}",
                f"Thumbnail source missing: {image_path}",
            )
            return self._placeholder_thumbnail("Missing")

        return self._placeholder_thumbnail("Loading")

    def _thumbnail_from_cache(self, page_relative_path: str) -> QPixmap | None:
        if self._project_root is None:
            return None

        image_path = self._project_root / page_relative_path
        cache_key = self._thumbnail_cache_key(image_path)
        if not image_path.exists():
            self._thumbnail_cache.pop(cache_key, None)
            return None

        mtime = None
        try:
            mtime = image_path.stat().st_mtime
        except Exception:
            mtime = None

        cached = self._thumbnail_cache.get(cache_key)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        return None

    def _load_thumbnail_for_page(self, page_relative_path: str) -> QPixmap:
        if self._project_root is None:
            return self._placeholder_thumbnail("No Project")

        image_path = self._project_root / page_relative_path
        cache_key = self._thumbnail_cache_key(image_path)
        if not image_path.exists():
            self._thumbnail_cache.pop(cache_key, None)
            self._emit_thumbnail_warning_once(
                f"missing:{cache_key}",
                f"Thumbnail source missing: {image_path}",
            )
            return self._placeholder_thumbnail("Missing")

        try:
            mtime = image_path.stat().st_mtime
        except Exception:
            mtime = None

        cached = self._thumbnail_cache.get(cache_key)
        if cached is not None and cached[0] == mtime:
            return cached[1]

        try:
            reader = QImageReader(str(image_path))
            image_size = reader.size()
            if image_size.isValid() and image_size.width() > 0 and image_size.height() > 0:
                scaled_size = QSize(
                    max(16, int(image_size.width() * (self._thumbnail_size.height() / max(image_size.height(), 1)))),
                    self._thumbnail_size.height(),
                )
                if scaled_size.width() > self._thumbnail_size.width():
                    scaled_size = QSize(
                        self._thumbnail_size.width(),
                        max(16, int(image_size.height() * (self._thumbnail_size.width() / max(image_size.width(), 1)))),
                    )
                reader.setScaledSize(scaled_size)
            image = reader.read()
            if image.isNull():
                raise ValueError(reader.errorString() or "Unknown image decode failure.")
            pixmap = QPixmap.fromImage(image)
        except Exception as exc:
            self._emit_thumbnail_warning_once(
                f"decode:{cache_key}:{mtime}",
                f"Thumbnail load failed for {image_path.name}: {exc}",
            )
            pixmap = self._placeholder_thumbnail("Preview")
        else:
            self._thumbnail_warning_tokens.discard(f"missing:{cache_key}")

        self._thumbnail_cache[cache_key] = (mtime, pixmap)
        return pixmap

    def _thumbnail_load_order(self, page_relative_paths: list[str], selected_index: int | None) -> list[str]:
        if not page_relative_paths:
            return []

        if selected_index is None or selected_index < 0 or selected_index >= len(page_relative_paths):
            return list(page_relative_paths)

        ordered_paths = [page_relative_paths[selected_index]]
        for offset in range(1, len(page_relative_paths)):
            right_index = selected_index + offset
            left_index = selected_index - offset
            if right_index < len(page_relative_paths):
                ordered_paths.append(page_relative_paths[right_index])
            if left_index >= 0:
                ordered_paths.append(page_relative_paths[left_index])
        return ordered_paths

    def _queue_thumbnail_paths(self, page_relative_paths: list[str]) -> None:
        for page_relative_path in page_relative_paths:
            if page_relative_path in self._pending_thumbnail_set:
                continue
            if self._thumbnail_from_cache(page_relative_path) is not None:
                continue
            self._pending_thumbnail_paths.append(page_relative_path)
            self._pending_thumbnail_set.add(page_relative_path)

        if self._pending_thumbnail_paths and not self._thumbnail_timer.isActive():
            self._thumbnail_timer.start()

    def _load_thumbnail_batch(self) -> None:
        if self._project_root is None or not self._pending_thumbnail_paths:
            self._thumbnail_timer.stop()
            return

        updated = False
        batch_count = 0
        while self._pending_thumbnail_paths and batch_count < THUMBNAIL_BATCH_SIZE:
            page_relative_path = self._pending_thumbnail_paths.pop(0)
            self._pending_thumbnail_set.discard(page_relative_path)
            row = self._page_rows.get(page_relative_path)
            if row is None or row < 0 or row >= self.list_widget.count():
                continue

            pixmap = self._load_thumbnail_for_page(page_relative_path)
            item = self.list_widget.item(row)
            if item is None:
                continue
            item.setData(PIXMAP_ROLE, pixmap)
            updated = True
            batch_count += 1

        if updated:
            self.list_widget.viewport().update()

        if not self._pending_thumbnail_paths:
            self._thumbnail_timer.stop()

    def _placeholder_thumbnail(self, text: str) -> QPixmap:
        pixmap = QPixmap(self._thumbnail_size)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = pixmap.rect().marginsRemoved(QMargins(2, 2, 2, 2))
        gradient = QLinearGradient(
            float(rect.left()),
            float(rect.top()),
            float(rect.right()),
            float(rect.bottom()),
        )
        if self.palette().window().color().lightness() < 128:
            gradient.setColorAt(0.0, QColor("#111116"))
            gradient.setColorAt(1.0, QColor("#0a0a0c"))
            text_color = QColor("#e2e8f0")
            outline = QColor(255, 255, 255, 40)
        else:
            gradient.setColorAt(0.0, QColor("#eef4ff"))
            gradient.setColorAt(1.0, QColor("#dbe8ff"))
            text_color = QColor("#334155")
            outline = QColor(15, 23, 42, 40)
        path = QPainterPath()
        path.addRect(QRectF(rect))
        painter.setPen(QPen(outline, 1))
        painter.setBrush(gradient)
        painter.drawPath(path)

        painter.setPen(text_color)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
        painter.end()
        return pixmap

    @staticmethod
    def _thumbnail_tooltip(page_relative_path: str, status: str) -> str:
        normalized_status = normalize_stage_status(status)
        status_label = {
            "missing": "Untouched",
            "ready": "Working",
            "done": "Complete",
            "error": "Error",
        }.get(normalized_status, normalized_status.title())
        return f"{Path(page_relative_path).name}\nStatus: {status_label}"

    def _emit_thumbnail_warning_once(self, token: str, message: str) -> None:
        if token in self._thumbnail_warning_tokens:
            return
        self._thumbnail_warning_tokens.add(token)
        self.thumbnail_load_failed.emit(message)

    def _thumbnail_cache_key(self, image_path: Path) -> str:
        return f"{image_path.as_posix()}::{self._thumbnail_size.width()}x{self._thumbnail_size.height()}"

    def _compute_metrics_for_height(self, height: int) -> tuple[QSize, QSize]:
        margins = self.layout().contentsMargins() if self.layout() is not None else QMargins()
        available_height = max(80, height - margins.top() - margins.bottom())
        item_height = max(92, min(360, available_height - 8))
        item_width = max(84, min(280, int(item_height * 0.78)))
        thumbnail_height = max(48, item_height - 46)
        thumbnail_width = max(48, item_width - 28)
        return QSize(thumbnail_width, thumbnail_height), QSize(item_width, item_height)

    def _apply_dynamic_metrics(self, *, force: bool = False) -> None:
        thumbnail_size, item_size = self._compute_metrics_for_height(self.height())
        if not force and thumbnail_size == self._thumbnail_size and item_size == self._item_size:
            return

        old_thumbnail_size = QSize(self._thumbnail_size)
        self._thumbnail_size = thumbnail_size
        self._item_size = item_size
        self.list_widget.set_metrics(self._thumbnail_size, self._item_size)

        significant_change = (
            abs(old_thumbnail_size.width() - self._thumbnail_size.width()) >= 16
            or abs(old_thumbnail_size.height() - self._thumbnail_size.height()) >= 16
        )

        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            if item is not None:
                item.setSizeHint(self._item_size)

        if significant_change:
            self._thumbnail_cache.clear()
            self.refresh_thumbnails()
        else:
            self.list_widget.viewport().update()


__all__ = ["PageFilmstripWidget"]
