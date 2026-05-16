"""Compact studio toolbar for page navigation, zoom, preview mode, and log access."""

from __future__ import annotations

from collections.abc import Sequence

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPainter
from PyQt6.QtWidgets import QCheckBox, QComboBox, QFrame, QLabel, QSizePolicy, QStyle, QToolButton, QVBoxLayout, QWidget

def get_colorized_icon(style: QStyle, icon_enum: QStyle.StandardPixmap, color_hex: str = "#e5e7eb") -> QIcon:
    icon = style.standardIcon(icon_enum)
    pixmap = icon.pixmap(24, 24)
    if pixmap.isNull():
        return icon
    painter = QPainter(pixmap)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    painter.fillRect(pixmap.rect(), QColor(color_hex))
    painter.end()
    return QIcon(pixmap)

MODE_LABELS = {
    "Source": "Src",
    "Detection Overlay": "Detect",
    "Mask Overlay": "Mask",
    "Inpaint Result": "Inp",
    "Render Result": "Rnd",
}


class LeftToolBar(QFrame):
    """Narrow vertical toolbar that replaces the old workflow/page sidebar."""

    preview_mode_changed = pyqtSignal(str)
    fit_requested = pyqtSignal()
    zoom_in_requested = pyqtSignal()
    zoom_out_requested = pyqtSignal()
    reset_zoom_requested = pyqtSignal()
    auto_preview_changed = pyqtSignal(bool)
    follow_batch_progress_changed = pyqtSignal(bool)
    previous_page_requested = pyqtSignal()
    next_page_requested = pyqtSignal()
    first_page_requested = pyqtSignal()
    last_page_requested = pyqtSignal()
    developer_log_toggled = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("LeftToolBar")
        self.setMinimumWidth(100)
        self.setMaximumWidth(130)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title_label = QLabel("Review", self)
        title_label.setProperty("role", "sectionTitle")
        layout.addWidget(title_label)

        self.first_page_button = self._build_button(
            "",
            "Jump to the first page.",
            self.first_page_requested.emit,
            icon_kind=QStyle.StandardPixmap.SP_MediaSkipBackward,
            icon_only=True,
        )
        self.previous_page_button = self._build_button(
            "",
            "Go to the previous page.",
            self.previous_page_requested.emit,
            icon_kind=QStyle.StandardPixmap.SP_MediaSeekBackward,
            icon_only=True,
        )
        self.next_page_button = self._build_button(
            "",
            "Go to the next page.",
            self.next_page_requested.emit,
            icon_kind=QStyle.StandardPixmap.SP_MediaSeekForward,
            icon_only=True,
        )
        self.last_page_button = self._build_button(
            "",
            "Jump to the last page.",
            self.last_page_requested.emit,
            icon_kind=QStyle.StandardPixmap.SP_MediaSkipForward,
            icon_only=True,
        )

        for button in (
            self.first_page_button,
            self.previous_page_button,
            self.next_page_button,
            self.last_page_button,
        ):
            layout.addWidget(button)

        layout.addSpacing(4)

        mode_label = QLabel("Mode", self)
        mode_label.setProperty("role", "muted")
        layout.addWidget(mode_label)

        self.mode_combo = QComboBox(self)
        self.mode_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContentsOnFirstShow)
        self.mode_combo.setMinimumContentsLength(4)
        self.mode_combo.currentIndexChanged.connect(self._emit_current_mode_changed)
        self.mode_combo.setToolTip("Choose which preview layer to show.")
        layout.addWidget(self.mode_combo)

        self.auto_preview_checkbox = QCheckBox("Auto", self)
        self.auto_preview_checkbox.setToolTip("Automatically switch to the newest useful preview result.")
        self.auto_preview_checkbox.setChecked(True)
        self.auto_preview_checkbox.toggled.connect(self.auto_preview_changed.emit)
        layout.addWidget(self.auto_preview_checkbox)

        self.follow_batch_checkbox = QCheckBox("Follow", self)
        self.follow_batch_checkbox.setToolTip("Follow the page currently being processed during batch work.")
        self.follow_batch_checkbox.setChecked(False)
        self.follow_batch_checkbox.toggled.connect(self.follow_batch_progress_changed.emit)
        layout.addWidget(self.follow_batch_checkbox)

        layout.addSpacing(4)

        self.fit_button = self._build_button("Fit", "Fit the preview to the available view.", self.fit_requested.emit)
        self.reset_button = self._build_button("1:1", "Reset preview zoom to 100%.", self.reset_zoom_requested.emit)
        self.zoom_out_button = self._build_button("-", "Zoom out.", self.zoom_out_requested.emit)
        self.zoom_in_button = self._build_button("+", "Zoom in.", self.zoom_in_requested.emit)

        for button in (
            self.fit_button,
            self.reset_button,
            self.zoom_out_button,
            self.zoom_in_button,
        ):
            layout.addWidget(button)

        layout.addStretch(1)

        self.developer_log_button = self._build_button(
            "Logs",
            "Show or hide the developer log.",
            self.developer_log_toggled.emit,
            checkable=True,
            icon_kind=QStyle.StandardPixmap.SP_FileDialogDetailedView,
        )
        layout.addWidget(self.developer_log_button)

    def _build_button(
        self,
        text: str,
        tooltip: str,
        callback,
        *,
        checkable: bool = False,
        icon_kind: QStyle.StandardPixmap | None = None,
        icon_only: bool = False,
    ) -> QToolButton:
        button = QToolButton(self)
        button.setText(text)
        button.setToolTip(tooltip)
        button.setCheckable(checkable)
        button.setProperty("leftToolButton", True)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setMinimumHeight(30)
        button.setMaximumHeight(34)
        button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        if icon_kind is not None:
            colorized = get_colorized_icon(self.style(), icon_kind)
            button.setIcon(colorized)
            if icon_only:
                button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
            else:
                button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        else:
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        button.clicked.connect(lambda _checked=False: callback())
        return button

    def set_modes(self, modes: Sequence[str], current_mode: str | None = None) -> None:
        unique_modes = [str(mode) for mode in modes if str(mode).strip()]
        if not unique_modes:
            unique_modes = ["Source"]

        preferred_mode = str(current_mode or "").strip()
        if preferred_mode and preferred_mode not in unique_modes:
            preferred_mode = ""

        if not preferred_mode:
            current_text = self.mode_combo.currentText().strip()
            if current_text in unique_modes:
                preferred_mode = current_text

        if not preferred_mode:
            preferred_mode = unique_modes[0]

        self.mode_combo.blockSignals(True)
        self.mode_combo.clear()
        for mode in unique_modes:
            display_label = MODE_LABELS.get(mode, mode)
            self.mode_combo.addItem(display_label, mode)
            combo_index = self.mode_combo.count() - 1
            self.mode_combo.setItemData(combo_index, mode, Qt.ItemDataRole.ToolTipRole)
        current_index = max(0, self.mode_combo.findData(preferred_mode))
        self.mode_combo.setCurrentIndex(current_index)
        self.mode_combo.blockSignals(False)

    def current_mode(self) -> str:
        data = self.mode_combo.currentData()
        if isinstance(data, str) and data.strip():
            return data.strip()
        return self.mode_combo.currentText().strip() or "Source"

    def set_current_mode(self, mode: str) -> None:
        normalized = str(mode or "").strip()
        if normalized:
            index = self.mode_combo.findData(normalized)
            if index >= 0:
                self.mode_combo.setCurrentIndex(index)

    def auto_preview_enabled(self) -> bool:
        return self.auto_preview_checkbox.isChecked()

    def set_auto_preview_enabled(self, enabled: bool) -> None:
        self.auto_preview_checkbox.setChecked(bool(enabled))

    def follow_batch_progress_enabled(self) -> bool:
        return self.follow_batch_checkbox.isChecked()

    def set_follow_batch_progress_enabled(self, enabled: bool) -> None:
        self.follow_batch_checkbox.setChecked(bool(enabled))

    def set_log_button_checked(self, visible: bool) -> None:
        self.developer_log_button.blockSignals(True)
        self.developer_log_button.setChecked(bool(visible))
        self.developer_log_button.blockSignals(False)

    def set_log_alert(self, has_alert: bool, *, error: bool = False) -> None:
        self.developer_log_button.setProperty("hasAlert", bool(has_alert))
        self.developer_log_button.setProperty("hasErrorAlert", bool(has_alert and error))
        self.developer_log_button.style().unpolish(self.developer_log_button)
        self.developer_log_button.style().polish(self.developer_log_button)

    def _emit_current_mode_changed(self, index: int) -> None:
        if index < 0:
            return
        self.preview_mode_changed.emit(self.current_mode())


__all__ = ["LeftToolBar"]
