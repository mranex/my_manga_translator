"""Reusable crop preview block for OCR/translation item editors."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from .image_preview import ImagePreviewWidget


class CropPreviewPanel(QWidget):
    """Small helper widget that shows a crop preview plus details text."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("CropPreviewPanel")
        self._project_root: Path | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        title_label = QLabel("Crop Preview")
        title_label.setProperty("role", "sectionTitle")
        layout.addWidget(title_label)

        self.preview = ImagePreviewWidget()
        self.preview.setMinimumHeight(220)
        layout.addWidget(self.preview)

        self.details_label = QLabel("Select an item to preview its crop.")
        self.details_label.setProperty("role", "muted")
        self.details_label.setWordWrap(True)
        self.details_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.details_label)

    def set_project_root(self, project_root: Path | None) -> None:
        self._project_root = project_root.resolve() if isinstance(project_root, Path) else None

    def clear(self, message: str = "Select an item to preview its crop.") -> None:
        self.preview.clear_image()
        self.details_label.setText(message)

    def set_crop(
        self,
        crop_path: str | Path | None,
        *,
        details: str,
    ) -> None:
        crop_value = str(crop_path or "").strip()
        if not crop_value:
            self.clear(details)
            return

        crop_file = Path(crop_value)
        if not crop_file.is_absolute() and self._project_root is not None:
            crop_file = (self._project_root / crop_file).resolve()

        if not crop_file.exists():
            self.clear(f"{details}\nCrop missing: {crop_file}")
            return

        if not self.preview.set_image(crop_file):
            self.clear(f"{details}\nFailed to load crop: {crop_file}")
            return

        self.details_label.setText(details)


__all__ = ["CropPreviewPanel"]
