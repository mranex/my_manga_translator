"""Models and normalization helpers for final export configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

EXPORT_SOURCE_CHOICES = ("render", "inpaint", "source")
PAGE_SCOPE_CHOICES = ("current", "all", "selected")
OUTPUT_FORMAT_CHOICES = ("original", "png", "jpg", "jpeg", "webp")
DEFAULT_NAMING_PATTERN = "{index:04d}_{stem}"


def normalize_export_source(value: str) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in EXPORT_SOURCE_CHOICES else "render"


def normalize_page_scope(value: str) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in PAGE_SCOPE_CHOICES else "current"


def normalize_output_format(value: str) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in OUTPUT_FORMAT_CHOICES else "original"


@dataclass(slots=True)
class ExportConfig:
    """Serializable export settings for final output packaging."""

    export_source: str = "render"
    page_scope: str = "current"
    selected_image_relative_paths: list[str] = field(default_factory=list)
    output_dir: str = ""
    output_format: str = "original"
    quality: int = 95
    create_zip: bool = False
    zip_name: str = ""
    naming_pattern: str = DEFAULT_NAMING_PATTERN
    overwrite: bool = False
    open_output_folder: bool = False
    include_manifest: bool = True

    @classmethod
    def from_value(cls, value: "ExportConfig | dict[str, Any] | None") -> "ExportConfig":
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            return cls()
        return cls(
            export_source=normalize_export_source(value.get("export_source", "render")),
            page_scope=normalize_page_scope(value.get("page_scope", "current")),
            selected_image_relative_paths=[
                str(item)
                for item in value.get("selected_image_relative_paths", [])
                if isinstance(item, str) and item.strip()
            ],
            output_dir=str(value.get("output_dir", "") or ""),
            output_format=normalize_output_format(value.get("output_format", "original")),
            quality=_coerce_quality(value.get("quality"), 95),
            create_zip=bool(value.get("create_zip", False)),
            zip_name=str(value.get("zip_name", "") or ""),
            naming_pattern=str(value.get("naming_pattern", DEFAULT_NAMING_PATTERN) or DEFAULT_NAMING_PATTERN),
            overwrite=bool(value.get("overwrite", False)),
            open_output_folder=bool(value.get("open_output_folder", False)),
            include_manifest=bool(value.get("include_manifest", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "export_source": normalize_export_source(self.export_source),
            "page_scope": normalize_page_scope(self.page_scope),
            "selected_image_relative_paths": list(self.selected_image_relative_paths),
            "output_dir": str(self.output_dir or ""),
            "output_format": normalize_output_format(self.output_format),
            "quality": _coerce_quality(self.quality, 95),
            "create_zip": bool(self.create_zip),
            "zip_name": str(self.zip_name or ""),
            "naming_pattern": str(self.naming_pattern or DEFAULT_NAMING_PATTERN),
            "overwrite": bool(self.overwrite),
            "open_output_folder": bool(self.open_output_folder),
            "include_manifest": bool(self.include_manifest),
        }


def _coerce_quality(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        return int(default)
    return max(1, min(100, parsed))


__all__ = [
    "DEFAULT_NAMING_PATTERN",
    "EXPORT_SOURCE_CHOICES",
    "OUTPUT_FORMAT_CHOICES",
    "PAGE_SCOPE_CHOICES",
    "ExportConfig",
    "normalize_export_source",
    "normalize_output_format",
    "normalize_page_scope",
]
