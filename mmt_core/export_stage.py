"""Final export stage for packaging cached results into user output folders."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import shutil
from pathlib import Path
from typing import Any, Protocol
import zipfile

from .export_io import (
    EXPORT_SCHEMA_VERSION,
    build_export_basename,
    ensure_export_dir,
    ensure_unique_path,
    export_manifest_path,
    export_zip_path,
    sanitize_windows_filename,
    save_export_manifest,
    timestamp,
)
from .export_models import DEFAULT_NAMING_PATTERN, ExportConfig
from .image_io import ensure_path
from .inpaint_io import inpaint_image_path
from .render_io import render_image_path


class ProjectLike(Protocol):
    root_dir: Path
    data: Any
    page_count: int

    def page_relative_path_for_index(self, index: int) -> str | None: ...
    def page_for_source_path(self, image_relative_path: str) -> Any: ...
    def image_path_for_index(self, index: int) -> Path | None: ...


def resolve_export_pages(
    project: ProjectLike,
    current_page: str | None,
    page_scope: str,
    selected_pages: Sequence[str] | None = None,
) -> list[str]:
    if project.page_count <= 0:
        return []

    normalized_scope = str(page_scope or "current").strip().lower()
    if normalized_scope == "all":
        return [
            relative_path
            for relative_path in (
                project.page_relative_path_for_index(index)
                for index in range(project.page_count)
            )
            if relative_path is not None
        ]

    if normalized_scope == "selected":
        candidate_pages = [
            str(item)
            for item in (selected_pages or [])
            if isinstance(item, str) and item.strip()
        ]
        seen: set[str] = set()
        resolved_pages: list[str] = []
        for candidate_page in candidate_pages:
            page = project.page_for_source_path(candidate_page)
            if page is None:
                continue
            source_path = str(page.source_path)
            if source_path in seen:
                continue
            seen.add(source_path)
            resolved_pages.append(source_path)
        return resolved_pages

    if current_page:
        page = project.page_for_source_path(current_page)
        if page is not None:
            return [str(page.source_path)]
    return []


def run_export(
    project: ProjectLike,
    *,
    current_page: str | None = None,
    selected_pages: Sequence[str] | None = None,
    config: ExportConfig | dict[str, Any] | None,
    logger: Callable[[str], None] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if project is None:
        raise ValueError("Create or open a project before exporting pages.")
    if getattr(project, "page_count", 0) <= 0:
        raise ValueError("Import images before exporting pages.")

    export_config = ExportConfig.from_value(config)
    if not str(export_config.output_dir or "").strip():
        raise ValueError("Choose an output folder before exporting.")

    output_dir = ensure_export_dir(export_config.output_dir)
    page_list = resolve_export_pages(
        project,
        current_page=current_page,
        page_scope=export_config.page_scope,
        selected_pages=selected_pages or export_config.selected_image_relative_paths,
    )
    if not page_list:
        if export_config.page_scope == "current":
            raise ValueError("Select a page before exporting the current page.")
        if export_config.page_scope == "selected":
            raise ValueError("No selected pages are available for export.")
        raise ValueError("No project pages are available for export.")

    project_name = str(getattr(getattr(project, "data", None), "name", "") or "project")
    created_at = timestamp()
    items: list[dict[str, Any]] = []
    exported_files: list[Path] = []
    invalid_pattern_logged = False
    total_pages = len(page_list)

    _emit_message(logger, f"Exporting {total_pages} page(s) to {output_dir}")
    _emit_progress(progress_callback, event="export_start", page_total=total_pages, progress=0)

    for export_index, image_relative_path in enumerate(page_list, start=1):
        page_number = _page_number_for_image(project, image_relative_path)
        source_path, stage_name, missing_message = _resolve_source_path(
            project,
            image_relative_path,
            export_source=export_config.export_source,
        )

        item_payload: dict[str, Any] = {
            "image_relative_path": str(image_relative_path),
            "source_path": str(source_path) if source_path is not None else "",
            "output_path": "",
            "status": "skipped",
            "error": "",
            "width": 0,
            "height": 0,
        }

        _emit_message(
            logger,
            f"[{export_index}/{total_pages}] Exporting {Path(image_relative_path).name} from {stage_name}",
        )

        if source_path is None or not source_path.exists():
            item_payload["status"] = "skipped"
            item_payload["error"] = missing_message
            items.append(item_payload)
            _emit_message(logger, f"Skipped {Path(image_relative_path).name}: {missing_message}")
            _emit_progress(
                progress_callback,
                event="page_done",
                image_relative_path=image_relative_path,
                page_index=export_index,
                page_total=total_pages,
                progress=int((export_index / total_pages) * 100),
                status="skipped",
            )
            continue

        output_suffix = _output_suffix_for_path(source_path, export_config.output_format)
        base_name, used_fallback = build_export_basename(
            export_config.naming_pattern,
            index=export_index,
            page_number=page_number,
            stem=source_path.stem,
            source_stem=Path(image_relative_path).stem,
            project_name=project_name,
            stage_name=stage_name,
        )
        if used_fallback and not invalid_pattern_logged:
            _emit_message(
                logger,
                f"Invalid naming pattern '{export_config.naming_pattern}'. Falling back to {DEFAULT_NAMING_PATTERN}.",
            )
            invalid_pattern_logged = True

        target_path = ensure_unique_path(output_dir / f"{base_name}{output_suffix}", export_config.overwrite)
        try:
            width, height = _export_one_file(
                source_path,
                target_path,
                export_format=export_config.output_format,
                quality=export_config.quality,
            )
        except Exception as exc:
            item_payload["status"] = "error"
            item_payload["error"] = str(exc)
            items.append(item_payload)
            _emit_message(logger, f"Export failed for {Path(image_relative_path).name}: {exc}")
            _emit_progress(
                progress_callback,
                event="page_done",
                image_relative_path=image_relative_path,
                page_index=export_index,
                page_total=total_pages,
                progress=int((export_index / total_pages) * 100),
                status="error",
            )
            continue

        item_payload["source_path"] = str(source_path)
        item_payload["output_path"] = str(target_path)
        item_payload["status"] = "exported"
        item_payload["width"] = width
        item_payload["height"] = height
        items.append(item_payload)
        exported_files.append(target_path)
        _emit_message(logger, f"Exported {Path(image_relative_path).name} -> {target_path.name}")
        _emit_progress(
            progress_callback,
            event="page_done",
            image_relative_path=image_relative_path,
            output_path=str(target_path),
            page_index=export_index,
            page_total=total_pages,
            progress=int((export_index / total_pages) * 100),
            status="exported",
        )

    exported_count = len([item for item in items if item["status"] == "exported"])
    skipped_count = len([item for item in items if item["status"] == "skipped"])
    error_count = len([item for item in items if item["status"] == "error"])

    planned_zip_path: Path | None = None
    if export_config.create_zip:
        default_zip_name = f"{sanitize_windows_filename(project_name, 'project')}_export.zip"
        planned_zip_path = export_zip_path(
            output_dir,
            export_config.zip_name.strip() or default_zip_name,
            overwrite=export_config.overwrite,
        )

    manifest: dict[str, Any] = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "stage": "export",
        "project_name": project_name,
        "exported_at": created_at,
        "export_source": export_config.export_source,
        "output_format": export_config.output_format,
        "quality": int(export_config.quality),
        "create_zip": bool(export_config.create_zip),
        "zip_path": "",
        "manifest_path": "",
        "output_dir": str(output_dir),
        "total_pages": total_pages,
        "exported_count": exported_count,
        "skipped_count": skipped_count,
        "error_count": error_count,
        "items": items,
    }

    manifest_path: Path | None = None
    manifest_error = ""
    if export_config.include_manifest:
        try:
            manifest_path = save_export_manifest(
                export_manifest_path(output_dir, overwrite=export_config.overwrite),
                manifest,
            )
            manifest["manifest_path"] = str(manifest_path)
        except Exception as exc:
            manifest_error = str(exc)
            _emit_message(logger, f"Failed to write export manifest: {exc}")

    zip_error = ""
    if export_config.create_zip and exported_files:
        try:
            assert planned_zip_path is not None
            _create_zip_archive(planned_zip_path, exported_files, manifest_path)
            manifest["zip_path"] = str(planned_zip_path)
            _emit_message(logger, f"Created ZIP archive: {planned_zip_path}")
        except Exception as exc:
            zip_error = str(exc)
            manifest["zip_path"] = ""
            _emit_message(logger, f"Failed to create ZIP archive: {exc}")

    if manifest_path is not None:
        try:
            save_export_manifest(manifest_path, manifest)
        except Exception as exc:
            if not manifest_error:
                manifest_error = str(exc)
                _emit_message(logger, f"Failed to update export manifest: {exc}")

    if manifest_error:
        manifest["manifest_error"] = manifest_error
    if zip_error:
        manifest["zip_error"] = zip_error

    _emit_progress(progress_callback, event="export_done", page_total=total_pages, progress=100)
    _emit_message(
        logger,
        f"Export finished. Exported: {exported_count}, skipped: {skipped_count}, errors: {error_count}",
    )
    return manifest


def _resolve_source_path(
    project: ProjectLike,
    image_relative_path: str,
    *,
    export_source: str,
) -> tuple[Path | None, str, str]:
    stage_name = str(export_source or "render").strip().lower()
    relative_path = Path(image_relative_path)
    if stage_name == "source":
        source_path = ensure_path(project.root_dir / relative_path)
        if source_path.exists():
            return source_path, "source", ""
        return None, "source", "Source image missing. Choose a different export source or re-import the page."

    if stage_name == "inpaint":
        source_path = inpaint_image_path(project, relative_path)
        if source_path.exists():
            return source_path, "inpaint", ""
        return None, "inpaint", "Inpaint result missing. Run Inpaint first or choose a different export source."

    source_path = render_image_path(project, relative_path)
    if source_path.exists():
        return source_path, "render", ""
    return None, "render", "Render result missing. Run Render first or choose a different export source."


def _output_suffix_for_path(source_path: Path, export_format: str) -> str:
    normalized_format = str(export_format or "original").strip().lower()
    if normalized_format == "original":
        return source_path.suffix or ".png"
    if normalized_format in {"jpg", "jpeg"}:
        return ".jpg" if normalized_format == "jpg" else ".jpeg"
    if normalized_format in {"png", "webp"}:
        return f".{normalized_format}"
    raise ValueError(f"Unsupported output format: {export_format}")


def _export_one_file(
    source_path: Path,
    target_path: Path,
    *,
    export_format: str,
    quality: int,
) -> tuple[int, int]:
    normalized_format = str(export_format or "original").strip().lower()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if normalized_format == "original":
        width, height = _open_image_dimensions(source_path)
        shutil.copy2(source_path, target_path)
        return width, height

    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow is required to convert exported images.") from exc

    with Image.open(source_path) as image:
        width, height = image.size
        image.load()
        export_image = image
        if normalized_format in {"jpg", "jpeg"}:
            if image.mode in {"RGBA", "LA"}:
                background = Image.new("RGB", image.size, (255, 255, 255))
                alpha_channel = image.getchannel("A")
                background.paste(image.convert("RGBA"), mask=alpha_channel)
                export_image = background
            else:
                export_image = image.convert("RGB")
            export_image.save(target_path, format="JPEG", quality=int(quality), optimize=True)
        elif normalized_format == "png":
            if image.mode not in {"RGB", "RGBA", "L", "LA", "P"}:
                export_image = image.convert("RGBA")
            export_image.save(target_path, format="PNG")
        elif normalized_format == "webp":
            export_image.save(target_path, format="WEBP", quality=int(quality))
        else:
            raise ValueError(f"Unsupported output format: {export_format}")
    return int(width), int(height)


def _open_image_dimensions(source_path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow is required to inspect exported images.") from exc
    with Image.open(source_path) as image:
        return int(image.size[0]), int(image.size[1])


def _page_number_for_image(project: ProjectLike, image_relative_path: str) -> int:
    for index in range(project.page_count):
        candidate = project.page_relative_path_for_index(index)
        if candidate == image_relative_path:
            return index + 1
    return 1


def _create_zip_archive(zip_path: Path, files: Sequence[Path], manifest_path: Path | None) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in files:
            archive.write(file_path, arcname=file_path.name)
        if manifest_path is not None and manifest_path.exists():
            archive.write(manifest_path, arcname=manifest_path.name)


def _emit_message(logger: Callable[[str], None] | None, message: str) -> None:
    if logger is not None:
        logger(str(message))


def _emit_progress(progress_callback: Callable[[dict[str, Any]], None] | None, **payload: Any) -> None:
    if progress_callback is not None:
        progress_callback(dict(payload))


__all__ = [
    "resolve_export_pages",
    "run_export",
]
