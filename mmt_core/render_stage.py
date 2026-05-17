"""Render-stage preparation and compositing for translated pages."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

from text_rendering import (
    alpha_composite_onto_bgr,
    choose_text_color_for_region,
    find_fallback_font_for_text,
    fit_text_layout,
    get_cached_font,
)

from .canon_state import canon_item_bbox, ensure_canon_state, get_canon_item, resolve_canon_item_for_stage_item
from .detection_io import detection_json_path, load_detection_json, save_detection_json
from .image_io import load_image_bgr, project_relative_path, save_png_image
from .inpaint_io import inpaint_image_path
from .ocr_io import load_ocr_json, ocr_json_path
from .render_io import (
    RENDER_SCHEMA_VERSION,
    load_render_json,
    normalize_render_item,
    render_image_path,
    render_json_path,
    render_sprite_dir,
    save_render_json,
    summarize_render_json,
    timestamp,
)
from .render_layout import (
    bbox_area,
    bbox_to_list,
    choose_render_bbox,
    choose_writing_mode,
    clamp_bbox_to_image,
    iter_vertical_tokens,
)
from .render_models import RenderConfig
from .render_styles import coerce_serializable_color, resolve_font_path
from .translation_io import load_translation_json, translation_json_path

Logger = Callable[[str], None]
ProgressCallback = Callable[[dict[str, Any]], None]


class ProjectLike(Protocol):
    root_dir: Path
    cache_dir: Path


def prepare_render_for_page(
    project: ProjectLike,
    image_relative_path: str | Path,
    *,
    force: bool = False,
    logger: Logger | None = None,
) -> Path:
    """Build render metadata from cached translation/OCR and inpaint outputs."""

    image_relative = str(Path(image_relative_path).as_posix())
    detection_path = detection_json_path(project, image_relative)
    if not detection_path.exists():
        raise FileNotFoundError(
            f"Detection cache is missing for {Path(image_relative).name}. Run Detection first."
        )
    detection_data = load_detection_json(detection_path)
    had_canon_state = isinstance(detection_data.get("canon_state"), dict)
    ensure_canon_state(detection_data)
    if not had_canon_state:
        save_detection_json(detection_path, detection_data)
    canon_state = detection_data["canon_state"]
    active_canon_items = _active_renderable_canon_items(canon_state)
    no_text_page = len(active_canon_items) == 0

    inpaint_path = inpaint_image_path(project, image_relative)
    translation_path = translation_json_path(project, image_relative)
    ocr_path = ocr_json_path(project, image_relative)
    ocr_data = load_ocr_json(ocr_path) if ocr_path.exists() else None
    translation_data: dict[str, Any] = {"source_image": image_relative, "items": []}
    base_image = None

    if no_text_page:
        _log(logger, f"Render no-op preparation for {Path(image_relative).name}: no active text items found.")
        _base_path, base_image = _resolve_render_base_image(project, image_relative)
        if translation_path.exists():
            translation_data = load_translation_json(translation_path)
    else:
        if not inpaint_path.exists():
            raise FileNotFoundError(
                f"Inpaint output is missing for {Path(image_relative).name}. Run Inpaint first."
            )
        if not translation_path.exists():
            raise FileNotFoundError(
                f"Translation cache is missing for {Path(image_relative).name}. Run Translation first."
            )
        translation_data = load_translation_json(translation_path)
        base_image = load_image_bgr(inpaint_path)

    render_items = (
        []
        if no_text_page
        else build_render_items_from_translation(
            translation_data,
            ocr_data,
            canon_state,
            base_image.shape,
        )
    )

    existing_data = None
    output_json_path = render_json_path(project, image_relative)
    if output_json_path.exists():
        try:
            existing_data = load_render_json(output_json_path)
        except Exception:
            existing_data = None

    payload = {
        "schema_version": RENDER_SCHEMA_VERSION,
        "stage": "render",
        "source_image": str(translation_data.get("source_image", "") or image_relative),
        "inpaint_image_path": project_relative_path(project.root_dir, inpaint_path) if inpaint_path.exists() else "",
        "translation_cache_path": project_relative_path(project.root_dir, translation_path) if translation_path.exists() else "",
        "ocr_cache_path": project_relative_path(project.root_dir, ocr_path) if ocr_path.exists() else "",
        "output_image_path": project_relative_path(project.root_dir, render_image_path(project, image_relative)),
        "image_width": int(base_image.shape[1]),
        "image_height": int(base_image.shape[0]),
        "item_count": len(render_items),
        "rendered_item_count": 0,
        "skipped_item_count": len([item for item in render_items if item.get("status") == "skipped"]),
        "no_text_page": no_text_page,
        "status": "pending",
        "error": "",
        "created_at": str((existing_data or {}).get("created_at", "") or timestamp()),
        "updated_at": timestamp(),
        "settings": {
            "font_name": "",
            "font_path": "",
            "font_size_mode": "fit",
            "min_font_size": 12,
            "max_font_size": 72,
            "text_color": None,
            "stroke_enabled": True,
            "stroke_color": None,
            "stroke_width": None,
            "auto_color": True,
            "auto_direction": True,
            "vertical_cjk": True,
            "save_sprites": True,
            "force": bool(force),
        },
        "items": [normalize_render_item(item) for item in render_items],
    }
    save_render_json(output_json_path, payload)
    _log(logger, f"Prepared render metadata for {Path(image_relative).name}: {output_json_path}")
    return output_json_path


def run_render_for_page(
    project: ProjectLike,
    image_relative_path: str | Path,
    *,
    force: bool = False,
    font_name: str | None = None,
    font_path: str | None = None,
    min_font_size: int = 12,
    max_font_size: int = 72,
    stroke_enabled: bool = True,
    stroke_width: float | None = None,
    text_color: tuple[int, int, int] | None = None,
    stroke_color: tuple[int, int, int] | None = None,
    auto_color: bool = True,
    auto_direction: bool = True,
    vertical_cjk: bool = True,
    save_sprites: bool = True,
    logger: Logger | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    """Render translated text onto the cached inpainted page image."""

    image_relative = str(Path(image_relative_path).as_posix())
    output_image = render_image_path(project, image_relative)
    output_json = render_json_path(project, image_relative)
    existing_metadata = _load_existing_render_metadata(output_json)
    if (
        not force
        and output_image.exists()
        and existing_metadata is not None
        and str(existing_metadata.get("status", "") or "").strip().lower() == "done"
        and not bool(existing_metadata.get("needs_render", False))
    ):
        _log(logger, f"Reusing cached render output: {output_image}")
        _emit_progress(
            progress_callback,
            event="page_done",
            image_relative_path=image_relative,
            output_path=str(output_image),
            summary=summarize_render_json(existing_metadata),
            message=f"Reused cached render output for {Path(image_relative).name}",
        )
        return output_image

    metadata = existing_metadata
    metadata_path = output_json
    preserve_existing_metadata = bool(
        metadata is not None
        and (
            bool(metadata.get("edited", False))
            or bool(metadata.get("needs_render", False))
        )
    )
    if metadata is None or not preserve_existing_metadata:
        metadata_path = prepare_render_for_page(project, image_relative, force=force, logger=logger)
        metadata = load_render_json(metadata_path)
    else:
        metadata = load_render_json(metadata_path)
    config = RenderConfig(
        font_name=str(font_name or ""),
        font_path=str(font_path or ""),
        min_font_size=max(1, int(min_font_size)),
        max_font_size=max(max(1, int(min_font_size)), int(max_font_size)),
        stroke_enabled=bool(stroke_enabled),
        stroke_width=stroke_width if stroke_width and stroke_width > 0 else None,
        text_color=text_color,
        stroke_color=stroke_color,
        auto_color=bool(auto_color),
        auto_direction=bool(auto_direction),
        vertical_cjk=bool(vertical_cjk),
        save_sprites=bool(save_sprites),
        force=bool(force),
    )
    resolved_font_path = resolve_font_path(
        project.root_dir,
        font_name=config.font_name,
        font_path=config.font_path,
    )
    config.font_path = resolved_font_path
    metadata["settings"] = config.to_metadata()
    metadata["status"] = "running"
    metadata["error"] = ""
    metadata["updated_at"] = timestamp()
    metadata["needs_render"] = False
    save_render_json(metadata_path, metadata)

    detection_path = detection_json_path(project, image_relative)
    if not detection_path.exists():
        raise FileNotFoundError(
            f"Detection cache is missing for {Path(image_relative).name}. Run Detection first."
        )
    detection_data = load_detection_json(detection_path)
    had_canon_state = isinstance(detection_data.get("canon_state"), dict)
    ensure_canon_state(detection_data)
    if not had_canon_state:
        save_detection_json(detection_path, detection_data)
    canon_state = detection_data["canon_state"]
    active_canon_items = _active_renderable_canon_items(canon_state)
    if not active_canon_items:
        output_path = _finalize_no_text_render_output(
            project,
            image_relative,
            metadata,
            metadata_path,
            logger=logger,
            progress_callback=progress_callback,
        )
        return output_path

    inpaint_path = project.root_dir / str(metadata.get("inpaint_image_path", "") or "")
    if not inpaint_path.exists():
        raise FileNotFoundError(
            f"Inpaint output is missing for {Path(image_relative).name}. Run Inpaint first."
        )
    base_image = load_image_bgr(inpaint_path)

    sprite_directory = render_sprite_dir(project, image_relative)
    if sprite_directory.exists():
        for existing_sprite in sprite_directory.glob("item_*.png"):
            existing_sprite.unlink()
    if save_sprites:
        sprite_directory.mkdir(parents=True, exist_ok=True)
    metadata["no_text_page"] = False

    items = [normalize_render_item(item) for item in metadata.get("items", [])]
    unresolved_item_ids: list[int] = []
    for index, item in enumerate(items):
        canon_item = resolve_canon_item_for_stage_item(canon_state, item, active_only=False)
        if canon_item is None:
            unresolved_item_ids.append(int(item.get("id", index)))
            continue
        item["canon_id"] = str(canon_item.get("canon_id", "") or "")
        item["bbox"] = canon_item_bbox(canon_item, "bbox")
        item["render_bbox"] = canon_item_bbox(canon_item, "render_bbox")
        item["excluded"] = not bool(canon_item.get("enabled", True))
    if unresolved_item_ids:
        unresolved_text = ", ".join(str(item_id) for item_id in unresolved_item_ids[:5])
        raise RuntimeError(
            f"Render cache for {Path(image_relative).name} could not be matched to canon_state "
            f"(item ids: {unresolved_text}). Re-prepare Render first."
        )
    renderable_items = [
        item
        for item in items
        if not bool(item.get("excluded", False))
        and str(item.get("status", "") or "").lower() != "skipped"
    ]
    if not renderable_items:
        raise RuntimeError("No translated text is available to render for this page.")

    rendered_count = 0
    skipped_count = len(
        [
            item
            for item in items
            if bool(item.get("excluded", False)) or str(item.get("status", "") or "").lower() == "skipped"
        ]
    )
    for item_index, item in enumerate(items):
        if bool(item.get("excluded", False)) or str(item.get("status", "") or "").lower() == "skipped":
            continue

        canon_id = str(item.get("canon_id", "") or "").strip()
        if not canon_id:
            raise RuntimeError(
                f"Render item {item.get('id', item_index)} on {Path(image_relative).name} is missing canon_id. "
                "Re-prepare Render first."
            )
        canon_item = get_canon_item(canon_state, canon_id)
        if not bool(canon_item.get("enabled", True)):
            item["excluded"] = True
            skipped_count += 1
            metadata["items"] = items
            metadata["skipped_item_count"] = skipped_count
            save_render_json(metadata_path, metadata)
            continue
        item["bbox"] = canon_item_bbox(canon_item, "bbox")
        item["render_bbox"] = canon_item_bbox(canon_item, "render_bbox")

        render_bbox = clamp_bbox_to_image(item.get("render_bbox"), base_image.shape)
        translated_text = str(item.get("translated_text", "") or "").strip()
        if render_bbox is None or bbox_area(render_bbox) < 64:
            item["status"] = "skipped"
            item["error"] = "Render box is invalid or too small."
            skipped_count += 1
            save_render_json(metadata_path, {**metadata, "items": items, "skipped_item_count": skipped_count})
            continue
        if not translated_text:
            item["status"] = "skipped"
            item["error"] = ""
            skipped_count += 1
            save_render_json(metadata_path, {**metadata, "items": items, "skipped_item_count": skipped_count})
            continue

        writing_mode = choose_writing_mode(
            translated_text,
            render_bbox,
            source_direction=str(item.get("source_direction", "") or ""),
            auto_direction=config.auto_direction,
            vertical_cjk=config.vertical_cjk,
        )
        try:
            sprite_rgba, render_details = _render_item_sprite(
                base_image,
                translated_text,
                render_bbox,
                kind=str(item.get("kind", "") or "bubble"),
                source_direction=str(item.get("source_direction", "") or ""),
                writing_mode=writing_mode,
                font_path=resolved_font_path,
                min_font_size=config.min_font_size,
                max_font_size=config.max_font_size,
                stroke_enabled=config.stroke_enabled,
                stroke_width=config.stroke_width,
                text_color=config.text_color,
                stroke_color=config.stroke_color,
                auto_color=config.auto_color,
            )
        except Exception as exc:
            item["status"] = "error"
            item["error"] = str(exc)
            save_render_json(metadata_path, {**metadata, "items": items})
            _emit_progress(
                progress_callback,
                event="item_error",
                image_relative_path=image_relative,
                message=f"Render item failed on {Path(image_relative).name}: {exc}",
            )
            continue

        alpha_composite_onto_bgr(base_image, sprite_rgba, render_bbox)
        item["writing_mode"] = writing_mode
        item["font_size"] = int(render_details.get("font_size", 0) or 0)
        item["font_path"] = str(render_details.get("font_path", "") or resolved_font_path)
        item["text_color"] = coerce_serializable_color(render_details.get("text_color"))
        item["stroke_color"] = coerce_serializable_color(render_details.get("stroke_color"))
        item["stroke_width"] = float(render_details.get("stroke_width", 0.0) or 0.0)
        item["sprite_transform"] = {
            "x": int(render_bbox[0]),
            "y": int(render_bbox[1]),
            "width": int(render_bbox[2] - render_bbox[0]),
            "height": int(render_bbox[3] - render_bbox[1]),
            "rotation_deg": 0,
        }
        item["status"] = "rendered"
        item["error"] = str(render_details.get("warning", "") or "")
        item["needs_render"] = False

        if save_sprites:
            sprite_path = sprite_directory / f"item_{item_index:03d}.png"
            save_png_image(sprite_rgba, sprite_path)
            item["sprite_path"] = project_relative_path(project.root_dir, sprite_path)
        else:
            item["sprite_path"] = ""

        rendered_count += 1
        metadata["items"] = items
        metadata["rendered_item_count"] = rendered_count
        metadata["skipped_item_count"] = skipped_count
        metadata["updated_at"] = timestamp()
        save_render_json(metadata_path, metadata)

        _emit_progress(
            progress_callback,
            event="item_done",
            image_relative_path=image_relative,
            message=f"Rendered item {item_index + 1}/{len(items)} for {Path(image_relative).name}",
        )

    if rendered_count <= 0:
        metadata["items"] = items
        metadata["status"] = "error"
        metadata["error"] = "No translated text could be rendered for this page."
        metadata["no_text_page"] = False
        metadata["updated_at"] = timestamp()
        save_render_json(metadata_path, metadata)
        raise RuntimeError(metadata["error"])

    save_png_image(base_image, output_image)
    metadata["items"] = items
    metadata["status"] = "done"
    metadata["error"] = ""
    metadata["no_text_page"] = False
    metadata["output_image_path"] = project_relative_path(project.root_dir, output_image)
    metadata["rendered_item_count"] = rendered_count
    metadata["skipped_item_count"] = skipped_count
    metadata["updated_at"] = timestamp()
    metadata["needs_render"] = False
    save_render_json(metadata_path, metadata)

    _emit_progress(
        progress_callback,
        event="page_done",
        image_relative_path=image_relative,
        output_path=str(output_image),
        summary=summarize_render_json(metadata),
        message=f"Render complete for {Path(image_relative).name}",
    )
    _log(logger, f"Rendered page saved to {output_image}")
    return output_image


def run_render_for_pages(
    project: ProjectLike,
    image_relative_paths: Sequence[str | Path],
    *,
    force: bool = False,
    font_name: str | None = None,
    font_path: str | None = None,
    min_font_size: int = 12,
    max_font_size: int = 72,
    stroke_enabled: bool = True,
    stroke_width: float | None = None,
    text_color: tuple[int, int, int] | None = None,
    stroke_color: tuple[int, int, int] | None = None,
    auto_color: bool = True,
    auto_direction: bool = True,
    vertical_cjk: bool = True,
    save_sprites: bool = True,
    logger: Logger | None = None,
    progress_callback: ProgressCallback | None = None,
) -> list[Path]:
    """Render multiple pages sequentially to keep memory usage low."""

    if not image_relative_paths:
        raise ValueError("No pages were provided for rendering.")

    output_paths: list[Path] = []
    total_pages = len(image_relative_paths)
    for page_index, image_relative_path in enumerate(image_relative_paths, start=1):
        image_relative = str(Path(image_relative_path).as_posix())
        _emit_progress(
            progress_callback,
            event="batch_page_start",
            page_index=page_index,
            page_total=total_pages,
            image_relative_path=image_relative,
            message=f"[{page_index}/{total_pages}] Rendering {Path(image_relative).name}",
        )
        output_paths.append(
            run_render_for_page(
                project,
                image_relative,
                force=force,
                font_name=font_name,
                font_path=font_path,
                min_font_size=min_font_size,
                max_font_size=max_font_size,
                stroke_enabled=stroke_enabled,
                stroke_width=stroke_width,
                text_color=text_color,
                stroke_color=stroke_color,
                auto_color=auto_color,
                auto_direction=auto_direction,
                vertical_cjk=vertical_cjk,
                save_sprites=save_sprites,
                logger=logger,
                progress_callback=progress_callback,
            )
        )
    return output_paths


def build_render_items_from_translation(
    translation_data: dict[str, Any],
    ocr_data: dict[str, Any] | None,
    canon_state: dict[str, Any],
    image_shape: Sequence[int],
) -> list[dict[str, Any]]:
    """Build lightweight render items from cached translation/OCR plus canon_state."""

    ocr_items_by_id: dict[int, dict[str, Any]] = {}
    if isinstance(ocr_data, dict):
        for index, ocr_item in enumerate(ocr_data.get("items", [])):
            if not isinstance(ocr_item, dict):
                continue
            try:
                ocr_item_id = int(ocr_item.get("id", index))
            except Exception:
                ocr_item_id = index
            ocr_items_by_id[ocr_item_id] = dict(ocr_item)

    render_items: list[dict[str, Any]] = []
    for item_index, translation_item in enumerate(translation_data.get("items", [])):
        if not isinstance(translation_item, dict):
            continue

        normalized_translation_item = dict(translation_item)
        try:
            ocr_item_id = int(normalized_translation_item.get("ocr_item_id", item_index))
        except Exception:
            ocr_item_id = item_index
        ocr_item = ocr_items_by_id.get(ocr_item_id, {})
        canon_item = resolve_canon_item_for_stage_item(canon_state, normalized_translation_item, active_only=False)
        if canon_item is None and ocr_item:
            canon_item = resolve_canon_item_for_stage_item(canon_state, ocr_item, active_only=False)
        if canon_item is None or not bool(canon_item.get("enabled", True)):
            continue
        kind = str(normalized_translation_item.get("kind") or ocr_item.get("kind") or "")
        translated_text = str(normalized_translation_item.get("translated_text", "") or "").strip()
        render_bbox = clamp_bbox_to_image(
            canon_item_bbox(canon_item, "render_bbox"),
            image_shape,
        )
        base_bbox = clamp_bbox_to_image(canon_item_bbox(canon_item, "bbox"), image_shape)

        item_payload = {
            "id": item_index,
            "translation_item_id": int(normalized_translation_item.get("id", item_index) or item_index),
            "ocr_item_id": ocr_item_id,
            "canon_id": str(canon_item.get("canon_id", "") or ""),
            "kind": kind,
            "source_text": str(normalized_translation_item.get("source_text", "") or ""),
            "translated_text": translated_text,
            "bbox": bbox_to_list(base_bbox),
            "render_bbox": bbox_to_list(render_bbox),
            "source_direction": str(canon_item.get("source_direction", "") or ocr_item.get("source_direction", "") or ""),
            "writing_mode": "horizontal",
            "font_size": 0,
            "font_path": "",
            "text_color": None,
            "stroke_color": None,
            "stroke_width": 0.0,
            "sprite_path": "",
            "sprite_transform": {},
            "status": "pending",
            "error": "",
            # Snapshot only. canon_state remains the source of truth.
            "excluded": False,
        }

        if not translated_text:
            item_payload["status"] = "skipped"
        elif render_bbox is None or bbox_area(render_bbox) < 64:
            item_payload["status"] = "skipped"
            item_payload["error"] = "Render box is invalid or too small."

        render_items.append(normalize_render_item(item_payload))

    return render_items


def _active_renderable_canon_items(canon_state: dict[str, Any]) -> list[dict[str, Any]]:
    items = canon_state.get("items", []) if isinstance(canon_state, dict) else []
    active_items: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if not bool(item.get("enabled", True)):
            continue
        active_items.append(item)
    return active_items


def _resolve_render_base_image(project: ProjectLike, image_relative_path: str) -> tuple[Path, Any]:
    inpaint_path = inpaint_image_path(project, image_relative_path)
    if inpaint_path.exists():
        return inpaint_path, load_image_bgr(inpaint_path)

    source_path = project.root_dir / image_relative_path
    if source_path.exists():
        return source_path, load_image_bgr(source_path)

    raise FileNotFoundError(
        f"No base image is available for {Path(image_relative_path).name}. "
        "Expected an inpaint output or the source image."
    )


def _finalize_no_text_render_output(
    project: ProjectLike,
    image_relative_path: str,
    metadata: dict[str, Any],
    metadata_path: Path,
    *,
    logger: Logger | None,
    progress_callback: ProgressCallback | None,
) -> Path:
    base_image_path, base_image = _resolve_render_base_image(project, image_relative_path)
    output_image = render_image_path(project, image_relative_path)
    save_png_image(base_image, output_image)

    inpaint_path = inpaint_image_path(project, image_relative_path)
    translation_path = translation_json_path(project, image_relative_path)
    ocr_path = ocr_json_path(project, image_relative_path)

    metadata["source_image"] = str(metadata.get("source_image", "") or image_relative_path)
    metadata["inpaint_image_path"] = project_relative_path(project.root_dir, inpaint_path) if inpaint_path.exists() else ""
    metadata["translation_cache_path"] = (
        project_relative_path(project.root_dir, translation_path) if translation_path.exists() else ""
    )
    metadata["ocr_cache_path"] = project_relative_path(project.root_dir, ocr_path) if ocr_path.exists() else ""
    metadata["output_image_path"] = project_relative_path(project.root_dir, output_image)
    metadata["image_width"] = int(base_image.shape[1])
    metadata["image_height"] = int(base_image.shape[0])
    metadata["item_count"] = 0
    metadata["items"] = []
    metadata["status"] = "done"
    metadata["error"] = ""
    metadata["no_text_page"] = True
    metadata["rendered_item_count"] = 0
    metadata["skipped_item_count"] = 0
    metadata["updated_at"] = timestamp()
    metadata["needs_render"] = False
    save_render_json(metadata_path, metadata)

    message = (
        f"Render no-op: no active text items found for {Path(image_relative_path).name}. "
        f"Used {base_image_path.name} as the base image."
    )
    _emit_progress(
        progress_callback,
        event="page_done",
        image_relative_path=image_relative_path,
        output_path=str(output_image),
        summary=summarize_render_json(metadata),
        message=message,
    )
    _log(logger, message)
    return output_image


def _render_item_sprite(
    image_bgr,
    text: str,
    render_bbox: tuple[int, int, int, int],
    *,
    kind: str,
    source_direction: str,
    writing_mode: str,
    font_path: str,
    min_font_size: int,
    max_font_size: int,
    stroke_enabled: bool,
    stroke_width: float | None,
    text_color: tuple[int, int, int] | None,
    stroke_color: tuple[int, int, int] | None,
    auto_color: bool,
) -> tuple[Any, dict[str, Any]]:
    try:
        import numpy as np
        from PIL import Image, ImageDraw
    except Exception as exc:
        raise RuntimeError("Pillow and NumPy are required for rendering translated text.") from exc

    x1, y1, x2, y2 = render_bbox
    sprite_width = max(1, x2 - x1)
    sprite_height = max(1, y2 - y1)
    overlay = Image.new("RGBA", (sprite_width, sprite_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    resolved_font_path = find_fallback_font_for_text(text, font_path)
    if font_path and not resolved_font_path:
        raise RuntimeError(f"No usable font could be resolved for {font_path}")

    resolved_text_color = text_color
    resolved_stroke_color = stroke_color
    effective_stroke_width = stroke_width
    if auto_color or resolved_text_color is None:
        auto_text_color, auto_stroke_color, auto_stroke_width = choose_text_color_for_region(
            image_bgr,
            render_bbox,
            prefer_stroke=(kind == "layout_text"),
            is_dark=None,
        )
        resolved_text_color = resolved_text_color or auto_text_color
        if resolved_stroke_color is None:
            resolved_stroke_color = auto_stroke_color
        if effective_stroke_width is None:
            effective_stroke_width = float(auto_stroke_width)

    if resolved_text_color is None:
        resolved_text_color = (0, 0, 0)
    if effective_stroke_width is None:
        effective_stroke_width = max(1.0, min(sprite_width, sprite_height) / 36.0)

    if not stroke_enabled:
        resolved_stroke_color = None
        effective_stroke_width = 0.0

    if resolved_stroke_color is None and stroke_enabled:
        resolved_stroke_color = (255, 255, 255) if resolved_text_color == (0, 0, 0) else (0, 0, 0)

    integer_stroke_width = max(0, int(round(effective_stroke_width)))
    fill_rgba = tuple(int(channel) for channel in resolved_text_color) + (255,)
    stroke_rgba = None if resolved_stroke_color is None else tuple(int(channel) for channel in resolved_stroke_color) + (255,)

    if writing_mode == "vertical_rl":
        render_details = _draw_vertical_text(
            draw,
            text,
            sprite_width,
            sprite_height,
            font_path=resolved_font_path,
            min_font_size=min_font_size,
            max_font_size=max_font_size,
            fill_rgba=fill_rgba,
            stroke_rgba=stroke_rgba,
            stroke_width=integer_stroke_width,
        )
    else:
        render_details = _draw_horizontal_text(
            draw,
            text,
            sprite_width,
            sprite_height,
            font_path=resolved_font_path,
            min_font_size=min_font_size,
            max_font_size=max_font_size,
            fill_rgba=fill_rgba,
            stroke_rgba=stroke_rgba,
            stroke_width=integer_stroke_width,
        )

    return np.asarray(overlay, dtype=np.uint8), {
        "font_size": render_details.get("font_size", min_font_size),
        "font_path": resolved_font_path,
        "text_color": resolved_text_color,
        "stroke_color": resolved_stroke_color,
        "stroke_width": float(integer_stroke_width),
        "warning": render_details.get("warning", ""),
    }


def _draw_horizontal_text(
    draw,
    text: str,
    width: int,
    height: int,
    *,
    font_path: str,
    min_font_size: int,
    max_font_size: int,
    fill_rgba: tuple[int, int, int, int],
    stroke_rgba: tuple[int, int, int, int] | None,
    stroke_width: int,
) -> dict[str, Any]:
    padding = max(2, int(round(min(width, height) * 0.04)))
    layout = fit_text_layout(
        text,
        font_path,
        width,
        height,
        align="center",
        padding=padding,
        min_font_size=min_font_size,
        max_font_size=max_font_size,
        stroke_width=stroke_width,
        line_spacing_ratio=0.18,
        min_line_spacing=1,
    )

    available_x1 = padding
    available_y1 = padding
    available_width = max(1, width - (padding * 2))
    available_height = max(1, height - (padding * 2))

    text_bbox = layout.text_bbox
    raw_x = available_x1 + ((available_width - layout.text_width) / 2.0) - text_bbox[0]
    raw_y = available_y1 + ((available_height - layout.text_height) / 2.0) - text_bbox[1]
    text_x = int(round(raw_x))
    text_y = int(round(raw_y))

    draw.multiline_text(
        (text_x, text_y),
        layout.wrapped_text,
        font=layout.font,
        fill=fill_rgba,
        align="center",
        spacing=layout.line_spacing,
        stroke_width=stroke_width,
        stroke_fill=stroke_rgba,
    )

    warning = "Rendered at minimum font size; text may overflow." if layout.overflow else ""
    return {
        "font_size": int(layout.font_size),
        "warning": warning,
    }


def _draw_vertical_text(
    draw,
    text: str,
    width: int,
    height: int,
    *,
    font_path: str,
    min_font_size: int,
    max_font_size: int,
    fill_rgba: tuple[int, int, int, int],
    stroke_rgba: tuple[int, int, int, int] | None,
    stroke_width: int,
) -> dict[str, Any]:
    padding = max(2, int(round(min(width, height) * 0.04)))
    usable_width = max(1, width - (padding * 2))
    usable_height = max(1, height - (padding * 2))

    paragraphs = [iter_vertical_tokens(paragraph) for paragraph in (text.splitlines() or [text])]
    best_layout: dict[str, Any] | None = None

    for font_size in range(max_font_size, max(min_font_size - 1, 0), -1):
        font = get_cached_font(font_path, font_size)
        line_gap = max(1, int(round(font_size * 0.12)))
        column_gap = max(1, int(round(font_size * 0.18)))

        token_metrics: dict[str, tuple[int, int, tuple[int, int, int, int]]] = {}
        cell_width = 0
        cell_height = 0
        for paragraph in paragraphs:
            for token in paragraph:
                bbox = draw.textbbox((0, 0), token, font=font, stroke_width=stroke_width)
                token_width = max(1, int(bbox[2] - bbox[0]))
                token_height = max(1, int(bbox[3] - bbox[1]))
                token_metrics[token] = (token_width, token_height, tuple(int(value) for value in bbox))
                cell_width = max(cell_width, token_width)
                cell_height = max(cell_height, token_height)

        if cell_width <= 0 or cell_height <= 0:
            continue

        rows_per_column = max(1, (usable_height + line_gap) // max(1, cell_height + line_gap))
        columns: list[list[str]] = []
        for paragraph in paragraphs:
            if not paragraph:
                if columns:
                    columns.append([])
                continue
            for start_index in range(0, len(paragraph), rows_per_column):
                columns.append(paragraph[start_index:start_index + rows_per_column])

        if not columns:
            columns = [[]]

        column_count = len(columns)
        total_width = column_count * cell_width + max(0, column_count - 1) * column_gap
        if total_width > usable_width:
            continue

        best_layout = {
            "font_size": font_size,
            "font": font,
            "line_gap": line_gap,
            "column_gap": column_gap,
            "cell_width": cell_width,
            "cell_height": cell_height,
            "columns": columns,
            "token_metrics": token_metrics,
            "overflow": False,
        }
        break

    if best_layout is None:
        fallback_size = min_font_size
        font = get_cached_font(font_path, fallback_size)
        line_gap = max(1, int(round(fallback_size * 0.12)))
        column_gap = max(1, int(round(fallback_size * 0.18)))
        columns = [paragraph for paragraph in paragraphs if paragraph] or [[]]
        token_metrics: dict[str, tuple[int, int, tuple[int, int, int, int]]] = {}
        cell_width = 1
        cell_height = 1
        for paragraph in paragraphs:
            for token in paragraph:
                bbox = draw.textbbox((0, 0), token, font=font, stroke_width=stroke_width)
                token_width = max(1, int(bbox[2] - bbox[0]))
                token_height = max(1, int(bbox[3] - bbox[1]))
                token_metrics[token] = (token_width, token_height, tuple(int(value) for value in bbox))
                cell_width = max(cell_width, token_width)
                cell_height = max(cell_height, token_height)
        best_layout = {
            "font_size": fallback_size,
            "font": font,
            "line_gap": line_gap,
            "column_gap": column_gap,
            "cell_width": cell_width,
            "cell_height": cell_height,
            "columns": columns,
            "token_metrics": token_metrics,
            "overflow": True,
        }

    font = best_layout["font"]
    line_gap = int(best_layout["line_gap"])
    column_gap = int(best_layout["column_gap"])
    cell_width = int(best_layout["cell_width"])
    cell_height = int(best_layout["cell_height"])
    columns = list(best_layout["columns"])
    token_metrics = dict(best_layout["token_metrics"])

    total_width = len(columns) * cell_width + max(0, len(columns) - 1) * column_gap
    start_x = padding + max(0, (usable_width - total_width) // 2)

    for column_index, column_tokens in enumerate(columns):
        column_height = (
            len(column_tokens) * cell_height + max(0, len(column_tokens) - 1) * line_gap
            if column_tokens
            else 0
        )
        column_x = start_x + (len(columns) - 1 - column_index) * (cell_width + column_gap)
        column_y = padding + max(0, (usable_height - column_height) // 2)

        for row_index, token in enumerate(column_tokens):
            token_width, token_height, bbox = token_metrics[token]
            text_x = column_x + max(0, (cell_width - token_width) // 2) - bbox[0]
            text_y = column_y + row_index * (cell_height + line_gap) + max(0, (cell_height - token_height) // 2) - bbox[1]
            draw.text(
                (text_x, text_y),
                token,
                font=font,
                fill=fill_rgba,
                stroke_width=stroke_width,
                stroke_fill=stroke_rgba,
            )

    warning = "Rendered at minimum font size; vertical text may overflow." if best_layout.get("overflow") else ""
    return {"font_size": int(best_layout["font_size"]), "warning": warning}


def _load_existing_render_metadata(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return load_render_json(path)
    except Exception:
        return None


def _log(logger: Logger | None, message: str) -> None:
    if logger is not None and message:
        logger(str(message))


def _emit_progress(callback: ProgressCallback | None, **payload: Any) -> None:
    if callback is not None:
        callback(payload)


__all__ = [
    "build_render_items_from_translation",
    "prepare_render_for_page",
    "run_render_for_page",
    "run_render_for_pages",
]
