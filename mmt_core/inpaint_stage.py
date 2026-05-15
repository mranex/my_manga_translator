"""Inpaint stage helpers for GUI-driven mask preparation and LaMa execution."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
from pathlib import Path
import threading
from typing import Any, Protocol

from .canon_state import ensure_canon_state, get_active_canon_items
from .detection_io import detection_json_path, load_detection_json, save_detection_json
from .image_io import load_image_bgr, load_image_grayscale, save_png_image
from .inpaint_io import (
    build_inpaint_metadata,
    bubble_mask_path,
    inpaint_image_path,
    inpaint_json_path,
    inpaint_preview_mask_path,
    load_inpaint_json,
    save_inpaint_json,
    text_mask_path,
)
from .inpaint_masks import (
    build_bubble_guidance_mask,
    build_crop_windows_from_boxes,
    build_preview_mask,
    build_text_mask_from_canon_ocr_bboxes,
)
from .ocr_io import load_ocr_json, ocr_json_path

DEFAULT_MASK_PADDING = 0
DEFAULT_CROP_TRIGGER_SIZE = 800
DEFAULT_CROP_MARGIN = 128
DEFAULT_RESIZE_LIMIT = 1280
DEFAULT_PAD_MOD = 8

Logger = Callable[[str], None]
ProgressCallback = Callable[[dict[str, Any]], None]


class ProjectLike(Protocol):
    root_dir: Path
    cache_dir: Path


@dataclass(slots=True)
class _PreparedMaskBundle:
    source_image_path: Path
    ocr_cache_path: Path
    detection_cache_path: Path | None
    output_image_path: Path
    metadata_path: Path
    text_mask_file: Path
    bubble_mask_file: Path | None
    preview_mask_file: Path
    image_shape: tuple[int, int, int]
    item_count: int
    masked_pixel_count: int
    valid_boxes: list[tuple[int, int, int, int]]
    metadata: dict[str, Any]


class LamaInpainterManager:
    """Resident cache for a single long-lived LaMa Manga inpainter."""

    def __init__(self) -> None:
        self._inpainter: Any | None = None
        self._signature: tuple[str | None, str | None, int, int, int, int] | None = None
        self._loaded = False
        self._last_device: str | None = None
        self._owner_thread_id: int | None = None
        self._active_use_count = 0
        self._load_count = 0
        self._reload_count = 0
        self._lock = threading.RLock()

    def load_once(
        self,
        *,
        device: str | None,
        crop_trigger_size: int = DEFAULT_CROP_TRIGGER_SIZE,
        crop_margin: int = DEFAULT_CROP_MARGIN,
        resize_limit: int = DEFAULT_RESIZE_LIMIT,
        pad_mod: int = DEFAULT_PAD_MOD,
        model_path: str | None = None,
        explicit_reload: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            signature = _normalized_lama_signature(
                device=device,
                crop_trigger_size=crop_trigger_size,
                crop_margin=crop_margin,
                resize_limit=resize_limit,
                pad_mod=pad_mod,
                model_path=model_path,
            )
            self._ensure_thread_ownership_locked()

            if self._inpainter is not None and self._signature == signature and self._loaded and not explicit_reload:
                resolved_device = str(self._last_device or signature[1] or "")
                return {
                    "loaded": True,
                    "device": resolved_device,
                    "reused": True,
                    "reloaded": False,
                    "signature": signature,
                    "message": f"LaMa Manga model already loaded on {resolved_device or 'auto'}.",
                }

            if self._inpainter is not None and self._signature != signature and not explicit_reload:
                raise RuntimeError(
                    "Inpaint runtime model signature mismatch: "
                    f"loaded={_format_lama_signature(self._signature)} "
                    f"requested={_format_lama_signature(signature)}. "
                    "Reload Inpaint Service before running."
                )

            if self._active_use_count > 0:
                raise RuntimeError(
                    "LaMa Manga is busy and cannot change configuration until the current inpaint task finishes."
                )

            had_loaded_model = self._inpainter is not None or self._loaded
            if explicit_reload and had_loaded_model:
                self._unload_unlocked(clear_cuda_cache=True)

            if self._inpainter is None or self._signature != signature:
                inpainting_module = _import_inpainting()
                self._inpainter = inpainting_module.LamaMangaInpainter(
                    model_path=signature[0],
                    device=signature[1],
                    crop_trigger_size=signature[2],
                    crop_margin=signature[3],
                    resize_limit=signature[4],
                    pad_mod=signature[5],
                )
                self._signature = signature
                self._loaded = False
                self._owner_thread_id = threading.get_ident()

            if not self._loaded:
                self._inpainter.load()
                self._loaded = True
                self._load_count += 1
                if explicit_reload and had_loaded_model:
                    self._reload_count += 1
                self._last_device = str(
                    getattr(self._inpainter, "device", signature[1] or "") or signature[1] or ""
                )

            resolved_device = str(self._last_device or signature[1] or "")
            return {
                "loaded": True,
                "device": resolved_device,
                "reused": False,
                "reloaded": bool(explicit_reload and had_loaded_model),
                "signature": signature,
                "message": (
                    f"LaMa Manga model reloaded explicitly on {resolved_device or 'auto'}."
                    if explicit_reload and had_loaded_model
                    else f"LaMa Manga model loaded on {resolved_device or 'auto'}."
                ),
            }

    def get_loaded_inpainter(
        self,
        *,
        device: str | None,
        crop_trigger_size: int = DEFAULT_CROP_TRIGGER_SIZE,
        crop_margin: int = DEFAULT_CROP_MARGIN,
        resize_limit: int = DEFAULT_RESIZE_LIMIT,
        pad_mod: int = DEFAULT_PAD_MOD,
        model_path: str | None = None,
    ) -> Any:
        with self._lock:
            requested_signature = _normalized_lama_signature(
                device=device,
                crop_trigger_size=crop_trigger_size,
                crop_margin=crop_margin,
                resize_limit=resize_limit,
                pad_mod=pad_mod,
                model_path=model_path,
            )
            self._ensure_thread_ownership_locked()
            if self._inpainter is None or not self._loaded:
                raise RuntimeError("LaMa Manga is not loaded. Reload Inpaint Service.")
            if self._signature != requested_signature:
                raise RuntimeError(
                    "Inpaint runtime model signature mismatch: "
                    f"loaded={_format_lama_signature(self._signature)} "
                    f"requested={_format_lama_signature(requested_signature)}. "
                    "Reload Inpaint Service before running."
                )
            return self._inpainter

    def begin_inpaint_use(self) -> None:
        with self._lock:
            self._active_use_count += 1

    def end_inpaint_use(self) -> None:
        with self._lock:
            if self._active_use_count > 0:
                self._active_use_count -= 1

    def get_inpainter(
        self,
        *,
        device: str | None,
        crop_trigger_size: int = DEFAULT_CROP_TRIGGER_SIZE,
        crop_margin: int = DEFAULT_CROP_MARGIN,
        resize_limit: int = DEFAULT_RESIZE_LIMIT,
        pad_mod: int = DEFAULT_PAD_MOD,
        model_path: str | None = None,
        preload: bool = False,
    ) -> Any:
        if preload:
            self.load_once(
                device=device,
                crop_trigger_size=crop_trigger_size,
                crop_margin=crop_margin,
                resize_limit=resize_limit,
                pad_mod=pad_mod,
                model_path=model_path,
            )
            return self.get_loaded_inpainter(
                device=device,
                crop_trigger_size=crop_trigger_size,
                crop_margin=crop_margin,
                resize_limit=resize_limit,
                pad_mod=pad_mod,
                model_path=model_path,
            )
        return self.get_loaded_inpainter(
            device=device,
            crop_trigger_size=crop_trigger_size,
            crop_margin=crop_margin,
            resize_limit=resize_limit,
            pad_mod=pad_mod,
            model_path=model_path,
        )

    def load(
        self,
        *,
        device: str | None,
        crop_trigger_size: int = DEFAULT_CROP_TRIGGER_SIZE,
        crop_margin: int = DEFAULT_CROP_MARGIN,
        resize_limit: int = DEFAULT_RESIZE_LIMIT,
        pad_mod: int = DEFAULT_PAD_MOD,
        model_path: str | None = None,
    ) -> dict[str, Any]:
        return self.load_once(
            device=device,
            crop_trigger_size=crop_trigger_size,
            crop_margin=crop_margin,
            resize_limit=resize_limit,
            pad_mod=pad_mod,
            model_path=model_path,
        )

    def reload(
        self,
        *,
        device: str | None,
        crop_trigger_size: int = DEFAULT_CROP_TRIGGER_SIZE,
        crop_margin: int = DEFAULT_CROP_MARGIN,
        resize_limit: int = DEFAULT_RESIZE_LIMIT,
        pad_mod: int = DEFAULT_PAD_MOD,
        model_path: str | None = None,
    ) -> dict[str, Any]:
        return self.load_once(
            device=device,
            crop_trigger_size=crop_trigger_size,
            crop_margin=crop_margin,
            resize_limit=resize_limit,
            pad_mod=pad_mod,
            model_path=model_path,
            explicit_reload=True,
        )

    def unload(self) -> dict[str, Any]:
        with self._lock:
            if self._active_use_count > 0:
                return {
                    "loaded": bool(self._inpainter is not None and self._loaded),
                    "device": self._last_device or "",
                    "message": "LaMa Manga is busy running inpaint. Wait for the current task to finish before unloading.",
                }
            return self._unload_unlocked(clear_cuda_cache=True)

    def _ensure_thread_ownership_locked(self) -> None:
        current_thread_id = threading.get_ident()
        if self._owner_thread_id is None:
            self._owner_thread_id = current_thread_id
            return
        if self._owner_thread_id != current_thread_id:
            raise RuntimeError(
                "LaMa Manga manager is owned by another thread. Reload Inpaint Service before running."
            )

    def _unload_unlocked(self, *, clear_cuda_cache: bool) -> dict[str, Any]:
        had_model = self._inpainter is not None
        self._inpainter = None
        self._signature = None
        self._loaded = False
        self._owner_thread_id = None

        if clear_cuda_cache:
            try:
                import torch
            except Exception:
                torch = None

            if torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        gc.collect()
        return {
            "loaded": False,
            "device": self._last_device or "",
            "message": "LaMa Manga model cache cleared." if had_model else "LaMa Manga model was not loaded.",
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "loaded": bool(self._inpainter is not None and self._loaded),
                "device": self._last_device or "",
                "busy": self._active_use_count > 0,
                "load_count": self._load_count,
                "reload_count": self._reload_count,
                "signature": _format_lama_signature(self._signature),
                "message": (
                    f"LaMa Manga model busy on {self._last_device or 'auto'}."
                    if self._active_use_count > 0
                    else (
                        f"LaMa Manga model ready on {self._last_device or 'auto'}."
                        if self._inpainter is not None and self._loaded
                        else "LaMa Manga model is not loaded."
                    )
                ),
            }


_LAMA_MANAGER = LamaInpainterManager()


def get_lama_model_manager() -> LamaInpainterManager:
    """Return the process-level LaMa inpainter manager."""

    return _LAMA_MANAGER


def load_lama_model(
    *,
    device: str | None = None,
    crop_trigger_size: int = DEFAULT_CROP_TRIGGER_SIZE,
    crop_margin: int = DEFAULT_CROP_MARGIN,
    resize_limit: int = DEFAULT_RESIZE_LIMIT,
    pad_mod: int = DEFAULT_PAD_MOD,
    model_path: str | None = None,
    explicit_reload: bool = False,
    logger: Logger | None = None,
    manager: LamaInpainterManager | None = None,
) -> dict[str, Any]:
    """Preload the cached LaMa Manga model without running inpaint yet."""

    manager = manager or get_lama_model_manager()
    try:
        result = (
            manager.reload(
                device=device,
                crop_trigger_size=crop_trigger_size,
                crop_margin=crop_margin,
                resize_limit=resize_limit,
                pad_mod=pad_mod,
                model_path=model_path,
            )
            if explicit_reload
            else manager.load_once(
                device=device,
                crop_trigger_size=crop_trigger_size,
                crop_margin=crop_margin,
                resize_limit=resize_limit,
                pad_mod=pad_mod,
                model_path=model_path,
            )
        )
    except Exception as exc:
        raise RuntimeError(_friendly_inpaint_error(exc)) from exc

    _log(logger, result["message"])
    return result


def unload_lama_model(
    logger: Logger | None = None,
    *,
    manager: LamaInpainterManager | None = None,
) -> dict[str, Any]:
    """Release the cached LaMa Manga model and free CUDA cache when available."""

    manager = manager or get_lama_model_manager()
    result = manager.unload()
    _log(logger, result["message"])
    return result


def prepare_inpaint_mask_for_page(
    project: ProjectLike,
    image_relative_path: str | Path,
    *,
    mask_padding: int = DEFAULT_MASK_PADDING,
    use_bubble_mask: bool = True,
    force: bool = False,
    logger: Logger | None = None,
) -> Path:
    """Build and cache the inpaint masks for one page."""

    try:
        bundle = _prepare_mask_bundle(
            project,
            image_relative_path,
            mask_padding=mask_padding,
            use_bubble_mask=use_bubble_mask,
            use_crop_windows=True,
            force=force,
            device="",
            logger=logger,
        )
    except Exception as exc:
        raise RuntimeError(_friendly_inpaint_error(exc)) from exc

    return bundle.text_mask_file


def run_inpaint_for_page(
    project: ProjectLike,
    image_relative_path: str | Path,
    *,
    force: bool = False,
    mask_padding: int = DEFAULT_MASK_PADDING,
    use_bubble_mask: bool = True,
    use_crop_windows: bool = True,
    device: str | None = None,
    logger: Logger | None = None,
    progress_callback: ProgressCallback | None = None,
    manager: LamaInpainterManager | None = None,
    require_loaded_model: bool = False,
) -> Path:
    """Run LaMa Manga inpainting for one page using cached OCR/detection data."""

    image_relative = str(Path(image_relative_path).as_posix())
    page_name = Path(image_relative).name
    _log(logger, f"[inpaint] before prepare mask {page_name}")
    _emit_progress(
        progress_callback,
        event="page_start",
        image_relative_path=image_relative,
        message=f"Preparing inpaint mask for {page_name}",
    )

    bundle = _prepare_mask_bundle(
        project,
        image_relative,
        mask_padding=mask_padding,
        use_bubble_mask=use_bubble_mask,
        use_crop_windows=use_crop_windows,
        force=force,
        device=device or "",
        logger=logger,
    )
    _log(logger, f"[inpaint] after prepare mask {page_name}")
    current_settings = _settings_payload(
        mask_padding=mask_padding,
        use_bubble_mask=use_bubble_mask,
        use_crop_windows=use_crop_windows,
    )
    if _can_reuse_inpaint_output(
        bundle.metadata,
        bundle.output_image_path,
        current_settings=current_settings,
        force=force,
    ):
        _log(logger, f"Reusing cached inpaint result: {bundle.output_image_path}")
        _emit_progress(
            progress_callback,
            event="page_done",
            image_relative_path=image_relative,
            output_path=str(bundle.output_image_path),
            summary={
                "item_count": int(bundle.metadata.get("item_count", 0) or 0),
                "masked_pixel_count": int(bundle.metadata.get("masked_pixel_count", 0) or 0),
            },
            message=f"Reused cached inpaint result for {page_name}",
        )
        return bundle.output_image_path

    _log(logger, f"[inpaint] before load image/mask {page_name}")
    source_image = load_image_bgr(bundle.source_image_path)
    text_mask = load_image_grayscale(bundle.text_mask_file)
    bubble_mask = load_image_grayscale(bundle.bubble_mask_file) if bundle.bubble_mask_file else None
    _log(logger, f"[inpaint] after load image/mask {page_name}")
    _log(logger, f"[inpaint] before crop window build {page_name}")
    crop_windows = build_crop_windows_from_boxes(bundle.valid_boxes, source_image.shape) if use_crop_windows else []
    _log(logger, f"[inpaint] after crop window build {page_name}")

    metadata = dict(bundle.metadata)
    metadata["status"] = "running"
    metadata["error"] = ""
    metadata["needs_inpaint"] = True
    metadata["device"] = _normalize_device_value(device) or ""
    metadata["strict_ocr_bbox_mask"] = True
    metadata["updated_at"] = _timestamp()
    metadata["inpaint_created_at"] = str(metadata.get("inpaint_created_at", "") or _timestamp())
    metadata["inpaint_updated_at"] = metadata["updated_at"]
    save_inpaint_json(bundle.metadata_path, metadata)

    _emit_progress(
        progress_callback,
        event="mask_ready",
        image_relative_path=image_relative,
        output_path=str(bundle.text_mask_file),
        summary={
            "item_count": bundle.item_count,
            "masked_pixel_count": bundle.masked_pixel_count,
        },
        message=f"Inpaint mask ready for {page_name}",
    )

    manager = manager or get_lama_model_manager()
    inpainter = None
    try:
        _log(logger, f"[inpaint] before resident model acquire {page_name}")
        model_result = (
            {
                "loaded": True,
                "device": manager.status().get("device", "") if hasattr(manager, "status") else "",
                "reused": True,
                "reloaded": False,
            }
            if require_loaded_model
            else {}
        )
        if require_loaded_model:
            inpainter = manager.get_loaded_inpainter(
                device=device,
                crop_trigger_size=DEFAULT_CROP_TRIGGER_SIZE,
                crop_margin=DEFAULT_CROP_MARGIN,
                resize_limit=DEFAULT_RESIZE_LIMIT,
                pad_mod=DEFAULT_PAD_MOD,
            )
        else:
            model_result = manager.load_once(
                device=device,
                crop_trigger_size=DEFAULT_CROP_TRIGGER_SIZE,
                crop_margin=DEFAULT_CROP_MARGIN,
                resize_limit=DEFAULT_RESIZE_LIMIT,
                pad_mod=DEFAULT_PAD_MOD,
            )
            inpainter = manager.get_loaded_inpainter(
                device=device,
                crop_trigger_size=DEFAULT_CROP_TRIGGER_SIZE,
                crop_margin=DEFAULT_CROP_MARGIN,
                resize_limit=DEFAULT_RESIZE_LIMIT,
                pad_mod=DEFAULT_PAD_MOD,
            )
        _log(logger, f"[inpaint] after resident model acquire {page_name}")
        _emit_progress(
            progress_callback,
            event="model_ready",
            image_relative_path=image_relative,
            message=f"Using resident LaMa model for {page_name}",
        )
        _log(logger, f"Using resident LaMa model for {page_name}")
        manager.begin_inpaint_use()
        _log(logger, f"[inpaint] before LaMa inference {page_name}")
        output_image = inpainter.inpaint(
            source_image,
            text_mask,
            bubble_mask=bubble_mask,
            crop_windows=crop_windows if use_crop_windows else None,
            require_loaded=require_loaded_model,
        )
        _log(logger, f"[inpaint] after LaMa inference {page_name}")
        _log(logger, f"[inpaint] before save output {page_name}")
        save_png_image(output_image, bundle.output_image_path)
        _log(logger, f"[inpaint] after save output {page_name}")
    except Exception as exc:
        readable_error = _friendly_inpaint_error(exc)
        metadata["status"] = "error"
        metadata["error"] = readable_error
        metadata["needs_inpaint"] = True
        metadata["updated_at"] = _timestamp()
        metadata["inpaint_updated_at"] = metadata["updated_at"]
        save_inpaint_json(bundle.metadata_path, metadata)
        _emit_progress(
            progress_callback,
            event="page_error",
            image_relative_path=image_relative,
            output_path=str(bundle.output_image_path),
            summary={
                "item_count": bundle.item_count,
                "masked_pixel_count": bundle.masked_pixel_count,
            },
            message=f"Inpaint failed for {page_name}: {readable_error}",
        )
        raise RuntimeError(readable_error) from exc
    finally:
        if inpainter is not None:
            manager.end_inpaint_use()

    manager_status = manager.status() if hasattr(manager, "status") else {}
    metadata["status"] = "done"
    metadata["error"] = ""
    metadata["needs_inpaint"] = False
    metadata["device"] = str(getattr(inpainter, "device", _normalize_device_value(device) or "") or "")
    metadata["output_mask_hash"] = str(metadata.get("text_mask_hash", "") or "")
    metadata["output_bubble_mask_hash"] = str(metadata.get("bubble_mask_hash", "") or "")
    metadata["model_loaded_at_service_start"] = bool(require_loaded_model)
    metadata["model_reused"] = bool(model_result.get("reused", True))
    metadata["model_reload_count"] = int(manager_status.get("reload_count", 0) or 0)
    metadata["updated_at"] = _timestamp()
    metadata["inpaint_created_at"] = str(metadata.get("inpaint_created_at", "") or metadata["updated_at"])
    metadata["inpaint_updated_at"] = metadata["updated_at"]
    save_inpaint_json(bundle.metadata_path, metadata)

    _log(logger, f"Inpainted page saved to {bundle.output_image_path}")
    _log(logger, f"[inpaint] before emit page_done {page_name}")
    _emit_progress(
        progress_callback,
        event="page_done",
        image_relative_path=image_relative,
        output_path=str(bundle.output_image_path),
        summary={
            "item_count": bundle.item_count,
            "masked_pixel_count": bundle.masked_pixel_count,
        },
        message=f"Inpaint complete for {page_name}",
    )
    _log(logger, f"[inpaint] after emit page_done {page_name}")
    return bundle.output_image_path


def run_inpaint_for_pages(
    project: ProjectLike,
    image_relative_paths: Sequence[str | Path],
    *,
    force: bool = False,
    mask_padding: int = DEFAULT_MASK_PADDING,
    use_bubble_mask: bool = True,
    use_crop_windows: bool = True,
    device: str | None = None,
    logger: Logger | None = None,
    progress_callback: ProgressCallback | None = None,
    manager: LamaInpainterManager | None = None,
    require_loaded_model: bool = False,
) -> list[Path]:
    """Run inpainting sequentially across multiple pages."""

    if not image_relative_paths:
        raise ValueError("No pages were provided for inpainting.")

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
            message=f"[{page_index}/{total_pages}] Inpainting {Path(image_relative).name}",
        )
        output_path = run_inpaint_for_page(
            project,
            image_relative,
            force=force,
            mask_padding=mask_padding,
            use_bubble_mask=use_bubble_mask,
            use_crop_windows=use_crop_windows,
            device=device,
            logger=logger,
            progress_callback=progress_callback,
            manager=manager,
            require_loaded_model=require_loaded_model,
        )
        output_paths.append(output_path)

    return output_paths


def _prepare_mask_bundle(
    project: ProjectLike,
    image_relative_path: str | Path,
    *,
    mask_padding: int,
    use_bubble_mask: bool,
    use_crop_windows: bool,
    force: bool,
    device: str,
    logger: Logger | None,
) -> _PreparedMaskBundle:
    image_relative = str(Path(image_relative_path).as_posix())
    source_image_path = _project_source_path(project, image_relative)
    if not source_image_path.exists():
        raise FileNotFoundError(f"Source image is missing: {source_image_path}")

    ocr_cache_file = ocr_json_path(project, image_relative)
    if not ocr_cache_file.exists():
        raise FileNotFoundError(
            "OCR cache is missing for this page. Prepare OCR items first before inpainting."
        )

    detection_cache_file = detection_json_path(project, image_relative)
    if not detection_cache_file.exists():
        raise FileNotFoundError(
            f"Detection cache is missing for {Path(image_relative).name}. Run Detection first."
        )
    detection_cache_path_value = detection_cache_file

    metadata_path = inpaint_json_path(project, image_relative)
    output_image_file = inpaint_image_path(project, image_relative)
    text_mask_file = text_mask_path(project, image_relative)
    bubble_mask_file = bubble_mask_path(project, image_relative)
    preview_mask_file = inpaint_preview_mask_path(project, image_relative)
    existing_metadata = _load_existing_metadata(metadata_path)

    source_image = load_image_bgr(source_image_path)
    try:
        load_ocr_json(ocr_cache_file)
    except Exception as exc:
        raise ValueError(f"Invalid OCR cache: {exc}") from exc

    detection_data: dict[str, Any] | None = None
    if detection_cache_path_value is not None:
        try:
            detection_data = load_detection_json(detection_cache_path_value)
            had_canon_state = isinstance(detection_data.get("canon_state"), dict)
            needs_text_mask_backfill = _canon_state_needs_text_mask_backfill(detection_data)
            ensure_canon_state(detection_data, image_shape=source_image.shape)
            if not had_canon_state or needs_text_mask_backfill:
                save_detection_json(detection_cache_path_value, detection_data)
        except Exception as exc:
            _log(logger, f"Detection cache unavailable for bubble mask guidance: {exc}")
            detection_data = None

    if detection_data is None:
        raise RuntimeError(
            f"Detection cache is missing for {Path(image_relative).name}. Run Detection first."
        )

    text_mask, valid_boxes, masked_pixel_count, mask_stats = build_text_mask_from_canon_ocr_bboxes(
        source_image.shape,
        get_active_canon_items(detection_data["canon_state"]),
        padding=0,
        strict_mask=True,
        return_stats=True,
    )
    if not valid_boxes or masked_pixel_count <= 0:
        error_message = "No active canon OCR target boxes were available to build an inpaint mask."
        error_metadata = build_inpaint_metadata(
            project_root=project.root_dir,
            image_relative_path=image_relative,
            ocr_cache_path=ocr_cache_file,
            detection_cache_path=detection_cache_path_value,
            output_image_path_value=output_image_file,
            text_mask_path_value=text_mask_file,
            bubble_mask_path_value=bubble_mask_file if use_bubble_mask else None,
            text_mask_hash="",
            bubble_mask_hash="",
            output_mask_hash=str((existing_metadata or {}).get("output_mask_hash", "") or ""),
            output_bubble_mask_hash=str((existing_metadata or {}).get("output_bubble_mask_hash", "") or ""),
            image_shape=source_image.shape,
            item_count=0,
            active_item_count=int(mask_stats.get("active_item_count", 0) or 0),
            text_mask_box_count=0,
            skipped_item_count=int(mask_stats.get("skipped_invalid_bbox_count", 0) or 0),
            masked_pixel_count=0,
            bubble_mask_pixel_count=0,
            device=device,
            status="error",
            error=error_message,
            needs_inpaint=True,
            created_at=(existing_metadata or {}).get("created_at"),
            updated_at=_timestamp(),
            mask_created_at=(existing_metadata or {}).get("mask_created_at"),
            mask_updated_at=_timestamp(),
            inpaint_created_at=(existing_metadata or {}).get("inpaint_created_at"),
            inpaint_updated_at=(existing_metadata or {}).get("inpaint_updated_at"),
            settings=_settings_payload(
                mask_padding=mask_padding,
                use_bubble_mask=use_bubble_mask,
                use_crop_windows=use_crop_windows,
            ),
            strict_ocr_bbox_mask=True,
            model_loaded_at_service_start=False,
            model_reused=False,
            model_reload_count=int((existing_metadata or {}).get("model_reload_count", 0) or 0),
        )
        save_inpaint_json(metadata_path, error_metadata)
        raise ValueError(error_message)

    bubble_guidance = None
    saved_bubble_mask_file: Path | None = None
    bubble_mask_hash = ""
    bubble_mask_pixel_count = 0
    if use_bubble_mask and detection_data is not None:
        bubble_guidance = build_bubble_guidance_mask(
            source_image.shape,
            detection_data,
            project_root=project.root_dir,
        )
        if bubble_guidance is not None:
            save_png_image(bubble_guidance, bubble_mask_file)
            saved_bubble_mask_file = bubble_mask_file
            bubble_mask_hash = _hash_mask_image(bubble_guidance)
            bubble_mask_pixel_count = _count_mask_pixels(bubble_guidance)
        elif bubble_mask_file.exists():
            bubble_mask_file.unlink()
    elif bubble_mask_file.exists():
        bubble_mask_file.unlink()

    text_mask_hash = _hash_mask_image(text_mask)
    save_png_image(text_mask, text_mask_file)
    save_png_image(build_preview_mask(text_mask, bubble_guidance), preview_mask_file)
    current_settings = _settings_payload(
        mask_padding=mask_padding,
        use_bubble_mask=use_bubble_mask,
        use_crop_windows=use_crop_windows,
    )
    reusable_output = _existing_output_matches_current_inputs(
        existing_metadata,
        output_image_file,
        current_settings=current_settings,
        text_mask_hash=text_mask_hash,
        bubble_mask_hash=bubble_mask_hash,
    )
    mask_timestamp = _timestamp()
    created_at = str((existing_metadata or {}).get("created_at", "") or mask_timestamp)
    mask_created_at = str((existing_metadata or {}).get("mask_created_at", "") or mask_timestamp)
    inpaint_created_at = str((existing_metadata or {}).get("inpaint_created_at", "") or "")

    metadata = build_inpaint_metadata(
        project_root=project.root_dir,
        image_relative_path=image_relative,
        ocr_cache_path=ocr_cache_file,
        detection_cache_path=detection_cache_path_value,
        output_image_path_value=output_image_file,
        text_mask_path_value=text_mask_file,
        bubble_mask_path_value=saved_bubble_mask_file,
        text_mask_hash=text_mask_hash,
        bubble_mask_hash=bubble_mask_hash,
        output_mask_hash=(
            text_mask_hash
            if reusable_output
            else str((existing_metadata or {}).get("output_mask_hash", "") or "")
        ),
        output_bubble_mask_hash=(
            bubble_mask_hash
            if reusable_output
            else str((existing_metadata or {}).get("output_bubble_mask_hash", "") or "")
        ),
        image_shape=source_image.shape,
        item_count=len(valid_boxes),
        active_item_count=int(mask_stats.get("active_item_count", 0) or 0),
        text_mask_box_count=int(mask_stats.get("used_ocr_bboxes", 0) or 0),
        skipped_item_count=int(mask_stats.get("skipped_invalid_bbox_count", 0) or 0),
        masked_pixel_count=masked_pixel_count,
        bubble_mask_pixel_count=bubble_mask_pixel_count,
        device=device,
        status="done" if reusable_output else "prepared",
        error="",
        needs_inpaint=not reusable_output,
        created_at=created_at,
        updated_at=mask_timestamp,
        mask_created_at=mask_created_at,
        mask_updated_at=mask_timestamp,
        inpaint_created_at=inpaint_created_at,
        inpaint_updated_at=(existing_metadata or {}).get("inpaint_updated_at"),
        settings=current_settings,
        strict_ocr_bbox_mask=True,
        model_loaded_at_service_start=False,
        model_reused=False,
        model_reload_count=int((existing_metadata or {}).get("model_reload_count", 0) or 0),
    )
    save_inpaint_json(metadata_path, metadata)
    _log(logger, f"Prepared inpaint mask for {Path(image_relative).name}: {text_mask_file}")
    for box_log in list(mask_stats.get("box_logs", []) or []):
        _log(
            logger,
            "Inpaint mask box: "
            f"item_id={box_log.get('item_id')} "
            f"raw_ocr_bbox={box_log.get('raw_ocr_bbox')} "
            f"clamped_ocr_bbox={box_log.get('clamped_ocr_bbox')} "
            f"fallback_used={bool(box_log.get('fallback_used', False))} "
            f"requested_mask_padding={int(box_log.get('requested_mask_padding', 0) or 0)} "
            f"effective_mask_padding={int(box_log.get('effective_mask_padding', 0) or 0)} "
            f"final_mask_bbox={box_log.get('final_mask_bbox')}"
        )
    _log(
        logger,
        "Inpaint mask stats: "
        f"{int(mask_stats.get('active_item_count', 0) or 0)} active items, "
        f"{int(mask_stats.get('used_ocr_bboxes', 0) or 0)} OCR boxes, "
        f"{int(mask_stats.get('fallback_to_bbox_count', 0) or 0)} bbox fallbacks, "
        f"{int(mask_stats.get('skipped_invalid_bbox_count', 0) or 0)} skipped invalid, "
        f"{masked_pixel_count} masked pixels, "
        f"{bubble_mask_pixel_count} bubble-mask pixels.",
    )

    return _PreparedMaskBundle(
        source_image_path=source_image_path,
        ocr_cache_path=ocr_cache_file,
        detection_cache_path=detection_cache_path_value,
        output_image_path=output_image_file,
        metadata_path=metadata_path,
        text_mask_file=text_mask_file,
        bubble_mask_file=saved_bubble_mask_file,
        preview_mask_file=preview_mask_file,
        image_shape=tuple(int(value) for value in source_image.shape),
        item_count=len(valid_boxes),
        masked_pixel_count=masked_pixel_count,
        valid_boxes=valid_boxes,
        metadata=metadata,
    )


def _project_source_path(project: ProjectLike, image_relative_path: str) -> Path:
    return project.root_dir / Path(image_relative_path)


def _import_inpainting():
    try:
        import inpainting
    except Exception as exc:
        raise RuntimeError(
            "LaMa Manga dependencies are not available. Install the inpainting runtime dependencies first."
        ) from exc
    return inpainting


def _settings_payload(
    *,
    mask_padding: int,
    use_bubble_mask: bool,
    use_crop_windows: bool,
) -> dict[str, Any]:
    requested_mask_padding = int(mask_padding)
    return {
        "mask_padding": requested_mask_padding,
        "text_removal_mask_padding": 0,
        "mask_padding_ignored_for_strict_ocr_bbox": bool(requested_mask_padding > 0),
        "use_bubble_mask": bool(use_bubble_mask),
        "use_crop_windows": bool(use_crop_windows),
        "crop_trigger_size": DEFAULT_CROP_TRIGGER_SIZE,
        "crop_margin": DEFAULT_CROP_MARGIN,
        "resize_limit": DEFAULT_RESIZE_LIMIT,
        "strict_ocr_bbox_mask": True,
    }


def _normalize_device_value(device: str | None) -> str | None:
    if device is None:
        return None
    normalized = str(device).strip()
    if not normalized or normalized.lower() == "auto":
        return None
    return normalized


def _normalized_lama_signature(
    *,
    device: str | None,
    crop_trigger_size: int,
    crop_margin: int,
    resize_limit: int,
    pad_mod: int,
    model_path: str | None,
) -> tuple[str | None, str | None, int, int, int, int]:
    normalized_model_path = str(model_path).strip() if model_path is not None and str(model_path).strip() else None
    return (
        normalized_model_path,
        _normalize_device_value(device),
        int(crop_trigger_size),
        int(crop_margin),
        int(resize_limit),
        int(pad_mod),
    )


def _format_lama_signature(signature: tuple[str | None, str | None, int, int, int, int] | None) -> str:
    if signature is None:
        return "unloaded"
    model_path, device, crop_trigger_size, crop_margin, resize_limit, pad_mod = signature
    return (
        f"model_path={model_path or 'default'}, "
        f"device={device or 'auto'}, "
        f"crop_trigger_size={crop_trigger_size}, "
        f"crop_margin={crop_margin}, "
        f"resize_limit={resize_limit}, "
        f"pad_mod={pad_mod}"
    )


def _canon_state_needs_text_mask_backfill(detection_data: dict[str, Any]) -> bool:
    canon_state = detection_data.get("canon_state")
    if not isinstance(canon_state, dict):
        return False
    for item in canon_state.get("items", []):
        if isinstance(item, dict) and "text_mask_bboxes" not in item:
            return True
    return False


def _existing_output_matches_current_inputs(
    metadata: dict[str, Any] | None,
    output_path: Path,
    *,
    current_settings: dict[str, Any],
    text_mask_hash: str,
    bubble_mask_hash: str,
) -> bool:
    if metadata is None or not output_path.exists():
        return False
    if str(metadata.get("status", "") or "").strip().lower() != "done":
        return False
    if bool(metadata.get("needs_inpaint", False)):
        return False
    if str(metadata.get("output_mask_hash", "") or "") != str(text_mask_hash or ""):
        return False
    if str(metadata.get("output_bubble_mask_hash", "") or "") != str(bubble_mask_hash or ""):
        return False
    return _settings_match(metadata.get("settings", {}), current_settings)


def _can_reuse_inpaint_output(
    metadata: dict[str, Any],
    output_path: Path,
    *,
    current_settings: dict[str, Any],
    force: bool,
) -> bool:
    if force:
        return False
    return _existing_output_matches_current_inputs(
        metadata,
        output_path,
        current_settings=current_settings,
        text_mask_hash=str(metadata.get("text_mask_hash", "") or ""),
        bubble_mask_hash=str(metadata.get("bubble_mask_hash", "") or ""),
    )


def _settings_match(existing_settings: Any, current_settings: dict[str, Any]) -> bool:
    if not isinstance(existing_settings, dict):
        return False
    for key, value in current_settings.items():
        if existing_settings.get(key) != value:
            return False
    return True


def _count_mask_pixels(mask: Any | None) -> int:
    if mask is None:
        return 0
    try:
        import numpy as np
    except Exception:
        return 0
    return int(np.count_nonzero(mask))


def _hash_mask_image(mask: Any) -> str:
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("NumPy is required to hash inpaint masks.") from exc
    mask_array = np.asarray(mask)
    hasher = hashlib.sha256()
    hasher.update(str(mask_array.shape).encode("utf-8"))
    hasher.update(mask_array.tobytes())
    return hasher.hexdigest()


def _friendly_inpaint_error(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    lower_message = message.lower()
    if "is not loaded. reload inpaint service" in lower_message:
        return "LaMa Manga is not loaded. Reload Inpaint Service."
    if "runtime model signature mismatch" in lower_message:
        return message
    if "no valid ocr bounding boxes" in lower_message or "no active canon ocr target boxes" in lower_message:
        return "No valid OCR target boxes were available to build an inpaint mask."
    if "ocr cache is missing" in lower_message:
        return message
    if "source image is missing" in lower_message:
        return message
    if "cuda out of memory" in lower_message or "out of memory" in lower_message:
        return "LaMa Manga ran out of GPU memory. Try CPU mode or process fewer pages."
    if "huggingface_hub" in lower_message:
        return "LaMa Manga weights could not be downloaded. Check huggingface_hub and network access."
    if "safetensors" in lower_message or "torch" in lower_message or "numpy" in lower_message:
        return f"LaMa Manga dependencies are missing or failed to load. {message}"
    return message


def _emit_progress(callback: ProgressCallback | None, **payload: Any) -> None:
    if callback is None:
        return
    callback(payload)


def _log(logger: Logger | None, message: str) -> None:
    if logger is not None and message:
        logger(str(message))


def _load_existing_metadata(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return load_inpaint_json(path)
    except Exception:
        return None


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


__all__ = [
    "DEFAULT_CROP_MARGIN",
    "DEFAULT_CROP_TRIGGER_SIZE",
    "DEFAULT_MASK_PADDING",
    "DEFAULT_PAD_MOD",
    "DEFAULT_RESIZE_LIMIT",
    "LamaInpainterManager",
    "get_lama_model_manager",
    "load_lama_model",
    "prepare_inpaint_mask_for_page",
    "run_inpaint_for_page",
    "run_inpaint_for_pages",
    "unload_lama_model",
]
