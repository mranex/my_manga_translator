"""OCR preparation and OCR inference stages backed by on-disk cache files."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .canon_state import (
    CANON_STATE_SCHEMA_VERSION,
    canon_item_bbox,
    ensure_canon_state,
    get_canon_item,
    resolve_canon_item_for_stage_item,
)
from .detection_io import detection_json_path, load_detection_json, save_detection_json
from .image_io import load_image_bgr, project_relative_path, save_png_image
from .ocr_image_preprocess import (
    DEFAULT_IMAGE_PAD,
    DEFAULT_MAX_LONG_SIDE,
    DEFAULT_MAX_UPSCALE_FACTOR,
    DEFAULT_MIN_SHORT_SIDE,
    save_ocr_provider_image,
)
from .deepseek_ocr_client import DeepSeekOCRClientError
from .ocr_models import DEFAULT_OCR_PROVIDER, OCRConfig
from .ocr_io import (
    load_ocr_json,
    normalize_ocr_item,
    ocr_crop_dir_for_page,
    ocr_provider_crop_dir_for_page,
    ocr_json_path,
    save_ocr_items_result,
    save_ocr_payload,
)
from .ocr_items import build_ocr_items_from_canon_state, crop_image_to_bbox
from .paddleocr_vl_client import PaddleOCRVLClientError
from .ocr_providers import OCRProvider, OCRProviderError, create_ocr_provider
from .ocr_text_filter import filter_ocr_text_for_pipeline


ProgressCallback = Callable[[int, int, dict[str, Any]], None]


def prepare_ocr_items_for_image(
    project,
    image_relative_path,
    force: bool = False,
    save_crops: bool = True,
    logger: Callable[[str], None] | None = None,
) -> Path:
    """Prepare OCR items and crop files from cached detection JSON canon_state."""

    relative_path = Path(str(image_relative_path))
    source_image_path = project.root_dir / relative_path
    if not source_image_path.exists():
        raise FileNotFoundError(f"Source image does not exist: {source_image_path}")

    output_json_path = ocr_json_path(project, relative_path)
    crop_dir = ocr_crop_dir_for_page(project, relative_path)

    if not force and output_json_path.exists():
        _log(logger, f"Reusing cached OCR items for {relative_path.name}")
        return output_json_path

    detection_path = detection_json_path(project, relative_path)
    if not detection_path.exists():
        raise RuntimeError(
            f"Detection cache is missing for {relative_path.name}. Run Detection first."
        )

    _log(logger, f"Loading detection cache for OCR prep: {detection_path}")
    detection_data = load_detection_json(detection_path)

    _log(logger, f"Loading source image for OCR prep: {source_image_path.name}")
    image = load_image_bgr(source_image_path)
    existing_canon_schema = _canon_state_schema_version(detection_data.get("canon_state"))
    ensure_canon_state(detection_data, image_shape=image.shape)
    if existing_canon_schema != CANON_STATE_SCHEMA_VERSION:
        save_detection_json(detection_path, detection_data)

    _log(logger, f"Preparing OCR items from canon_state: {relative_path.name}")
    items = build_ocr_items_from_canon_state(detection_data["canon_state"], image.shape, logger=logger)

    if save_crops:
        if crop_dir.exists():
            for existing_crop in crop_dir.glob("*.png"):
                existing_crop.unlink()
        crop_dir.mkdir(parents=True, exist_ok=True)

        for item in items:
            crop_bbox = item.get("ocr_bbox")
            crop_image = crop_image_to_bbox(image, crop_bbox)
            crop_path = crop_dir / f"item_{int(item['id']):03d}.png"
            save_png_image(crop_image, crop_path)
            item["crop_path"] = project_relative_path(project.root_dir, crop_path)
    else:
        for item in items:
            item["crop_path"] = None

    output_path = save_ocr_items_result(
        items,
        image_path=source_image_path,
        detection_cache_path=detection_path,
        image_shape=image.shape,
        output_path=output_json_path,
        project_root=project.root_dir,
    )
    _log(logger, f"Saved OCR cache: {output_path}")
    return output_path


def run_ocr_for_page(
    project,
    image_relative_path,
    server_url: str | None = None,
    *,
    ocr_provider: str = DEFAULT_OCR_PROVIDER,
    provider_config: OCRConfig | dict[str, Any] | None = None,
    provider_instance: OCRProvider | None = None,
    force: bool = False,
    selected_item_ids: Sequence[int] | None = None,
    timeout: float = 120,
    logger: Callable[[str], None] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    """Run OCR inference for one prepared page using the selected OCR provider."""

    relative_path = Path(str(image_relative_path))
    ocr_path = ocr_json_path(project, relative_path)
    if not ocr_path.exists():
        raise RuntimeError(
            f"OCR items are not prepared for {relative_path.name}. Prepare OCR items first."
        )

    detection_path = detection_json_path(project, relative_path)
    if not detection_path.exists():
        raise RuntimeError(
            f"Detection cache is missing for {relative_path.name}. Run Detection first."
        )

    payload = load_ocr_json(ocr_path)
    raw_items = payload.get("items", [])
    if not isinstance(raw_items, list):
        raise ValueError(f"OCR cache field 'items' must be a list in {ocr_path}")

    payload["items"] = [normalize_ocr_item(item) for item in raw_items]
    detection_data = load_detection_json(detection_path)
    existing_canon_schema = _canon_state_schema_version(detection_data.get("canon_state"))
    ensure_canon_state(detection_data)
    if existing_canon_schema != CANON_STATE_SCHEMA_VERSION:
        save_detection_json(detection_path, detection_data)
    canon_state = detection_data["canon_state"]

    source_image_path = project.root_dir / relative_path
    if not source_image_path.exists():
        raise FileNotFoundError(f"Source image does not exist: {source_image_path}")
    source_image = load_image_bgr(source_image_path)
    crop_dir = ocr_crop_dir_for_page(project, relative_path)
    provider_crop_dir = ocr_provider_crop_dir_for_page(project, relative_path)
    crop_dir.mkdir(parents=True, exist_ok=True)
    provider_crop_dir.mkdir(parents=True, exist_ok=True)

    config = OCRConfig.from_value(provider_config)
    config.ocr_provider = str(ocr_provider or config.ocr_provider or DEFAULT_OCR_PROVIDER)
    if server_url is not None:
        config.server_url = str(server_url or "").strip()
    if timeout:
        config.timeout = float(timeout)

    target_item_ids = _normalize_selected_item_ids(selected_item_ids)
    unresolved_item_ids: list[int] = []
    for index, item in enumerate(payload["items"]):
        if not isinstance(item, dict):
            continue
        if target_item_ids is not None and _safe_int(item.get("id")) not in target_item_ids:
            continue
        canon_item = resolve_canon_item_for_stage_item(canon_state, item, active_only=False)
        if canon_item is None:
            unresolved_item_ids.append(_safe_int(item.get("id")) or index)
            continue
        item["canon_id"] = str(canon_item.get("canon_id", "") or "")
        item["bbox"] = canon_item_bbox(canon_item, "bbox")
        item["ocr_bbox"] = canon_item_bbox(canon_item, "ocr_bbox")
        item["excluded"] = not bool(canon_item.get("enabled", True))

    if unresolved_item_ids:
        unresolved_text = ", ".join(str(item_id) for item_id in unresolved_item_ids[:5])
        raise RuntimeError(
            f"OCR cache for {relative_path.name} could not be matched to canon_state "
            f"(item ids: {unresolved_text}). Re-prepare OCR items first."
        )

    items_to_process = [
        item
        for item in payload["items"]
        if not bool(item.get("excluded", False))
        if target_item_ids is None or _safe_int(item.get("id")) in target_item_ids
    ]
    if not items_to_process:
        _log(logger, f"No OCR items selected for {relative_path.name}; nothing to run.")
        return save_ocr_payload(payload, ocr_path)

    total_items = len(items_to_process)
    provider = provider_instance
    created_provider = False
    try:
        if provider is None:
            provider = create_ocr_provider(config)
            created_provider = True
            provider.validate()

        provider_name = getattr(provider, "provider_label", config.provider_label)
        _log(logger, f"Running OCR for {relative_path.name} with {provider_name}")
        _log(logger, f"Processing {total_items} OCR item(s) for {relative_path.name}")
        _log(logger, "Provider image preprocess source: old PaddleOCRVLOCR logic")
        _save_ocr_progress(payload, ocr_path)

        processed_count = 0
        for item in items_to_process:
            item_id = _safe_int(item.get("id"))
            canon_id = str(item.get("canon_id", "") or "").strip()
            if not canon_id:
                raise RuntimeError(
                    f"OCR item {item_id} on {relative_path.name} is missing canon_id. Re-prepare OCR items first."
                )

            canon_item = get_canon_item(canon_state, canon_id)
            if not bool(canon_item.get("enabled", True)):
                item["excluded"] = True
                item["updated_at"] = _timestamp()
                _save_ocr_progress(payload, ocr_path)
                processed_count += 1
                _emit_progress(
                    progress_callback,
                    processed_count,
                    total_items,
                    item_id=item_id,
                    status="skipped",
                    message=f"Skipped OCR item {item_id}: disabled in canon_state.",
                )
                continue

            item["bbox"] = canon_item_bbox(canon_item, "bbox")
            item["ocr_bbox"] = canon_item_bbox(canon_item, "ocr_bbox")
            item["excluded"] = False
            current_status = str(item.get("status", "") or "").strip().lower()
            current_text = str(item.get("text", "") or "").strip()
            needs_ocr = bool(item.get("needs_ocr", False))

            if not force and current_status == "done" and current_text and not needs_ocr:
                processed_count += 1
                _emit_progress(
                    progress_callback,
                    processed_count,
                    total_items,
                    item_id=item_id,
                    status="skipped",
                    message=f"Skipped OCR item {item_id}: already recognized.",
                )
                _log(logger, f"Skipping OCR item {item_id} for {relative_path.name}: already done.")
                continue

            crop_bbox = item.get("ocr_bbox") or item.get("bbox")
            if not isinstance(crop_bbox, (list, tuple)) or len(crop_bbox) < 4:
                item["status"] = "error"
                item["error"] = f"OCR crop bbox is missing for item {item_id}."
                item["updated_at"] = _timestamp()
                item["needs_ocr"] = True
                _apply_provider_metadata(item, provider)
                _save_ocr_progress(payload, ocr_path)
                processed_count += 1
                _emit_progress(
                    progress_callback,
                    processed_count,
                    total_items,
                    item_id=item_id,
                    status="error",
                    message=item["error"],
                )
                _log(logger, f"OCR item {item_id} failed: {item['error']}")
                continue

            crop_image = crop_image_to_bbox(source_image, crop_bbox)
            crop_path = crop_dir / f"item_{item_id:03d}.png"
            save_png_image(crop_image, crop_path)
            item["crop_path"] = project_relative_path(project.root_dir, crop_path)
            _log(
                logger,
                f"Saved original OCR crop for item {item_id}: {crop_path} "
                f"({_image_width(crop_image)}x{_image_height(crop_image)})",
            )

            provider_crop_path = provider_crop_dir / f"item_{item_id:03d}.png"
            save_ocr_provider_image(crop_image, provider_crop_path)
            provider_width, provider_height = _png_image_size(provider_crop_path)
            item["provider_crop_path"] = project_relative_path(project.root_dir, provider_crop_path)
            item["provider_crop_preprocess"] = {
                "source": "ocr/paddleocr_vl_ocr.py::_preprocess_ocr_image",
                "image_pad": DEFAULT_IMAGE_PAD,
                "min_short_side": DEFAULT_MIN_SHORT_SIDE,
                "max_long_side": DEFAULT_MAX_LONG_SIDE,
                "max_upscale_factor": DEFAULT_MAX_UPSCALE_FACTOR,
            }
            _log(
                logger,
                f"Saved provider OCR crop for item {item_id}: {provider_crop_path} "
                f"({provider_width}x{provider_height})",
            )

            item["status"] = "running"
            item["error"] = ""
            item["skip_reason"] = ""
            item["ocr_rejection_reason"] = ""
            item["raw_ocr_text"] = ""
            item["updated_at"] = _timestamp()
            item["needs_ocr"] = False
            _apply_provider_metadata(item, provider)
            _save_ocr_progress(payload, ocr_path)
            _emit_progress(
                progress_callback,
                processed_count,
                total_items,
                item_id=item_id,
                status="running",
                message=f"Running OCR item {item_id}",
            )

            try:
                recognized_text = provider.recognize_image(provider_crop_path)
            except TimeoutError:
                item["status"] = "error"
                item["error"] = _timeout_message(provider, item_id)
                item["skip_reason"] = ""
                item["ocr_rejection_reason"] = ""
                item["raw_ocr_text"] = ""
                item["updated_at"] = _timestamp()
                item["needs_ocr"] = True
                _log(logger, f"OCR item {item_id} failed: {item['error']}")
            except (OCRProviderError, PaddleOCRVLClientError, DeepSeekOCRClientError, FileNotFoundError) as exc:
                item["status"] = "error"
                item["error"] = str(exc)
                item["skip_reason"] = ""
                item["ocr_rejection_reason"] = ""
                item["raw_ocr_text"] = ""
                item["updated_at"] = _timestamp()
                item["needs_ocr"] = True
                _log(logger, f"OCR item {item_id} failed: {exc}")
            except Exception as exc:
                item["status"] = "error"
                item["error"] = str(exc)
                item["skip_reason"] = ""
                item["ocr_rejection_reason"] = ""
                item["raw_ocr_text"] = ""
                item["updated_at"] = _timestamp()
                item["needs_ocr"] = True
                _log(logger, f"OCR item {item_id} failed with an unexpected error: {exc}")
            else:
                filter_result = filter_ocr_text_for_pipeline(
                    recognized_text,
                    provider_key=str(getattr(provider, "provider_key", "") or ""),
                )
                item["error"] = ""
                item["updated_at"] = _timestamp()
                item["needs_ocr"] = False
                if filter_result.rejected:
                    item["raw_ocr_text"] = str(recognized_text or "")
                    item["text"] = ""
                    item["status"] = "skipped"
                    item["skip_reason"] = filter_result.reason
                    item["ocr_rejection_reason"] = filter_result.reason
                    _log(logger, f"OCR item {item_id} skipped: {filter_result.reason}")
                else:
                    filtered_text = filter_result.text
                    item["raw_ocr_text"] = str(recognized_text or "") if filtered_text != str(recognized_text or "") else ""
                    item["text"] = filtered_text
                    item["status"] = "done"
                    item["skip_reason"] = ""
                    item["ocr_rejection_reason"] = ""
                    _log(logger, f"OCR item {item_id} complete.")

            _apply_provider_metadata(item, provider)
            _save_ocr_progress(payload, ocr_path)
            processed_count += 1
            _emit_progress(
                progress_callback,
                processed_count,
                total_items,
                item_id=item_id,
                status=str(item.get("status", "")),
                message=_item_progress_message(item_id, item),
            )
    finally:
        if created_provider:
            provider.close()

    return ocr_path


def _item_progress_message(item_id: int, item: dict[str, Any]) -> str:
    status = str(item.get("status", "") or "prepared")
    error = str(item.get("error", "") or "").strip()
    rejection_reason = str(item.get("ocr_rejection_reason", "") or "").strip()
    if error:
        return f"OCR item {item_id}: {status} ({error})"
    if status == "skipped" and rejection_reason:
        return f"OCR item {item_id}: skipped ({rejection_reason})"
    return f"OCR item {item_id}: {status}"


def _apply_provider_metadata(item: dict[str, Any], provider: OCRProvider) -> None:
    metadata = provider.item_metadata()
    for key, value in metadata.items():
        item[key] = value


def _timeout_message(provider: OCRProvider, item_id: int) -> str:
    provider_key = str(getattr(provider, "provider_key", "") or "")
    if provider_key == "chrome_lens":
        return f"Chrome Lens OCR timed out for item {item_id}"
    return f"OCR request timed out for item {item_id}"


def _normalize_selected_item_ids(selected_item_ids: Sequence[int] | None) -> set[int] | None:
    if selected_item_ids is None:
        return None

    normalized = {_safe_int(item_id) for item_id in selected_item_ids}
    return normalized if normalized else set()


def _save_ocr_progress(payload: dict[str, Any], output_path: Path) -> None:
    try:
        save_ocr_payload(payload, output_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to save OCR cache: {output_path}. {exc}") from exc


def _emit_progress(
    callback: ProgressCallback | None,
    current: int,
    total: int,
    *,
    item_id: int,
    status: str,
    message: str,
) -> None:
    if callback is None:
        return

    callback(
        current,
        total,
        {
            "item_id": item_id,
            "status": status,
            "message": message,
        },
    )


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _image_width(image: Any) -> int:
    shape = getattr(image, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[1])
    size = getattr(image, "size", None)
    if isinstance(size, tuple) and len(size) >= 2:
        return int(size[0])
    return 0


def _image_height(image: Any) -> int:
    shape = getattr(image, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[0])
    size = getattr(image, "size", None)
    if isinstance(size, tuple) and len(size) >= 2:
        return int(size[1])
    return 0


def _png_image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow is required to inspect OCR provider crop size.") from exc

    with Image.open(path) as image:
        width, height = image.size
    return int(width), int(height)


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _log(logger: Callable[[str], None] | None, message: str) -> None:
    if logger is not None:
        logger(message)


def _canon_state_schema_version(value: Any) -> int | None:
    if not isinstance(value, dict):
        return None
    try:
        return int(value.get("schema_version"))
    except Exception:
        return None


__all__ = ["prepare_ocr_items_for_image", "run_ocr_for_page"]
