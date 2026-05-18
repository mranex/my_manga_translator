"""GUI-facing wrapper for running and caching the detection stage."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .detection_config import DetectionConfig, detection_configs_match
from .detection_io import load_detection_json, save_detection_result
from .image_io import ensure_path, load_image_bgr


def run_detection_for_image(
    image_path: Path,
    detection_cache_dir: Path,
    masks_cache_dir: Path,
    *,
    force: bool = False,
    detection_config: DetectionConfig | dict | None = None,
    logger: Callable[[str], None] | None = None,
) -> Path:
    """Run the existing page detection pipeline and cache its outputs on disk."""

    source_image_path = ensure_path(image_path)
    detection_dir = ensure_path(detection_cache_dir)
    masks_dir = ensure_path(masks_cache_dir)

    detection_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    detection_json_path = detection_dir / f"{source_image_path.stem}.json"
    page_mask_dir = masks_dir / source_image_path.stem
    project_root = detection_dir.parents[1]
    active_config = DetectionConfig.from_value(detection_config)

    if not force and detection_json_path.exists():
        cached_payload: dict | None = None
        try:
            cached_payload = load_detection_json(detection_json_path)
        except Exception:
            cached_payload = None
        if detection_configs_match(active_config, (cached_payload or {}).get("detection_config")):
            _log(logger, f"Reusing cached detection for {source_image_path.name}")
            return detection_json_path
        _log(logger, f"Detection config changed for {source_image_path.name}. Regenerating detection cache.")

    if active_config.engine != "classic":
        raise RuntimeError(
            "Manga RT-DETR detection requires the resident Detection service. "
            "Open Config and click Load Manga Detector first."
        )

    _log(logger, f"Loading image for detection: {source_image_path.name}")
    image = load_image_bgr(source_image_path)

    _log(logger, f"Running detection: {source_image_path.name}")
    from detectors import detect_page_regions_layout_first

    result = detect_page_regions_layout_first(image)
    output_path = save_detection_result(
        result,
        image_path=source_image_path,
        image_shape=image.shape,
        detection_json_output_path=detection_json_path,
        mask_output_dir=page_mask_dir,
        project_root=project_root,
        detection_config=active_config,
        logger=logger,
    )
    _log(logger, f"Saved detection cache: {output_path}")
    return output_path


def _log(logger: Callable[[str], None] | None, message: str) -> None:
    if logger is not None:
        logger(message)
