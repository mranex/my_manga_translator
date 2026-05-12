from __future__ import annotations

from dataclasses import replace
import importlib
from pathlib import Path
from typing import Any, Iterable, Sequence

from .base import PageDetectionResult, TextRegion
from .matching import assign_text_regions_to_bubbles
from .runtime_utils import (
    clamp_bbox_to_image,
    crop_bbox,
    map_text_region_from_roi_to_page,
    merge_duplicate_text_regions,
)


HF_REPO_ID = "mayocream/comic-text-detector"
WEIGHT_FILENAMES = {
    "yolo_v5": "yolo-v5.safetensors",
    "unet": "unet.safetensors",
    "dbnet": "dbnet.safetensors",
}
REQUIRED_RUNTIME_MODULES = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("numpy", "numpy"),
    ("cv2", "opencv-python / opencv-contrib-python / opencv-python-headless"),
    ("huggingface_hub", "huggingface_hub"),
    ("safetensors.torch", "safetensors"),
]
_TEXT_DETECTOR_CACHE = {
    "key": None,
    "detector": None,
}


class ComicTextDetectorUnavailable(RuntimeError):
    pass


def _default_model_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "model" / "comic_text_detector"


def _check_runtime_dependencies() -> None:
    failures: list[tuple[str, str, str]] = []

    for module_name, package_hint in REQUIRED_RUNTIME_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            failures.append((module_name, package_hint, repr(exc)))

    if failures:
        details = "\n".join(
            f"- import {module_name} failed; install/check {package_hint}: {error}"
            for module_name, package_hint, error in failures
        )
        raise ComicTextDetectorUnavailable(
            "Comic text detector runtime dependencies are not available:\n"
            f"{details}"
        )


def _extract_bbox(raw: Any) -> tuple[int, int, int, int] | None:
    bbox_values: Sequence[Any] | None = None

    if isinstance(raw, dict):
        for key in ("bbox", "box", "xyxy"):
            if key in raw:
                bbox_values = raw[key]
                break
    elif isinstance(raw, (list, tuple)) and len(raw) >= 4:
        bbox_values = raw[:4]

    if bbox_values is None or len(bbox_values) < 4:
        return None

    x1, y1, x2, y2 = bbox_values[:4]
    return (int(x1), int(y1), int(x2), int(y2))


def _extract_confidence(raw: Any) -> float:
    if isinstance(raw, dict):
        for key in ("confidence", "score", "conf"):
            if key in raw:
                return float(raw[key])
        return 1.0

    if isinstance(raw, (list, tuple)) and len(raw) >= 5:
        return float(raw[4])

    return 1.0


def normalize_text_detection(
    raw: Any,
    confidence_threshold: float = 0.3,
) -> TextRegion | None:
    bbox = _extract_bbox(raw)
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None

    confidence = _extract_confidence(raw)
    if confidence < confidence_threshold:
        return None

    text = ""
    mask = None
    reading_order = None
    bubble_id = None

    if isinstance(raw, dict):
        text = str(raw.get("text", ""))
        mask = raw.get("mask")
        reading_order = raw.get("reading_order")
        bubble_id = raw.get("bubble_id")

    return TextRegion(
        bbox=bbox,
        score=confidence,
        mask=mask,
        text=text,
        confidence=confidence,
        bubble_id=bubble_id,
        reading_order=reading_order,
    )


def normalize_text_detections(
    raw_detections: Iterable[Any],
    confidence_threshold: float = 0.3,
) -> list[TextRegion]:
    text_regions: list[TextRegion] = []

    for raw_detection in raw_detections:
        if isinstance(raw_detection, TextRegion):
            region = raw_detection
            if region.confidence < confidence_threshold:
                continue
            if region.bbox[2] <= region.bbox[0] or region.bbox[3] <= region.bbox[1]:
                continue
            text_regions.append(region)
            continue

        normalized = normalize_text_detection(
            raw_detection,
            confidence_threshold=confidence_threshold,
        )
        if normalized is not None:
            text_regions.append(normalized)

    return text_regions


def ensure_comic_text_detector_weights(model_dir=None):
    target_dir = Path(model_dir) if model_dir is not None else _default_model_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    weights: dict[str, Path] = {"model_dir": target_dir}
    missing = {
        key: filename
        for key, filename in WEIGHT_FILENAMES.items()
        if not (target_dir / filename).exists()
    }

    if missing:
        try:
            huggingface_hub = importlib.import_module("huggingface_hub")
        except ImportError as exc:
            raise ComicTextDetectorUnavailable(
                "huggingface_hub is required to download comic text detector weights."
            ) from exc

        for key, filename in missing.items():
            download_path = huggingface_hub.hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
            )
            weights[key] = Path(download_path)

    for key, filename in WEIGHT_FILENAMES.items():
        weights[key] = target_dir / filename

    return weights


def _build_comic_text_backend(
    *,
    weights,
    device: str,
    confidence_threshold: float,
    **kwargs,
):
    try:
        backend_module = importlib.import_module(
            "detectors.comic_text_backend.inference"
        )
    except Exception as exc:
        raise ComicTextDetectorUnavailable(
            "Comic text detector backend import failed:\n"
            "- import detectors.comic_text_backend.inference failed after dependency "
            f"check: {exc!r}"
        ) from exc

    return backend_module.PyTorchComicTextDetectorBackend(
        yolo_weights_path=str(weights["yolo_v5"]),
        unet_weights_path=str(weights["unet"]),
        dbnet_weights_path=str(weights["dbnet"]),
        device=device,
        confidence_threshold=confidence_threshold,
        **kwargs,
    )


class ComicTextDetector:
    def __init__(
        self,
        model_path: str | None = None,
        device: str | None = None,
        confidence_threshold: float = 0.3,
        lazy_load: bool = True,
        **kwargs,
    ):
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = float(confidence_threshold)
        self.lazy_load = lazy_load
        self.kwargs = kwargs
        self._backend = None

        if not self.lazy_load:
            self.load()

    def load(self) -> None:
        if self._backend is not None:
            return

        _check_runtime_dependencies()
        torch = importlib.import_module("torch")

        weights = ensure_comic_text_detector_weights(model_dir=self.model_path)
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._backend = _build_comic_text_backend(
            weights=weights,
            device=device,
            confidence_threshold=self.confidence_threshold,
            **self.kwargs,
        )
        self.device = device

    def detect_text_regions(self, image) -> list[TextRegion]:
        if self._backend is None:
            self.load()

        raw_detections = self._backend.detect(image)
        return normalize_text_detections(
            raw_detections,
            confidence_threshold=self.confidence_threshold,
        )

    def detect_text_regions_in_rois(self, image, layout_rois) -> list[TextRegion]:
        mapped_regions: list[TextRegion] = []
        reading_order = 0

        ordered_rois = sorted(
            enumerate(layout_rois),
            key=lambda item: (
                getattr(item[1], "reading_order", None)
                if getattr(item[1], "reading_order", None) is not None
                else 10**9,
                item[1].bbox[1],
                item[1].bbox[0],
                item[0],
            ),
        )

        for _, layout_roi in ordered_rois:
            roi_bbox = clamp_bbox_to_image(layout_roi.bbox, image.shape)
            if roi_bbox[2] <= roi_bbox[0] or roi_bbox[3] <= roi_bbox[1]:
                continue

            roi_image = crop_bbox(image, roi_bbox)
            local_regions = self.detect_text_regions(roi_image)
            for local_region in local_regions:
                mapped_region = map_text_region_from_roi_to_page(
                    local_region,
                    roi_bbox,
                    image.shape,
                )
                mapped_regions.append(
                    replace(mapped_region, reading_order=reading_order)
                )
                reading_order += 1

        return merge_duplicate_text_regions(
            mapped_regions,
            image_shape=image.shape,
        )

    def detect_page_regions(
        self,
        image,
        bubbles=None,
        assign_to_bubbles: bool = True,
    ) -> PageDetectionResult:
        bubble_regions = list(bubbles) if bubbles is not None else []
        text_regions = self.detect_text_regions(image)

        if assign_to_bubbles and bubble_regions:
            text_regions = assign_text_regions_to_bubbles(
                text_regions,
                bubble_regions,
            )

        return PageDetectionResult(
            bubbles=bubble_regions,
            text_regions=text_regions,
            method="comic_text_detector",
        )


def get_comic_text_detector(
    model_path: str | None = None,
    device: str | None = None,
    confidence_threshold: float = 0.3,
    lazy_load: bool = True,
    **kwargs,
) -> ComicTextDetector:
    cache_key = (
        model_path,
        device,
        float(confidence_threshold),
        bool(lazy_load),
        tuple(sorted(kwargs.items())),
    )
    if _TEXT_DETECTOR_CACHE["key"] != cache_key or _TEXT_DETECTOR_CACHE["detector"] is None:
        _TEXT_DETECTOR_CACHE["key"] = cache_key
        _TEXT_DETECTOR_CACHE["detector"] = ComicTextDetector(
            model_path=model_path,
            device=device,
            confidence_threshold=confidence_threshold,
            lazy_load=lazy_load,
            **kwargs,
        )
    return _TEXT_DETECTOR_CACHE["detector"]


def detect_text_regions_in_rois(
    image,
    layout_rois,
    *,
    detector: ComicTextDetector | None = None,
) -> list[TextRegion]:
    active_detector = detector if detector is not None else get_comic_text_detector()
    return active_detector.detect_text_regions_in_rois(image, layout_rois)


__all__ = [
    "ComicTextDetector",
    "ComicTextDetectorUnavailable",
    "detect_text_regions_in_rois",
    "ensure_comic_text_detector_weights",
    "get_comic_text_detector",
    "normalize_text_detection",
    "normalize_text_detections",
]
