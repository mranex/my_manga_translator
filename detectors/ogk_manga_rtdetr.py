from __future__ import annotations

import gc
import importlib
from pathlib import Path
import threading
from typing import Any, Callable, Sequence

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

from mmt_core.detection_config import DEFAULT_MANGA_MODEL_ID, DetectionConfig

from .base import BubbleRegion, LayoutRegion, PageDetectionResult
from .runtime_utils import clamp_bbox_to_image


DETECTOR_NAME = "ogk_manga_rtdetr"
RUNTIME_MODULES = [
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("numpy", "numpy"),
]
_BUBBLE_LABEL_PARTS = ("bubble", "balloon", "speech")
_LAYOUT_LABEL_PARTS = ("text", "caption", "letter")

Logger = Callable[[str], None] | None


class OgkMangaRTDetrUnavailable(RuntimeError):
    pass


def _default_cache_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "model" / DETECTOR_NAME


def _require_numpy():
    if np is None:
        raise ModuleNotFoundError("numpy is required for OGK Manga RT-DETR inference")
    return np


def _log(logger: Logger, message: str) -> None:
    if logger is not None:
        logger(str(message or ""))


def _check_runtime_dependencies() -> None:
    failures: list[tuple[str, str, str]] = []
    for module_name, package_hint in RUNTIME_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            failures.append((module_name, package_hint, repr(exc)))

    if failures:
        details = "\n".join(
            f"- import {module_name} failed; install/check {package_hint}: {error}"
            for module_name, package_hint, error in failures
        )
        raise OgkMangaRTDetrUnavailable(
            "OGK Manga RT-DETR runtime dependencies are not available:\n"
            f"{details}"
        )


def _normalize_device(preferred_device: str) -> str:
    torch = importlib.import_module("torch")
    normalized = str(preferred_device or "auto").strip().lower()
    if normalized in {"", "auto"}:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if normalized == "cuda" and torch.cuda.is_available():
        return "cuda:0"
    return str(preferred_device or "cpu").strip() or "cpu"


def _label_key(label: str) -> str:
    return str(label or "").strip().lower().replace("-", "_").replace(" ", "_")


def _resolve_label_role(label: str) -> str:
    normalized = _label_key(label)
    if any(part in normalized for part in _LAYOUT_LABEL_PARTS):
        return "layout"
    if any(part in normalized for part in _BUBBLE_LABEL_PARTS):
        return "bubble"
    return ""


def _normalize_id2label(value: Any) -> dict[int, str]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[int, str] = {}
    for raw_key, raw_label in value.items():
        try:
            label_id = int(raw_key)
        except Exception:
            continue
        normalized[label_id] = str(raw_label or "").strip()
    return normalized


def _normalize_label_overrides(labels: Sequence[str]) -> set[str]:
    return {
        _label_key(label)
        for label in labels
        if str(label or "").strip()
    }


def _tensor_to_numpy(value: Any):
    np_module = _require_numpy()
    current = value
    if hasattr(current, "detach"):
        current = current.detach()
    if hasattr(current, "cpu"):
        current = current.cpu()
    if hasattr(current, "numpy"):
        current = current.numpy()
    return np_module.asarray(current)


def _prepare_rgb_array(image: Any):
    np_module = _require_numpy()
    array = np_module.asarray(image)
    if array.size == 0:
        raise ValueError("OGK Manga RT-DETR received an empty image array.")
    if array.dtype != np_module.uint8:
        array = np_module.clip(array, 0, 255).astype(np_module.uint8)
    if array.ndim == 2:
        return np_module.stack([array, array, array], axis=-1)
    if array.ndim == 3:
        channels = int(array.shape[2])
        if channels >= 3:
            return np_module.ascontiguousarray(array[:, :, [2, 1, 0]])
    raise ValueError(f"Unsupported OGK Manga RT-DETR image shape: {array.shape}")


def _sort_regions(regions: Sequence[Any]) -> list[Any]:
    return [
        region
        for _, region in sorted(
            enumerate(regions),
            key=lambda item: (item[1].bbox[1], item[1].bbox[0], item[0]),
        )
    ]


def _resolve_label_roles(
    id2label: dict[int, str],
    config: DetectionConfig,
) -> dict[int, str]:
    bubble_overrides = _normalize_label_overrides(config.manga_bubble_labels)
    layout_overrides = _normalize_label_overrides(config.manga_text_labels)
    roles: dict[int, str] = {}

    for label_id, label in id2label.items():
        label_key = _label_key(label)
        if label_key in layout_overrides:
            roles[label_id] = "layout"
            continue
        if label_key in bubble_overrides:
            roles[label_id] = "bubble"
            continue
        inferred = _resolve_label_role(label)
        if inferred:
            roles[label_id] = inferred

    if not any(role == "bubble" for role in roles.values()) or not any(role == "layout" for role in roles.values()):
        labels_text = ", ".join(
            f"{label_id}:{label}"
            for label_id, label in sorted(id2label.items())
        ) or "<empty>"
        raise RuntimeError(
            "Could not infer OGK Manga detector label mapping from id2label: "
            f"{labels_text}"
        )
    return roles


def _clear_cuda_cache() -> None:
    try:
        torch = importlib.import_module("torch")
    except Exception:
        return
    if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


class OgkMangaRTDetrDetector:
    def __init__(self, *, cache_dir: str | Path | None = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
        self.model_id = DEFAULT_MANGA_MODEL_ID
        self.device = "cpu"
        self.confidence_threshold = 0.35
        self._image_processor = None
        self._model = None
        self._config_signature: tuple[Any, ...] | None = None
        self._id2label: dict[int, str] = {}
        self._label_roles: dict[int, str] = {}

    def load(self, config: DetectionConfig, *, logger: Logger = None) -> None:
        active_config = DetectionConfig.from_value(config)
        if self._model is not None and self._image_processor is not None:
            if self._config_signature == active_config.signature():
                return
            raise RuntimeError("OGK Manga RT-DETR detector is already loaded with a different config.")

        _check_runtime_dependencies()
        transformers = importlib.import_module("transformers")

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        resolved_device = _normalize_device(active_config.manga_device)
        _log(logger, f"Loading OGK Manga RT-DETR detector: {active_config.manga_model_id}")

        image_processor = transformers.AutoImageProcessor.from_pretrained(
            active_config.manga_model_id,
            cache_dir=str(self.cache_dir),
        )
        model = transformers.AutoModelForObjectDetection.from_pretrained(
            active_config.manga_model_id,
            cache_dir=str(self.cache_dir),
        )
        if hasattr(model, "to"):
            model.to(resolved_device)
        if hasattr(model, "eval"):
            model.eval()

        id2label = _normalize_id2label(getattr(getattr(model, "config", None), "id2label", {}))
        label_roles = _resolve_label_roles(id2label, active_config)

        self._image_processor = image_processor
        self._model = model
        self._id2label = id2label
        self._label_roles = label_roles
        self._config_signature = active_config.signature()
        self.model_id = active_config.manga_model_id
        self.device = resolved_device
        self.confidence_threshold = active_config.manga_confidence_threshold
        _log(
            logger,
            "OGK Manga RT-DETR detector loaded with labels: "
            + ", ".join(f"{label_id}:{label}" for label_id, label in sorted(id2label.items())),
        )

    def unload(self) -> None:
        self._image_processor = None
        self._model = None
        self._id2label = {}
        self._label_roles = {}
        self._config_signature = None
        gc.collect()
        _clear_cuda_cache()

    def detect_page(self, image: Any, config: DetectionConfig) -> PageDetectionResult:
        active_config = DetectionConfig.from_value(config)
        if self._model is None or self._image_processor is None or self._config_signature is None:
            raise RuntimeError("OGK Manga RT-DETR detector is not loaded.")
        if self._config_signature != active_config.signature():
            raise RuntimeError("OGK Manga RT-DETR detector config changed.")

        np_module = _require_numpy()
        torch = importlib.import_module("torch")
        rgb = _prepare_rgb_array(image)
        inputs = self._image_processor(images=rgb, return_tensors="pt")
        prepared_inputs = {
            key: (value.to(self.device) if hasattr(value, "to") else value)
            for key, value in dict(inputs).items()
        }

        with torch.inference_mode():
            outputs = self._model(**prepared_inputs)

        post_process = getattr(self._image_processor, "post_process_object_detection", None)
        if not callable(post_process):
            raise RuntimeError("OGK Manga RT-DETR image processor does not expose object-detection post-processing.")

        target_sizes = torch.tensor([[int(rgb.shape[0]), int(rgb.shape[1])]])
        processed = post_process(
            outputs,
            threshold=float(active_config.manga_confidence_threshold),
            target_sizes=target_sizes,
        )
        processed_output = processed[0] if isinstance(processed, (list, tuple)) and processed else {}
        boxes = _tensor_to_numpy(processed_output.get("boxes", []))
        scores = _tensor_to_numpy(processed_output.get("scores", []))
        labels = _tensor_to_numpy(processed_output.get("labels", []))

        if boxes.ndim == 1 and boxes.size >= 4:
            boxes = boxes.reshape(1, -1)

        bubbles: list[BubbleRegion] = []
        layout_regions: list[LayoutRegion] = []

        for index, raw_box in enumerate(boxes):
            if len(raw_box) < 4:
                continue
            label_id = int(labels[index]) if index < len(labels) else None
            if label_id is None:
                continue
            role = self._label_roles.get(label_id, "")
            if not role:
                continue
            bbox = clamp_bbox_to_image(
                (
                    int(round(float(raw_box[0]))),
                    int(round(float(raw_box[1]))),
                    int(round(float(raw_box[2]))),
                    int(round(float(raw_box[3]))),
                ),
                rgb.shape,
            )
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            score = float(scores[index]) if index < len(scores) else 1.0
            label = self._id2label.get(label_id, str(label_id))
            if role == "bubble":
                bubbles.append(
                    BubbleRegion(
                        bbox=bbox,
                        score=score,
                        class_id=label_id,
                        mask=None,
                        detector=DETECTOR_NAME,
                    )
                )
                continue
            layout_regions.append(
                LayoutRegion(
                    bbox=bbox,
                    score=score,
                    class_id=label_id,
                    label=label,
                    label_id=label_id,
                    detector=DETECTOR_NAME,
                )
            )

        return PageDetectionResult(
            bubbles=_sort_regions(bubbles),
            layout_regions=_sort_regions(layout_regions),
            method=DETECTOR_NAME,
            stats={
                "engine": "manga_rtdetr",
                "model_id": self.model_id,
                "device": self.device,
                "confidence_threshold": float(active_config.manga_confidence_threshold),
                "raw_detections": int(len(boxes)) if hasattr(boxes, "__len__") else 0,
                "bubbles": len(bubbles),
                "layout_regions": len(layout_regions),
                "labels": [self._id2label[key] for key in sorted(self._id2label)],
            },
        )


class OgkMangaRTDetrManager:
    def __init__(self) -> None:
        self._detector: OgkMangaRTDetrDetector | None = None
        self._loaded = False
        self._config_signature: tuple[Any, ...] | None = None
        self._loaded_config: dict[str, Any] | None = None
        self._model_id = DEFAULT_MANGA_MODEL_ID
        self._device = "auto"
        self._confidence_threshold = 0.35
        self._last_device = ""
        self._load_count = 0
        self._reload_count = 0
        self._busy = False
        self._last_error = ""
        self._lock = threading.RLock()

    def load_once(self, config: DetectionConfig, *, logger: Logger = None) -> dict[str, Any]:
        active_config = DetectionConfig.from_value(config)
        with self._lock:
            signature = active_config.signature()
            if self._loaded and self._config_signature == signature and self._detector is not None:
                return {
                    "loaded": True,
                    "device": self._last_device or self._device,
                    "model_id": self._model_id,
                    "message": f"OGK Manga RT-DETR detector already loaded on {self._last_device or self._device}.",
                    "reused": True,
                    "reloaded": False,
                }
            if self._loaded and self._config_signature != signature:
                raise RuntimeError(
                    "Manga RT-DETR detector config changed. Reload Manga Detector before running."
                )
            if self._busy:
                raise RuntimeError("Manga RT-DETR detector is busy.")
            detector = self._detector if self._detector is not None else OgkMangaRTDetrDetector()
            try:
                detector.load(active_config, logger=logger)
            except Exception as exc:
                self._last_error = str(exc)
                raise
            self._detector = detector
            self._loaded = True
            self._config_signature = signature
            self._loaded_config = active_config.to_settings_dict()
            self._model_id = active_config.manga_model_id
            self._device = active_config.manga_device
            self._confidence_threshold = active_config.manga_confidence_threshold
            self._last_device = str(detector.device or self._device or "auto")
            self._last_error = ""
            self._load_count += 1
            return {
                "loaded": True,
                "device": self._last_device,
                "model_id": self._model_id,
                "detection_config": dict(self._loaded_config or active_config.to_settings_dict()),
                "error": "",
                "message": f"OGK Manga RT-DETR detector loaded on {self._last_device}.",
                "reused": False,
                "reloaded": False,
            }

    def reload(self, config: DetectionConfig, *, logger: Logger = None) -> dict[str, Any]:
        with self._lock:
            if self._busy:
                raise RuntimeError("Manga RT-DETR detector is busy.")
            had_loaded = self._loaded or self._detector is not None
            self._unload_locked()
            result = self.load_once(config, logger=logger)
            if had_loaded:
                self._reload_count += 1
            result["reloaded"] = bool(had_loaded)
            result["message"] = f"OGK Manga RT-DETR detector reloaded on {result.get('device', 'auto')}."
            return result

    def unload(self) -> dict[str, Any]:
        with self._lock:
            if self._busy:
                return {
                    "loaded": self._loaded,
                    "device": self._last_device or self._device,
                    "model_id": self._model_id,
                    "error": self._last_error,
                    "message": "Manga RT-DETR detector is busy and cannot be unloaded right now.",
                }
            had_loaded = self._loaded or self._detector is not None
            self._unload_locked()
            return {
                "loaded": False,
                "device": self._last_device or "",
                "model_id": self._model_id,
                "detection_config": {},
                "error": "",
                "message": (
                    "OGK Manga RT-DETR detector unloaded."
                    if had_loaded
                    else "OGK Manga RT-DETR detector was not loaded."
                ),
            }

    def status(self) -> dict[str, Any]:
        with self._lock:
            if self._loaded and self._detector is not None:
                message = f"OGK Manga RT-DETR detector loaded on {self._last_device or self._device or 'auto'}."
            elif self._last_error:
                message = self._last_error
            else:
                message = "OGK Manga RT-DETR detector is not loaded."
            return {
                "loaded": self._loaded and self._detector is not None,
                "busy": self._busy,
                "device": self._last_device or self._device,
                "model_id": self._model_id,
                "confidence_threshold": float(self._confidence_threshold),
                "detection_config": dict(self._loaded_config or {}),
                "error": self._last_error,
                "load_count": self._load_count,
                "reload_count": self._reload_count,
                "message": message,
            }

    def set_error(self, message: str) -> None:
        with self._lock:
            self._last_error = str(message or "").strip()

    def detect_page(
        self,
        image: Any,
        config: DetectionConfig,
        *,
        logger: Logger = None,
    ) -> PageDetectionResult:
        active_config = DetectionConfig.from_value(config)
        with self._lock:
            if not self._loaded or self._detector is None or self._config_signature is None:
                raise RuntimeError(
                    "Manga RT-DETR detector is not loaded. Open Config and click Load Manga Detector first."
                )
            if self._config_signature != active_config.signature():
                raise RuntimeError(
                    "Manga RT-DETR detector config changed. Reload Manga Detector in Config before running Detection."
                )
            if self._busy:
                raise RuntimeError("Manga RT-DETR detector is busy.")
            self._busy = True
            detector = self._detector
        try:
            _log(logger, f"Running OGK Manga RT-DETR detection on {self._last_device or detector.device}.")
            return detector.detect_page(image, active_config)
        finally:
            with self._lock:
                self._busy = False

    def _unload_locked(self) -> None:
        detector = self._detector
        self._detector = None
        self._loaded = False
        self._config_signature = None
        self._loaded_config = None
        self._last_error = ""
        if detector is not None:
            detector.unload()
        gc.collect()
        _clear_cuda_cache()


__all__ = [
    "DETECTOR_NAME",
    "OgkMangaRTDetrDetector",
    "OgkMangaRTDetrManager",
    "OgkMangaRTDetrUnavailable",
]
