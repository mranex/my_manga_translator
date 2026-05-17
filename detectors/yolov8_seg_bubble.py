from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Sequence

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

from .base import BubbleRegion, PageDetectionResult
from .runtime_utils import (
    clamp_bbox_to_image,
    crop_bbox,
    detect_dark_bubble_from_mask,
    map_bubble_region_from_roi_to_page,
    merge_duplicate_bubble_regions,
    normalize_binary_mask,
)


HF_REPO_ID = "kitsumed/yolov8m_seg-speech-bubble"
MODEL_FILENAME = "model.pt"
RUNTIME_MODULES = [
    ("torch", "torch"),
    ("ultralytics", "ultralytics"),
    ("numpy", "numpy"),
    ("cv2", "opencv-python / opencv-contrib-python / opencv-python-headless"),
    ("huggingface_hub", "huggingface_hub"),
]
_DETECTOR_CACHE = {
    "key": None,
    "detector": None,
}


class YoloSegBubbleDetectorUnavailable(RuntimeError):
    pass


def _default_model_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "model" / "yolov8m_seg_speech_bubble"


def _require_numpy():
    if np is None:
        raise ModuleNotFoundError("numpy is required for YOLOv8 bubble segmentation")
    return np


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
        raise YoloSegBubbleDetectorUnavailable(
            "YOLOv8 speech bubble detector runtime dependencies are not available:\n"
            f"{details}"
        )


def ensure_yolov8_seg_bubble_weights(model_dir=None):
    target_dir = Path(model_dir) if model_dir is not None else _default_model_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    model_path = target_dir / MODEL_FILENAME
    if model_path.exists():
        return {
            "model_dir": target_dir,
            "model_path": model_path,
        }

    try:
        huggingface_hub = importlib.import_module("huggingface_hub")
    except Exception as exc:
        raise YoloSegBubbleDetectorUnavailable(
            "huggingface_hub is required to download YOLOv8 speech bubble weights."
        ) from exc

    download_path = huggingface_hub.hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=MODEL_FILENAME,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
    )
    return {
        "model_dir": target_dir,
        "model_path": Path(download_path),
    }


def _tensor_to_numpy(value: Any) -> np.ndarray:
    np_module = _require_numpy()
    current = value
    if hasattr(current, "detach"):
        current = current.detach()
    if hasattr(current, "cpu"):
        current = current.cpu()
    if hasattr(current, "numpy"):
        current = current.numpy()
    return np_module.asarray(current)


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    np_module = _require_numpy()
    ys, xs = np_module.nonzero(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None

    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def normalize_yolov8_segmentation_result(
    result: Any,
    image: np.ndarray,
) -> list[BubbleRegion]:
    np_module = _require_numpy()
    height, width = image.shape[:2]
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)

    if boxes is None or masks is None or getattr(masks, "data", None) is None:
        return []

    xyxy_values = _tensor_to_numpy(getattr(boxes, "xyxy", []))
    conf_values = _tensor_to_numpy(getattr(boxes, "conf", []))
    cls_values = _tensor_to_numpy(getattr(boxes, "cls", []))
    mask_values = _tensor_to_numpy(masks.data)

    if mask_values.size == 0:
        return []

    if mask_values.ndim == 2:
        mask_values = mask_values[np_module.newaxis, ...]

    bubbles: list[BubbleRegion] = []

    for index, raw_mask in enumerate(mask_values):
        full_mask = normalize_binary_mask(raw_mask, image.shape)
        if not np_module.any(full_mask):
            continue

        bbox: tuple[int, int, int, int] | None = None
        if xyxy_values.ndim >= 2 and index < len(xyxy_values):
            box_values = xyxy_values[index][:4]
            bbox = clamp_bbox_to_image(
                (
                    int(box_values[0]),
                    int(box_values[1]),
                    int(box_values[2]),
                    int(box_values[3]),
                ),
                (height, width),
            )

        if bbox is None or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            bbox = _mask_bbox(full_mask)
            if bbox is None:
                continue
            bbox = clamp_bbox_to_image(bbox, (height, width))

        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue

        score = 1.0
        if conf_values.ndim >= 1 and index < len(conf_values):
            score = float(conf_values[index])

        class_id = None
        if cls_values.ndim >= 1 and index < len(cls_values):
            class_id = int(cls_values[index])

        bubbles.append(
            BubbleRegion(
                bbox=bbox,
                score=score,
                class_id=class_id,
                mask=full_mask.astype(np_module.uint8),
                is_dark=detect_dark_bubble_from_mask(image, full_mask),
            )
        )

    return [
        bubble
        for _, bubble in sorted(
            enumerate(bubbles),
            key=lambda item: (item[1].bbox[1], item[1].bbox[0], item[0]),
        )
    ]


def _ordered_layout_rois(layout_rois):
    return [
        roi
        for _, roi in sorted(
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
    ]


class YoloSegBubbleDetector:
    def __init__(
        self,
        model_path: str | None = None,
        confidence: float = 0.25,
        iou: float = 0.5,
        device: str | None = None,
    ):
        self.model_path = model_path
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.device = device
        self._model = None
        self.last_raw_bubble_count = 0
        self.last_merged_bubble_count = 0

    def _resolve_model_path(self) -> Path:
        if self.model_path:
            candidate = Path(self.model_path)
            if candidate.suffix.lower() == ".pt" and candidate.exists():
                return candidate
            if candidate.suffix.lower() != ".pt":
                weights = ensure_yolov8_seg_bubble_weights(model_dir=candidate)
                return weights["model_path"]

        weights = ensure_yolov8_seg_bubble_weights()
        return weights["model_path"]

    def load(self) -> None:
        if self._model is not None:
            return

        _check_runtime_dependencies()
        torch = importlib.import_module("torch")
        ultralytics = importlib.import_module("ultralytics")

        model_path = self._resolve_model_path()
        device = self.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

        model = ultralytics.YOLO(str(model_path))
        if hasattr(model, "to"):
            model.to(device)

        self._model = model
        self.device = device

    def detect_segmented_bubble_regions(self, image: np.ndarray) -> list[BubbleRegion]:
        if self._model is None:
            self.load()

        results = self._model.predict(
            source=image,
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            verbose=False,
            retina_masks=True,
        )

        bubbles: list[BubbleRegion] = []
        for result in results:
            bubbles.extend(normalize_yolov8_segmentation_result(result, image))

        self.last_raw_bubble_count = len(bubbles)
        bubbles = merge_duplicate_bubble_regions(
            bubbles,
            image_shape=image.shape,
        )
        self.last_merged_bubble_count = len(bubbles)

        return [
            bubble
            for _, bubble in sorted(
                enumerate(bubbles),
                key=lambda item: (item[1].bbox[1], item[1].bbox[0], item[0]),
            )
        ]

    def detect_bubble_regions_in_rois(self, image, layout_rois) -> list[BubbleRegion]:
        mapped_bubbles: list[BubbleRegion] = []

        for layout_roi in _ordered_layout_rois(layout_rois):
            roi_bbox = clamp_bbox_to_image(layout_roi.bbox, image.shape)
            if roi_bbox[2] <= roi_bbox[0] or roi_bbox[3] <= roi_bbox[1]:
                continue

            roi_image = crop_bbox(image, roi_bbox)
            local_bubbles = self.detect_segmented_bubble_regions(roi_image)
            for local_bubble in local_bubbles:
                mapped_bubbles.append(
                    map_bubble_region_from_roi_to_page(
                        local_bubble,
                        roi_bbox,
                        image.shape,
                    )
                )

        return merge_duplicate_bubble_regions(
            mapped_bubbles,
            image_shape=image.shape,
        )

    def detect_page_regions(self, image: np.ndarray) -> PageDetectionResult:
        return PageDetectionResult(
            bubbles=self.detect_segmented_bubble_regions(image),
            layout_regions=[],
            method="yolov8_seg_speech_bubble",
        )


def get_yolov8_seg_bubble_detector(
    model_path: str | None = None,
    confidence: float = 0.25,
    iou: float = 0.5,
    device: str | None = None,
) -> YoloSegBubbleDetector:
    cache_key = (model_path, float(confidence), float(iou), device)
    if _DETECTOR_CACHE["key"] != cache_key or _DETECTOR_CACHE["detector"] is None:
        _DETECTOR_CACHE["key"] = cache_key
        _DETECTOR_CACHE["detector"] = YoloSegBubbleDetector(
            model_path=model_path,
            confidence=confidence,
            iou=iou,
            device=device,
        )
    return _DETECTOR_CACHE["detector"]


def detect_segmented_bubble_regions(
    image: np.ndarray,
    *,
    model_path: str | None = None,
    confidence: float = 0.25,
    iou: float = 0.5,
    device: str | None = None,
) -> list[BubbleRegion]:
    detector = get_yolov8_seg_bubble_detector(
        model_path=model_path,
        confidence=confidence,
        iou=iou,
        device=device,
    )
    return detector.detect_segmented_bubble_regions(image)


def detect_bubble_regions_in_rois(
    image,
    layout_rois,
    *,
    model_path: str | None = None,
    confidence: float = 0.25,
    iou: float = 0.5,
    device: str | None = None,
) -> list[BubbleRegion]:
    detector = get_yolov8_seg_bubble_detector(
        model_path=model_path,
        confidence=confidence,
        iou=iou,
        device=device,
    )
    return detector.detect_bubble_regions_in_rois(image, layout_rois)


def detect_page_regions(
    image: np.ndarray,
    *,
    model_path: str | None = None,
    confidence: float = 0.25,
    iou: float = 0.5,
    device: str | None = None,
) -> PageDetectionResult:
    detector = get_yolov8_seg_bubble_detector(
        model_path=model_path,
        confidence=confidence,
        iou=iou,
        device=device,
    )
    return detector.detect_page_regions(image)


__all__ = [
    "YoloSegBubbleDetector",
    "YoloSegBubbleDetectorUnavailable",
    "detect_bubble_regions_in_rois",
    "detect_page_regions",
    "detect_segmented_bubble_regions",
    "ensure_yolov8_seg_bubble_weights",
    "get_yolov8_seg_bubble_detector",
    "normalize_yolov8_segmentation_result",
]
