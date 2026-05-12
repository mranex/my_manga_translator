from __future__ import annotations

import importlib
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

from .base import LayoutRegion
from .matching import bbox_iou
from .runtime_utils import clamp_bbox_to_image


MODEL_ID = "PaddlePaddle/PP-DocLayoutV3_safetensors"
RUNTIME_MODULES = [
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("numpy", "numpy"),
    ("PIL.Image", "pillow"),
]
_DETECTOR_CACHE = {
    "key": None,
    "detector": None,
}


class PPDocLayoutV3Unavailable(RuntimeError):
    pass


def _default_cache_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "model" / "pp_doclayout_v3"


def _require_numpy():
    if np is None:
        raise ModuleNotFoundError("numpy is required for PP-DocLayoutV3 inference")
    return np


def ensure_pp_doclayout_v3_available() -> None:
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
        raise PPDocLayoutV3Unavailable(
            "PP-DocLayoutV3 runtime dependencies are not available:\n"
            f"{details}"
        )


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


def _normalize_optional_sequence(value, index: int):
    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        if index >= len(value):
            return None
        return value[index]

    try:
        return value[index]
    except Exception:
        return None


def _full_page_layout_region(image_shape, *, label: str = "full_page", reading_order: int = 0):
    height = int(image_shape[0])
    width = int(image_shape[1])
    return LayoutRegion(
        bbox=(0, 0, width, height),
        score=1.0,
        label=label,
        label_id=None,
        reading_order=reading_order,
    )


def normalize_layout_detections(
    raw_output: dict[str, Any] | None,
    image_shape: Sequence[int],
    *,
    id2label: dict[int, str] | None = None,
    confidence_threshold: float = 0.25,
    min_region_area: int = 64,
    max_full_page_region_ratio: float = 0.92,
) -> list[LayoutRegion]:
    np_module = _require_numpy()
    if raw_output is None:
        return [_full_page_layout_region(image_shape)]

    page_height = int(image_shape[0])
    page_width = int(image_shape[1])
    page_area = max(page_height * page_width, 1)

    raw_boxes = raw_output.get("boxes", [])
    raw_scores = raw_output.get("scores", [])
    raw_labels = raw_output.get("labels", [])
    raw_polygons = raw_output.get("polygon_points", raw_output.get("polygons"))
    raw_reading_orders = raw_output.get(
        "reading_order",
        raw_output.get("reading_orders"),
    )

    boxes = _tensor_to_numpy(raw_boxes) if raw_boxes is not None else np_module.zeros((0, 4))
    scores = _tensor_to_numpy(raw_scores) if raw_scores is not None else np_module.zeros((0,))
    labels = _tensor_to_numpy(raw_labels) if raw_labels is not None else np_module.zeros((0,))

    if boxes.ndim == 1 and boxes.size >= 4:
        boxes = boxes.reshape(1, -1)

    layout_regions: list[LayoutRegion] = []

    for index, box_values in enumerate(boxes):
        if len(box_values) < 4:
            continue

        score = float(scores[index]) if index < len(scores) else 1.0
        if score < float(confidence_threshold):
            continue

        bbox = clamp_bbox_to_image(
            (
                int(box_values[0]),
                int(box_values[1]),
                int(box_values[2]),
                int(box_values[3]),
            ),
            image_shape,
        )
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue

        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area < int(min_region_area):
            continue

        label_id = int(labels[index]) if index < len(labels) else None
        label = (
            id2label.get(label_id, str(label_id))
            if label_id is not None and id2label is not None
            else (str(label_id) if label_id is not None else "")
        )
        polygon_points = _normalize_optional_sequence(raw_polygons, index)
        reading_order = _normalize_optional_sequence(raw_reading_orders, index)

        if polygon_points is not None:
            try:
                polygon_points = _tensor_to_numpy(polygon_points).tolist()
            except Exception:
                pass

        layout_regions.append(
            LayoutRegion(
                bbox=bbox,
                score=score,
                class_id=label_id,
                label=label,
                label_id=label_id,
                reading_order=None if reading_order is None else int(reading_order),
                polygon_points=polygon_points,
            )
        )

    if len(layout_regions) > 1:
        smaller_regions = [
            region
            for region in layout_regions
            if ((region.bbox[2] - region.bbox[0]) * (region.bbox[3] - region.bbox[1])) / page_area
            <= float(max_full_page_region_ratio)
        ]
        if smaller_regions:
            layout_regions = smaller_regions

    if not layout_regions:
        return [_full_page_layout_region(image_shape)]

    sorted_regions = sorted(
        enumerate(layout_regions),
        key=lambda item: (
            item[1].reading_order if item[1].reading_order is not None else 10**9,
            item[1].bbox[1],
            item[1].bbox[0],
            item[0],
        ),
    )

    normalized_regions: list[LayoutRegion] = []
    for order_index, (_, region) in enumerate(sorted_regions):
        normalized_regions.append(
            replace(
                region,
                reading_order=(
                    region.reading_order
                    if region.reading_order is not None
                    else order_index
                ),
            )
        )

    return normalized_regions


def build_layout_rois(
    layout_regions: Sequence[LayoutRegion],
    image_shape: Sequence[int],
    *,
    padding: int = 24,
    merge_iou: float = 0.35,
) -> list[LayoutRegion]:
    if not layout_regions:
        return [_full_page_layout_region(image_shape)]

    ordered_regions = sorted(
        enumerate(layout_regions),
        key=lambda item: (
            item[1].reading_order if item[1].reading_order is not None else 10**9,
            item[1].bbox[1],
            item[1].bbox[0],
            item[0],
        ),
    )

    merged_rois: list[LayoutRegion] = []
    for _, region in ordered_regions:
        x1, y1, x2, y2 = region.bbox
        expanded_bbox = clamp_bbox_to_image(
            (x1 - padding, y1 - padding, x2 + padding, y2 + padding),
            image_shape,
        )
        candidate = replace(region, bbox=expanded_bbox)

        match_index = None
        best_iou = 0.0
        for index, existing in enumerate(merged_rois):
            current_iou = bbox_iou(existing.bbox, candidate.bbox)
            if current_iou >= float(merge_iou) and current_iou >= best_iou:
                best_iou = current_iou
                match_index = index

        if match_index is None:
            merged_rois.append(candidate)
            continue

        existing = merged_rois[match_index]
        merged_rois[match_index] = LayoutRegion(
            bbox=(
                min(existing.bbox[0], candidate.bbox[0]),
                min(existing.bbox[1], candidate.bbox[1]),
                max(existing.bbox[2], candidate.bbox[2]),
                max(existing.bbox[3], candidate.bbox[3]),
            ),
            score=max(existing.score, candidate.score),
            class_id=existing.class_id if existing.class_id is not None else candidate.class_id,
            label=existing.label or candidate.label,
            label_id=existing.label_id if existing.label_id is not None else candidate.label_id,
            reading_order=min(
                value
                for value in (
                    existing.reading_order,
                    candidate.reading_order,
                )
                if value is not None
            )
            if (existing.reading_order is not None or candidate.reading_order is not None)
            else None,
            polygon_points=existing.polygon_points or candidate.polygon_points,
        )

    if not merged_rois:
        return [_full_page_layout_region(image_shape)]

    return [
        replace(
            region,
            reading_order=(
                region.reading_order if region.reading_order is not None else index
            ),
        )
        for index, region in enumerate(
            sorted(
                merged_rois,
                key=lambda region: (
                    region.reading_order if region.reading_order is not None else 10**9,
                    region.bbox[1],
                    region.bbox[0],
                ),
            )
        )
    ]


class PPDocLayoutV3Detector:
    def __init__(
        self,
        model_id: str = MODEL_ID,
        cache_dir: str = "model/pp_doclayout_v3",
        confidence_threshold: float = 0.25,
        device: str | None = None,
        roi_padding: int = 24,
        min_region_area: int = 64,
        max_full_page_region_ratio: float = 0.92,
    ):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.confidence_threshold = float(confidence_threshold)
        self.device = device
        self.roi_padding = int(roi_padding)
        self.min_region_area = int(min_region_area)
        self.max_full_page_region_ratio = float(max_full_page_region_ratio)
        self._image_processor = None
        self._model = None

    def load(self) -> None:
        if self._image_processor is not None and self._model is not None:
            return

        ensure_pp_doclayout_v3_available()
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")

        cache_dir = str(Path(self.cache_dir))
        device = self.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

        self._image_processor = transformers.AutoImageProcessor.from_pretrained(
            self.model_id,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        self._model = transformers.AutoModelForObjectDetection.from_pretrained(
            self.model_id,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        if hasattr(self._model, "to"):
            self._model.to(device)
        if hasattr(self._model, "eval"):
            self._model.eval()
        self.device = device

    def _prepare_pil_image(self, image):
        _require_numpy()
        pil_module = importlib.import_module("PIL.Image")
        if image.ndim == 2:
            return pil_module.fromarray(image)
        return pil_module.fromarray(image[..., ::-1])

    def detect_layout_regions(self, image) -> list[LayoutRegion]:
        _require_numpy()
        if self._model is None or self._image_processor is None:
            self.load()

        torch = importlib.import_module("torch")
        pil_image = self._prepare_pil_image(image)

        inputs = self._image_processor(images=pil_image, return_tensors="pt")
        prepared_inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = self._model(**prepared_inputs)

        processed_output = None
        post_process = getattr(self._image_processor, "post_process_object_detection", None)
        if callable(post_process):
            target_sizes = torch.tensor([pil_image.size[::-1]])
            processed = post_process(
                outputs,
                threshold=self.confidence_threshold,
                target_sizes=target_sizes,
            )
            if isinstance(processed, (list, tuple)) and processed:
                processed_output = processed[0]

        if processed_output is None:
            processed_output = {
                "boxes": getattr(outputs, "boxes", []),
                "scores": getattr(outputs, "scores", []),
                "labels": getattr(outputs, "labels", []),
                "polygon_points": getattr(outputs, "polygon_points", None),
                "reading_order": getattr(outputs, "reading_order", None),
            }

        id2label = getattr(getattr(self._model, "config", None), "id2label", None)
        return normalize_layout_detections(
            processed_output,
            image.shape,
            id2label=id2label,
            confidence_threshold=self.confidence_threshold,
            min_region_area=self.min_region_area,
            max_full_page_region_ratio=self.max_full_page_region_ratio,
        )


def get_pp_doclayout_v3_detector(
    model_id: str = MODEL_ID,
    cache_dir: str = "model/pp_doclayout_v3",
    confidence_threshold: float = 0.25,
    device: str | None = None,
    roi_padding: int = 24,
    min_region_area: int = 64,
    max_full_page_region_ratio: float = 0.92,
) -> PPDocLayoutV3Detector:
    cache_key = (
        model_id,
        cache_dir,
        float(confidence_threshold),
        device,
        int(roi_padding),
        int(min_region_area),
        float(max_full_page_region_ratio),
    )
    if _DETECTOR_CACHE["key"] != cache_key or _DETECTOR_CACHE["detector"] is None:
        _DETECTOR_CACHE["key"] = cache_key
        _DETECTOR_CACHE["detector"] = PPDocLayoutV3Detector(
            model_id=model_id,
            cache_dir=cache_dir,
            confidence_threshold=confidence_threshold,
            device=device,
            roi_padding=roi_padding,
            min_region_area=min_region_area,
            max_full_page_region_ratio=max_full_page_region_ratio,
        )
    return _DETECTOR_CACHE["detector"]


def detect_layout_regions(
    image,
    *,
    detector: PPDocLayoutV3Detector | None = None,
) -> list[LayoutRegion]:
    active_detector = detector if detector is not None else get_pp_doclayout_v3_detector()
    return active_detector.detect_layout_regions(image)


__all__ = [
    "MODEL_ID",
    "PPDocLayoutV3Detector",
    "PPDocLayoutV3Unavailable",
    "build_layout_rois",
    "detect_layout_regions",
    "ensure_pp_doclayout_v3_available",
    "get_pp_doclayout_v3_detector",
    "normalize_layout_detections",
]
