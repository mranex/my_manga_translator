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
FIGURE_LIKE_LABEL_PARTS = (
    "figure",
    "image",
    "background",
    "photo",
    "illustration",
    "chart",
    "diagram",
    "graphic",
    "picture",
)
RUNTIME_MODULES = [
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("numpy", "numpy"),
    ("cv2", "opencv-python"),
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
        detector="pp_doclayout_v3",
    )


def is_pp_text_block_label(label: str) -> bool:
    normalized = (label or "").strip().lower()
    return (
        normalized == "content"
        or "text" in normalized
        or "title" in normalized
        or "caption" in normalized
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
                detector="pp_doclayout_v3",
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
            detector=existing.detector or candidate.detector or "pp_doclayout_v3",
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

    def _prepare_rgb_array(self, image):
        np_module = _require_numpy()
        array = np_module.asarray(image)

        if array.size == 0:
            raise ValueError("PPLayout received an empty image array.")

        if array.dtype != np_module.uint8:
            array = np_module.clip(array, 0, 255).astype(np_module.uint8)

        if array.ndim == 2:
            rgb = np_module.stack([array, array, array], axis=-1)
        elif array.ndim == 3:
            channels = int(array.shape[2])
            if channels == 3:
                rgb = array[:, :, [2, 1, 0]]
            elif channels == 4:
                rgb = array[:, :, [2, 1, 0]]
            else:
                raise ValueError(f"Unsupported PPLayout channel count: {array.shape}")
        else:
            raise ValueError(f"Unsupported PPLayout image shape: {array.shape}")

        rgb = np_module.ascontiguousarray(rgb)
        if rgb.dtype != np_module.uint8:
            rgb = rgb.astype(np_module.uint8, copy=False)

        if rgb.ndim != 3 or int(rgb.shape[2]) != 3:
            raise ValueError(f"PPLayout RGB preparation failed for shape: {rgb.shape}")
        return rgb

    def _resolve_processor_size(self) -> tuple[int, int]:
        size_value = getattr(self._image_processor, "size", None)
        default_size = (1024, 1024)

        if isinstance(size_value, dict):
            height = size_value.get("height")
            width = size_value.get("width")
            if height is not None and width is not None:
                return max(1, int(height)), max(1, int(width))
            shortest_edge = size_value.get("shortest_edge")
            if shortest_edge is not None:
                edge = max(1, int(shortest_edge))
                return edge, edge
            longest_edge = size_value.get("longest_edge")
            if longest_edge is not None:
                edge = max(1, int(longest_edge))
                return edge, edge

        if isinstance(size_value, int):
            edge = max(1, int(size_value))
            return edge, edge

        if isinstance(size_value, (list, tuple)):
            if len(size_value) >= 2:
                return max(1, int(size_value[0])), max(1, int(size_value[1]))
            if len(size_value) == 1:
                edge = max(1, int(size_value[0]))
                return edge, edge

        return default_size

    def _resolve_image_mean(self) -> tuple[float, float, float]:
        return self._resolve_normalization_triplet(
            getattr(self._image_processor, "image_mean", None),
            default=(0.485, 0.456, 0.406),
        )

    def _resolve_image_std(self) -> tuple[float, float, float]:
        return self._resolve_normalization_triplet(
            getattr(self._image_processor, "image_std", None),
            default=(0.229, 0.224, 0.225),
        )

    def _resolve_normalization_triplet(
        self,
        value: Any,
        *,
        default: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        np_module = _require_numpy()
        if value is None:
            return default

        if isinstance(value, (int, float)):
            scalar = float(value)
            return scalar, scalar, scalar

        try:
            arr = np_module.asarray(value, dtype=np_module.float32).reshape(-1)
        except Exception:
            return default

        if arr.size == 0:
            return default
        if arr.size == 1:
            scalar = float(arr[0])
            return scalar, scalar, scalar
        if arr.size >= 3:
            return float(arr[0]), float(arr[1]), float(arr[2])
        return default

    def _prepare_model_inputs(self, image) -> tuple[dict[str, Any], tuple[int, int]]:
        np_module = _require_numpy()
        torch = importlib.import_module("torch")
        cv2 = importlib.import_module("cv2")

        _write_pp_layout_breadcrumb("before manual PPLayout preprocess")
        rgb = self._prepare_rgb_array(image)
        original_size = (int(rgb.shape[0]), int(rgb.shape[1]))
        input_h, input_w = self._resolve_processor_size()
        mean = np_module.asarray(self._resolve_image_mean(), dtype=np_module.float32).reshape(1, 1, 3)
        std = np_module.asarray(self._resolve_image_std(), dtype=np_module.float32).reshape(1, 1, 3)

        resized = cv2.resize(rgb, (int(input_w), int(input_h)), interpolation=cv2.INTER_LINEAR)
        resized = np_module.ascontiguousarray(resized)
        arr = resized.astype(np_module.float32) / np_module.float32(255.0)
        arr = (arr - mean) / std
        arr = arr.transpose(2, 0, 1)
        arr = np_module.ascontiguousarray(arr, dtype=np_module.float32)
        pixel_values = torch.from_numpy(arr).unsqueeze(0)

        _write_pp_layout_breadcrumb(
            "after manual PPLayout preprocess",
            tensor_shape=tuple(int(value) for value in pixel_values.shape),
            tensor_dtype=str(pixel_values.dtype),
            input_size=(int(input_h), int(input_w)),
            original_size=original_size,
        )
        return {"pixel_values": pixel_values}, original_size

    def detect_layout_regions(self, image) -> list[LayoutRegion]:
        if self._model is None or self._image_processor is None:
            self.load()

        torch = importlib.import_module("torch")
        inputs, original_size = self._prepare_model_inputs(image)
        target_device = str(self.device or "")
        use_cuda_transfer = target_device.startswith("cuda")

        _write_pp_layout_breadcrumb(
            "before PPLayout input transfer block",
            target_device=target_device,
            input_keys=tuple(str(key) for key in inputs.keys()),
        )
        prepared_inputs: dict[str, Any] = {}
        for key, value in inputs.items():
            if not hasattr(value, "to"):
                prepared_inputs[key] = value
                continue

            _write_pp_layout_breadcrumb(
                "before PPLayout tensor prepare",
                **_tensor_breadcrumb_details(value, key=key, target_device=target_device),
            )
            tensor = value.contiguous() if hasattr(value, "contiguous") else value
            _write_pp_layout_breadcrumb(
                "after PPLayout tensor contiguous",
                **_tensor_breadcrumb_details(tensor, key=key, target_device=target_device),
            )

            if use_cuda_transfer:
                if _tensor_device_type(tensor) == "cpu" and hasattr(tensor, "pin_memory"):
                    _write_pp_layout_breadcrumb(
                        "before PPLayout tensor pin_memory",
                        **_tensor_breadcrumb_details(tensor, key=key, target_device=target_device),
                    )
                    try:
                        tensor = tensor.pin_memory()
                        _write_pp_layout_breadcrumb(
                            "after PPLayout tensor pin_memory",
                            **_tensor_breadcrumb_details(tensor, key=key, target_device=target_device),
                        )
                    except Exception as exc:
                        _write_pp_layout_breadcrumb(
                            "PPLayout tensor pin_memory failed; continuing without pinning",
                            level="warning",
                            error=str(exc),
                            **_tensor_breadcrumb_details(tensor, key=key, target_device=target_device),
                        )

                _write_pp_layout_breadcrumb(
                    "before PPLayout tensor to cuda",
                    **_tensor_breadcrumb_details(tensor, key=key, target_device=target_device),
                )
                tensor = tensor.to(target_device, non_blocking=True)
                _write_pp_layout_breadcrumb(
                    "after PPLayout tensor to cuda",
                    **_tensor_breadcrumb_details(tensor, key=key, target_device=target_device),
                )
                if hasattr(torch, "cuda") and torch.cuda.is_available():
                    _write_pp_layout_breadcrumb(
                        "before PPLayout cuda synchronize after transfer",
                        **_tensor_breadcrumb_details(tensor, key=key, target_device=target_device),
                    )
                    torch.cuda.current_stream(device=tensor.device).synchronize()
                    _write_pp_layout_breadcrumb(
                        "after PPLayout cuda synchronize after transfer",
                        **_tensor_breadcrumb_details(tensor, key=key, target_device=target_device),
                    )
            else:
                _write_pp_layout_breadcrumb(
                    "before PPLayout tensor to device",
                    **_tensor_breadcrumb_details(tensor, key=key, target_device=target_device),
                )
                tensor = tensor.to(target_device)
                _write_pp_layout_breadcrumb(
                    "after PPLayout tensor to device",
                    **_tensor_breadcrumb_details(tensor, key=key, target_device=target_device),
                )

            prepared_inputs[key] = tensor

        _write_pp_layout_breadcrumb(
            "after PPLayout input transfer block",
            target_device=target_device,
            input_devices={key: _tensor_device_string(value) for key, value in prepared_inputs.items()},
        )

        model_param_device = _model_parameter_device(self._model)
        if target_device and model_param_device and model_param_device != target_device and hasattr(self._model, "to"):
            _write_pp_layout_breadcrumb(
                "before PPLayout model.to(self.device)",
                model_device=model_param_device,
                target_device=target_device,
            )
            self._model.to(target_device)
            model_param_device = _model_parameter_device(self._model)
            _write_pp_layout_breadcrumb(
                "after PPLayout model.to(self.device)",
                model_device=model_param_device,
                target_device=target_device,
            )

        _write_pp_layout_breadcrumb(
            "before PPLayout model forward",
            target_device=target_device,
            model_device=model_param_device,
            input_devices={key: _tensor_device_string(value) for key, value in prepared_inputs.items()},
        )
        with torch.inference_mode():
            outputs = self._model(**prepared_inputs)
        _write_pp_layout_breadcrumb(
            "after PPLayout model forward",
            target_device=target_device,
            model_device=model_param_device,
        )

        processed_output = None
        post_process = getattr(self._image_processor, "post_process_object_detection", None)
        if callable(post_process):
            _write_pp_layout_breadcrumb("before PPLayout postprocess", original_size=original_size)
            try:
                target_sizes = torch.tensor([[int(original_size[0]), int(original_size[1])]])
                processed = post_process(
                    outputs,
                    threshold=self.confidence_threshold,
                    target_sizes=target_sizes,
                )
                if isinstance(processed, (list, tuple)) and processed:
                    processed_output = processed[0]
                _write_pp_layout_breadcrumb("after PPLayout postprocess", original_size=original_size)
            except Exception as exc:
                _write_pp_layout_breadcrumb(
                    "PPLayout postprocess failed; falling back to raw outputs",
                    level="warning",
                    error=str(exc),
                )

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


def _write_pp_layout_breadcrumb(message: str, **details: Any) -> None:
    try:
        from mmt_core.crash_logging import write_crash_breadcrumb

        write_crash_breadcrumb(message, detector="pp_doclayout_v3", **details)
    except Exception:
        pass


def _tensor_device_string(value: Any) -> str:
    try:
        device = getattr(value, "device", None)
        if device is None:
            return ""
        return str(device)
    except Exception:
        return ""


def _tensor_device_type(value: Any) -> str:
    try:
        device = getattr(value, "device", None)
        if device is None:
            return ""
        return str(getattr(device, "type", "") or "")
    except Exception:
        return ""


def _tensor_is_pinned(value: Any) -> bool | None:
    try:
        is_pinned = getattr(value, "is_pinned", None)
        if callable(is_pinned):
            return bool(is_pinned())
    except Exception:
        return None
    return None


def _tensor_breadcrumb_details(value: Any, *, key: str, target_device: str) -> dict[str, Any]:
    details: dict[str, Any] = {
        "key": str(key),
        "source_device": _tensor_device_string(value),
        "target_device": str(target_device or ""),
        "shape": tuple(int(dim) for dim in getattr(value, "shape", ()) or ()),
        "dtype": str(getattr(value, "dtype", "")),
        "is_contiguous": bool(value.is_contiguous()) if hasattr(value, "is_contiguous") else None,
    }
    is_pinned = _tensor_is_pinned(value)
    if is_pinned is not None:
        details["is_pinned"] = is_pinned
    return details


def _model_parameter_device(model: Any) -> str:
    try:
        first_param = next(model.parameters())
    except Exception:
        return ""
    return _tensor_device_string(first_param)


__all__ = [
    "MODEL_ID",
    "PPDocLayoutV3Detector",
    "PPDocLayoutV3Unavailable",
    "build_layout_rois",
    "detect_layout_regions",
    "ensure_pp_doclayout_v3_available",
    "get_pp_doclayout_v3_detector",
    "is_pp_text_block_label",
    "normalize_layout_detections",
]
