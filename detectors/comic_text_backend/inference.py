from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch

from .basemodel import TextDetBase
from .yolov5_utils import non_max_suppression


def letterbox(
    image: np.ndarray,
    new_shape=(1024, 1024),
    color=(0, 0, 0),
    scaleup=True,
):
    shape = image.shape[:2]
    if not isinstance(new_shape, tuple):
        new_shape = (new_shape, new_shape)

    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        ratio = min(ratio, 1.0)

    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw = int(dw)
    dh = int(dh)

    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    image = cv2.copyMakeBorder(image, 0, dh, 0, dw, cv2.BORDER_CONSTANT, value=color)
    return image, (ratio, ratio), (dw, dh)


def preprocess_image(
    image: np.ndarray,
    input_size=(1024, 1024),
    device: str = "cpu",
) -> tuple[torch.Tensor, tuple[int, int]]:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_in, _, (dw, dh) = letterbox(image_rgb, new_shape=input_size)
    image_in = image_in.transpose((2, 0, 1))[::-1]
    image_in = np.ascontiguousarray(image_in)[None].astype(np.float32) / 255.0
    tensor = torch.from_numpy(image_in).to(device)
    return tensor, (dw, dh)


def _resize_map_to_original(
    pred_map: np.ndarray,
    original_shape: tuple[int, int],
    padding: tuple[int, int],
) -> np.ndarray:
    dw, dh = padding
    if dh > 0:
        pred_map = pred_map[: pred_map.shape[0] - dh, :]
    if dw > 0:
        pred_map = pred_map[:, : pred_map.shape[1] - dw]
    original_height, original_width = original_shape
    return cv2.resize(pred_map, (original_width, original_height), interpolation=cv2.INTER_LINEAR)


def _mean_contour_score(prob_map: np.ndarray, contour: np.ndarray) -> float:
    x, y, w, h = cv2.boundingRect(contour)
    if w <= 0 or h <= 0:
        return 0.0
    mask = np.zeros((h, w), dtype=np.uint8)
    shifted = contour.copy()
    shifted[:, 0, 0] -= x
    shifted[:, 0, 1] -= y
    cv2.fillPoly(mask, [shifted], 1)
    region = prob_map[y : y + h, x : x + w]
    return float(cv2.mean(region, mask)[0])


def _extract_line_regions(
    line_pred: torch.Tensor,
    original_shape: tuple[int, int],
    padding: tuple[int, int],
    *,
    text_map_thresh: float,
    confidence_threshold: float,
    min_text_area: float,
) -> list[dict[str, Any]]:
    line_prob_map = line_pred[0, 0].detach().float().cpu().numpy()
    line_prob_map = _resize_map_to_original(line_prob_map, original_shape, padding)

    binary_map = (line_prob_map > text_map_thresh).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    raw_regions: list[dict[str, Any]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_text_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w <= 1 or h <= 1:
            continue

        score = _mean_contour_score(line_prob_map, contour)
        if score < confidence_threshold:
            continue

        raw_regions.append(
            {
                "bbox": [x, y, x + w, y + h],
                "confidence": score,
            }
        )

    raw_regions.sort(key=lambda region: (region["bbox"][1], region["bbox"][0]))
    for index, region in enumerate(raw_regions):
        region["reading_order"] = index

    return raw_regions


def _extract_block_regions(
    block_pred: torch.Tensor,
    original_shape: tuple[int, int],
    input_shape: tuple[int, int],
    padding: tuple[int, int],
    *,
    confidence_threshold: float,
    nms_threshold: float,
) -> list[dict[str, Any]]:
    detections = non_max_suppression(
        block_pred,
        conf_thres=confidence_threshold,
        iou_thres=nms_threshold,
    )[0]
    if detections.device != torch.device("cpu"):
        detections = detections.detach().cpu()
    detections = detections.numpy()

    original_height, original_width = original_shape
    input_height, input_width = input_shape
    dw, dh = padding
    scale_x = original_width / float(input_width - dw)
    scale_y = original_height / float(input_height - dh)

    raw_regions: list[dict[str, Any]] = []
    for index, detection in enumerate(detections):
        x1, y1, x2, y2, score, _ = detection[:6]
        x1 = max(0, min(int(x1 * scale_x), original_width))
        y1 = max(0, min(int(y1 * scale_y), original_height))
        x2 = max(0, min(int(x2 * scale_x), original_width))
        y2 = max(0, min(int(y2 * scale_y), original_height))
        if x2 <= x1 or y2 <= y1:
            continue
        raw_regions.append(
            {
                "bbox": [x1, y1, x2, y2],
                "confidence": float(score),
                "reading_order": index,
            }
        )

    raw_regions.sort(key=lambda region: (region["bbox"][1], region["bbox"][0]))
    for index, region in enumerate(raw_regions):
        region["reading_order"] = index

    return raw_regions


class PyTorchComicTextDetectorBackend:
    def __init__(
        self,
        *,
        yolo_weights_path: str,
        unet_weights_path: str,
        dbnet_weights_path: str,
        device: str = "cpu",
        input_size: int = 1024,
        confidence_threshold: float = 0.3,
        nms_threshold: float = 0.35,
        text_map_threshold: float = 0.3,
        min_text_area: float = 9.0,
        act: str = "leaky",
    ):
        self.device = device
        self.input_size = (input_size, input_size)
        self.confidence_threshold = float(confidence_threshold)
        self.nms_threshold = float(nms_threshold)
        self.text_map_threshold = float(text_map_threshold)
        self.min_text_area = float(min_text_area)

        self.model = TextDetBase(
            yolo_weights_path=yolo_weights_path,
            unet_weights_path=unet_weights_path,
            dbnet_weights_path=dbnet_weights_path,
            device=device,
            act=act,
        ).eval()

    @torch.no_grad()
    def detect(self, image: np.ndarray) -> list[dict[str, Any]]:
        image_tensor, padding = preprocess_image(
            image,
            input_size=self.input_size,
            device=self.device,
        )
        block_pred, _, line_pred = self.model(image_tensor)

        original_shape = image.shape[:2]
        raw_regions = _extract_line_regions(
            line_pred,
            original_shape,
            padding,
            text_map_thresh=self.text_map_threshold,
            confidence_threshold=self.confidence_threshold,
            min_text_area=self.min_text_area,
        )
        if raw_regions:
            return raw_regions

        return _extract_block_regions(
            block_pred,
            original_shape,
            self.input_size,
            padding,
            confidence_threshold=self.confidence_threshold,
            nms_threshold=self.nms_threshold,
        )


__all__ = ["PyTorchComicTextDetectorBackend"]
