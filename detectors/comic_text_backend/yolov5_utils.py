from __future__ import annotations

import math
import time

import numpy as np
import pkg_resources as pkg
import torch
import torch.nn as nn
import torchvision


def fuse_conv_and_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    fusedconv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        groups=conv.groups,
        bias=True,
    ).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device)
    if conv.bias is not None:
        b_conv = conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
        torch.sqrt(bn.running_var + bn.eps)
    )
    fusedconv.bias.copy_(
        torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn
    )

    return fusedconv


def check_anchor_order(module) -> None:
    anchor_areas = module.anchors.prod(-1).view(-1)
    delta_anchor = anchor_areas[-1] - anchor_areas[0]
    delta_stride = module.stride[-1] - module.stride[0]
    if delta_anchor.sign() != delta_stride.sign():
        module.anchors[:] = module.anchors.flip(0)


def initialize_weights(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eps = 1e-3
            module.momentum = 0.03
        elif isinstance(
            module,
            (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU),
        ):
            module.inplace = True


def make_divisible(x: float, divisor: int | torch.Tensor) -> int:
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())
    return math.ceil(x / divisor) * divisor


def check_version(
    current: str = "0.0.0",
    minimum: str = "0.0.0",
    pinned: bool = False,
) -> bool:
    current_version, minimum_version = (
        pkg.parse_version(version) for version in (current, minimum)
    )
    return (
        current_version == minimum_version
        if pinned
        else current_version >= minimum_version
    )


def xywh2xyxy(boxes):
    converted = boxes.clone() if isinstance(boxes, torch.Tensor) else np.copy(boxes)
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return converted


def box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)
    inter = (
        torch.min(box1[:, None, 2:], box2[:, 2:])
        - torch.max(box1[:, None, :2], box2[:, :2])
    ).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)


def non_max_suppression(
    prediction,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300,
):
    if isinstance(prediction, np.ndarray):
        prediction = torch.from_numpy(prediction)

    nc = prediction.shape[2] - 5
    xc = prediction[..., 4] > conf_thres

    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    time_limit = 10.0
    start = time.time()

    for image_index, image_pred in enumerate(prediction):
        image_pred = image_pred[xc[image_index]]
        if not image_pred.shape[0]:
            continue

        image_pred[:, 5:] *= image_pred[:, 4:5]
        boxes = xywh2xyxy(image_pred[:, :4])
        conf, classes = image_pred[:, 5:].max(1, keepdim=True)
        image_pred = torch.cat((boxes, conf, classes.float()), 1)[
            conf.view(-1) > conf_thres
        ]

        num_boxes = image_pred.shape[0]
        if not num_boxes:
            continue

        class_offsets = image_pred[:, 5:6] * 4096
        nms_boxes = image_pred[:, :4] + class_offsets
        scores = image_pred[:, 4]
        keep = torchvision.ops.nms(nms_boxes, scores, iou_thres)
        if keep.shape[0] > max_det:
            keep = keep[:max_det]

        output[image_index] = image_pred[keep]
        if (time.time() - start) > time_limit:
            print(f"WARNING: NMS time limit {time_limit}s exceeded")
            break

    return output
