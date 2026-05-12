from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from .yolov5_common import C3, Concat, Conv, DWConv, SPPF
from .yolov5_utils import (
    check_anchor_order,
    check_version,
    fuse_conv_and_bn,
    initialize_weights,
    make_divisible,
)


class Detect(nn.Module):
    stride = None
    onnx_dynamic = False

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        super().__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.nl
        self.anchor_grid = [torch.zeros(1)] * self.nl
        self.register_buffer(
            "anchors",
            torch.tensor(anchors).float().view(self.nl, -1, 2),
        )
        self.m = nn.ModuleList(
            nn.Conv2d(channels, self.no * self.na, 1) for channels in ch
        )
        self.inplace = inplace

    def forward(self, x):
        outputs = []
        for index in range(self.nl):
            x[index] = self.m[index](x[index])
            batch_size, _, ny, nx = x[index].shape
            x[index] = (
                x[index]
                .view(batch_size, self.na, self.no, ny, nx)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )

            if not self.training:
                if self.onnx_dynamic or self.grid[index].shape[2:4] != x[index].shape[2:4]:
                    self.grid[index], self.anchor_grid[index] = self._make_grid(nx, ny, index)

                pred = x[index].sigmoid()
                if self.inplace:
                    pred[..., 0:2] = (
                        (pred[..., 0:2] * 2 - 0.5 + self.grid[index]) * self.stride[index]
                    )
                    pred[..., 2:4] = (
                        (pred[..., 2:4] * 2) ** 2 * self.anchor_grid[index]
                    )
                else:
                    xy = (
                        (pred[..., 0:2] * 2 - 0.5 + self.grid[index]) * self.stride[index]
                    )
                    wh = (pred[..., 2:4] * 2) ** 2 * self.anchor_grid[index]
                    pred = torch.cat((xy, wh, pred[..., 4:]), -1)
                outputs.append(pred.view(batch_size, -1, self.no))

        return x if self.training else (torch.cat(outputs, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        device = self.anchors[i].device
        if check_version(torch.__version__, "1.10.0"):
            yv, xv = torch.meshgrid(
                [torch.arange(ny, device=device), torch.arange(nx, device=device)],
                indexing="ij",
            )
        else:
            yv, xv = torch.meshgrid(
                [torch.arange(ny, device=device), torch.arange(nx, device=device)]
            )
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (
            self.anchors[i].clone() * self.stride[i]
        ).view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid


class Model(nn.Module):
    def __init__(self, cfg, ch=3, nc=None, anchors=None):
        super().__init__()
        self.out_indices = None
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            import yaml

            self.yaml_file = Path(cfg).name
            with open(cfg, encoding="ascii", errors="ignore") as file_handle:
                self.yaml = yaml.safe_load(file_handle)

        ch = self.yaml["ch"] = self.yaml.get("ch", ch)
        if nc and nc != self.yaml["nc"]:
            self.yaml["nc"] = nc
        if anchors:
            self.yaml["anchors"] = round(anchors)
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])
        self.names = [str(index) for index in range(self.yaml["nc"])]
        self.inplace = self.yaml.get("inplace", True)

        detect_layer = self.model[-1]
        if isinstance(detect_layer, Detect):
            stride_probe = 256
            detect_layer.inplace = self.inplace
            detect_layer.stride = torch.tensor(
                [
                    stride_probe / output.shape[-2]
                    for output in self.forward(torch.zeros(1, ch, stride_probe, stride_probe))
                ]
            )
            detect_layer.anchors /= detect_layer.stride.view(-1, 1, 1)
            check_anchor_order(detect_layer)
            self.stride = detect_layer.stride
            self._initialize_biases()

        initialize_weights(self)

    def forward(self, x, detect=False):
        return self._forward_once(x, detect=detect)

    def _forward_once(self, x, detect=False):
        saved_outputs = []
        indexed_outputs = []
        for module in self.model:
            if module.f != -1:
                x = (
                    saved_outputs[module.f]
                    if isinstance(module.f, int)
                    else [x if index == -1 else saved_outputs[index] for index in module.f]
                )
            x = module(x)
            saved_outputs.append(x if module.i in self.save else None)
            if self.out_indices is not None and module.i in self.out_indices:
                indexed_outputs.append(x)

        if self.out_indices is not None:
            if detect:
                return x, indexed_outputs
            return indexed_outputs
        return x

    def _initialize_biases(self, cf=None):
        detect_layer = self.model[-1]
        for conv, stride in zip(detect_layer.m, detect_layer.stride):
            bias = conv.bias.view(detect_layer.na, -1)
            bias.data[:, 4] += math.log(8 / (640 / stride) ** 2)
            bias.data[:, 5:] += (
                math.log(0.6 / (detect_layer.nc - 0.999999))
                if cf is None
                else torch.log(cf / cf.sum())
            )
            conv.bias = torch.nn.Parameter(bias.view(-1), requires_grad=True)

    def fuse(self):
        for module in self.model.modules():
            if isinstance(module, (Conv, DWConv)) and hasattr(module, "bn"):
                module.conv = fuse_conv_and_bn(module.conv, module.bn)
                delattr(module, "bn")
                module.forward = module.forward_fuse
        return self

    def _apply(self, fn):
        self = super()._apply(fn)
        detect_layer = self.model[-1]
        if isinstance(detect_layer, Detect):
            detect_layer.stride = fn(detect_layer.stride)
            detect_layer.grid = list(map(fn, detect_layer.grid))
            if isinstance(detect_layer.anchor_grid, list):
                detect_layer.anchor_grid = list(map(fn, detect_layer.anchor_grid))
        return self


def parse_model(model_dict, ch):
    anchors = model_dict["anchors"]
    nc = model_dict["nc"]
    gd = model_dict["depth_multiple"]
    gw = model_dict["width_multiple"]
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    no = na * (nc + 5)

    layers = []
    save = []
    c2 = ch[-1]
    for index, (f, n, m, args) in enumerate(model_dict["backbone"] + model_dict["head"]):
        m = eval(m) if isinstance(m, str) else m
        for arg_index, arg in enumerate(args):
            try:
                args[arg_index] = eval(arg) if isinstance(arg, str) else arg
            except NameError:
                pass

        n = n_ = max(round(n * gd), 1) if n > 1 else n
        if m in [Conv, C3, SPPF]:
            c1, c2 = ch[f], args[0]
            if c2 != no:
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]
            if m in [C3]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[layer_index] for layer_index in f)
        elif m is Detect:
            args.append([ch[layer_index] for layer_index in f])
            if isinstance(args[1], int):
                args[1] = [list(range(args[1] * 2))] * len(f)
        else:
            c2 = ch[f]

        module = (
            nn.Sequential(*(m(*args) for _ in range(n)))
            if n > 1
            else m(*args)
        )
        module.i = index
        module.f = f
        module.type = str(m)[8:-2].replace("__main__.", "")
        module.np = sum(parameter.numel() for parameter in module.parameters())
        save.extend(
            layer_index % index
            for layer_index in ([f] if isinstance(f, int) else f)
            if layer_index != -1
        )
        layers.append(module)
        if index == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


__all__ = ["Detect", "Model"]
