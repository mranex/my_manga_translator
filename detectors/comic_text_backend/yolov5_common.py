from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [value // 2 for value in k]
    return p


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            c1,
            c2,
            k,
            s,
            autopad(k, p),
            groups=g,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(c2)
        if isinstance(act, bool):
            self.act = (
                nn.SiLU()
                if act is True
                else (act if isinstance(act, nn.Module) else nn.Identity())
            )
        elif isinstance(act, str):
            if act == "leaky":
                self.act = nn.LeakyReLU(0.1, inplace=True)
            elif act == "relu":
                self.act = nn.ReLU(inplace=True)
            else:
                self.act = nn.Identity()
        else:
            self.act = act

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DWConv(Conv):
    def __init__(self, c1, c2, k=1, s=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, act=True):
        super().__init__()
        hidden = int(c2 * e)
        self.cv1 = Conv(c1, hidden, 1, 1, act=act)
        self.cv2 = Conv(hidden, c2, 3, 1, g=g, act=act)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out


class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, act=True):
        super().__init__()
        hidden = int(c2 * e)
        self.cv1 = Conv(c1, hidden, 1, 1, act=act)
        self.cv2 = Conv(c1, hidden, 1, 1, act=act)
        self.cv3 = Conv(2 * hidden, c2, 1, act=act)
        self.m = nn.Sequential(
            *(Bottleneck(hidden, hidden, shortcut, g, e=1.0, act=act) for _ in range(n))
        )

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        hidden = c1 // 2
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.cv2 = Conv(hidden * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


__all__ = [
    "Bottleneck",
    "C3",
    "Concat",
    "Conv",
    "DWConv",
    "SPPF",
]
