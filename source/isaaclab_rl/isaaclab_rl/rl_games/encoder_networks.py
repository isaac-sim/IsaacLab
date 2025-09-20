# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch
import torch.nn as nn


class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor):
        return x.permute(*self.dims).contiguous()


class CNN(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int], *args, **kwargs):
        super().__init__()
        channels = kwargs["channels"]
        kernel_sizes = kwargs["kernel_sizes"]
        strides = kwargs["strides"]
        paddings = kwargs["paddings"]
        activation = kwargs.get("activation", "relu")
        use_maxpool = kwargs.get("use_maxpool", False)
        pool_size = kwargs.get("pool_size", 2)
        gap = kwargs.get("gap", True)
        feature_size = kwargs.get("feature_size", None)
        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings), "lists must match length"
        act_cls = {
            "relu": nn.ReLU, "elu": nn.ELU, "gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh, "none": nn.Identity
        }[activation.lower()]

        C, H, W = input_shape
        layers: list[nn.Module] = []
        in_c = C
        for i, out_c in enumerate(channels):
            layers.append(nn.Conv2d(in_c, out_c, kernel_size=kernel_sizes[i], stride=strides[i], padding=paddings[i]))
            layers.append(nn.BatchNorm2d(out_c))
            layers.append(act_cls())
            if use_maxpool:
                layers.append(nn.MaxPool2d(pool_size))
            in_c = out_c

        if gap:
            layers.append(nn.AdaptiveAvgPool2d((1, 1)))  # -> (B, C, 1, 1)

        self.encoder = nn.Sequential(*layers)

        # figure out flattened dim with a dummy forward
        with torch.no_grad():
            p = next(self.encoder.parameters(), None)
            dummy = torch.zeros(1, C, H, W, device=p.device)
            enc = self.encoder(dummy)
            flat_dim = enc.shape[1] if gap else enc.view(1, -1).shape[1]

        self.flatten = (lambda t: t.view(t.size(0), -1)) if not gap else (lambda t: t.view(t.size(0), t.size(1)))
        self.projector = nn.Identity() if feature_size is None else nn.Linear(flat_dim, feature_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.encoder(x)  # expect x as (B, C, H, W)
        y = self.flatten(y)  # (B, flat_dim)
        return self.projector(y)  # (B, out_dim)


class MLP(nn.Module):
    def __init__(self, input_shape: tuple[int, ...], *args, **kwargs):
        super().__init__()
        layers = kwargs["layers"]                          # e.g., [512, 256, 128]
        activation = kwargs.get("activation", "relu")
        norm = kwargs.get("norm", None)                    # None | "batch" | "layer"
        dropout = kwargs.get("dropout", 0.0)               # float or List[float]
        bias = kwargs.get("bias", True)
        feature_size = kwargs.get("feature_size", None)    # optionally project to this size

        act_cls = {
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
            "none": nn.Identity,
        }[activation.lower()]

        # Flattened input dimension
        in_dim = math.prod(input_shape)

        # Allow scalar or per-layer dropout
        if isinstance(dropout, (list, tuple)):
            assert len(dropout) == len(layers), "dropout list must match number of layers"
            dropout_list = list(dropout)
        else:
            dropout_list = [float(dropout)] * len(layers)

        mods: list[nn.Module] = []
        curr = in_dim
        for i, h in enumerate(layers):
            mods.append(nn.Linear(curr, h, bias=bias))
            if norm == "batch":
                mods.append(nn.BatchNorm1d(h))
            elif norm == "layer":
                mods.append(nn.LayerNorm(h))
            mods.append(act_cls())
            if dropout_list[i] > 0:
                mods.append(nn.Dropout(p=float(dropout_list[i])))
            curr = h

        self.encoder = nn.Sequential(*mods)
        self.flatten = lambda t: t.view(t.size(0), -1)

        # Figure out flattened dim with a dummy forward
        with torch.no_grad():
            p = next(self.encoder.parameters(), None)
            device = p.device if p is not None else None
            dummy = torch.zeros(1, *input_shape, device=device)
            x = self.flatten(dummy)
            if len(layers) == 0:
                flat_dim = x.shape[1]
            else:
                y = self.encoder(x)
                flat_dim = y.shape[1]

        self.projector = nn.Identity() if feature_size is None else nn.Linear(flat_dim, feature_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)             # (B, in_dim)
        y = self.encoder(x) if len(self.encoder) > 0 else x
        return self.projector(y)        # (B, out_dim)
