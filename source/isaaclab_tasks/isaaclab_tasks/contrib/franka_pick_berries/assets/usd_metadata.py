# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Typed USD storage for berry task metadata (no JSON or pickle payloads)."""

import numpy as np

from pxr import Usd


def read_data(stage: Usd.Stage, path: str):
    """Read nested metadata written as typed attributes and ordered child prims."""
    prim = stage.GetPrimAtPath(path)
    if not prim:
        raise ValueError(f"Missing task metadata: {path}")
    kind = prim.GetAttribute("data:kind").Get()
    if kind in {"dict", "list"}:
        values = [read_data(stage, f"{path}/item_{i}") for i in range(prim.GetAttribute("data:count").Get())]
        return dict(zip(prim.GetAttribute("data:keys").Get(), values, strict=True)) if kind == "dict" else values
    if kind == "null":
        return None
    if kind != "scalar":
        raise ValueError(f"Unknown task metadata kind at {path}: {kind}")
    return prim.GetAttribute("data:value").Get()


def read_array(prim: Usd.Prim, name: str) -> np.ndarray:
    """Read a numeric array written with its shape and dtype as sibling attributes."""
    return np.asarray(prim.GetAttribute(name).Get(), dtype=prim.GetAttribute(name + ":dtype").Get()).reshape(
        tuple(prim.GetAttribute(name + ":shape").Get())
    )
