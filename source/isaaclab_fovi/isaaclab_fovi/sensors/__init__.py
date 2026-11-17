# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Foveated camera sensors and simulator-independent projection helpers.

Names are imported on first access, so reading the configuration or the geometry
helpers does not load a renderer runtime.
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_EXPORTS = {
    "FoveatedCamera": "foveated_camera",
    "FoveatedCameraCfg": "foveated_camera_cfg",
    "FoveatedNewtonCamera": "foveated_newton_camera",
    "FoveatedRTXCamera": "foveated_rtx_camera",
    "compute_foveated_ray_array": "foveated_rays",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.{module}"), name)
