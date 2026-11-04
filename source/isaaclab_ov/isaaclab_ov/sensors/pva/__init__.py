# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for OVPhysX PVA sensor."""

import warnings

from isaaclab.utils.module import lazy_export

_lazy_getattr, _, _ = lazy_export()


def __getattr__(name: str):
    value = _lazy_getattr(name)
    if name == "PvaData":
        warnings.warn(
            "isaaclab_ov.sensors.pva.PvaData is deprecated and will be removed in a future release."
            " Use sensor.data or PvaData from isaaclab.sensors instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        globals()[name] = value
    return value
