# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for PhysX PVA sensor."""

from isaaclab.utils.module import lazy_export, warn_on_lazy_exports

_lazy_getattr, _, _ = lazy_export()
warn_on_lazy_exports(_lazy_getattr, {"PvaData": "sensor.data or PvaData from isaaclab.sensors"})
