# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for spawners that spawn lights in the simulation.

There are various different kinds of lights that can be spawned into the USD stage.
Please check the Omniverse documentation for `lighting overview
<https://docs.omniverse.nvidia.com/materials-and-rendering/latest/lighting.html>`_.
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from .lights import spawn_light
    from .lights_cfg import CylinderLightCfg, DiskLightCfg, DistantLightCfg, DomeLightCfg, LightCfg, SphereLightCfg

from isaaclab.utils.module import lazy_export

lazy_export(
    ("lights", "spawn_light"),
    ("lights_cfg", [
        "CylinderLightCfg",
        "DiskLightCfg",
        "DistantLightCfg",
        "DomeLightCfg",
        "LightCfg",
        "SphereLightCfg",
    ]),
)
