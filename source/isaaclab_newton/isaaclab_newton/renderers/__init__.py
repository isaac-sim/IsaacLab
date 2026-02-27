# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for Newton renderer backends (Newton Warp)."""

from __future__ import annotations

from .newton_warp_renderer import NewtonWarpRenderer

import typing

if typing.TYPE_CHECKING:
    from .newton_warp_renderer_cfg import NewtonWarpRendererCfg

from isaaclab.utils.module import lazy_export

lazy_export(
    ("newton_warp_renderer_cfg", "NewtonWarpRendererCfg"),
)

Renderer = NewtonWarpRenderer
