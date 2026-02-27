# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for PhysX renderer backends (Isaac RTX / Omniverse Replicator)."""

from __future__ import annotations

from .isaac_rtx_renderer import IsaacRtxRenderer

import typing

if typing.TYPE_CHECKING:
    from .isaac_rtx_renderer_cfg import IsaacRtxRendererCfg

from isaaclab.utils.module import lazy_export

lazy_export(
    ("isaac_rtx_renderer_cfg", "IsaacRtxRendererCfg"),
)

Renderer = IsaacRtxRenderer
