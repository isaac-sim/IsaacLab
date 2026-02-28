# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stub config for future RTX renderer integration."""

from isaaclab.renderers import RendererCfg
from isaaclab.utils import configclass


@configclass
class RTXRendererCfg(RendererCfg):
    """Stub config for future RTX renderer integration.

    TODO: Implement renderer lifecycle, sensor/render-product routing, and
    backend-specific settings application.
    """

    renderer_type: str = "rtx"
    rendering_mode: str | None = None
