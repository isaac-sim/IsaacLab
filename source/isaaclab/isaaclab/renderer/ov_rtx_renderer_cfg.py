# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for OV RTX Renderer."""

from __future__ import annotations

from isaaclab.utils import configclass

from .renderer_cfg import RendererCfg


@configclass
class OVRTXRendererCfg(RendererCfg):
    """Configuration for Omniverse RTX renderer.

    This renderer applies RTX settings from SimulationCfg.render_cfg.
    Unlike camera renderers, it doesn't produce image output - it configures
    the RTX rendering pipeline settings.
    """

    renderer_type: str = "ov_rtx"
    """Type identifier for registry lookup."""

    # Override defaults since this isn't a camera renderer
    height: int = 0
    width: int = 0
    num_envs: int = 0
    num_cameras: int = 0
    data_types: list[str] = []
