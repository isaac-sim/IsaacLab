# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for OVRTX Renderer."""

import tempfile
from pathlib import Path

from isaaclab.renderers.renderer_cfg import RendererCfg
from isaaclab.utils import configclass


@configclass
class OVRTXRendererCfg(RendererCfg):
    """Configuration for OVRTX Renderer.

    The OVRTX renderer uses the ovrtx library for high-fidelity RTX-based rendering.
    width, height, num_envs, and data_types are obtained from the sensor when
    create_render_data() is called (same pattern as Isaac RTX).
    """

    renderer_type: str = "ovrtx"
    """Type identifier for OVRTX renderer."""

    requires_newton_model: bool = True
    """Internal requirement flag; do not override in user configs."""

    requires_usd_stage: bool = True
    """Internal requirement flag; do not override in user configs."""

    simple_shading_mode: bool = True
    """Whether to use simple shading mode (default: True).

    When enabled, uses SimpleShadingSD RenderVar instead of LdrColor for RGB rendering.
    Provides faster, simpler rendering suitable for many vision-based tasks.
    Set to False to use full RTX path-traced rendering with LdrColor.
    """

    temp_usd_dir: str = str(Path(tempfile.gettempdir()) / "ovrtx")
    """Directory for temporary combined USD files (scene + injected cameras).
    Used by the OVRTX renderer when building the render scope; must be writable.
    """

    temp_usd_suffix: str = ".usda"
    """File suffix for temporary combined USD files (e.g. '.usda' or '.usdc')."""

    use_cloning: bool = False
    """When True, export only env_0 and use OVRTX clone_usd. When False, export full stage."""

    log_level: str = "verbose"
    """OVRTX carb log level: "verbose", "info", "warn", "error"."""

    log_file_path: str = "/tmp/ovrtx_renderer.log"
    """Path for OVRTX log file."""
