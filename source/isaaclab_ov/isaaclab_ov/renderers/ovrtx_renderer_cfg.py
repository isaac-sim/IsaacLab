# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for OVRTX Renderer."""

import tempfile
from pathlib import Path

from isaaclab.renderers.renderer_cfg import RendererCfg
from isaaclab.utils.configclass import configclass


@configclass
class OVRTXRendererCfg(RendererCfg):
    """Configuration for OVRTX Renderer.

    The OVRTX renderer uses the ovrtx library for high-fidelity RTX-based rendering.
    width, height, num_envs, and data_types are obtained from the
    :class:`~isaaclab.renderers.camera_render_spec.CameraRenderSpec` when
    :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.create_render_data` is called
    (same pattern as Isaac RTX).
    """

    renderer_type: str = "ovrtx"
    """Type identifier for OVRTX renderer."""

    temp_usd_dir: str = str(Path(tempfile.gettempdir()) / "ovrtx")
    """Directory for temporary combined USD files (scene + injected cameras).
    Used by the OVRTX renderer when building the render scope; must be writable.
    """

    temp_usd_suffix: str = ".usda"
    """File suffix for temporary combined USD files (e.g. '.usda' or '.usdc')."""

    use_ovrtx_cloning: bool = True
    """When True, export only env_0 and use OVRTX ``clone_usd``. When False, export full multi-environment stage.

    OVRTX cloning is only supported in OVRTX 0.3.0 or newer.

    If the simulation uses a heterogeneous env setup, the renderer disables this path and exports the full
    multi-environment stage instead (same effect as setting this to ``False`` for that run).
    """

    log_level: str = "verbose"
    """OVRTX carb log level: "verbose", "info", "warn", "error"."""

    log_file_path: str = "/tmp/ovrtx_renderer.log"
    """Path for OVRTX log file."""
