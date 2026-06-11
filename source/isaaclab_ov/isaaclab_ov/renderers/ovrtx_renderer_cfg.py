# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for OVRTX Renderer."""

import os
import tempfile

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

    temp_usd_dir: str | None = None
    """Directory for temporary USD debug dumps written during OVRTX stage preparation.

    When set, the renderer writes ``pre_ovrtx_renderer_stage.usda`` (raw stage before
    partition attributes and export trimming) and ``ovrtx_renderer_stage.usda`` (exported
    stage plus injected render products) under this directory. Must be writable.
    """

    log_level: str = "verbose"
    """OVRTX carb log level: "verbose", "info", "warn", "error"."""

    log_file_path: str = os.path.join(tempfile.gettempdir(), "ovrtx_renderer.log")
    """Path for OVRTX log file. Defaults to ``<system temp>/ovrtx_renderer.log``."""
