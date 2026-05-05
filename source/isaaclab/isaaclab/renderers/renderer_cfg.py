# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration for renderers."""

from isaaclab.utils import configclass


@configclass
class RendererCfg:
    """Configuration for a renderer."""

    renderer_type: str = "default"

    def provides_temporal_camera_data(self, sim_render_cfg) -> bool:
        """Whether this renderer's pipeline introduces temporal frame correlation given
        ``sim_render_cfg`` (a :class:`~isaaclab.sim.simulation_cfg.RenderCfg` or ``None``).

        Defaults to ``True``; override for pure-rasterization renderers or to inspect
        renderer-specific runtime state such as anti-aliasing mode.
        """
        return True
