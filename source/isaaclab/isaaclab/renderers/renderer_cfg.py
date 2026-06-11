# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration for renderers."""

from isaaclab.utils.configclass import configclass


@configclass
class RendererCfg:
    """Configuration for a renderer."""

    renderer_type: str = "default"

    def provides_temporal_camera_data(self, data_type: str) -> bool:
        """Whether this renderer's ``data_type`` output carries temporal information.

        Under a physics backend without implicit damping (e.g. Newton), a camera policy
        needs a temporal cue to infer velocity. Renderers that accumulate frames over time
        (temporal AA / DLSS) supply it; pure rasterizers and non-beauty AOVs do not.

        The base default is ``False`` (assume no temporal information); renderer subclasses
        override per output type.
        """
        return False
