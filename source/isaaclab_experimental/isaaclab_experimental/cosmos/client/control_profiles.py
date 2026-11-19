# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Control images for Cosmos cameras, as camera tensor processors."""

from __future__ import annotations

from ...image_transfer import FunctionProcessorCfg, depth_to_control

# A tabletop workspace in front of a robot's camera lies within 0.1 m to 2 m. Set the range to match the scene:
# depth outside it clips to white or black, and a wider range leaves less contrast for the objects.
DEPTH_NEAR = 0.1
"""Default depth rendered white [m]."""

DEPTH_FAR = 2.0
"""Default depth rendered black [m]."""


def depth_control(near: float = DEPTH_NEAR, far: float = DEPTH_FAR) -> FunctionProcessorCfg:
    """Return the processor that writes the depth control from the camera's ``distance_to_image_plane`` output.

    Near hits become white and far hits black. Set the metric range to match the camera's scene.

    Args:
        near: Depth rendered white [m].
        far: Depth rendered black [m].
    """
    return FunctionProcessorCfg(
        function=depth_to_control,
        params={"near": near, "far": far},
        inputs=["distance_to_image_plane"],
        outputs=["control"],
    )
