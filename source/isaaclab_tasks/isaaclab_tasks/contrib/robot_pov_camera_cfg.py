# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared recorded robot-PoV camera configuration for contributed tasks."""

from isaaclab_physx.renderers import IsaacRtxRendererCfg

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg

from isaaclab_tasks.utils.presets import MultiBackendRendererCfg


def robot_pov_camera_cfg(
    *,
    parent_prim_path: str,
    offset_pos: tuple[float, float, float],
    offset_rot: tuple[float, float, float, float],
) -> CameraCfg:
    """Return a recorded robot-PoV camera under a prim that follows physical-body motion.

    Args:
        parent_prim_path: Path of the robot prim whose transform inherits physical-body motion.
        offset_pos: Camera position in the parent body frame.
        offset_rot: Camera XYZW quaternion in the parent body frame using the ROS camera convention.
    """
    return CameraCfg(
        prim_path=f"{parent_prim_path}/RobotPOVCam",
        update_period=0.0,
        height=450,
        width=720,
        data_types=["rgb"],
        renderer_cfg=MultiBackendRendererCfg(default=IsaacRtxRendererCfg(), isaacsim_rtx=IsaacRtxRendererCfg()),
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.15, clipping_range=(0.1, 2.0)),
        offset=CameraCfg.OffsetCfg(pos=offset_pos, rot=offset_rot, convention="ros"),
    )
