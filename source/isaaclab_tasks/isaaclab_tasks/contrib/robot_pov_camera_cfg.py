# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared recorded robot-PoV camera configuration for contributed tasks."""

from isaaclab_physx.renderers import IsaacRtxRendererCfg

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils.presets import MultiBackendRendererCfg


@configclass
class _RobotPovCameraRendererCfg(MultiBackendRendererCfg):
    """Renderer presets shared by the recorded robot-PoV cameras."""

    default: IsaacRtxRendererCfg = IsaacRtxRendererCfg(
        enable_dlss_ray_reconstruction=True,
        dlss_exec_mode="quality",
    )
    isaacsim_rtx = default


def robot_pov_camera_cfg() -> CameraCfg:
    """Return the shared recorded robot-PoV camera configuration."""
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/RobotPOVCam",
        update_period=0.0,
        height=450,
        width=720,
        data_types=["rgb"],
        renderer_cfg=_RobotPovCameraRendererCfg(),
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.15, clipping_range=(0.1, 2.0)),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.12, 1.67675),
            rot=(0.9801, 0.0, 0.0, -0.19848),
            convention="ros",
        ),
    )
