# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the direct-workflow Cartpole lidar environment."""

from isaaclab_ov.renderers import OVRTXLidarCfg

from isaaclab.utils import configclass

from isaaclab_tasks.core.cartpole.cartpole_direct_camera_env_cfg import CartpoleCameraEnvCfg
from isaaclab_tasks.core.cartpole.cartpole_direct_env_cfg import CartpoleSceneCfg

from .cartpole_lidar_common import (
    CARTPOLE_LIDAR_AZIMUTH_COUNT,
    CARTPOLE_LIDAR_ELEVATION_COUNT,
    cartpole_lidar_cfg,
)


@configclass
class CartpoleLidarSceneCfg(CartpoleSceneCfg):
    """Cartpole scene with a tiled lidar and no view-obstructing ground plane."""

    ground = None
    lidar: OVRTXLidarCfg = cartpole_lidar_cfg()


@configclass
class CartpoleLidarEnvCfg(CartpoleCameraEnvCfg.BaseCartpoleCameraEnvCfg):
    """Cartpole environment whose policy reads stacked lidar range images.

    The policy observation is a stacked range image of the scan, and the scene defaults to 64
    environments. The critic reads the joint state.
    """

    scene: CartpoleLidarSceneCfg = CartpoleLidarSceneCfg(num_envs=64, env_spacing=20.0)
    # One range channel. Env init stacks ``frame_stack`` scans on the channel axis.
    observation_space = [1, CARTPOLE_LIDAR_ELEVATION_COUNT, CARTPOLE_LIDAR_AZIMUTH_COUNT]
