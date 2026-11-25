# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the manager-based Cartpole lidar environment."""

from isaaclab_ov.renderers import OVRTXLidarCfg

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.core.cartpole.cartpole_manager_camera_env_cfg import CartpoleCameraEnvCfg
from isaaclab_tasks.core.cartpole.cartpole_manager_env_cfg import CartpoleSceneCfg, ObservationsCfg

from .cartpole_lidar_common import cartpole_lidar_cfg
from .mdp.observations import _LidarRangeImage


@configclass
class CartpoleLidarSceneCfg(CartpoleSceneCfg):
    """Cartpole scene with a tiled lidar."""

    lidar: OVRTXLidarCfg = cartpole_lidar_cfg()


def lidar_observations_cfg():
    """Build policy observations from stacked lidar range images.

    The critic keeps the privileged joint state. The policy is a channel-first range image: misses
    are 0 and a hit at the far range is 1. Joint terms stay off the policy group so they are not
    flattened into that image.

    Returns:
        An observations config with the range-image policy and the privileged critic.
    """

    @configclass
    class LidarObservationsCfg:
        @configclass
        class PolicyCfg(ObsGroup):
            lidar_range = ObsTerm(
                func=_LidarRangeImage,
                params={"sensor_cfg": SceneEntityCfg("lidar")},
            )

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True

        policy: ObsGroup = PolicyCfg()
        critic: ObsGroup = ObservationsCfg.PolicyCfg()

    return LidarObservationsCfg()


@configclass
class CartpoleLidarEnvCfg(CartpoleCameraEnvCfg.BaseCartpoleCameraEnvCfg):
    """Cartpole environment whose policy reads stacked lidar range images."""

    scene: CartpoleLidarSceneCfg = CartpoleLidarSceneCfg(num_envs=64, env_spacing=20.0)
    observations = lidar_observations_cfg()
