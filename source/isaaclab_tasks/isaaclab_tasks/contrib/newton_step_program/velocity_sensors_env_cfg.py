# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flat ANYmal-D velocity tracking whose policy also observes Newton IMU and frame-transformer sensors."""

import torch

from isaaclab.envs import ManagerBasedEnv, mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg, ImuCfg
from isaaclab.utils import configclass

from isaaclab_tasks.core.velocity.config.anymal_d.flat_env_cfg import AnymalDFlatEnvCfg


def feet_pos_base(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Foot positions in the base frame [m], flattened to shape (num_envs, 3 * num_feet)."""
    return env.scene[sensor_cfg.name].data.target_pos_source.torch.flatten(1)


@configclass
class AnymalDSensorsEnvCfg(AnymalDFlatEnvCfg):
    """Flat ANYmal-D with a base IMU, a foot frame transformer, and foot contact sensors in the policy input.

    Newton updates the IMU, frame-transformer, and contact sensors at the end of every step program; the policy
    reads them through Isaac Lab's sensor data.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.imu = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/base")
        self.scene.feet_frames = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            target_frames=[FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/.*_FOOT")],
        )
        policy = self.observations.policy
        policy.imu_ang_vel = ObsTerm(func=mdp.imu_ang_vel, params={"asset_cfg": SceneEntityCfg("imu")})
        policy.imu_lin_acc = ObsTerm(func=mdp.imu_lin_acc, params={"asset_cfg": SceneEntityCfg("imu")})
        policy.feet_pos_base = ObsTerm(func=feet_pos_base, params={"sensor_cfg": SceneEntityCfg("feet_frames")})
