# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2024, The IsaacSim RL Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul, quat_rotate_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import RLTaskEnv


def feet_air_time(env: RLTaskEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    first_contact = last_air_time > 0.0
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


class feet_air_time_l2(ManagerTermBase):
    """Reward long steps taken by the feet."""

    def __init__(self, env: RLTaskEnv, cfg: RewardTermCfg):
        # initialize the base class
        super().__init__(cfg, env)
        # create history buffers
        self.contact_sensor: ContactSensor = env.scene.sensors[self.cfg.params["sensor_cfg"].name]
        asset_cfg: SceneEntityCfg = self.cfg.params["sensor_cfg"]
        self.asset = env.scene[asset_cfg.name]
        self.body_ids = asset_cfg.body_ids
        self.air_time = torch.zeros((env.num_envs, len(self.body_ids)), device=env.device)
        self.was_contact = torch.zeros(env.num_envs, len(self.body_ids), dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor):
        self.air_time[env_ids] = 0.0

    def __call__(
        self,
        env: RLTaskEnv,
        threshold: float,
        command_name: str,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        contact = self.contact_sensor.data.net_forces_w_history[:, :, self.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
        touch_down = (self.air_time > 0.0) * contact
        self.air_time += env.step_dt
        # reward only on first contact with the ground
        reward = torch.sum(self.air_time.clip(max=threshold).square() * touch_down, dim=1)
        reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
        self.air_time *= ~contact
        return reward



