# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
import warp as wp
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import TerminationTermCfg
from isaaclab.utils.warp.utils import resolve_asset_cfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers.command_manager import CommandTerm

"""
MDP terminations.
"""

@wp.kernel
def time_out_kernel(
    episode_length_buf: wp.array(dtype=wp.float32),
    max_episode_length: wp.float32,
    done: wp.array(dtype=wp.bool)
) -> None:
    i = wp.tid()
    done[i] = episode_length_buf[i] >= max_episode_length

class time_out(ManagerTermBase):
    """Terminate the episode when the episode length exceeds the maximum episode length."""
    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._done_buf = wp.zeros((env.num_envs,), dtype=wp.bool, device=env.device)
    def update_config(self) -> None:
        pass

    def __call__(self, env: ManagerBasedRLEnv, **kwargs) -> wp.array(dtype=wp.bool):
        wp.launch(
            time_out_kernel,
            dim=env.num_envs,
            inputs=[
                env.episode_length_buf,
                env.max_episode_length,
                self._done_buf,
            ],
        )
        return self._done_buf

@wp.kernel
def command_resample_kernel(
    time_left: wp.array(dtype=wp.float32),
    command_counter: wp.array(dtype=wp.int32),
    num_resamples: wp.int32,
    dt: wp.float32,
    done: wp.array(dtype=wp.bool)
) -> None:
    i = wp.tid()
    done[i] = (time_left[i] <= dt) and (command_counter[i] == num_resamples)

class command_resample(ManagerTermBase):
    """Terminate the episode based on the total number of times commands have been re-sampled."""
    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._done_buf = wp.zeros((env.num_envs,), dtype=wp.bool, device=env.device)
        self._command_name = "None"
        self._num_resamples = 1
        self.update_config(**cfg.params)
        self._command: CommandTerm = env.command_manager.get_term(self._command_name)

    def update_config(self, command_name: str, num_resamples: int = 1) -> None:
        self._command_name = command_name
        self._num_resamples = num_resamples

    def __call__(self, env: ManagerBasedRLEnv, **kwargs) -> wp.array(dtype=wp.bool):
        wp.launch(
            command_resample_kernel,
            dim=env.num_envs,
            inputs=[
                self._command.time_left,
                self._command.command_counter,
                self._num_resamples,
                env.step_dt,
                self._done_buf,
            ],
        )
        return self._done_buf

"""
Root terminations.
"""


@wp.kernel
def bad_orientation_kernel(
    projected_gravity_b: wp.array(dtype=wp.vec3f),
    limit_angle: wp.float32,
    done: wp.array(dtype=wp.bool)
) -> None:
    i = wp.tid()
    done[i] = wp.abs(wp.acos(-projected_gravity_b[i, 2])) > limit_angle

class bad_orientation(ManagerTermBase):
    """Terminate when the asset's orientation is too far from the desired orientation limits."""
    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = resolve_asset_cfg(cfg.params, env)
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._done_buf = wp.zeros((env.num_envs,), dtype=wp.bool, device=env.device)
        self._limit_angle = 0.0

        self.update_config(**cfg.params)

    def update_config(self, limit_angle: float, asset_cfg: SceneEntityCfg | None = None) -> None:
        self._limit_angle = limit_angle

    def __call__(self, env: ManagerBasedRLEnv, **kwargs) -> wp.array(dtype=wp.bool):
        wp.launch(
            bad_orientation_kernel,
            dim=env.num_envs,
            inputs=[
                self._asset.data.projected_gravity_b,
                self._limit_angle,
            ],
        )
        return self._done_buf

@wp.kernel
def root_height_below_minimum_kernel(
    root_pose_w: wp.array(dtype=wp.transformf),
    minimum_height: wp.float32,
    done: wp.array(dtype=wp.bool)
) -> None:
    i = wp.tid()
    done[i] = root_pos_w[i][2] < minimum_height

class root_height_below_minimum(ManagerTermBase):
    """Terminate when the asset's root height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = resolve_asset_cfg(cfg.params, env)
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._done_buf = wp.zeros((env.num_envs,), dtype=wp.bool, device=env.device)
        self._minimum_height = 0.0

        self.update_config(**cfg.params)

    def update_config(self, minimum_height: float, asset_cfg: SceneEntityCfg | None = None) -> None:
        self._minimum_height = minimum_height

    def __call__(self, env: ManagerBasedRLEnv, **kwargs) -> wp.array(dtype=wp.bool):
        wp.launch(
            root_height_below_minimum_kernel,
            dim=env.num_envs,
            inputs=[
                self._asset.data.root_pose_w,
                self._minimum_height,
                self._done_buf,
            ],
        )
        return self._done_buf

@wp.kernel
def root_height_above_maximum_kernel(
    root_pose_w: wp.array(dtype=wp.transformf),
    maximum_height: wp.float32,
    done: wp.array(dtype=wp.bool)
) -> None:
    i = wp.tid()
    done[i] = root_pose_w[i][2] > maximum_height

class root_height_above_maximum(ManagerTermBase):
    """Terminate when the asset's root height is above the maximum height."""
    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = resolve_asset_cfg(cfg.params, env)
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._done_buf = wp.zeros((env.num_envs,), dtype=wp.bool, device=env.device)
        self._maximum_height = 0.0

        self.update_config(**cfg.params)

    def update_config(self, maximum_height: float, asset_cfg: SceneEntityCfg | None = None) -> None:
        self._maximum_height = maximum_height

    def __call__(self, env: ManagerBasedRLEnv, **kwargs) -> wp.array(dtype=wp.bool):
        wp.launch(
            root_height_above_maximum_kernel,
            dim=env.num_envs,
            inputs=[
                self._asset.data.root_pose_w,
                self._maximum_height,
                self._done_buf,
            ],
        )
        return self._done_buf


"""
Joint terminations.
"""


def joint_pos_out_of_limit(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate when the asset's joint positions are outside of the soft joint limits."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute any violations
    out_of_upper_limits = torch.any(asset.data.joint_pos > asset.data.soft_joint_pos_limits[..., 1], dim=1)
    out_of_lower_limits = torch.any(asset.data.joint_pos < asset.data.soft_joint_pos_limits[..., 0], dim=1)
    return torch.logical_or(out_of_upper_limits[:, asset_cfg.joint_ids], out_of_lower_limits[:, asset_cfg.joint_ids])


def joint_pos_out_of_manual_limit(
    env: ManagerBasedRLEnv, bounds: tuple[float, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's joint positions are outside of the configured bounds.

    Note:
        This function is similar to :func:`joint_pos_out_of_limit` but allows the user to specify the bounds manually.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.joint_ids is None:
        asset_cfg.joint_ids = slice(None)
    # compute any violations
    out_of_upper_limits = torch.any(asset.data.joint_pos[:, asset_cfg.joint_ids] > bounds[1], dim=1)
    out_of_lower_limits = torch.any(asset.data.joint_pos[:, asset_cfg.joint_ids] < bounds[0], dim=1)
    return torch.logical_or(out_of_upper_limits, out_of_lower_limits)


def joint_vel_out_of_limit(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate when the asset's joint velocities are outside of the soft joint limits."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute any violations
    limits = asset.data.soft_joint_vel_limits
    return torch.any(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]) > limits[:, asset_cfg.joint_ids], dim=1)


def joint_vel_out_of_manual_limit(
    env: ManagerBasedRLEnv, max_velocity: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's joint velocities are outside the provided limits."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute any violations
    return torch.any(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]) > max_velocity, dim=1)


def joint_effort_out_of_limit(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when effort applied on the asset's joints are outside of the soft joint limits.

    In the actuators, the applied torque are the efforts applied on the joints. These are computed by clipping
    the computed torques to the joint limits. Hence, we check if the computed torques are equal to the applied
    torques.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # check if any joint effort is out of limit
    out_of_limits = torch.isclose(
        asset.data.computed_torque[:, asset_cfg.joint_ids], asset.data.applied_torque[:, asset_cfg.joint_ids]
    )
    return torch.any(out_of_limits, dim=1)


"""
Contact sensor.
"""


#def illegal_contact(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
#    """Terminate when the contact force on the sensor exceeds the force threshold."""
#    # extract the used quantities (to enable type-hinting)
#    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#    net_contact_forces = contact_sensor.data.net_forces_w_history
#    # check if any contact force exceeds the threshold
#    return torch.any(
#        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold, dim=1
#    )
