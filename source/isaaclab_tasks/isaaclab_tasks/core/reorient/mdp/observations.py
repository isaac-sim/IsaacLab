# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions specific to the in-hand dexterous manipulation environments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils.noise import NoiseModelCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import JointWrenchSensor

    from .commands import ReorientCommand


def goal_quat_diff(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, make_quat_unique: bool
) -> torch.Tensor:
    """Goal orientation relative to the asset's root frame.

    The quaternion is represented as (x, y, z, w). The real part is always positive.
    """
    # extract useful elements
    asset: RigidObject = env.scene[asset_cfg.name]
    command_term: ReorientCommand = env.command_manager.get_term(command_name)

    # obtain the orientations
    goal_quat_w = command_term.command[:, 3:7]
    asset_quat_w = asset.data.root_quat_w.torch

    # compute quaternion difference
    quat = math_utils.quat_mul(asset_quat_w, math_utils.quat_conjugate(goal_quat_w))
    # make sure the quaternion real-part is always positive
    return math_utils.quat_unique(quat) if make_quat_unique else quat


def fingertip_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return flattened fingertip positions in the environment frame [m].

    Args:
        env: Environment containing the hand.
        asset_cfg: Hand entity with resolved fingertip body indices.

    Returns:
        Fingertip positions [m], shape ``(num_envs, num_fingertips * 3)``.
    """
    asset = env.scene[asset_cfg.name]
    positions = asset.data.body_pos_w.torch[:, asset_cfg.body_ids]
    positions = positions - env.scene.env_origins[:, None, :]
    return positions.flatten(start_dim=1)


def fingertip_quat(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return flattened fingertip ``(x, y, z, w)`` orientations.

    Args:
        env: Environment containing the hand.
        asset_cfg: Hand entity with resolved fingertip body indices.

    Returns:
        Unit quaternions, shape ``(num_envs, num_fingertips * 4)``.
    """
    asset = env.scene[asset_cfg.name]
    return asset.data.body_quat_w.torch[:, asset_cfg.body_ids].flatten(start_dim=1)


def fingertip_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return flattened fingertip spatial velocities in the world frame.

    Args:
        env: Environment containing the hand.
        asset_cfg: Hand entity with resolved fingertip body indices.

    Returns:
        Spatial velocities [m/s, rad/s], shape ``(num_envs, num_fingertips * 6)``.
    """
    asset = env.scene[asset_cfg.name]
    return asset.data.body_vel_w.torch[:, asset_cfg.body_ids].flatten(start_dim=1)


def fingertip_wrench(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return fingertip reaction wrenches with Direct-compatible zero fallback.

    Args:
        env: Environment containing the joint-wrench sensor.
        sensor_cfg: Joint-wrench sensor entity with resolved fingertip body indices.

    Returns:
        Fingertip reaction wrenches [N, N·m], shape ``(num_envs, num_fingertips * 6)``.
    """
    sensor: JointWrenchSensor = env.scene.sensors[sensor_cfg.name]
    force_data = sensor.data.force
    torque_data = sensor.data.torque
    if force_data is None or torque_data is None:
        body_count = len(sensor_cfg.body_ids)
        return torch.zeros(env.num_envs, body_count * 6, device=env.device)
    force = force_data.torch[:, sensor_cfg.body_ids]
    torque = torque_data.torch[:, sensor_cfg.body_ids]
    return torch.cat((force, torque), dim=-1).flatten(start_dim=1)


def reorient_last_action(env: ManagerBasedRLEnv, action_name: str) -> torch.Tensor:
    """Return the Direct-compatible last action across same-step autoreset.

    Args:
        env: Environment containing the action term and reset buffers.
        action_name: Action term whose raw action is observed.

    Returns:
        Raw actions, retaining each terminal action in its same-step reset observation.
    """
    raw_action = env.action_manager.get_term(action_name).raw_actions
    reset_action = getattr(env, "_reorient_reset_action", None)
    reset_step = getattr(env, "_reorient_reset_step", None)
    common_step_counter = getattr(env, "common_step_counter", None)
    if reset_action is None or reset_step is None or common_step_counter is None:
        return raw_action
    return torch.where((reset_step == common_step_counter).unsqueeze(-1), reset_action, raw_action)


def openai_policy_observation(
    env: ManagerBasedRLEnv,
    command_name: str,
    action_name: str,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Build the Direct OpenAI actor observation before corruption.

    Args:
        env: Environment containing the hand, object, command, and action term.
        command_name: Goal command term name.
        action_name: Action term whose raw action is observed.
        robot_cfg: Hand entity with resolved fingertip body indices.
        object_cfg: Object scene entity.

    Returns:
        Actor observation in Direct order, shape ``(num_envs, 42)``.
    """
    object_asset: RigidObject = env.scene[object_cfg.name]
    object_pos = object_asset.data.root_pos_w.torch - env.scene.env_origins
    return torch.cat(
        (
            fingertip_pos(env, robot_cfg),
            object_pos,
            goal_quat_diff(env, object_cfg, command_name, make_quat_unique=False),
            reorient_last_action(env, action_name),
        ),
        dim=-1,
    )


class OpenAIPolicyObservation(ManagerTermBase):
    """Apply one stateful noise model to the concatenated OpenAI actor observation."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        noise_model: NoiseModelCfg = cfg.params["noise_model"]
        self._noise_model = noise_model.class_type(noise_model, num_envs=self.num_envs, device=self.device)
        # ObservationManager probes callable terms once for their shape and then
        # calls reset. Keep that probe side-effect free so initialization matches
        # DirectRLEnv's first noise-model reset and application.
        self._shape_probe_pending = True

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the actor observation bias for selected environments.

        Args:
            env_ids: Environment indices to reset, or ``None`` for every environment.
        """
        if self._shape_probe_pending:
            self._shape_probe_pending = False
            return
        self._noise_model.reset(env_ids)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        action_name: str,
        noise_model: NoiseModelCfg,
        robot_cfg: SceneEntityCfg,
        object_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Return the corrupted 42-dimensional actor observation."""
        del noise_model
        observation = openai_policy_observation(env, command_name, action_name, robot_cfg, object_cfg)
        if self._shape_probe_pending:
            return observation
        return self._noise_model(observation)
