# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions specific to the in-hand dexterous manipulation environments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils.noise import NoiseModelCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import JointWrenchSensor

    from .commands import ReorientCommand


wp.init()


@wp.kernel
def _goal_quat_error_kernel(
    asset_quat: wp.array(dtype=wp.vec4),
    goal_quat: wp.array(dtype=wp.vec4),
    make_unique: int,
    out: wp.array(dtype=wp.vec4),
):
    """Per-environment quaternion error ``asset * conjugate(goal)`` in (x, y, z, w) order."""
    i = wp.tid()
    q1 = asset_quat[i]
    q2 = goal_quat[i]
    # Hamilton product against the conjugate, matching isaaclab.utils.math.quat_mul/quat_conjugate;
    # quaternions are stored (x, y, z, w)
    w = q1[3] * q2[3] + q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2]
    x = q1[0] * q2[3] - q1[3] * q2[0] - q1[1] * q2[2] + q1[2] * q2[1]
    y = q1[1] * q2[3] - q1[3] * q2[1] - q1[2] * q2[0] + q1[0] * q2[2]
    z = q1[2] * q2[3] - q1[3] * q2[2] - q1[0] * q2[1] + q1[1] * q2[0]
    sign = 1.0
    # make_unique keeps the real part non-negative (isaaclab.utils.math.quat_unique)
    if make_unique != 0 and w < 0.0:
        sign = -1.0
    out[i] = wp.vec4(sign * x, sign * y, sign * z, sign * w)


def _as_wp(tensor: torch.Tensor, dtype) -> wp.array:
    """View a contiguous float tensor as a Warp array of *dtype*."""
    return wp.from_torch(tensor.contiguous(), dtype=dtype)


def compute_goal_quat_error(
    asset_quat: torch.Tensor, goal_quat: torch.Tensor, make_quat_unique: bool, out: torch.Tensor
) -> torch.Tensor:
    """Compute the quaternion error between asset and goal orientations.

    Args:
        asset_quat: Asset ``(x, y, z, w)`` orientations, shape ``(num_envs, 4)``.
        goal_quat: Goal ``(x, y, z, w)`` orientations, shape ``(num_envs, 4)``.
        make_quat_unique: Flip the sign so the real part is always non-negative.
        out: Caller-owned output buffer, shape ``(num_envs, 4)``, float32.

    Returns:
        ``out`` filled with per-environment quaternion errors.
    """
    wp.launch(
        _goal_quat_error_kernel,
        dim=out.shape[0],
        inputs=[_as_wp(asset_quat, wp.vec4), _as_wp(goal_quat, wp.vec4), int(make_quat_unique)],
        outputs=[wp.from_torch(out, dtype=wp.vec4)],
        device=wp.device_from_torch(out.device),
    )
    return out


class goal_quat_diff(ManagerTermBase):
    """Goal orientation relative to the asset's root frame.

    The real part is always positive when ``make_quat_unique`` is set.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._out = torch.empty(env.num_envs, 4, dtype=torch.float32, device=env.device)

    def __call__(
        self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, make_quat_unique: bool
    ) -> torch.Tensor:
        """Return the per-environment quaternion error, shape ``(num_envs, 4)``."""
        asset: RigidObject = env.scene[asset_cfg.name]
        command_term: ReorientCommand = env.command_manager.get_term(command_name)
        return compute_goal_quat_error(
            asset.data.root_quat_w.torch, command_term.command[:, 3:7], make_quat_unique, self._out
        )


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


class OpenAIPolicyObservation(ManagerTermBase):
    """Apply one stateful noise model to the concatenated OpenAI actor observation."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        noise_model: NoiseModelCfg = cfg.params["noise_model"]
        self._noise_model = noise_model.class_type(noise_model, num_envs=self.num_envs, device=self.device)
        self._quat_error = torch.empty(env.num_envs, 4, dtype=torch.float32, device=env.device)
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
        object_asset: RigidObject = env.scene[object_cfg.name]
        object_pos = object_asset.data.root_pos_w.torch - env.scene.env_origins
        command_term: ReorientCommand = env.command_manager.get_term(command_name)
        compute_goal_quat_error(
            object_asset.data.root_quat_w.torch, command_term.command[:, 3:7], False, self._quat_error
        )
        # Direct actor-observation order: fingertips, object position, goal quat error, last action
        observation = torch.cat(
            (fingertip_pos(env, robot_cfg), object_pos, self._quat_error, reorient_last_action(env, action_name)),
            dim=-1,
        )
        if self._shape_probe_pending:
            return observation
        return self._noise_model(observation)
