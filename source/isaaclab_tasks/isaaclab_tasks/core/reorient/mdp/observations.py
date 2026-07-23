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


CUBE_HALF_SIZE: tuple[float, float, float] = (0.03, 0.03, 0.03)
"""Half side lengths [m] of the reorientation cube."""


def _cube_corner_offsets(
    size: tuple[float, float, float], num_keypoints: int, device: torch.device | str
) -> torch.Tensor:
    """Corner offsets [m] from the cube center; corner index bits select the +/- half side per axis."""
    signs = torch.tensor(
        [[1 - 2 * ((corner >> axis) & 1) for axis in range(3)] for corner in range(num_keypoints)],
        dtype=torch.float32,
        device=device,
    )
    half_size = torch.tensor(size, dtype=torch.float32, device=device) / 2.0
    return signs * half_size


def compute_cube_keypoints(
    pose: torch.Tensor,
    num_keypoints: int = 8,
    size: tuple[float, float, float] = (2 * 0.03, 2 * 0.03, 2 * 0.03),
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute cube-corner positions for batched poses.

    Args:
        pose: Cube center poses ``(x, y, z, qx, qy, qz, qw)`` [m, unit quaternion].
        num_keypoints: Number of binary-sign corners to compute.
        size: Cube side lengths along each axis [m].
        out: Optional output buffer [m], shape ``(num_envs, num_keypoints, 3)``.

    Returns:
        Cube-corner positions [m], shape ``(num_envs, num_keypoints, 3)``.
    """
    num_envs = pose.shape[0]
    corners = _cube_corner_offsets(size, num_keypoints, pose.device)
    rotated = math_utils.quat_apply(
        pose[:, None, 3:7].expand(num_envs, num_keypoints, 4), corners.expand(num_envs, num_keypoints, 3)
    )
    keypoints = pose[:, None, 0:3] + rotated
    if out is None:
        return keypoints
    out.copy_(keypoints)
    return out


def cube_keypoints_from_quat(
    quat: torch.Tensor,
    half_size: tuple[float, float, float] = CUBE_HALF_SIZE,
    num_keypoints: int = 8,
) -> torch.Tensor:
    """Rotation-only cube-corner offsets [m] from batched ``(x, y, z, w)`` orientations.

    Args:
        quat: Cube orientations, shape ``(num_envs, 4)``.
        half_size: Cube half side lengths along each axis [m].
        num_keypoints: Number of binary-sign corners to compute.

    Returns:
        Flattened corner offsets [m], shape ``(num_envs, num_keypoints * 3)``.
    """
    num_envs = quat.shape[0]
    size = (2.0 * half_size[0], 2.0 * half_size[1], 2.0 * half_size[2])
    corners = _cube_corner_offsets(size, num_keypoints, quat.device)
    rotated = math_utils.quat_apply(
        quat[:, None, :].expand(num_envs, num_keypoints, 4), corners.expand(num_envs, num_keypoints, 3)
    )
    return rotated.reshape(num_envs, num_keypoints * 3)


def goal_quat_diff(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, make_quat_unique: bool
) -> torch.Tensor:
    """Goal orientation relative to the asset's root frame.

    The real part is always positive when ``make_quat_unique`` is set.

    Args:
        env: The environment object.
        asset_cfg: The scene entity whose root orientation is compared.
        command_name: The command term to be used for extracting the goal.
        make_quat_unique: Whether to keep the quaternion real part non-negative.

    Returns:
        Per-environment quaternion error ``asset * conjugate(goal)`` in ``(x, y, z, w)`` order.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command_term: ReorientCommand = env.command_manager.get_term(command_name)
    quat_error = math_utils.quat_mul(
        asset.data.root_quat_w.torch, math_utils.quat_conjugate(command_term.quat_command_w)
    )
    return math_utils.quat_unique(quat_error) if make_quat_unique else quat_error


def fingertip_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Flattened fingertip positions in the environment frame [m], shape ``(num_envs, num_fingertips * 3)``."""
    asset = env.scene[asset_cfg.name]
    positions = asset.data.body_pos_w.torch[:, asset_cfg.body_ids] - env.scene.env_origins.unsqueeze(1)
    return positions.reshape(env.num_envs, -1)


def fingertip_quat(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Flattened fingertip ``(x, y, z, w)`` orientations, shape ``(num_envs, num_fingertips * 4)``."""
    asset = env.scene[asset_cfg.name]
    return asset.data.body_quat_w.torch[:, asset_cfg.body_ids].reshape(env.num_envs, -1)


def fingertip_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Flattened fingertip spatial velocities [m/s, rad/s], shape ``(num_envs, num_fingertips * 6)``."""
    asset = env.scene[asset_cfg.name]
    return asset.data.body_vel_w.torch[:, asset_cfg.body_ids].reshape(env.num_envs, -1)


class fingertip_wrench(ManagerTermBase):
    """Fingertip reaction wrenches [N, N·m] with Direct-compatible zero fallback."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        body_ids = cfg.params["sensor_cfg"].body_ids
        # Direct-compatible fallback: report zero wrenches until the sensor produces data
        self._zeros = torch.zeros(env.num_envs, len(body_ids) * 6, dtype=torch.float32, device=env.device)

    def __call__(self, env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
        """Return the flattened wrench block, shape ``(num_envs, num_fingertips * 6)``."""
        sensor: JointWrenchSensor = env.scene.sensors[sensor_cfg.name]
        force_data = sensor.data.force
        torque_data = sensor.data.torque
        if force_data is None or torque_data is None:
            return self._zeros
        force = force_data.torch[:, sensor_cfg.body_ids]
        torque = torque_data.torch[:, sensor_cfg.body_ids]
        return torch.cat((force, torque), dim=-1).reshape(env.num_envs, -1)


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
        quat_error = math_utils.quat_mul(
            object_asset.data.root_quat_w.torch, math_utils.quat_conjugate(command_term.quat_command_w)
        )
        fingertips = fingertip_pos(env, robot_cfg)
        # Direct actor-observation order: fingertips, object position, goal quat error, last action
        observation = torch.cat(
            (fingertips, object_pos, quat_error, reorient_last_action(env, action_name)),
            dim=-1,
        )
        if self._shape_probe_pending:
            return observation
        return self._noise_model(observation)


# ---------------------------------------------------------------------------
# Shadow Hand camera observation terms.
#
# These terms wrap the CNN feature pipeline defined in the shadow-hand config
# package. The config layer imports the mdp layer, so the FeatureExtractor
# machinery is imported lazily at term construction time.
# ---------------------------------------------------------------------------
