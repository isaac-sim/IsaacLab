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
    from isaaclab.sensors import Camera, JointWrenchSensor

    from isaaclab_tasks.core.reorient.config.shadow_hand.feature_extractor import FeatureExtractorCfg

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


class ShadowHandCameraFeatures(ManagerTermBase):
    """Run the Direct camera feature pipeline as one Manager observation term."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        camera: Camera = env.scene.sensors[sensor_cfg.name]
        # Runtime-only import: the mdp layer must not import the task-config layer
        # at module load (config modules import mdp; see the layering note above).
        from isaaclab_tasks.core.reorient.config.shadow_hand.feature_extractor import FeatureExtractor

        feature_extractor_cfg: FeatureExtractorCfg = env.cfg.feature_extractor
        self._feature_extractor = FeatureExtractor(
            feature_extractor_cfg,
            env.device,
            camera.cfg.data_types,
            env.cfg.log_dir,
            height=camera.cfg.height,
            width=camera.cfg.width,
        )
        # ObservationManager calls terms once to infer their shape. Do not train
        # or save a CNN checkpoint during that initialization probe.
        self._shape_probe_pending = True
        self._keypoints_buf = torch.empty(env.num_envs, 8, 3, dtype=torch.float32, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Finish the shape-probe phase on the first Manager reset.

        Args:
            env_ids: Environment indices being reset. The feature extractor
                has no per-environment state, so the indices are unused.
        """
        del env_ids
        if self._shape_probe_pending:
            self._shape_probe_pending = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        feature_extractor_cfg: FeatureExtractorCfg,
        sensor_cfg: SceneEntityCfg,
        object_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Return the detached 27-dimensional cube-pose embedding.

        Args:
            env: Environment containing the object and tiled camera.
            feature_extractor_cfg: Feature-extractor configuration captured by
                the observation term. The initialized extractor owns its copy.
            sensor_cfg: Tiled-camera scene entity.
            object_cfg: Reoriented-object scene entity.

        Returns:
            Predicted object position and cube keypoints [m], shape
            ``(num_envs, 27)``.
        """
        del feature_extractor_cfg
        if self._shape_probe_pending:
            embeddings = torch.zeros(env.num_envs, 27, dtype=torch.float32, device=env.device)
            env._shadow_hand_camera_embeddings = embeddings
            return embeddings

        camera: Camera = env.scene.sensors[sensor_cfg.name]
        object_asset: RigidObject = env.scene[object_cfg.name]
        object_pos = object_asset.data.root_pos_w.torch - env.scene.env_origins
        object_pose = torch.cat((object_pos, object_asset.data.root_quat_w.torch), dim=-1)
        keypoints = compute_cube_keypoints(object_pose, out=self._keypoints_buf)
        target = torch.cat((object_pos, keypoints.flatten(start_dim=1)), dim=-1)
        camera_output = {
            data_type: value if isinstance(value, torch.Tensor) else value.torch
            for data_type, value in camera.data.output.items()
        }
        pose_loss, embeddings = self._feature_extractor.step(camera_output, target)
        embeddings = embeddings.clone().detach()
        env._shadow_hand_camera_embeddings = embeddings
        if pose_loss is not None:
            env.extras.setdefault("log", {})["pose_loss"] = pose_loss
        return embeddings


def shadow_hand_camera_cached_features(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return camera features computed by the preceding policy observation group.

    Args:
        env: Environment whose policy group cached the current camera embedding.

    Returns:
        Detached camera embeddings, shape ``(num_envs, 27)``.
    """
    embeddings = getattr(env, "_shadow_hand_camera_embeddings", None)
    if embeddings is None:
        raise RuntimeError("Shadow Hand camera policy features must be computed before critic observations.")
    return embeddings


def shadow_hand_goal_keypoints(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Flattened zero-origin cube keypoints [m] for the current goal orientation.

    Args:
        env: Environment containing the goal command term.
        command_name: Goal command term name.

    Returns:
        Flattened zero-origin cube keypoints [m], shape ``(num_envs, 24)``.
    """
    command_term = env.command_manager.get_term(command_name)
    return cube_keypoints_from_quat(command_term.quat_command_w)
