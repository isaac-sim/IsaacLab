# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions specific to the in-hand dexterous manipulation environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from collections.abc import Sequence

    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import Camera

    from isaaclab_tasks.core.reorient.config.shadow_hand.feature_extractor import FeatureExtractorCfg

    from .commands import ReorientCommand


CUBE_HALF_SIZE: tuple[float, float, float] = (0.03, 0.03, 0.03)
"""Half side lengths [m] of the reorientation cube."""


# -- cube keypoint helpers, shared by the camera and state observation terms
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
    # Vectorized over corners: the earlier implementation looped over the eight corners,
    # allocating a tensor and calling quat_apply once per corner. The corner sign-offsets
    # are pose-independent, so they are built once and all num_keypoints corners are rotated
    # by the pose in a single batched quat_apply — mathematically identical, no Python loop.
    num_envs = pose.shape[0]
    corners = _cube_corner_offsets(size, num_keypoints, pose.device)
    # Broadcast each env's quaternion across its corners and rotate every offset at once.
    rotated = math_utils.quat_apply(
        pose[:, None, 3:7].expand(num_envs, num_keypoints, 4), corners.expand(num_envs, num_keypoints, 3)
    )
    # Translate the rotated offsets by the cube-center position to get world-frame corners.
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


# -- command terms
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


# -- fingertip terms
# Task-local because the framework provides body_pose_w but no body_pos_w or body_vel_w.
def fingertip_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Flattened fingertip positions in the environment frame [m], shape ``(num_envs, num_fingertips * 3)``."""
    asset = env.scene[asset_cfg.name]
    positions = asset.data.body_pos_w.torch[:, asset_cfg.body_ids] - env.scene.env_origins.unsqueeze(1)
    return positions.reshape(env.num_envs, -1)


def fingertip_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Flattened fingertip spatial velocities [m/s, rad/s], shape ``(num_envs, num_fingertips * 6)``."""
    asset = env.scene[asset_cfg.name]
    return asset.data.body_vel_w.torch[:, asset_cfg.body_ids].reshape(env.num_envs, -1)


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


class ShadowHandCameraFeatures(ManagerTermBase):
    """Run the Direct camera feature pipeline as one Manager observation term."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        camera: Camera = env.scene.sensors[sensor_cfg.name]
        # Runtime-only import: the mdp layer must not import the task-config layer
        # at module load (config modules import mdp; see the layering note above).
        from isaaclab_tasks.core.reorient.config.shadow_hand.feature_extractor import FeatureExtractor

        feature_extractor_cfg: FeatureExtractorCfg = cfg.params["feature_extractor_cfg"]
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
            feature_extractor_cfg: Feature-extractor configuration.
            sensor_cfg: Tiled-camera scene entity.
            object_cfg: Reoriented-object scene entity.

        Returns:
            Predicted object position and cube keypoints [m], shape
            ``(num_envs, 27)``.
        """
        del feature_extractor_cfg  # consumed in __init__
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
