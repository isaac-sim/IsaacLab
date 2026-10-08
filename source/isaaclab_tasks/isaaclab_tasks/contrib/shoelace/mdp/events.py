# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset randomization for the dual-Franka shoelace task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from pxr import Gf, Usd, UsdGeom

from isaaclab.assets import Articulation, CableObject, RigidObject
from isaaclab.envs.mdp.events import reset_joints_by_offset
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_mul, sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def install_settled_default_state(env: ManagerBasedEnv, env_ids: torch.Tensor | None) -> None:
    """Install the offline gravity-settled cable poses as defaults at startup.

    Args:
        env: Environment containing both shoelace cables.
        env_ids: Unused; startup installs defaults for every environment.
    """
    for name, side in (("shoelace_left", "Left"), ("shoelace_right", "Right")):
        cable: CableObject = env.scene[name]
        curve = _source_curve(env, side)
        positions = _startup_array(curve, "settledPositions", (cable.num_segments, 3))
        orientations = _startup_array(curve, "settledOrientations", (cable.num_segments, 4), quaternion=True)
        transform = _source_transform(curve)
        matrix = np.asarray(transform)
        positions = positions @ matrix[:3, :3] + matrix[3, :3]
        rotation = transform.ExtractRotationQuat()
        default_pose = cable.data.default_segment_pose_w.torch
        # Remove the source environment origin before broadcasting to replicated environments.
        positions -= env.scene.env_origins[0].cpu().numpy()
        default_pose[..., :3] = default_pose.new_tensor(positions)
        default_pose[..., :3] += env.scene.env_origins.unsqueeze(1)
        local_rotation = default_pose.new_tensor(orientations)
        world_rotation = (*rotation.GetImaginary(), rotation.GetReal())
        # Preserve the baked float32 state exactly when no rotation needs composing.
        if world_rotation == (0.0, 0.0, 0.0, 1.0):
            default_pose[..., 3:] = local_rotation
        else:
            default_pose[..., 3:] = quat_mul(
                default_pose.new_tensor(world_rotation).expand(cable.num_segments, -1), local_rotation
            )
        velocity = cable.data.default_segment_velocity_w.torch
        velocity.zero_()
        cable.write_segment_pose_to_sim_index(segment_pose=default_pose)
        cable.write_segment_velocity_to_sim_index(segment_velocity=velocity)


def reset_arm_joints(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice,
    position_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Perturb arm joints about their defaults and synchronize position targets.

    Args:
        env: Environment containing the arm.
        env_ids: Environments to reset.
        position_range: Uniform joint position offset bounds [rad].
        asset_cfg: Arm articulation and joints to randomize; exclude finger joints.
    """
    reset_joints_by_offset(env, env_ids, position_range, (0.0, 0.0), asset_cfg)
    robot: Articulation = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos.torch[env_ids][:, asset_cfg.joint_ids]
    robot.actuators.target_command.set_position_index(value=joint_pos, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


def reset_shoe_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice,
    position_range: dict[str, tuple[float, float]],
) -> None:
    """Translate the shoe and both laces together about their default positions.

    The pinned lace mesh is a child of the shoe; cable segment poses include the
    fixed anchors. Sampling from defaults avoids accumulating offsets across resets.
    Run after ``reset_scene_to_default`` to restore default velocities as well.

    Args:
        env: Shoelace environment.
        env_ids: Environments to reset.
        position_range: Uniform translation offset bounds [m], keyed by ``x``, ``y``,
            or ``z``. Omitted axes have zero offset.
    """
    shoe: RigidObject = env.scene["shoe"]
    bounds = torch.tensor([position_range.get(axis, (0.0, 0.0)) for axis in "xyz"], device=env.device)
    root_pose = shoe.data.default_root_pose.torch[env_ids].clone()
    offset = sample_uniform(bounds[:, 0], bounds[:, 1], (root_pose.shape[0], 3), env.device)
    root_pose[:, :3] += env.scene.env_origins[env_ids] + offset
    shoe.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)

    for name in ("shoelace_left", "shoelace_right"):
        cable: CableObject = env.scene[name]
        segment_pose = cable.data.default_segment_pose_w.torch[env_ids].clone()
        segment_pose[..., :3] += offset.unsqueeze(1)
        cable.write_segment_pose_to_sim_index(segment_pose=segment_pose, env_ids=env_ids)


def _source_curve(env: ManagerBasedEnv, side: str) -> Usd.Prim:
    """Return the first environment's ``Left`` or ``Right`` cable prim.

    Physics-only replication retains this source prim. Raise ``ValueError`` if missing.
    """
    path = f"{env.scene.env_prim_paths[0]}/ShoelaceScene/Shoelace{side}/geometry/mesh"
    curve = env.sim.stage.GetPrimAtPath(path)
    if not curve:
        raise ValueError(f"Missing shoelace source curve: {path}")
    return curve


def _startup_array(curve: Usd.Prim, name: str, shape: tuple[int, ...], quaternion: bool = False) -> np.ndarray:
    """Read and validate task-local startup data baked into a cable prim.

    Args:
        curve: Source cable geometry prim.
        name: Attribute name without the ``shoelace:`` prefix.
        shape: Expected array shape; ``()`` denotes a scalar.
        quaternion: Whether to require unit-length xyzw quaternions.

    Returns:
        Float64 values with the requested shape. Positions and lengths are in [m];
        orientations are dimensionless xyzw quaternions.

    Raises:
        ValueError: If the attribute is missing, has an unexpected shape, contains
            nonfinite values, or fails the requested quaternion validation.
    """
    value = curve.GetAttribute(f"shoelace:{name}").Get()
    if value is None:
        raise ValueError(f"{curve.GetPath()}: missing shoelace:{name}; regenerate shoelace.usda")
    values = np.asarray(value, dtype=np.float64)
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError(f"{curve.GetPath()}: shoelace:{name} must have finite shape {shape}")
    if quaternion and not np.allclose(np.linalg.norm(values, axis=-1), 1.0, atol=2.0e-5):
        raise ValueError(f"{curve.GetPath()}: shoelace:{name} must contain unit xyzw quaternions")
    return values


def _source_transform(curve: Usd.Prim) -> Gf.Matrix4d:
    """Return the default-time curve-to-world transform, with translation in [m].

    The generated asset is meter-based. Raise ``ValueError`` for scale, shear, or
    reflection, which violate the baked startup data's rigid-transform assumption.
    """
    transform = UsdGeom.Xformable(curve).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    linear = np.asarray(transform)[:3, :3]
    if not np.allclose(linear @ linear.T, np.eye(3), atol=1.0e-6) or np.linalg.det(linear) < 0.0:
        raise ValueError(f"{curve.GetPath()}: bake scale/shear before using shoelace startup data")
    return transform
