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

from ..shoelace_assets import read_shoelace_attribute

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
        positions = read_shoelace_attribute(curve, "settledPositions", (cable.num_segments, 3))
        orientations = read_shoelace_attribute(curve, "settledOrientations", (cable.num_segments, 4), quaternion=True)
        source_env = env.sim.stage.GetPrimAtPath(env.scene.env_prim_paths[0])
        transform = _source_transform(curve, source_env)
        matrix = np.asarray(transform)
        positions = positions @ matrix[:3, :3] + matrix[3, :3]
        rotation = transform.ExtractRotationQuat()
        default_pose = cable.data.default_segment_pose_w.torch
        # USD retains the source prototype; world placement comes from the clone plan.
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


def mirror_arm_joints(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice,
    source_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    mirror_signs: tuple[float, ...],
) -> None:
    """Mirror an already-reset arm's state and synchronize the other arm's position targets.

    Run after the source arm reset. Both entity selections must preserve corresponding joint order.
    This retains reset noise while making its realization symmetric, including partial resets.

    Args:
        env: Environment containing the two arms.
        env_ids: Environments to reset.
        source_cfg: Arm whose reset positions and velocities are mirrored.
        asset_cfg: Arm receiving the mirrored state.
        mirror_signs: Axial reflection signs in the selections' common joint order.
    """
    source: Articulation = env.scene[source_cfg.name]
    target: Articulation = env.scene[asset_cfg.name]
    joint_pos = source.data.joint_pos.torch[env_ids][:, source_cfg.joint_ids]
    signs = joint_pos.new_tensor(mirror_signs)
    joint_pos = joint_pos * signs
    joint_vel = source.data.joint_vel.torch[env_ids][:, source_cfg.joint_ids] * signs
    target.write_joint_position_to_sim_index(position=joint_pos, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
    target.write_joint_velocity_to_sim_index(velocity=joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
    target.actuators.target_command.set_position_index(value=joint_pos, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


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


def _source_transform(curve: Usd.Prim, source_env: Usd.Prim) -> Gf.Matrix4d:
    """Return the default-time curve-to-environment transform, with translation in [m].

    The generated asset is meter-based. Raise ``ValueError`` for scale, shear, or
    reflection, which violate the baked startup data's rigid-transform assumption.
    """
    transform, _ = UsdGeom.XformCache(Usd.TimeCode.Default()).ComputeRelativeTransform(curve, source_env)
    linear = np.asarray(transform)[:3, :3]
    if not np.allclose(linear @ linear.T, np.eye(3), atol=1.0e-6) or np.linalg.det(linear) < 0.0:
        raise ValueError(f"{curve.GetPath()}: bake scale/shear before using shoelace startup data")
    return transform
