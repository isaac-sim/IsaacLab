# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batched DLS target guarding against G1 arm/torso overlap."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from pxr import Usd, UsdPhysics

from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.sim import get_current_stage
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .collision_ik_cfg import G1CollisionAwareIKActionCfg


_JOINT_PARTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
# Overlapping spheres fitted to source visual bounds, in their link frames. The
# shoulder attachment is deliberately excluded, since it meets the torso by design.
_ARM_SPHERES = (
    (2, (0.0075, 0.0, -0.02), 0.043),
    (2, (0.0075, 0.0, -0.05), 0.043),
    (2, (0.0075, 0.0, -0.08), 0.043),
    (3, (0.0, 0.0, -0.008), 0.040),
    (3, (0.03, 0.0, -0.008), 0.040),
    (3, (0.06, 0.0, -0.008), 0.040),
    (3, (0.09, 0.0, -0.008), 0.040),
    (4, (0.015, 0.0, 0.0), 0.034),
    (4, (0.04, 0.0, 0.0), 0.034),
    (5, (0.0, 0.0, 0.0), 0.033),
    (5, (0.035, 0.0, 0.0), 0.033),
    (5, (0.065, 0.0, 0.0), 0.033),
    (6, (0.0, 0.0, 0.0), 0.045),
    (6, (0.035, 0.0, 0.0), 0.045),
    (6, (0.07, 0.0, 0.0), 0.045),
    (6, (0.105, 0.0, 0.0), 0.045),
    (6, (0.14, 0.0, 0.0), 0.045),
)


def sphere_box_distance(
    points: torch.Tensor, radii: torch.Tensor, center: torch.Tensor, half_extents: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return signed surface distances and outward point gradients for an axis-aligned box.

    Inputs broadcast over leading dimensions; points have trailing dimension 3.
    Interior points use the nearest face, including a deterministic choice at ties.
    """
    relative = points - center
    face_distance = relative.abs() - half_extents
    outside = face_distance.clamp_min(0.0)
    outside_norm = outside.norm(dim=-1)
    nearest, axis = face_distance.max(dim=-1)
    sign = torch.where(relative >= 0.0, 1.0, -1.0)
    face_normal = torch.nn.functional.one_hot(axis, 3).to(points.dtype) * sign
    normal = torch.where(
        (outside_norm > 1.0e-9).unsqueeze(-1),
        outside * sign / outside_norm.unsqueeze(-1).clamp_min(1.0e-9),
        face_normal,
    )
    return outside_norm + nearest.clamp_max(0.0) - radii, normal


def project_bounded_step(
    delta: torch.Tensor,
    gradient: torch.Tensor,
    required: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    iterations: int,
) -> torch.Tensor:
    """Project toward violated distance half-spaces while respecting joint-step bounds.

    This bounded iteration is not an exact QP solver. The caller must validate the
    resulting target with nonlinear geometry and reject infeasible updates.
    """
    result = torch.maximum(torch.minimum(delta, upper), lower)
    norm_sq = gradient.square().sum(dim=-1).clamp_min(1.0e-12)
    for _ in range(iterations):
        violation = required - torch.einsum("nkj,nj->nk", gradient, result)
        index = (violation.clamp_min(0.0) / norm_sq.sqrt()).argmax(dim=-1)
        selected = gradient.gather(1, index[:, None, None].expand(-1, 1, delta.shape[-1])).squeeze(1)
        amount = violation.gather(1, index[:, None]).clamp_min(0.0) / norm_sq.gather(1, index[:, None])
        result = torch.maximum(torch.minimum(result + amount * selected, upper), lower)
    return result


class G1CollisionAwareIKAction(DifferentialInverseKinematicsAction):
    """Keep the existing 6D DLS action and guard each joint target with analytic FK.

    USD joint frames define the seven-joint chain. All per-step computations stay
    on the articulation device. Linearized corrections are followed by exact FK
    backtracking; already-unsafe measurements may recover but may not command a
    further decrease in any unsafe distance. Actual simulated tracking is checked
    separately by :func:`g1_arm_body_collision` after each environment step.
    """

    cfg: G1CollisionAwareIKActionCfg

    def __init__(self, cfg: G1CollisionAwareIKActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        if cfg.side not in ("left", "right") or self._num_joints != 7:
            raise ValueError("G1 collision-aware IK requires a left/right seven-joint arm")
        if min(cfg.minimum_clearance, cfg.max_joint_step, cfg.recovery_step) <= 0:
            raise ValueError("Clearance, joint step and recovery step must be positive")
        if cfg.projection_iterations < 1 or cfg.backtrack_steps < 1:
            raise ValueError("Projection and backtracking iterations must be positive")
        names = [f"{cfg.side}_{part}_joint" for part in _JOINT_PARTS]
        self._chain_order = [self._joint_names.index(name) for name in names]
        self._action_order = [names.index(name) for name in self._joint_names]
        self._chain_joint_ids = [self._asset.joint_names.index(name) for name in names]
        root = get_current_stage().GetPrimAtPath(
            f"{env.scene.env_prim_paths[0]}/{self._asset.cfg.prim_path.rsplit('/', 1)[-1]}"
        )
        joints = {
            p.GetName(): UsdPhysics.RevoluteJoint(p) for p in Usd.PrimRange(root) if p.IsA(UsdPhysics.RevoluteJoint)
        }
        parent_positions, parent_rotations, child_positions, child_rotations, axes = [], [], [], [], []
        for i, name in enumerate(names):
            joint = joints[name]
            parent = cfg.torso_body if i == 0 else f"{cfg.side}_{_JOINT_PARTS[i - 1]}_link"
            child = f"{cfg.side}_{_JOINT_PARTS[i]}_link"
            if joint.GetBody0Rel().GetTargets()[0].name != parent or joint.GetBody1Rel().GetTargets()[0].name != child:
                raise ValueError(f"Unexpected G1 joint topology at {name}")
            parent_positions.append(tuple(joint.GetLocalPos0Attr().Get()))
            child_positions.append(tuple(joint.GetLocalPos1Attr().Get()))
            for attr, target in (
                (joint.GetLocalRot0Attr(), parent_rotations),
                (joint.GetLocalRot1Attr(), child_rotations),
            ):
                rotation = attr.Get()
                target.append((*rotation.GetImaginary(), rotation.GetReal()))
            axes.append(tuple(float(j == "XYZ".index(joint.GetAxisAttr().Get())) for j in range(3)))

        def tensor(value):
            return torch.tensor(value, dtype=torch.float32, device=self.device)

        self._parent_positions = tensor(parent_positions)
        self._parent_rotations = tensor(parent_rotations)
        self._child_positions = tensor(child_positions)
        self._child_rotations_inv = quat_conjugate(tensor(child_rotations))
        self._joint_axes = tensor(axes)
        self._sphere_links = torch.tensor([s[0] for s in _ARM_SPHERES], device=self.device)
        self._sphere_offsets = tensor([s[1] for s in _ARM_SPHERES])
        self._sphere_radii = tensor([s[2] for s in _ARM_SPHERES])
        self._ancestor_mask = torch.arange(7, device=self.device)[None, :] <= self._sphere_links[:, None]
        self._box_center = tensor(cfg.torso_center)
        self._box_half_extents = tensor(cfg.torso_half_extents)
        self._fractions = tensor([0.5**i for i in range(cfg.backtrack_steps)] + [0.0])

    def forward_kinematics(self, joint_pos: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return link poses and joint origins/axes in the torso frame, for arbitrary batches."""
        position = torch.zeros_like(joint_pos[:, :3])
        rotation = torch.zeros_like(joint_pos[:, :4])
        rotation[:, 3] = 1.0
        positions, rotations, origins, axes = [], [], [], []
        for i in range(7):
            origin = position + quat_apply(rotation, self._parent_positions[i].expand_as(position))
            joint_rotation = quat_mul(rotation, self._parent_rotations[i].expand_as(rotation))
            axis = quat_apply(joint_rotation, self._joint_axes[i].expand_as(position))
            angle = joint_pos[:, i : i + 1] * 0.5
            motion = torch.cat((self._joint_axes[i] * angle.sin(), angle.cos()), dim=-1)
            rotation = quat_mul(quat_mul(joint_rotation, motion), self._child_rotations_inv[i].expand_as(rotation))
            position = origin - quat_apply(rotation, self._child_positions[i].expand_as(position))
            positions.append(position)
            rotations.append(rotation)
            origins.append(origin)
            axes.append(axis)
        return tuple(torch.stack(items, dim=1) for items in (positions, rotations, origins, axes))

    def clearance(self, joint_pos: torch.Tensor, *, with_gradient: bool = False):
        """Evaluate arm sphere distances, optionally with their analytic joint gradients."""
        positions, rotations, origins, axes = self.forward_kinematics(joint_pos)
        points = positions[:, self._sphere_links] + quat_apply(
            rotations[:, self._sphere_links], self._sphere_offsets.expand(joint_pos.shape[0], -1, -1)
        )
        distance, normal = sphere_box_distance(points, self._sphere_radii, self._box_center, self._box_half_extents)
        if not with_gradient:
            return distance
        lever = points[:, :, None, :] - origins[:, None, :, :]
        jacobian = torch.linalg.cross(axes[:, None, :, :].expand_as(lever), lever, dim=-1)
        gradient = (jacobian * normal[:, :, None, :]).sum(dim=-1) * self._ancestor_mask
        return distance, gradient

    def apply_actions(self):
        ee_pos, ee_quat = self._compute_frame_pose()
        measured = self._asset.data.joint_pos.torch[:, self._joint_ids]
        proposed = self._ik_controller.compute(ee_pos, ee_quat, self._compute_frame_jacobian(), measured)
        current = measured[:, self._chain_order]
        desired_delta = (proposed - measured)[:, self._chain_order]
        distance, gradient = self.clearance(current, with_gradient=True)
        limits = self._asset.data.soft_joint_pos_limits.torch[:, self._chain_joint_ids]
        # Include the current state in the box so an externally displaced joint can
        # recover monotonically instead of creating contradictory step bounds.
        lower = (torch.minimum(limits[..., 0], current) - current).clamp_min(-self.cfg.max_joint_step)
        upper = (torch.maximum(limits[..., 1], current) - current).clamp_max(self.cfg.max_joint_step)
        required = (self.cfg.minimum_clearance - distance).clamp_max(self.cfg.recovery_step)
        finite = torch.isfinite(desired_delta).all(dim=-1) & torch.isfinite(distance).all(dim=-1)
        delta = project_bounded_step(
            torch.nan_to_num(desired_delta), gradient, required, lower, upper, self.cfg.projection_iterations
        )
        candidates = current[:, None, :] + self._fractions[None, :, None] * delta[:, None, :]
        candidate_distance = self.clearance(candidates.flatten(0, 1)).reshape(self.num_envs, -1, distance.shape[-1])
        threshold = distance.clamp_max(self.cfg.minimum_clearance)
        valid = (candidate_distance >= threshold[:, None, :] - 1.0e-6).all(dim=-1) & finite[:, None]
        index = valid.to(torch.int32).argmax(dim=1)
        selected = candidates[torch.arange(self.num_envs, device=self.device), index]
        selected = torch.where(valid.any(dim=1, keepdim=True), selected, current)
        self._asset.set_joint_position_target_index(target=selected[:, self._action_order], joint_ids=self._joint_ids)


def g1_arm_body_collision(env: ManagerBasedRLEnv, penetration_limit: float = 0.005) -> torch.Tensor:
    """Terminate measured arm/torso proxy penetration, independently of commanded targets."""
    actions = [env.action_manager.get_term(name) for name in ("left_arm", "right_arm")]
    distances = torch.stack(
        [
            action.clearance(action._asset.data.joint_pos.torch[:, action._chain_joint_ids]).amin(dim=1)
            for action in actions
        ],
        dim=1,
    )
    return (~torch.isfinite(distances).all(dim=1)) | (distances < -penetration_limit).any(dim=1)
