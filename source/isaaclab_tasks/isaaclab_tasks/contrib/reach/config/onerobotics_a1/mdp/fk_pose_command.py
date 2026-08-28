# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Forward-kinematics-based reachable pose command implementation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands import UniformPoseCommand
from isaaclab.utils.math import quat_from_matrix, quat_unique

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .fk_pose_command_cfg import FkReachablePoseCommandCfg


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> torch.Tensor:
    """Convert a URDF fixed-axis roll-pitch-yaw rotation to a matrix."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = torch.tensor([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=torch.float64)
    ry = torch.tensor([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=torch.float64)
    rz = torch.tensor([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=torch.float64)
    return rz @ ry @ rx


class FkReachablePoseCommand(UniformPoseCommand):
    """Generate reachable end-effector poses from random in-limit A1 joint positions."""

    cfg: FkReachablePoseCommandCfg

    def __init__(self, cfg: FkReachablePoseCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        joint_patterns = [entry[0] for entry in cfg.chain]
        joint_ids, joint_names = self.robot.find_joints(joint_patterns, preserve_order=True)
        if len(joint_ids) != len(cfg.chain):
            raise ValueError(f"Expected {len(cfg.chain)} A1 chain joints, resolved {len(joint_ids)}: {joint_names}")
        self._chain_joint_ids = joint_ids

        origins = torch.eye(4, dtype=torch.float64).repeat(len(cfg.chain), 1, 1)
        axes = torch.empty((len(cfg.chain), 3), dtype=torch.float64)
        for index, entry in enumerate(cfg.chain):
            _, xyz, rpy = entry[:3]
            origins[index, :3, :3] = _rpy_to_matrix(*rpy)
            origins[index, :3, 3] = torch.tensor(xyz, dtype=torch.float64)
            axes[index] = torch.tensor(entry[3] if len(entry) == 4 else (0.0, 0.0, 1.0), dtype=torch.float64)
        self._origins = origins.to(self.device)
        self._joint_axes = torch.nn.functional.normalize(axes, dim=-1).to(self.device)

        fixed_transform = torch.eye(4, dtype=torch.float64)
        if cfg.fixed_transform is not None:
            xyz, rpy = cfg.fixed_transform
            fixed_transform[:3, :3] = _rpy_to_matrix(*rpy)
            fixed_transform[:3, 3] = torch.tensor(xyz, dtype=torch.float64)
        self._fixed_transform = fixed_transform.to(self.device)

        # Keep the original single-arm computation as a separate exact path. In
        # particular, do not replace its z-axis matrices with generic Rodrigues
        # matrices: even equivalent arithmetic could perturb existing commands.
        self._use_legacy_z_axis_fk = cfg.fixed_transform is None and all(len(entry) == 3 for entry in cfg.chain)

    def _forward_kinematics(self, joint_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the Link7 pose in the A1 base frame using XYZW quaternions."""
        # Unified training enables TF32, whose precision is insufficient to keep seven
        # accumulated rotation matrices inside quat_from_matrix's validity tolerance.
        # This command is resampled only every four seconds, so float64 keeps the
        # source kinematics robust with negligible runtime cost.
        output_dtype = joint_pos.dtype
        joint_pos = joint_pos.to(dtype=torch.float64)
        num_samples = joint_pos.shape[0]
        if self._use_legacy_z_axis_fk:
            transform = torch.eye(4, dtype=torch.float64, device=self.device).expand(num_samples, 4, 4).contiguous()
            joint_rotation = (
                torch.eye(4, dtype=torch.float64, device=self.device).expand(num_samples, 4, 4).contiguous()
            )
            for index in range(len(self._chain_joint_ids)):
                cosine = torch.cos(joint_pos[:, index])
                sine = torch.sin(joint_pos[:, index])
                joint_rotation[:, 0, 0] = cosine
                joint_rotation[:, 0, 1] = -sine
                joint_rotation[:, 1, 0] = sine
                joint_rotation[:, 1, 1] = cosine
                transform = transform @ self._origins[index].unsqueeze(0) @ joint_rotation
        else:
            transform = self._fixed_transform.unsqueeze(0).expand(num_samples, 4, 4).contiguous()
            identity = torch.eye(3, dtype=torch.float64, device=self.device).unsqueeze(0)
            joint_rotation = torch.zeros((num_samples, 4, 4), dtype=torch.float64, device=self.device)
            joint_rotation[:, 3, 3] = 1.0
            for index in range(len(self._chain_joint_ids)):
                axis = self._joint_axes[index]
                axis_outer = axis.unsqueeze(-1) @ axis.unsqueeze(0)
                axis_skew = torch.zeros((3, 3), dtype=torch.float64, device=self.device)
                axis_skew[0, 1] = -axis[2]
                axis_skew[0, 2] = axis[1]
                axis_skew[1, 0] = axis[2]
                axis_skew[1, 2] = -axis[0]
                axis_skew[2, 0] = -axis[1]
                axis_skew[2, 1] = axis[0]
                cosine = torch.cos(joint_pos[:, index]).view(-1, 1, 1)
                sine = torch.sin(joint_pos[:, index]).view(-1, 1, 1)
                joint_rotation[:, :3, :3] = (
                    cosine * identity + (1.0 - cosine) * axis_outer.unsqueeze(0) + sine * axis_skew.unsqueeze(0)
                )
                transform = transform @ self._origins[index].unsqueeze(0) @ joint_rotation
        position = transform[:, :3, 3].to(dtype=output_dtype)
        orientation = quat_from_matrix(transform[:, :3, :3]).to(dtype=output_dtype)
        return position, orientation

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        limits = self.robot.data.joint_pos_limits.torch[0, self._chain_joint_ids]
        lower, upper = limits[:, 0], limits[:, 1]
        center = 0.5 * (lower + upper)
        half_range = 0.5 * (upper - lower) * self.cfg.joint_range_scale
        lower, upper = center - half_range, center + half_range

        joint_pos = lower + (upper - lower) * torch.rand(len(env_ids), len(self._chain_joint_ids), device=self.device)
        position, orientation = self._forward_kinematics(joint_pos)
        self.pose_command_b[env_ids, :3] = position
        self.pose_command_b[env_ids, 3:] = quat_unique(orientation) if self.cfg.make_quat_unique else orientation
