# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime Newton IK objective implementations.

Each class is built by :class:`~isaaclab_newton.ik.NewtonIKSolver` from the
matching :class:`~isaaclab_newton.ik.newton_ik_objectives_cfg.NewtonIKObjectiveCfg`
and owns the concrete :class:`newton.ik.IKObjective` instances appended to the
solver. Command-driven objectives (currently pose) additionally own their
action contribution: an :attr:`~NewtonIKObjective.action_dim`, the coordinate
names for that slice, and the mapping from a raw action slice to a target pose.
The action term is therefore generic -- it sums :attr:`action_dim` across the
objective list and dispatches each slice without branching on command type.

Importing this module pulls ``newton`` (and ``pxr``), so it is loaded lazily via
the package ``lazy_export`` only after Kit has launched. Custom objectives
integrate by subclassing :class:`NewtonIKObjective`, taking
``(cfg, ctx)`` in ``__init__`` -- pulling only the :class:`NewtonIKBuildContext`
fields they need -- and populating :attr:`NewtonIKObjective.solver_objectives`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import newton.ik as ik
import torch
import warp as wp

import isaaclab.utils.math as math_utils

from .newton_ik_objectives_cfg import NewtonIKJointLimitObjectiveCfg, NewtonIKPoseObjectiveCfg


@dataclass(frozen=True)
class NewtonIKBuildContext:
    """Build-time inputs shared with every objective; each pulls what it needs."""

    model: object
    """Finalized Newton prototype model (e.g. for joint limits)."""

    num_envs: int
    """Number of parallel IK problems (target-array batch size)."""

    device: str
    """Warp device string for objective-owned arrays."""

    resolve_link: Callable[[str], int]
    """Maps a body name to its Newton link index in the prototype model."""


class NewtonIKObjective:
    """Base built IK objective.

    Owns the concrete :class:`newton.ik.IKObjective` instances in
    :attr:`solver_objectives`. Command-driven objectives (pose) also set a
    :attr:`name` and a non-zero :attr:`action_dim` and add ``compute_target_b``
    / ``command_coordinate_names``; constraint objectives leave the defaults.
    """

    name: str | None = None
    """Unique objective name, or ``None`` when the objective has no runtime target."""

    action_dim: int = 0
    """Number of action coordinates this objective consumes (0 for constraints)."""

    solver_objectives: list[ik.IKObjective]
    """Concrete Newton objectives appended to the solver's objective list."""


class NewtonIKPoseObjective(NewtonIKObjective):
    """Command-driven position + rotation objective tracking one end-effector body."""

    def __init__(self, cfg: NewtonIKPoseObjectiveCfg, ctx: NewtonIKBuildContext):
        self.name = cfg.name if cfg.name is not None else cfg.body_name
        self.command_type = cfg.command_type
        self.use_relative_mode = cfg.use_relative_mode
        self.link_index = ctx.resolve_link(cfg.body_name)
        self.action_dim = len(self.command_coordinate_names())

        scale = torch.as_tensor(cfg.scale, dtype=torch.float32, device=ctx.device)
        if scale.ndim == 0:
            scale = scale.repeat(self.action_dim)
        if scale.shape != (self.action_dim,):
            raise ValueError(
                f"Newton IK pose objective '{self.name}' scale must be a float or length-{self.action_dim} "
                f"sequence, got shape {tuple(scale.shape)}."
            )
        self._scale = scale

        target_positions = wp.zeros((ctx.num_envs,), dtype=wp.vec3, device=ctx.device)
        target_rotations = wp.array([(0.0, 0.0, 0.0, 1.0)] * ctx.num_envs, dtype=wp.vec4, device=ctx.device)
        self.position_objective = ik.IKObjectivePosition(
            link_index=self.link_index,
            link_offset=wp.vec3(*cfg.body_offset_pos),
            target_positions=target_positions,
            weight=cfg.position_weight,
        )
        self.rotation_objective = ik.IKObjectiveRotation(
            link_index=self.link_index,
            link_offset_rotation=wp.quat(*cfg.body_offset_rot),
            target_rotations=target_rotations,
            weight=cfg.rotation_weight,
        )
        self.solver_objectives = [self.position_objective, self.rotation_objective]

    def command_coordinate_names(self) -> list[str]:
        if self.command_type == "position":
            return ["x", "y", "z"]
        if self.command_type == "pose":
            if self.use_relative_mode:
                return ["x", "y", "z", "roll", "pitch", "yaw"]
            return ["x", "y", "z", "qx", "qy", "qz", "qw"]
        raise ValueError(f"Unsupported Newton IK command type: {self.command_type}")

    def compute_target_b(
        self, action: torch.Tensor, ee_pos_b: torch.Tensor, ee_quat_b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        processed = action * self._scale
        if self.command_type == "position":
            target_pos_b = ee_pos_b + processed if self.use_relative_mode else processed
            return target_pos_b, ee_quat_b
        if self.use_relative_mode:
            return math_utils.apply_delta_pose(ee_pos_b, ee_quat_b, processed)
        return processed[:, 0:3], processed[:, 3:7]

    def set_target_pose(self, target_pos_w: torch.Tensor, target_quat_w: torch.Tensor) -> None:
        """Update the batched world-frame target pose, shape ``[num_envs, 3]`` and ``[num_envs, 4]``."""
        self.position_objective.set_target_positions(wp.from_torch(target_pos_w.contiguous(), dtype=wp.vec3))
        self.rotation_objective.set_target_rotations(wp.from_torch(target_quat_w.contiguous(), dtype=wp.vec4))


class NewtonIKJointLimitObjective(NewtonIKObjective):
    """Soft joint-limit constraint reading the model's coordinate limits."""

    def __init__(self, cfg: NewtonIKJointLimitObjectiveCfg, ctx: NewtonIKBuildContext):
        self.objective = ik.IKObjectiveJointLimit(
            joint_limit_lower=ctx.model.joint_limit_lower,
            joint_limit_upper=ctx.model.joint_limit_upper,
            weight=cfg.weight,
        )
        self.solver_objectives = [self.objective]
