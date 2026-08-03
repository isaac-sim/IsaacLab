# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime Newton IK objective implementations.

Each class is built by :class:`~isaaclab_newton.ik.NewtonIKSolver` from the
matching :class:`~isaaclab_newton.ik.newton_ik_objectives_cfg.NewtonIKObjectiveCfg`
and owns the concrete :class:`newton.ik.IKObjective` instances appended to the
solver. Pose objectives also describe their action contribution as Warp data:
an :attr:`~NewtonIKObjective.action_dim`, the coordinate names for that slice,
a numeric :attr:`~NewtonIKPoseObjective.command_code` / relative flag, a Warp
``scale`` array and a target-frame ``offset`` transform. The action term reads
these directly into a Warp kernel; nothing here touches Torch.

Importing this module pulls ``newton`` (and ``pxr``), so it is loaded lazily via
the package ``lazy_export`` only after Kit has launched. Custom objectives
integrate by subclassing :class:`NewtonIKObjective`, taking ``(cfg, ctx)`` in
``__init__`` -- pulling only the :class:`NewtonIKBuildContext` fields they need --
and populating :attr:`NewtonIKObjective.solver_objectives`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import newton.ik as ik
import warp as wp

from .newton_ik_objectives_cfg import NewtonIKJointLimitObjectiveCfg, NewtonIKPoseObjectiveCfg

# Numeric command codes consumed by the action's Warp kernel.
COMMAND_POSITION = 0
COMMAND_POSE = 1


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
    :attr:`solver_objectives`. Pose objectives also set a :attr:`name` and a
    non-zero :attr:`action_dim`; constraint objectives leave the defaults.
    """

    name: str | None = None
    """Unique objective name, or ``None`` when the objective has no runtime target."""

    action_dim: int = 0
    """Number of action coordinates this objective consumes (0 for constraints)."""

    solver_objectives: list[ik.IKObjective]
    """Concrete Newton objectives appended to the solver's objective list."""


class NewtonIKPoseObjective(NewtonIKObjective):
    """Command-driven position + rotation objective tracking one end-effector body.

    Exposes its command convention to the action's Warp kernel as data:
    :attr:`command_code` / :attr:`use_relative`, the per-coordinate :attr:`scale`
    (``wp.float32``), the target-frame :attr:`offset` (``wp.transformf``), and the
    position/rotation target arrays the kernel writes into.
    """

    def __init__(self, cfg: NewtonIKPoseObjectiveCfg, ctx: NewtonIKBuildContext):
        self.name = cfg.name if cfg.name is not None else cfg.body_name
        self.command_type = cfg.command_type
        self.use_relative_mode = cfg.use_relative_mode
        self.link_index = ctx.resolve_link(cfg.body_name)
        self.action_dim = len(self.command_coordinate_names())

        self.command_code = COMMAND_POSITION if cfg.command_type == "position" else COMMAND_POSE
        self.use_relative = int(cfg.use_relative_mode)
        scale_values = [float(cfg.scale)] * self.action_dim if _is_scalar(cfg.scale) else [float(s) for s in cfg.scale]
        if len(scale_values) != self.action_dim:
            raise ValueError(
                f"Newton IK pose objective '{self.name}' scale must be a float or length-{self.action_dim} "
                f"sequence, got {len(scale_values)} values."
            )
        self.scale = wp.array(scale_values, dtype=wp.float32, device=ctx.device)
        self.offset = wp.transformf(wp.vec3f(*cfg.body_offset_pos), wp.quatf(*cfg.body_offset_rot))

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


class NewtonIKJointLimitObjective(NewtonIKObjective):
    """Soft joint-limit constraint reading the model's coordinate limits."""

    def __init__(self, cfg: NewtonIKJointLimitObjectiveCfg, ctx: NewtonIKBuildContext):
        self.objective = ik.IKObjectiveJointLimit(
            joint_limit_lower=ctx.model.joint_limit_lower,
            joint_limit_upper=ctx.model.joint_limit_upper,
            weight=cfg.weight,
        )
        self.solver_objectives = [self.objective]


def _is_scalar(value) -> bool:
    return isinstance(value, (int, float))
