# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-dataclass configuration for Newton IK objectives.

Inverse kinematics is expressed as a single ordered list of objectives. Each
entry is a :class:`NewtonIKObjectiveCfg` subclass describing one constraint:
a tracked end-effector pose, a joint limit, a collision penalty, and so on.
There is no separate notion of a "target" versus an "extra" objective -- a
single-body solve is a one-element list, a multi-body solve simply has more
pose entries, and constraints like joint limits are further entries that add
no action dimensions.

The solver builds each objective via
``string_to_callable(cfg.class_type)(cfg, model=..., num_envs=..., device=...,
link_resolver=...)`` and the action term reads each built objective's action
contribution, so an objective owns both its Newton residual and how a policy
command (if any) maps onto its target.

This module imports only the standard library and Isaac Lab's config utilities;
it must stay free of ``import newton`` so action/env configs remain importable
before Kit has launched. Matching runtime implementations live in
:mod:`isaaclab_newton.ik.newton_ik_objectives`.
"""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils.configclass import configclass


@configclass
class NewtonIKObjectiveCfg:
    """Base configuration for a Newton IK objective.

    Subclasses set :attr:`class_type` to the runtime implementation, which the
    solver instantiates as
    ``class_type(cfg, model=..., num_envs=..., device=..., link_resolver=...)``.
    The implementation exposes the concrete :class:`newton.ik.IKObjective`
    instances appended to the solver and, for command-driven objectives, an
    action-dimension contribution.
    """

    class_type: type | str = MISSING  # type: ignore[assignment]
    """Runtime objective implementation, as a type or a ``"module:Class"`` string."""


@configclass
class NewtonIKPoseObjectiveCfg(NewtonIKObjectiveCfg):
    """A pose objective tracking one end-effector body.

    This is the command-driven objective: it contributes action dimensions
    (3 for ``"position"``, 6 for relative ``"pose"``, 7 for absolute ``"pose"``)
    and maps its slice of the policy action onto a target pose for
    :attr:`body_name`. Multiple pose objectives drive a multi-body solve, each
    with its own body, command convention, weights and scale.
    """

    class_type: type | str = "isaaclab_newton.ik.newton_ik_objectives:NewtonIKPoseObjective"

    body_name: str = MISSING  # type: ignore[assignment]
    """Name of the controlled end-effector body."""

    name: str | None = None
    """Unique objective name used to update its target. Defaults to :attr:`body_name`."""

    body_offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Target-frame translation [m] relative to the body frame."""

    body_offset_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    """Target-frame quaternion ``(x, y, z, w)`` relative to the body frame."""

    command_type: str = "pose"
    """How the policy action is interpreted: ``"position"`` or ``"pose"``."""

    use_relative_mode: bool = True
    """Whether the command is a delta from the current end-effector pose."""

    scale: float | tuple[float, ...] = 1.0
    """Scale applied to this objective's raw action slice [m for position, rad for relative rotation]."""

    position_weight: float = 1.0
    """Residual weight [unitless] for the position component."""

    rotation_weight: float = 1.0
    """Residual weight [unitless] for the rotation component."""


@configclass
class NewtonIKJointLimitObjectiveCfg(NewtonIKObjectiveCfg):
    """Soft joint-limit constraint penalizing coordinates outside the model limits.

    A constraint-only objective: it adds a Newton residual but no action
    dimensions.
    """

    class_type: type | str = "isaaclab_newton.ik.newton_ik_objectives:NewtonIKJointLimitObjective"

    weight: float = 0.1
    """Residual weight [unitless] applied to limit violations."""
