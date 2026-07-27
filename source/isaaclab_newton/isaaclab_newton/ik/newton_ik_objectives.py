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

import math
from collections.abc import Callable
from dataclasses import dataclass

import newton.ik as ik
import numpy as np
import warp as wp

from .newton_ik_objectives_cfg import (
    NewtonIKJointLimitObjectiveCfg,
    NewtonIKJointPostureObjectiveCfg,
    NewtonIKPoseObjectiveCfg,
)

# Numeric command codes consumed by the action's Warp kernel.
COMMAND_POSITION = 0
COMMAND_POSE = 1


@wp.kernel
def _joint_posture_residuals(
    joint_q: wp.array2d(dtype=wp.float32),
    coordinate_indices: wp.array(dtype=wp.int32),
    target_positions: wp.array(dtype=wp.float32),
    weights: wp.array(dtype=wp.float32),
    start_idx: int,
    residuals: wp.array2d(dtype=wp.float32),
):
    row, posture_idx = wp.tid()
    coordinate_idx = coordinate_indices[posture_idx]
    residuals[row, start_idx + posture_idx] = weights[posture_idx] * (
        joint_q[row, coordinate_idx] - target_positions[posture_idx]
    )


@wp.kernel
def _joint_posture_jacobian_from_autodiff(
    q_grad: wp.array2d(dtype=wp.float32),
    dof_indices: wp.array(dtype=wp.int32),
    start_idx: int,
    jacobian: wp.array3d(dtype=wp.float32),
):
    row, posture_idx = wp.tid()
    dof_idx = dof_indices[posture_idx]
    jacobian[row, start_idx + posture_idx, dof_idx] = q_grad[row, dof_idx]


@wp.kernel
def _joint_posture_jacobian_analytic(
    dof_indices: wp.array(dtype=wp.int32),
    weights: wp.array(dtype=wp.float32),
    start_idx: int,
    jacobian: wp.array3d(dtype=wp.float32),
):
    row, posture_idx = wp.tid()
    dof_idx = dof_indices[posture_idx]
    jacobian[row, start_idx + posture_idx, dof_idx] = weights[posture_idx]


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

    resolve_joint: Callable[[str], tuple[int, int]] | None = None
    """Maps a scalar joint name to its Newton ``(coordinate, DoF)`` indices."""


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


class _IKObjectiveJointPosture(ik.IKObjective):
    """Newton objective penalizing selected scalar joint-coordinate errors."""

    def __init__(
        self,
        coordinate_indices: wp.array,
        dof_indices: wp.array,
        target_positions: wp.array,
        weights: wp.array,
    ) -> None:
        super().__init__()
        self.coordinate_indices = coordinate_indices
        self.dof_indices = dof_indices
        self.target_positions = target_positions
        self.weights = weights
        self.num_joints = len(coordinate_indices)
        self.e_array = None

    def residual_dim(self) -> int:
        return self.num_joints

    def init_buffers(self, model, jacobian_mode) -> None:
        self._require_batch_layout()
        if jacobian_mode == ik.IKJacobianType.AUTODIFF:
            e = np.zeros((self.n_batch, self.total_residuals), dtype=np.float32)
            e[:, self.residual_offset : self.residual_offset + self.num_joints] = 1.0
            self.e_array = wp.array(e.flatten(), dtype=wp.float32, device=self.device)

    def supports_analytic(self) -> bool:
        return True

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx) -> None:
        wp.launch(
            _joint_posture_residuals,
            dim=(joint_q.shape[0], self.num_joints),
            inputs=[joint_q, self.coordinate_indices, self.target_positions, self.weights, start_idx],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, dq_dof) -> None:
        self._require_batch_layout()
        tape.backward(grads={tape.outputs[0]: self.e_array})
        wp.launch(
            _joint_posture_jacobian_from_autodiff,
            dim=(self.n_batch, self.num_joints),
            inputs=[tape.gradients[dq_dof], self.dof_indices, start_idx],
            outputs=[jacobian],
            device=self.device,
        )

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx) -> None:
        wp.launch(
            _joint_posture_jacobian_analytic,
            dim=(joint_q.shape[0], self.num_joints),
            inputs=[self.dof_indices, self.weights, start_idx],
            outputs=[jacobian],
            device=self.device,
        )


class NewtonIKJointPostureObjective(NewtonIKObjective):
    """Soft reference-posture objective over explicitly named scalar joints."""

    def __init__(self, cfg: NewtonIKJointPostureObjectiveCfg, ctx: NewtonIKBuildContext):
        if ctx.resolve_joint is None:
            raise ValueError("Newton IK joint-posture objectives require a scalar-joint resolver.")
        if not cfg.joint_names:
            raise ValueError("Newton IK joint-posture objectives require at least one joint name.")
        if len(set(cfg.joint_names)) != len(cfg.joint_names):
            raise ValueError("Newton IK joint-posture objective joint names must be unique.")

        resolved = [ctx.resolve_joint(name) for name in cfg.joint_names]
        coordinate_indices = [int(indices[0]) for indices in resolved]
        dof_indices = [int(indices[1]) for indices in resolved]

        if cfg.target_positions is None:
            model_joint_q = ctx.model.joint_q.numpy()
            target_positions = [float(model_joint_q[index]) for index in coordinate_indices]
        else:
            target_positions = [float(value) for value in cfg.target_positions]
            if len(target_positions) != len(cfg.joint_names):
                raise ValueError(
                    "Newton IK joint-posture target_positions must contain one value per joint: "
                    f"got {len(target_positions)} values for {len(cfg.joint_names)} joints."
                )

        weight_values = (
            [float(cfg.weight)] * len(cfg.joint_names)
            if _is_scalar(cfg.weight)
            else [float(value) for value in cfg.weight]
        )
        if len(weight_values) != len(cfg.joint_names):
            raise ValueError(
                "Newton IK joint-posture weight must be a float or contain one value per joint: "
                f"got {len(weight_values)} values for {len(cfg.joint_names)} joints."
            )
        if any(not math.isfinite(value) for value in target_positions):
            raise ValueError("Newton IK joint-posture target positions must be finite.")
        if any(not math.isfinite(value) or value < 0.0 for value in weight_values):
            raise ValueError("Newton IK joint-posture weights must be finite and non-negative.")

        self.objective = _IKObjectiveJointPosture(
            coordinate_indices=wp.array(coordinate_indices, dtype=wp.int32, device=ctx.device),
            dof_indices=wp.array(dof_indices, dtype=wp.int32, device=ctx.device),
            target_positions=wp.array(target_positions, dtype=wp.float32, device=ctx.device),
            weights=wp.array(weight_values, dtype=wp.float32, device=ctx.device),
        )
        self.solver_objectives = [self.objective]


def _is_scalar(value) -> bool:
    return isinstance(value, (int, float))
