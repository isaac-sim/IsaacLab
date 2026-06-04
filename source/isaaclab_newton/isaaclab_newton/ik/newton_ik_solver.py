# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import newton.ik as ik
import torch
import warp as wp

from .newton_ik_solver_cfg import NewtonIKSolverCfg


@dataclass(frozen=True)
class NewtonIKPoseObjective:
    """Pose objective descriptor used to build Newton position/rotation objectives.

    Args:
        name: Unique objective name used when updating targets.
        link_index: Newton body index for the controlled link.
        link_offset_pos: Target frame translation in meters relative to the link frame.
        link_offset_rot: Target frame quaternion ``(x, y, z, w)`` relative to the link frame.
        position_weight: Optional residual weight overriding the solver default.
        rotation_weight: Optional residual weight overriding the solver default.
    """

    name: str
    link_index: int
    link_offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    link_offset_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    position_weight: float | None = None
    rotation_weight: float | None = None


class NewtonIKSolver:
    """Batched wrapper around Newton's inverse-kinematics solver.

    The solver mirrors Newton's objective-list design while adding torch/Warp
    target updates convenient for Isaac Lab action terms. Pose objectives are
    named so callers can update individual targets between solves. Additional
    custom Newton objectives can be passed through ``extra_objectives``.
    """

    cfg: NewtonIKSolverCfg

    def __init__(
        self,
        cfg: NewtonIKSolverCfg,
        *,
        model,
        num_envs: int,
        device: str,
        pose_objectives: Sequence[NewtonIKPoseObjective],
        extra_objectives: Sequence[ik.IKObjective] | None = None,
    ):
        if not pose_objectives and not extra_objectives:
            raise ValueError("NewtonIKSolver requires at least one pose or custom objective.")

        self.cfg = cfg
        self.model = model
        self.num_envs = num_envs
        self.device = device
        self.num_coords = model.joint_coord_count
        self.pose_objective_names = [objective.name for objective in pose_objectives]
        self.pose_objective_cfgs = {objective.name: objective for objective in pose_objectives}

        if len(set(self.pose_objective_names)) != len(self.pose_objective_names):
            raise ValueError(f"Newton IK pose objective names must be unique: {self.pose_objective_names}")

        self.position_objectives: dict[str, ik.IKObjectivePosition] = {}
        self.rotation_objectives: dict[str, ik.IKObjectiveRotation] = {}
        solver_objectives: list[ik.IKObjective] = []

        for objective_cfg in pose_objectives:
            target_positions = wp.zeros((num_envs,), dtype=wp.vec3, device=device)
            target_rotations = wp.array(
                [(0.0, 0.0, 0.0, 1.0)] * num_envs,
                dtype=wp.vec4,
                device=device,
            )
            position_objective = ik.IKObjectivePosition(
                link_index=objective_cfg.link_index,
                link_offset=wp.vec3(*objective_cfg.link_offset_pos),
                target_positions=target_positions,
                weight=cfg.position_weight if objective_cfg.position_weight is None else objective_cfg.position_weight,
            )
            rotation_objective = ik.IKObjectiveRotation(
                link_index=objective_cfg.link_index,
                link_offset_rotation=wp.quat(*objective_cfg.link_offset_rot),
                target_rotations=target_rotations,
                weight=cfg.rotation_weight if objective_cfg.rotation_weight is None else objective_cfg.rotation_weight,
            )
            self.position_objectives[objective_cfg.name] = position_objective
            self.rotation_objectives[objective_cfg.name] = rotation_objective
            solver_objectives.extend((position_objective, rotation_objective))

        if cfg.joint_limit_weight is not None:
            self.joint_limit_objective = ik.IKObjectiveJointLimit(
                joint_limit_lower=model.joint_limit_lower,
                joint_limit_upper=model.joint_limit_upper,
                weight=cfg.joint_limit_weight,
            )
            solver_objectives.append(self.joint_limit_objective)
        else:
            self.joint_limit_objective = None

        solver_objectives.extend(extra_objectives or [])

        self.joint_q_out = wp.zeros((num_envs, self.num_coords), dtype=wp.float32, device=device)
        self.joint_q_seed_t = torch.zeros((num_envs, self.num_coords), dtype=torch.float32, device=device)
        self.joint_q_seed = wp.from_torch(self.joint_q_seed_t, dtype=wp.float32)
        self._has_joint_q_seed = False
        self.solver = ik.IKSolver(
            model=model,
            n_problems=num_envs,
            objectives=solver_objectives,
            optimizer=ik.IKOptimizer(cfg.optimizer),
            jacobian_mode=ik.IKJacobianType(cfg.jacobian_mode),
            sampler=ik.IKSampler(cfg.sampler),
            n_seeds=cfg.n_seeds,
            noise_std=cfg.noise_std,
            rng_seed=cfg.rng_seed,
            lambda_initial=cfg.lambda_initial,
        )

    @property
    def action_dim(self) -> int:
        """Dimension of the IK command expected by this solver."""
        if self.cfg.command_type == "position":
            return 3
        if self.cfg.command_type == "pose" and self.cfg.use_relative_mode:
            return 6
        if self.cfg.command_type == "pose":
            return 7
        raise ValueError(f"Unsupported Newton IK command type: {self.cfg.command_type}")

    def set_target_pose(self, name: str, target_pos_w: torch.Tensor, target_quat_w: torch.Tensor) -> None:
        """Update batched world-frame target poses for a named pose objective."""
        try:
            position_objective = self.position_objectives[name]
            rotation_objective = self.rotation_objectives[name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown Newton IK pose objective '{name}'. Available objectives: {self.pose_objective_names}."
            ) from exc
        position_objective.set_target_positions(wp.from_torch(target_pos_w.contiguous(), dtype=wp.vec3))
        rotation_objective.set_target_rotations(wp.from_torch(target_quat_w.contiguous(), dtype=wp.vec4))

    def set_target_pose_from_body_q(
        self,
        name: str,
        body_q: torch.Tensor | wp.array | Any,
        *,
        env_origins: torch.Tensor | None = None,
    ) -> None:
        """Set one pose objective target from body transforms."""
        objective_cfg = self.pose_objective_cfgs[name]
        body_q_t = _as_torch(body_q, device=self.device)
        if body_q_t.ndim == 2:
            target_pos = body_q_t[objective_cfg.link_index, :3].reshape(1, 3).repeat(self.num_envs, 1)
            target_quat = body_q_t[objective_cfg.link_index, 3:7].reshape(1, 4).repeat(self.num_envs, 1)
        elif body_q_t.ndim == 3:
            if body_q_t.shape[0] != self.num_envs:
                raise ValueError(f"Expected body_q first dimension {self.num_envs}, got {body_q_t.shape[0]}.")
            target_pos = body_q_t[:, objective_cfg.link_index, :3]
            target_quat = body_q_t[:, objective_cfg.link_index, 3:7]
        else:
            raise ValueError(
                f"Expected body_q shape (num_bodies, 7) or (num_envs, num_bodies, 7), got {body_q_t.shape}."
            )
        if env_origins is not None:
            target_pos = target_pos - env_origins.to(device=self.device, dtype=torch.float32)
        self.set_target_pose(name, target_pos.to(torch.float32), target_quat.to(torch.float32))

    def set_pose_targets_from_body_q(
        self,
        body_q: torch.Tensor | wp.array | Any,
        names: Sequence[str] | None = None,
        *,
        env_origins: torch.Tensor | None = None,
    ) -> None:
        """Set multiple pose objective targets from body transforms."""
        for name in self.pose_objective_names if names is None else names:
            self.set_target_pose_from_body_q(name, body_q, env_origins=env_origins)

    def set_joint_seed(
        self, joint_pos: torch.Tensor, env_ids: Sequence[int] | torch.Tensor | slice | None = None
    ) -> None:
        """Set the persistent joint-coordinate seed used by ``solve()`` without an explicit seed."""
        joint_pos = joint_pos.to(device=self.device, dtype=torch.float32)
        if env_ids is None:
            if joint_pos.shape != (self.num_envs, self.num_coords):
                raise ValueError(
                    f"Expected joint seed shape {(self.num_envs, self.num_coords)}, got {tuple(joint_pos.shape)}."
                )
            self.joint_q_seed_t.copy_(joint_pos)
        else:
            ids = _env_ids_to_tensor(env_ids, self.num_envs, self.device)
            if joint_pos.shape != (ids.numel(), self.num_coords):
                raise ValueError(
                    f"Expected joint seed shape {(ids.numel(), self.num_coords)}, got {tuple(joint_pos.shape)}."
                )
            self.joint_q_seed_t[ids] = joint_pos
        self._has_joint_q_seed = True

    def reset(
        self,
        env_ids: Sequence[int] | torch.Tensor | slice | None = None,
        joint_pos: torch.Tensor | None = None,
    ) -> None:
        """Reset Newton solver state, selected seeds, and sampler RNG."""
        self.solver.reset()
        if joint_pos is not None:
            self.set_joint_seed(joint_pos, env_ids=env_ids)
        elif env_ids is None:
            self._has_joint_q_seed = False

    @property
    def costs(self) -> wp.array:
        """Expanded per-seed costs from the most recent Newton solve."""
        return self.solver.costs

    @property
    def joint_q(self) -> wp.array:
        """Expanded joint-coordinate buffer storing all sampled seeds."""
        return self.solver.joint_q

    def solve(self, joint_pos: torch.Tensor | None = None) -> torch.Tensor:
        """Solve IK from an explicit seed or the solver's persistent seed."""
        if joint_pos is None:
            if not self._has_joint_q_seed:
                raise RuntimeError("NewtonIKSolver.solve() needs joint_pos or a seed set with set_joint_seed().")
            joint_q_in = self.joint_q_seed
            update_seed = True
        else:
            if joint_pos.shape != (self.num_envs, self.num_coords):
                raise ValueError(
                    f"Expected joint seed shape {(self.num_envs, self.num_coords)}, got {tuple(joint_pos.shape)}."
                )
            if self.cfg.use_persistent_seed:
                self.set_joint_seed(joint_pos)
                joint_q_in = self.joint_q_seed
                update_seed = True
            else:
                joint_q_in = wp.from_torch(joint_pos.contiguous(), dtype=wp.float32)
                update_seed = False
        self.solver.step(
            joint_q_in,
            self.joint_q_out,
            iterations=self.cfg.iterations,
            step_size=self.cfg.step_size,
        )
        result = wp.to_torch(self.joint_q_out).clone()
        if update_seed:
            self.joint_q_seed_t.copy_(result)
            self._has_joint_q_seed = True
        return result

    def step(self) -> torch.Tensor:
        """Solve IK from the persistent seed and store the result as the next seed."""
        return self.solve()


def _as_torch(value, *, device: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float32)
    if hasattr(value, "numpy"):
        return wp.to_torch(value).to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, device=device, dtype=torch.float32)


def _env_ids_to_tensor(env_ids: Sequence[int] | torch.Tensor | slice, num_envs: int, device: str) -> torch.Tensor:
    if isinstance(env_ids, slice):
        start, stop, step = env_ids.indices(num_envs)
        return torch.arange(start, stop, step, device=device, dtype=torch.long)
    if isinstance(env_ids, Integral):
        return torch.as_tensor([env_ids], device=device, dtype=torch.long)
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=device, dtype=torch.long).flatten()
    return torch.as_tensor(list(env_ids), device=device, dtype=torch.long)
