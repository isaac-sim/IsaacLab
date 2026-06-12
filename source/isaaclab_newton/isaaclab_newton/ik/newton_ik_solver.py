# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable, Sequence

import newton.ik as ik
import warp as wp

from .newton_ik_objectives import NewtonIKBuildContext, NewtonIKObjective
from .newton_ik_objectives_cfg import NewtonIKObjectiveCfg
from .newton_ik_solver_cfg import NewtonIKSolverCfg


class NewtonIKSolver:
    """Batched wrapper around Newton's inverse-kinematics solver.

    The solver is configured by an ordered list of
    :class:`~isaaclab_newton.ik.newton_ik_objectives_cfg.NewtonIKObjectiveCfg`.
    Each cfg is resolved to its runtime
    :class:`~isaaclab_newton.ik.newton_ik_objectives.NewtonIKObjective` and its
    concrete Newton objectives are appended to the underlying
    :class:`newton.ik.IKSolver`. The built objectives are exposed via
    :attr:`objectives` / :attr:`objectives_by_name`; callers update a pose
    objective's target by calling :meth:`~..newton_ik_objectives.NewtonIKPoseObjective.set_target_pose`
    on it directly.

    The solver solves ``num_envs`` independent problems and is agnostic to how
    targets are produced -- the prototype-broadcast policy used by the Newton IK
    action term lives in the action, not here. ``link_resolver`` maps an
    objective's body name to a Newton link index; the caller owns it because the
    name-to-index mapping depends on the model layout (e.g. cloned env prefixes).
    """

    cfg: NewtonIKSolverCfg

    def __init__(
        self,
        cfg: NewtonIKSolverCfg,
        *,
        model,
        num_envs: int,
        device: str,
        objectives: Sequence[NewtonIKObjectiveCfg],
        link_resolver: Callable[[str], int],
    ):
        if not objectives:
            raise ValueError("NewtonIKSolver requires at least one objective cfg.")

        self.cfg = cfg
        ctx = NewtonIKBuildContext(model=model, num_envs=num_envs, device=device, resolve_link=link_resolver)

        self.objectives: list[NewtonIKObjective] = []
        self.objectives_by_name: dict[str, NewtonIKObjective] = {}
        solver_objectives: list[ik.IKObjective] = []
        for objective_cfg in objectives:
            objective = objective_cfg.class_type(objective_cfg, ctx)
            if objective.name is not None:
                if objective.name in self.objectives_by_name:
                    raise ValueError(f"Newton IK objective names must be unique: duplicate '{objective.name}'.")
                self.objectives_by_name[objective.name] = objective
            self.objectives.append(objective)
            solver_objectives.extend(objective.solver_objectives)

        self.joint_q_out = wp.zeros((num_envs, model.joint_coord_count), dtype=wp.float32, device=device)
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
    def costs(self) -> wp.array:
        """Expanded per-seed costs from the most recent Newton solve."""
        return self.solver.costs

    @property
    def joint_q(self) -> wp.array:
        """Expanded joint-coordinate buffer storing all sampled seeds."""
        return self.solver.joint_q

    def solve(self, joint_pos: wp.array) -> wp.array:
        """Solve IK from the Warp seed ``joint_pos``, shape ``[num_envs, joint_coord_count]``.

        Returns the solver's output buffer, overwritten on the next solve -- consume
        or copy it before solving again.
        """
        self.solver.step(
            joint_pos,
            self.joint_q_out,
            iterations=self.cfg.iterations,
            step_size=self.cfg.step_size,
        )
        return self.joint_q_out
