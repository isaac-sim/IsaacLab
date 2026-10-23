# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capture the native IK solve without replacing the manager-based action."""

from __future__ import annotations

import warp as wp
from isaaclab_newton.controllers.ik.newton_ik_solver import NewtonIKSolver
from isaaclab_newton.controllers.ik.newton_ik_solver_cfg import NewtonIKSolverCfg
from isaaclab_newton.physics.newton_manager import NewtonQueries

from isaaclab.utils.version import has_kit


class CapturedNewtonIKSolver(NewtonIKSolver):
    """Replay native IK with fixed solver settings and an owned CUDA seed buffer.

    CPU solves and calls inside an enclosing capture use the native eager path.
    Targets stay in the parent's objective buffers and can change on each solve.
    """

    def __init__(self, cfg: NewtonIKSolverCfg, **kwargs) -> None:
        super().__init__(cfg, **kwargs)
        self._graph: wp.Graph | None = None
        self._seed: wp.array | None = None

    def solve(self, joint_pos: wp.array) -> wp.array:
        """Solve from current joint positions [m or rad], using the native output buffer."""
        if not joint_pos.device.is_cuda or joint_pos.device.is_capturing:
            return super().solve(joint_pos)
        if joint_pos.shape != self.joint_q_out.shape:
            raise ValueError(f"Expected IK seed shape {self.joint_q_out.shape}, got {joint_pos.shape}.")
        if self._seed is None:
            self._seed = wp.empty_like(joint_pos)
        wp.copy(self._seed, joint_pos)
        if self._graph is None:
            self._graph = NewtonQueries.capture_graph(
                str(joint_pos.device),
                lambda: super(CapturedNewtonIKSolver, self).solve(self._seed),
                relaxed=has_kit(),
            )
        wp.capture_launch(self._graph)
        return self.joint_q_out
