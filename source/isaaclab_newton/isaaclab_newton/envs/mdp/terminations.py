# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton solver failure terminations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp
from newton.solvers import SolverMuJoCo

from isaaclab.managers import ManagerTermBase, TerminationTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@wp.func
def _nonfinite(values: wp.array2d(dtype=wp.float32), world: int, column: int) -> bool:
    return column < values.shape[1] and not wp.isfinite(values[world, column])


@wp.kernel(enable_backward=False)
def _find_nonfinite_solver_worlds(
    qpos: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    qacc: wp.array2d(dtype=wp.float32),
    qfrc_actuator: wp.array2d(dtype=wp.float32),
    qacc_warmstart: wp.array2d(dtype=wp.float32),
    qfrc_applied: wp.array2d(dtype=wp.float32),
    ctrl: wp.array2d(dtype=wp.float32),
    act: wp.array2d(dtype=wp.float32),
    xfrc_applied: wp.array2d(dtype=wp.spatial_vector),
    invalid_worlds: wp.array(dtype=wp.bool),
):
    world, column = wp.tid()
    invalid = (
        _nonfinite(qpos, world, column)
        or _nonfinite(qvel, world, column)
        or _nonfinite(qacc, world, column)
        or _nonfinite(qfrc_actuator, world, column)
        or _nonfinite(qacc_warmstart, world, column)
        or _nonfinite(qfrc_applied, world, column)
        or _nonfinite(ctrl, world, column)
        or _nonfinite(act, world, column)
    )
    if column < xfrc_applied.shape[1]:
        force = xfrc_applied[world, column]
        for component in range(6):
            if not wp.isfinite(force[component]):
                invalid = True
    if invalid:
        invalid_worlds[world] = True


class SolverResetRequired(ManagerTermBase):
    """Mark MJWarp worlds with nonfinite persistent solver state for an environment reset."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.invalid_worlds = wp.zeros(env.num_envs, dtype=wp.bool, device=env.device)
        self.invalid_worlds_torch = wp.to_torch(self.invalid_worlds)

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        solver = env.sim.backend_manager.backend.solver
        if not isinstance(solver, SolverMuJoCo) or solver.mjw_data is None:
            raise RuntimeError("SolverResetRequired requires an initialized GPU MJWarp solver.")
        data = solver.mjw_data
        arrays = (
            data.qpos,
            data.qvel,
            data.qacc,
            data.qfrc_actuator,
            data.qacc_warmstart,
            data.qfrc_applied,
            data.ctrl,
            data.act,
            data.xfrc_applied,
        )
        self.invalid_worlds.zero_()
        wp.launch(
            _find_nonfinite_solver_worlds,
            dim=(env.num_envs, max(array.shape[1] for array in arrays)),
            inputs=arrays,
            outputs=[self.invalid_worlds],
            device=self.invalid_worlds.device,
        )
        return self.invalid_worlds_torch
