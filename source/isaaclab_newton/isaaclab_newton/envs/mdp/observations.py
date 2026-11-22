# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for Newton physics faults."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import ManagerTermBaseCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def zero_observation_on_solver_fault(env: ManagerBasedRLEnv, observation: ManagerTermBaseCfg) -> torch.Tensor:
    """Evaluate a nested observation and zero only solver-faulted rows before history storage.

    Configure observation processing and history on the outer observation term. Unflagged rows and
    the source tensor remain unchanged, including nonfinite values unrelated to a solver fault.
    """
    data = observation.func(env, **observation.params)
    faults = wp.to_torch(env.sim.backend_manager.get_non_finite_worlds())
    return data.masked_fill(faults.reshape((-1,) + (1,) * (data.ndim - 1)), 0.0)
