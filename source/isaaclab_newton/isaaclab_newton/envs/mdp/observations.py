# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for Newton physics faults."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def zero_observation_on_solver_fault(
    env: ManagerBasedRLEnv, observations: torch.Tensor | dict[str, torch.Tensor]
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Zero complete faulted rows, including history, as an observation group postprocessor.

    The manager supplies owned tensors after assembly. Unflagged rows, including nonfinite values
    unrelated to solver faults, and the source and history buffers remain unchanged.
    """
    faults = wp.to_torch(env.sim.backend_manager.get_non_finite_worlds())
    for data in observations.values() if isinstance(observations, dict) else (observations,):
        data.masked_fill_(faults.reshape((-1,) + (1,) * (data.ndim - 1)), 0.0)
    return observations
