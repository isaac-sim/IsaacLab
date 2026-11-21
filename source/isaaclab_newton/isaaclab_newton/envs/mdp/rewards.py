# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for Newton physics faults."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def zero_reward_on_solver_fault(env: ManagerBasedRLEnv) -> float:
    """Zero rewards for faulted worlds before summation and episode accumulation.

    Configure this after every other nonzero reward term, with a nonzero weight. The lazy backend query
    shares its result with other fault consumers without requiring a named termination term.
    """
    faults = wp.to_torch(env.sim.backend_manager.get_non_finite_worlds())
    env.reward_manager._step_reward.masked_fill_(faults[:, None], 0.0)
    return 0.0
