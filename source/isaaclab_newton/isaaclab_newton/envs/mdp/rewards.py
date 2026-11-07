# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton solver failure rewards."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def zero_reward_on_solver_reset(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Clear all reward terms for worlds terminated by a nonfinite solver state.

    Configure this after every other nonzero reward term. The reward manager sums
    ``_step_reward`` only after evaluating all terms, so clearing ``_reward_buf``
    here would leave the NaN in the final reward.
    """
    invalid_worlds = env.termination_manager.get_term("solver_reset_required")
    env.reward_manager._step_reward.masked_fill_(invalid_worlds[:, None], 0.0)
    return torch.zeros(env.num_envs, device=env.device)
