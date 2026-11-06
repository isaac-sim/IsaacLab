# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DR-Legs-specific observation terms."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

__all__ = ["gait_phase", "joint_position_setpoints"]


def joint_position_setpoints(env: ManagerBasedEnv, action_term_name: str = "joint_pos") -> torch.Tensor:
    """Joint position setpoints (the action term's ``processed_actions``, in radians)."""
    action_term = env.action_manager.get_term(action_term_name)
    return action_term.processed_actions.clone()


def gait_phase(env: ManagerBasedRLEnv, period: float = 1.0, num_harmonics: int = 1) -> torch.Tensor:
    """Cyclic gait clock encoded with sine and cosine harmonics.

    ``phi`` is the episode time wrapped to ``period`` and normalized to ``[0, 1)``.
    """
    episode_time_s = env.episode_length_buf.float() * env.step_dt
    phase = (episode_time_s % period) / period
    two_pi = 2.0 * math.pi
    components = []
    for harmonic in range(1, num_harmonics + 1):
        components.extend([torch.sin(harmonic * two_pi * phase), torch.cos(harmonic * two_pi * phase)])
    return torch.stack(components, dim=-1)
