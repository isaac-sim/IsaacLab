# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Derived from HybridRobotics/SuperMimic, revision 5efbf114e11f725c31d83e217be987c13c0e4ce2.
# Copyright 2026 the SuperMimic authors. See docs/licenses/dependencies/supermimic-official-license.txt.
# Modified for Isaac Lab package boundaries and simulator interfaces.

"""Per-task specs for the multi-encoder actor.

A ``TaskSpec`` bundles everything needed to route one task through the actor:
the goal width and the function that turns a reference motion into the
``goal_obs`` tensor consumed by its encoder. Reward is motion-tracking for every
task and lives in the WM env, not here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from tensordict import TensorDict

from .interfaces import MotionReference

ExtractFromMotion = Callable[[MotionReference, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class TaskSpec:
    """Contract for one task slot of the multi-encoder actor."""

    name: str
    goal_dim: int
    extract_training_goal: ExtractFromMotion


def build_actor_obs_td(
    goals_by_name: dict[str, torch.Tensor],
    history: torch.Tensor,
    critic: torch.Tensor | None = None,
) -> TensorDict:
    """``history`` (B, H, obs_dim) (+ optional ``critic``) plus one ``goal_<name>``
    key per task. The newest frame ``history[:, -1]`` is the current obs."""
    obs: dict[str, torch.Tensor] = {"history": history}
    if critic is not None:
        obs["critic"] = critic
    obs.update({f"goal_{name}": goal for name, goal in goals_by_name.items()})
    return TensorDict(obs, batch_size=[history.shape[0]], device=history.device)
