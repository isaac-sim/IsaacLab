# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# TaskSpec and observation assembly derived from HybridRobotics/SuperMimic, revision
# 5efbf114e11f725c31d83e217be987c13c0e4ce2, Copyright 2026 the SuperMimic authors.
# See docs/licenses/dependencies/supermimic-official-license.txt.

"""Task contracts consumed by the model-based learners.

Tasks supply reference motion, physical context and differentiable objectives.
The learners do not depend on a particular robot, simulator, or observation layout.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch
from tensordict import TensorDict


class MotionReference(Protocol):
    """Reference channels consumed by state integration and goal encoding."""

    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor
    root_lin_vel_w: torch.Tensor
    root_ang_vel_w: torch.Tensor
    motion_starts: torch.Tensor
    motion_ends: torch.Tensor


class SimulationContext(Protocol):
    """Normalization layout shared by physical and imagined observations."""

    def find(self, name: str) -> slice | None: ...

    def denormalize(self, values: torch.Tensor, columns: slice) -> torch.Tensor: ...


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
