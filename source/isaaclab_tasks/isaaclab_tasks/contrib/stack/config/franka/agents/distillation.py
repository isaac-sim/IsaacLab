# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action clipping for Franka stack distillation."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from rsl_rl.algorithms import Distillation
from tensordict import TensorDict

if TYPE_CHECKING:
    from rsl_rl.env import VecEnv

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


class ClippedTeacherDistillation(Distillation):
    """Match teacher labels to the stack runner's action bounds."""

    _clip_actions: float | None

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> ClippedTeacherDistillation:
        """Build the distillation algorithm using the training wrapper's action bound."""
        algorithm = cast(ClippedTeacherDistillation, Distillation.construct_algorithm(obs, env, cfg, device))
        algorithm._clip_actions = cast("RslRlVecEnvWrapper", env).clip_actions
        return algorithm

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Collect a student action and clamp its teacher label to the action contract."""
        actions = super().act(obs)
        if self._clip_actions is not None:
            self.transition.privileged_actions = self.transition.privileged_actions.clamp(
                -self._clip_actions, self._clip_actions
            )
        return actions
