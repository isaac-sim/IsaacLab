# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Small task-specific correction to standard RSL-RL distillation."""

from __future__ import annotations

import torch
from rsl_rl.algorithms import Distillation
from tensordict import TensorDict


class ClippedTeacherDistillation(Distillation):
    """Train against the clipped teacher action that the environment executes."""

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Collect a student action and clamp its teacher label to the action contract."""
        actions = super().act(obs)
        self.transition.privileged_actions = self.transition.privileged_actions.clamp(-1.0, 1.0)
        return actions
