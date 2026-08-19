# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import torch

from isaaclab_tasks.contrib.navigation.mdp.pre_trained_policy_action import PreTrainedPolicyAction


class _LowLevelActionTerm:
    """Captures the generated low-level action."""

    def process_actions(self, actions: torch.Tensor) -> None:
        self.actions = actions

    def apply_actions(self) -> None:
        pass


def test_pretrained_policy_action_detaches_low_level_policy_output():
    """Low-level actions must be suitable for the Warp-backed action term."""
    action = object.__new__(PreTrainedPolicyAction)
    action._counter = 0
    action.cfg = SimpleNamespace(low_level_decimation=1)
    action._low_level_obs_manager = SimpleNamespace(compute_group=lambda _: torch.zeros(1, 2))
    action.policy = torch.nn.Linear(2, 3)
    action.low_level_actions = torch.zeros(1, 3)
    action._low_level_action_term = _LowLevelActionTerm()

    action.apply_actions()

    assert not action._low_level_action_term.actions.requires_grad
