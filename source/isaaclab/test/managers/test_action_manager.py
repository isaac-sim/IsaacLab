# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the action manager."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from isaaclab.managers import ActionManager, ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

pytestmark = pytest.mark.unit


@configclass
class DummyActionTermCfg(ActionTermCfg):
    """Configuration for a dummy action term."""

    action_dim: int = 1


class DummyActionTerm(ActionTerm):
    """Action term that records the raw actions passed by the manager."""

    def __init__(self, cfg: ActionTermCfg, env):
        self.cfg = cfg
        self._env = env
        self._raw_actions = torch.zeros((env.num_envs, cfg.action_dim), device=env.device)

    @property
    def action_dim(self) -> int:
        return self._raw_actions.shape[1]

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

    def apply_actions(self):
        pass


def _make_env(num_envs: int = 2):
    sim = MagicMock()
    sim.is_playing.return_value = True
    return SimpleNamespace(num_envs=num_envs, device="cpu", sim=sim)


def test_action_space_and_processing_follow_term_bounds():
    """Compose term bounds and enforce them before term-specific processing."""
    cfg = SimpleNamespace(
        bounded=DummyActionTermCfg(class_type=DummyActionTerm, raw_action_bounds=(-1.0, 1.0), action_dim=2),
        unbounded=DummyActionTermCfg(class_type=DummyActionTerm, action_dim=1),
    )
    manager = ActionManager(cfg, _make_env())

    np.testing.assert_array_equal(manager.action_space.low, [-1.0, -1.0, -np.inf])
    np.testing.assert_array_equal(manager.action_space.high, [1.0, 1.0, np.inf])

    manager.process_action(torch.tensor([[-2.0, 0.5, 4.0], [2.0, -0.5, -4.0]]))

    torch.testing.assert_close(manager.get_term("bounded").raw_actions, torch.tensor([[-1.0, 0.5], [1.0, -0.5]]))
    torch.testing.assert_close(manager.get_term("unbounded").raw_actions, torch.tensor([[4.0], [-4.0]]))


@pytest.mark.parametrize("bounds", [(1.0, -1.0), (0.0, 0.0), (np.nan, 1.0), (-1.0, np.nan)])
def test_invalid_raw_action_bounds_are_rejected(bounds):
    """Reject malformed bounds while constructing the action contract."""
    cfg = SimpleNamespace(action=DummyActionTermCfg(class_type=DummyActionTerm, raw_action_bounds=bounds, action_dim=1))

    with pytest.raises(ValueError, match="Invalid raw action bounds"):
        ActionManager(cfg, _make_env())
