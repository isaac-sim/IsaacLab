# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for observation managers."""

from __future__ import annotations

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none
from typing import TYPE_CHECKING, cast

import pytest
import torch

from isaaclab.managers import ObservationGroupCfg, ObservationManager, ObservationTermCfg
from isaaclab.utils.configclass import configclass

pytestmark = pytest.mark.unit

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def dummy_observation(env: DummyEnv) -> torch.Tensor:
    """Return the dummy environment observation."""
    return env.observation


class DummySimulation:
    """Minimal playing simulation double."""

    def is_playing(self) -> bool:
        """Return whether the simulated timeline is playing."""
        return True


class DummyEnv:
    """Minimal environment double used by :class:`ObservationManager`."""

    def __init__(self, num_envs: int = 2) -> None:
        self.num_envs = num_envs
        self.device = "cpu"
        self.sim = DummySimulation()
        self.observation = torch.arange(num_envs, dtype=torch.float32).unsqueeze(-1)


@configclass
class HistoryObservationsCfg:
    """Observation configuration with group-level history."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Policy observation group configuration."""

        dummy: ObservationTermCfg = ObservationTermCfg(func=dummy_observation)

        def __post_init__(self):
            self.history_length = 5

    policy: PolicyCfg = PolicyCfg()


def test_compute_updates_history_only_when_requested():
    """Observation history changes only when ``update_history`` is enabled."""
    env = DummyEnv()
    manager = ObservationManager(HistoryObservationsCfg(), cast("ManagerBasedEnv", env))
    history = manager._group_obs_term_history_buffer["policy"]["dummy"]

    torch.testing.assert_close(history.current_length, torch.zeros(env.num_envs, dtype=torch.int64))

    manager.compute()
    torch.testing.assert_close(history.current_length, torch.zeros(env.num_envs, dtype=torch.int64))

    manager.compute(update_history=True)
    torch.testing.assert_close(history.current_length, torch.ones(env.num_envs, dtype=torch.int64))
    history_after_update = history.buffer.clone()

    env.observation.add_(10.0)
    observations = manager.compute()
    policy_observation = observations["policy"]
    assert isinstance(policy_observation, torch.Tensor)
    torch.testing.assert_close(history.current_length, torch.ones(env.num_envs, dtype=torch.int64))
    torch.testing.assert_close(history.buffer, history_after_update)
    torch.testing.assert_close(policy_observation, history_after_update.reshape(env.num_envs, -1))

    manager.compute(update_history=True)
    torch.testing.assert_close(history.current_length, torch.full((env.num_envs,), 2, dtype=torch.int64))
    torch.testing.assert_close(history.buffer[:, -1], env.observation)
