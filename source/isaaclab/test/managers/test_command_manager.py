# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command terms reset and resample only the environments their mask selects."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import CommandManager, CommandTerm, CommandTermCfg
from isaaclab.utils import configclass, env_mask_from_ids

pytestmark = pytest.mark.unit


class _CountingCommand(CommandTerm):
    """Counts the resamples of each environment in its command."""

    def __init__(self, cfg: CommandTermCfg, env):
        super().__init__(cfg, env)
        self._command = torch.zeros(self.num_envs)
        self.metrics["steps"] = torch.zeros(self.num_envs)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self):
        self.metrics["steps"] += 1.0

    def _resample_command(self, env_mask):
        self._command += env_mask.torch

    def _update_command(self):
        pass


def test_reset_and_expiry_resample_only_the_selected_environments():
    """Resets and expired timers resample the selected environments; the others keep their command and timer."""
    env = SimpleNamespace(num_envs=4, device="cpu")
    term = _CountingCommand(CommandTermCfg(class_type=_CountingCommand, resampling_time_range=(1.0, 1.0)), env)

    term.reset(env_mask_from_ids([1, 3], env.num_envs, env.device))
    torch.testing.assert_close(term.command, torch.tensor([0.0, 1.0, 0.0, 1.0]))
    torch.testing.assert_close(term.time_left, torch.tensor([0.0, 1.0, 0.0, 1.0]))

    # only the timers of environments 0 and 2 have expired
    term.compute(dt=0.5)
    torch.testing.assert_close(term.command, torch.ones(4))
    torch.testing.assert_close(term.time_left, torch.tensor([1.0, 0.5, 1.0, 0.5]))
    torch.testing.assert_close(term.command_counter, torch.ones(4, dtype=torch.long))

    # metrics are averaged over the reset environments and cleared only there
    log = term.reset(env_mask_from_ids([0], env.num_envs, env.device))
    torch.testing.assert_close(log["steps"], torch.tensor(1.0))
    torch.testing.assert_close(term.metrics["steps"], torch.tensor([0.0, 1.0, 1.0, 1.0]))
    torch.testing.assert_close(term.command_counter, torch.ones(4, dtype=torch.long))
    torch.testing.assert_close(term.command, torch.tensor([2.0, 1.0, 1.0, 1.0]))


class _IndexCommand(_CountingCommand):
    """Records the indices it resamples."""

    def _resample_command(self, env_ids):
        self.resampled = env_ids


@configclass
class _IndexCommandsCfg:
    cmd = CommandTermCfg(class_type=_IndexCommand, resampling_time_range=(1.0, 1.0))


def test_index_resampling_reuses_the_reset_indices():
    """A term resampling from indices makes resets compute them, and receives the ones the reset was given."""
    env = SimpleNamespace(num_envs=4, device="cpu", sim=SimpleNamespace(is_playing=lambda: True))
    manager = CommandManager(_IndexCommandsCfg(), env)
    assert manager.takes_env_ids

    env_ids = torch.tensor([1, 3])
    manager.reset(env_ids, env_mask_from_ids(env_ids, env.num_envs, env.device))
    assert manager.get_term("cmd").resampled is env_ids
