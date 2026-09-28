# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from unittest.mock import MagicMock

import pytest
import torch

from isaaclab.managers import TerminationManager, TerminationTermCfg
from isaaclab.test.utils import test_devices

pytestmark = pytest.mark.unit


class DummyEnv:
    """Minimal mutable env stub for the termination manager tests."""

    def __init__(self, num_envs: int, device: str, sim: MagicMock):
        self.num_envs = num_envs
        self.device = device
        self.sim = sim
        self.counter = 0  # mutable step counter used by test terms


def fail_every_5_steps(env) -> torch.Tensor:
    """Returns True for all envs when counter is a positive multiple of 5."""
    cond = env.counter > 0 and (env.counter % 5 == 0)
    return torch.full((env.num_envs,), cond, dtype=torch.bool, device=env.device)


def fail_every_10_steps(env) -> torch.Tensor:
    """Returns True for all envs when counter is a positive multiple of 10."""
    cond = env.counter > 0 and (env.counter % 10 == 0)
    return torch.full((env.num_envs,), cond, dtype=torch.bool, device=env.device)


def fail_every_3_steps(env) -> torch.Tensor:
    """Returns True for all envs when counter is a positive multiple of 3."""
    cond = env.counter > 0 and (env.counter % 3 == 0)
    return torch.full((env.num_envs,), cond, dtype=torch.bool, device=env.device)


@pytest.fixture
def env():
    # simulation double that has not started playing
    sim = MagicMock()
    sim.is_playing.return_value = False
    return DummyEnv(num_envs=20, device="cpu", sim=sim)


def test_term_transitions_and_persistence(env):
    """Concise transitions: single fire, persist, switch, both, persist.

    Uses 3-step and 5-step terms and verifies current-step values and last-episode persistence.
    """
    cfg = {
        "term_3": TerminationTermCfg(func=fail_every_3_steps, time_out=False),
        "term_5": TerminationTermCfg(func=fail_every_5_steps, time_out=False),
    }
    tm = TerminationManager(cfg, env)

    # step 3: only term_3 -> last_episode [True, False]
    env.counter = 3
    out = tm.compute()
    assert torch.all(tm.get_term("term_3")) and torch.all(~tm.get_term("term_5"))
    assert torch.all(out)
    assert torch.all(tm._last_episode_dones[:, 0]) and torch.all(~tm._last_episode_dones[:, 1])

    # step 4: none -> last_episode persists [True, False]
    env.counter = 4
    out = tm.compute()
    assert torch.all(~out)
    assert torch.all(~tm.get_term("term_3")) and torch.all(~tm.get_term("term_5"))
    assert torch.all(tm._last_episode_dones[:, 0]) and torch.all(~tm._last_episode_dones[:, 1])

    # step 5: only term_5 -> last_episode [False, True]
    env.counter = 5
    out = tm.compute()
    assert torch.all(~tm.get_term("term_3")) and torch.all(tm.get_term("term_5"))
    assert torch.all(out)
    assert torch.all(~tm._last_episode_dones[:, 0]) and torch.all(tm._last_episode_dones[:, 1])

    # step 15: both -> last_episode [True, True]
    env.counter = 15
    out = tm.compute()
    assert torch.all(tm.get_term("term_3")) and torch.all(tm.get_term("term_5"))
    assert torch.all(out)
    assert torch.all(tm._last_episode_dones[:, 0]) and torch.all(tm._last_episode_dones[:, 1])

    # step 16: none -> persist [True, True]
    env.counter = 16
    out = tm.compute()
    assert torch.all(~out)
    assert torch.all(~tm.get_term("term_3")) and torch.all(~tm.get_term("term_5"))
    assert torch.all(tm._last_episode_dones[:, 0]) and torch.all(tm._last_episode_dones[:, 1])


def test_time_out_vs_terminated_split(env):
    cfg = {
        "term_5": TerminationTermCfg(func=fail_every_5_steps, time_out=False),  # terminated
        "term_10": TerminationTermCfg(func=fail_every_10_steps, time_out=True),  # timeout
    }
    tm = TerminationManager(cfg, env)

    # Before the first compute, every public done buffer has one entry per env and is False
    assert tm.active_terms == ["term_5", "term_10"]
    for dones in (tm.dones, tm.time_outs, tm.terminated):
        assert dones.shape == (env.num_envs,)
        assert torch.all(~dones)

    # Step 5: terminated fires, not timeout
    env.counter = 5
    out = tm.compute()
    assert torch.all(out)
    assert torch.all(tm.terminated) and torch.all(~tm.time_outs)

    # Step 10: both fire; timeout and terminated both True
    env.counter = 10
    out = tm.compute()
    assert torch.all(out)
    assert torch.all(tm.terminated) and torch.all(tm.time_outs)


def fail_first_quarter(env) -> torch.Tensor:
    """Returns True for the first quarter of the envs."""
    return torch.arange(env.num_envs, device=env.device) < env.num_envs // 4


@pytest.mark.parametrize("device", test_devices())
def test_reset_logs_term_rates_without_host_reads(device):
    """Reset logs each term's last-episode activation rate as a device scalar without synchronizing."""
    sim = MagicMock()
    sim.is_playing.return_value = False
    env = DummyEnv(num_envs=20, device=device, sim=sim)
    cfg = {
        "quarter": TerminationTermCfg(func=fail_first_quarter, time_out=False),
        "term_3": TerminationTermCfg(func=fail_every_3_steps, time_out=True),
    }
    tm = TerminationManager(cfg, env)
    env.counter = 1
    tm.compute()

    previous = torch.cuda.get_sync_debug_mode() if device.startswith("cuda") else None
    if previous is not None:
        torch.cuda.synchronize(device)
        torch.cuda.set_sync_debug_mode("error")
    try:
        extras = tm.reset()
    finally:
        if previous is not None:
            torch.cuda.set_sync_debug_mode(previous)

    assert set(extras) == {"Episode_Termination/quarter", "Episode_Termination/term_3"}
    assert all(value.ndim == 0 and value.device == torch.device(device) for value in extras.values())
    assert extras["Episode_Termination/quarter"].item() == 0.25
    assert extras["Episode_Termination/term_3"].item() == 0.0
