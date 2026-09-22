# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

from __future__ import annotations

import pytest
import torch

from isaaclab.managers import TerminationManager, TerminationTermCfg

pytestmark = pytest.mark.unit


def _fail_every(period: int):
    """Build a termination term that fires for all envs when the env counter is a positive multiple of ``period``."""

    def term(env) -> torch.Tensor:
        cond = env.counter > 0 and env.counter % period == 0
        return torch.full((env.num_envs,), cond, dtype=torch.bool, device=env.device)

    return term


fail_every_3_steps = _fail_every(3)
fail_every_5_steps = _fail_every(5)
fail_every_10_steps = _fail_every(10)


@pytest.fixture
def env(make_env):
    return make_env(counter=0)


def test_initial_state_and_shapes(env):
    """Buffers are allocated per term and start as all False."""
    cfg = {
        "term_5": TerminationTermCfg(func=fail_every_5_steps),
        "term_10": TerminationTermCfg(func=fail_every_10_steps),
    }
    tm = TerminationManager(cfg, env)

    assert tm.active_terms == ["term_5", "term_10"]
    assert tm._term_dones.shape == tm._last_episode_dones.shape == (env.num_envs, 2)
    assert tm.dones.shape == tm.time_outs.shape == tm.terminated.shape == (env.num_envs,)
    assert not tm._term_dones.any() and not tm._last_episode_dones.any()


def test_term_transitions_and_persistence(env):
    """Per-term dones reflect the current step while the last-episode dones persist until a term fires again."""
    cfg = {
        "term_3": TerminationTermCfg(func=fail_every_3_steps),
        "term_5": TerminationTermCfg(func=fail_every_5_steps),
    }
    tm = TerminationManager(cfg, env)

    # (counter, term_3 fired, term_5 fired, last-episode dones)
    steps = [
        (3, True, False, (True, False)),
        (4, False, False, (True, False)),
        (5, False, True, (False, True)),
        (15, True, True, (True, True)),
        (16, False, False, (True, True)),
    ]
    for counter, term_3, term_5, last_episode in steps:
        env.counter = counter
        dones = tm.compute()
        assert torch.all(dones == (term_3 or term_5))
        assert torch.all(tm.get_term("term_3") == term_3) and torch.all(tm.get_term("term_5") == term_5)
        assert torch.all(tm._last_episode_dones == torch.tensor(last_episode))

    extras = tm.reset()
    assert extras == {"Episode_Termination/term_3": 1.0, "Episode_Termination/term_5": 1.0}


def test_time_out_vs_terminated_split(env):
    """Time-out terms feed ``time_outs`` while the others feed ``terminated``; both feed the net signal."""
    cfg = {
        "term_5": TerminationTermCfg(func=fail_every_5_steps, time_out=False),
        "term_10": TerminationTermCfg(func=fail_every_10_steps, time_out=True),
    }
    tm = TerminationManager(cfg, env)

    env.counter = 5
    assert torch.all(tm.compute())
    assert torch.all(tm.terminated) and not tm.time_outs.any()

    env.counter = 10
    assert torch.all(tm.compute())
    assert torch.all(tm.terminated) and torch.all(tm.time_outs)
    assert tm.get_active_iterable_terms(env_idx=0) == [("term_5", [1.0]), ("term_10", [1.0])]
