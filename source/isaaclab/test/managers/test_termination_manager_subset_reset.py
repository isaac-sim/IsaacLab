# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for subset-reset statistics in the termination manager.

These tests run without Isaac Sim: the manager is exercised through a minimal
environment stub, mirroring the manager-only reproduction from the issue.
"""

import pytest
import torch

from isaaclab.managers import TerminationManager, TerminationTermCfg


class _PlayingSimStub:
    """Sim stub reporting as playing so manager init skips deferred resolution."""

    def is_playing(self) -> bool:
        return True


class _SubsetResetEnv:
    """Minimal env stub (no simulator) for the termination manager tests."""

    def __init__(self, num_envs: int, device: str):
        self.num_envs = num_envs
        self.device = device
        self.sim = _PlayingSimStub()
        # per-env termination causes written by the test
        self.term_states: dict[str, torch.Tensor] = {}


def _success_flag(env) -> torch.Tensor:
    """Returns the per-env success flag stored on the env by the test."""
    return env.term_states["success"].clone()


def _timeout_flag(env) -> torch.Tensor:
    """Returns the per-env timeout flag stored on the env by the test."""
    return env.term_states["timeout"].clone()


@pytest.fixture
def subset_env():
    """Four rows: row 0 stores success, rows 1-3 store timeout (the issue case)."""
    env = _SubsetResetEnv(num_envs=4, device="cpu")
    env.term_states = {
        "success": torch.tensor([True, False, False, False]),
        "timeout": torch.tensor([False, True, True, True]),
    }
    return env


def _make_manager(env) -> TerminationManager:
    cfg = {
        "success": TerminationTermCfg(func=_success_flag, time_out=False),
        "timeout": TerminationTermCfg(func=_timeout_flag, time_out=True),
    }
    tm = TerminationManager(cfg, env)
    # one observed step: fills the last-episode dones from the stored causes,
    # mirroring the step -> compute -> reset flow of the real env loop
    tm.compute()
    return tm


def test_reset_none_reports_all_envs(subset_env):
    """reset(None) keeps the existing all-environment behavior."""
    tm = _make_manager(subset_env)

    stats = tm.reset(None)
    assert stats["Episode_Termination/success"] == pytest.approx(0.25)
    assert stats["Episode_Termination/timeout"] == pytest.approx(0.75)


def test_reset_subset_reports_only_selected_rows(subset_env):
    """reset(env_ids) statistics cover only the selected rows.

    In asynchronous vectorized setups a single row can be reset while others keep
    their last termination causes; the reported means must not pick those up.
    """
    tm = _make_manager(subset_env)

    stats = tm.reset([0])
    assert stats["Episode_Termination/success"] == pytest.approx(1.0)
    assert stats["Episode_Termination/timeout"] == pytest.approx(0.0)

    stats = tm.reset([1, 2])
    assert stats["Episode_Termination/success"] == pytest.approx(0.0)
    assert stats["Episode_Termination/timeout"] == pytest.approx(1.0)

    stats = tm.reset([3])
    assert stats["Episode_Termination/success"] == pytest.approx(0.0)
    assert stats["Episode_Termination/timeout"] == pytest.approx(1.0)


def test_reset_all_ids_matches_reset_none(subset_env):
    """Passing every row explicitly agrees with the reset(None) selector."""
    tm = _make_manager(subset_env)

    stats_none = tm.reset(None)
    stats_all_ids = tm.reset([0, 1, 2, 3])
    assert stats_all_ids["Episode_Termination/success"] == pytest.approx(stats_none["Episode_Termination/success"])
    assert stats_all_ids["Episode_Termination/timeout"] == pytest.approx(stats_none["Episode_Termination/timeout"])
