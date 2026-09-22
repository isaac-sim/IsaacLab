# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the benchmark metrics helpers (Isaac-Sim-free)."""

import pytest

from isaaclab.benchmark.metrics import (
    RL_LIBRARY_DESCRIPTORS,
    SUCCESS_RATE_LOG_TAGS,
    SuccessRateTracker,
    check_convergence,
    ema,
    get_success_rate_log,
    mean_std,
    mean_std_peak,
    parse_tf_logs,
    success_rate_step_value,
)
from isaaclab.benchmark.schema import MeanStd

pytestmark = pytest.mark.benchmark


@pytest.mark.parametrize(
    ("framework", "tfevents_pattern", "reward_tag", "ep_length_tag"),
    [
        ("rsl_rl", "events*", "Train/mean_reward", "Train/mean_episode_length"),
        ("rl_games", "summaries/events*", "rewards/iter", "episode_lengths/iter"),
        ("skrl", "events*", "Reward / Total reward (mean)", "Episode / Total timesteps (mean)"),
        ("sb3", "PPO_*/events*", "rollout/ep_rew_mean", "rollout/ep_len_mean"),
    ],
)
def test_rl_library_descriptors(
    framework: str,
    tfevents_pattern: str,
    reward_tag: str,
    ep_length_tag: str,
):
    descriptor = RL_LIBRARY_DESCRIPTORS[framework]

    assert descriptor.framework == framework
    assert descriptor.tfevents_pattern == tfevents_pattern
    assert descriptor.reward_tag == reward_tag
    assert descriptor.ep_length_tag == ep_length_tag


def test_mean_std_aggregates():
    assert mean_std_peak([1.0, 2.0, 3.0]) == MeanStd(mean=2.0, std=1.0, peak=3.0)
    assert mean_std_peak([]) == MeanStd(mean=0.0, std=0.0, peak=0.0)
    assert mean_std([10.0, 20.0]) == MeanStd(mean=15.0, std=pytest.approx(50**0.5), peak=None)
    assert mean_std([]) == MeanStd(mean=0.0, std=0.0, peak=None)


def test_ema():
    # e = 0 -> 0.5 * 10 + 0.5 * 0 = 5 -> 0.5 * 10 + 0.5 * 5 = 7.5
    assert ema([0.0, 10.0, 10.0], 0.5) == pytest.approx(7.5)
    assert ema([3.0], 0.1) == 3.0
    assert ema([], 0.1) == 0.0


@pytest.mark.parametrize(
    ("rewards", "kwargs", "tail_mean", "passed"),
    [
        ([100.0] * 10, {"threshold": 50.0}, 100.0, True),
        ([1.0] * 10, {"threshold": 50.0}, 1.0, False),
        # The tail mean clears the threshold but the coefficient of variation is far above 20%.
        ([1.0, 1000.0], {"threshold": 2.0, "window_pct": 1.0}, 500.5, False),
    ],
)
def test_check_convergence(rewards, kwargs, tail_mean, passed):
    result = check_convergence(rewards, **kwargs)
    assert result["passed"] is passed
    assert result["tail_mean"] == pytest.approx(tail_mean)


def test_get_success_rate_log_prefers_first_tag():
    data = {"Episode/Metrics/success_rate": [0.5], "Metrics/success_rate": [0.9]}
    assert get_success_rate_log(data) == [0.9]
    assert get_success_rate_log({}) is None


def test_success_rate_tracker_convergence():
    t = SuccessRateTracker(threshold=0.5, window=2, num_steps_per_env=1)
    for v in (0.6, 0.7):
        t.record_step({"log": {"Metrics/success_rate": v}})
        t.end_iteration()
    assert t.converged is True
    assert t.tail_mean == pytest.approx(0.65)


@pytest.mark.parametrize("rewards", [[], [0.0] * 5])
def test_check_convergence_without_signal_reports_undefined_cv(rewards):
    assert check_convergence(rewards, threshold=0.3) == {"tail_mean": 0.0, "cv": 999.9, "passed": False}


def test_success_rate_tracker_multi_step_boundary():
    # num_steps_per_env=3: boundary fires after step 3, not after 1 or 2.
    t = SuccessRateTracker(threshold=0.5, window=1, num_steps_per_env=3)
    t.record_step({"log": {"Metrics/success_rate": 0.6}})
    assert t.at_iteration_boundary is False
    t.record_step({"log": {"Metrics/success_rate": 0.6}})
    assert t.at_iteration_boundary is False
    t.record_step({"log": {"Metrics/success_rate": 0.6}})
    assert t.at_iteration_boundary is True
    t.end_iteration()
    assert len(t.history) == 1


def test_success_rate_tracker_handles_tensor_and_missing_values():
    import torch

    t = SuccessRateTracker(threshold=0.5, window=1, num_steps_per_env=1)
    t.record_step({"log": {"Metrics/success_rate": torch.tensor(0.7)}})
    assert t.end_iteration() == pytest.approx(0.7)
    assert t.history == [pytest.approx(0.7)]

    t.record_step({"log": {}})
    assert t.at_iteration_boundary
    assert t.end_iteration() is None
    assert len(t.history) == 1


def test_parse_tf_logs_empty_dir_returns_empty(tmp_path, caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        result = parse_tf_logs(str(tmp_path))
    assert result == {}
    assert any("No TensorBoard event files" in r.getMessage() for r in caplog.records)


def test_success_rate_step_value_reads_scalar():
    import torch

    tag = SUCCESS_RATE_LOG_TAGS[0]
    assert success_rate_step_value({tag: torch.tensor(0.75)}) == pytest.approx(0.75)
    assert success_rate_step_value({tag: 0.5}) == pytest.approx(0.5)
    assert success_rate_step_value({"other/metric": 1.0}) is None
