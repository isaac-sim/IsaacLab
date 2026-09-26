# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the benchmark metrics helpers (Isaac-Sim-free)."""

import pytest

from isaaclab.benchmark.metrics import (
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


def test_mean_std_peak_computes_peak():
    ms = mean_std_peak([1.0, 2.0, 3.0])
    assert isinstance(ms, MeanStd)
    assert ms.mean == pytest.approx(2.0)
    assert ms.std == pytest.approx(1.0)
    assert ms.peak == pytest.approx(3.0)


def test_mean_std_omits_peak():
    ms = mean_std([10.0, 20.0])
    assert ms.peak is None
    assert ms.mean == pytest.approx(15.0)


def test_mean_std_empty_is_zero():
    ms = mean_std_peak([])
    assert ms.mean == 0.0 and ms.std == 0.0 and ms.peak == 0.0


def test_ema_matches_manual():
    # e0 = 2.0; e1 = 0.25 * 10 + 0.75 * 2.0 = 4.0; e2 = 0.25 * 10 + 0.75 * 4.0 = 5.5
    assert ema([2.0, 10.0, 10.0], 0.25) == pytest.approx(5.5)


def test_ema_empty_is_zero():
    assert ema([], 0.1) == 0.0


def test_check_convergence_passes_on_stable_high_rewards():
    res = check_convergence([100.0] * 10, threshold=50.0)
    assert res["passed"] is True
    assert res["tail_mean"] == pytest.approx(100.0)


def test_check_convergence_fails_when_below_threshold():
    res = check_convergence([1.0] * 10, threshold=50.0)
    assert res["passed"] is False


def test_get_success_rate_log_prefers_first_tag():
    data = {"Episode/Metrics/success_rate": [0.5], "Metrics/success_rate": [0.9]}
    assert get_success_rate_log(data) == [0.9]
    assert get_success_rate_log({}) is None


def test_success_rate_tracker_convergence():
    t = SuccessRateTracker(threshold=0.5, window=2, num_steps_per_env=1)
    for v in (0.6, 0.7):
        t.record_step({"log": {"Metrics/success_rate": v}})
        # No process group is initialized, so the reduction leaves the local samples unchanged.
        t.all_reduce_iteration("cpu")
        t.end_iteration()
    assert t.converged is True
    assert t.tail_mean == pytest.approx(0.65)


def test_check_convergence_high_cv_fails_despite_mean_above_threshold():
    # Series whose tail mean clears the threshold but CV is too high to pass.
    # With window_pct=1.0, the whole series is the tail.
    # [1, 1000] -> tail_mean ~500.5, threshold=2.0, cv >> 20 -> passed False.
    rewards = [1.0, 1000.0]
    res = check_convergence(rewards, threshold=2.0, window_pct=1.0)
    assert res["passed"] is False
    assert res["cv"] > 20.0
    assert res["tail_mean"] >= 2.0


def test_check_convergence_empty_rewards():
    res = check_convergence([], threshold=1.0)
    assert res == {"tail_mean": 0.0, "cv": 999.9, "passed": False}


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


def test_success_rate_tracker_item_tensor_path():
    class _T:
        def item(self):
            return 0.7

    t = SuccessRateTracker(threshold=0.5, window=1, num_steps_per_env=1)
    t.record_step({"log": {"Metrics/success_rate": _T()}})
    mean = t.end_iteration()
    assert mean == pytest.approx(0.7)
    assert t.history == [pytest.approx(0.7)]


def test_success_rate_tracker_no_data_end_iteration_returns_none():
    t = SuccessRateTracker(threshold=0.5, window=1, num_steps_per_env=2)
    t.record_step({"log": {}})
    # A step without success data still advances the iteration step count.
    assert t.at_iteration_boundary is False
    result = t.end_iteration()
    assert result is None
    assert t.history == []


def test_parse_tf_logs_empty_dir_returns_empty(tmp_path, caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        result = parse_tf_logs(str(tmp_path))
    assert result == {}
    assert any("No TensorBoard event files" in r.getMessage() for r in caplog.records)


def test_check_convergence_zero_mean_returns_high_cv():
    result = check_convergence([0.0, 0.0, 0.0, 0.0, 0.0], threshold=0.3)
    assert result["tail_mean"] == 0.0
    assert result["cv"] == 999.9
    assert result["passed"] is False


def test_success_rate_step_value_reads_scalar():
    import torch

    tag = SUCCESS_RATE_LOG_TAGS[0]
    assert success_rate_step_value({tag: torch.tensor(0.75)}) == pytest.approx(0.75)
    assert success_rate_step_value({tag: 0.5}) == pytest.approx(0.5)
    assert success_rate_step_value({"other/metric": 1.0}) is None
