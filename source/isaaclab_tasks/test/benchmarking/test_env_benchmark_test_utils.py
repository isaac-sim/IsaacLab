# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for training benchmark KPI evaluation."""

import math

import env_benchmark_test_utils as utils
import pytest

_SUSTAINED_SUCCESS_THRESHOLD = {"value": 0.3, "consecutive_samples": 20}


def _evaluate_success_rate(monkeypatch, workflow, tag, values):
    """Evaluate a success-rate series with the sustained-success requirement."""
    log_data = {tag: list(enumerate(values))}
    monkeypatch.setattr(utils, "_retrieve_logs", lambda workflow, task: log_data)
    return utils.evaluate_job(
        workflow,
        "Isaac-Test",
        {"lower_thresholds": {"success_rate": _SUSTAINED_SUCCESS_THRESHOLD}},
        duration=1.0,
    )


@pytest.mark.parametrize(
    "task_id,expected",
    [
        ("Isaac-Reorient-Cube-Shadow-Camera", True),
        ("Isaac-Reorient-Cube-Shadow-Camera-v0", True),
        ("Isaac-Reorient-Cube-Shadow-Camera-Play", False),
        ("Isaac-Reorient-Cube-Shadow-Camera-Play-v0", False),
        ("Isaac-Reorient-Cube-Shadow-Camera-Benchmark", False),
        ("Isaac-Reorient-Cube-Shadow-Camera-Benchmark-v1", False),
        ("Isaac-Reorient-Cube-Shadow-Camera-Benchmark-Direct", False),
        ("Isaac-Reorient-Cube-Shadow-Camera-Benchmark-Direct-v0", False),
    ],
)
def test_training_task_filter_excludes_play_and_benchmark(task_id: str, expected: bool):
    """Verify inference-only variants never enter the training benchmark matrix."""
    assert utils._is_training_task(task_id) is expected


@pytest.mark.parametrize(
    "log_data",
    [
        {"Train/mean_episode_length": [(0, 100.0)]},
        {"Train/mean_reward": [(0, math.nan)]},
    ],
    ids=["missing", "nan"],
)
def test_evaluate_job_fails_when_configured_reward_is_unavailable(monkeypatch, log_data):
    """Verify missing or invalid configured metrics cannot produce a successful KPI payload."""
    monkeypatch.setattr(utils, "_retrieve_logs", lambda workflow, task: log_data)

    payload = utils.evaluate_job(
        "rsl_rl",
        "Isaac-Test",
        {"lower_thresholds": {"reward": 1000.0}},
        duration=1.0,
    )

    assert payload["success"] is False
    assert payload["msg"] == "reward metric is missing or non-numeric"
    assert payload["reward"] is None
    assert payload["reward_threshold"] == 1000.0


@pytest.mark.parametrize(
    "workflow_name,tag",
    [
        ("rl_games", "Episode/Metrics/success_rate"),
        ("rsl_rl", "Metrics/success_rate"),
        ("skrl", "Metrics/success_rate"),
    ],
)
def test_evaluate_job_passes_sustained_success_rate(monkeypatch, workflow_name, tag):
    """Verify each supported workflow passes after 20 consecutive successful samples."""
    payload = _evaluate_success_rate(monkeypatch, workflow_name, tag, [0.2, *([0.3] * 20)])

    assert payload["success"] is True
    assert payload["success_rate"] == 0.3
    assert payload["success_rate_threshold"] == 0.3
    assert payload["success_rate_consecutive_samples"] == 20


def test_evaluate_job_fails_success_rate_below_threshold(monkeypatch):
    """Verify a sustained series below the configured success rate fails."""
    payload = _evaluate_success_rate(monkeypatch, "rsl_rl", "Metrics/success_rate", [0.29] * 20)

    assert payload["success"] is False
    assert payload["msg"] == "success_rate below threshold: 0.29 < 0.3"
    assert payload["success_rate"] == 0.29


def test_evaluate_job_fails_when_success_rate_is_missing(monkeypatch):
    """Verify a missing configured success-rate metric fails."""
    payload = _evaluate_success_rate(monkeypatch, "rsl_rl", "Train/mean_reward", [1000.0] * 20)

    assert payload["success"] is False
    assert payload["msg"] == "success_rate metric is missing or non-numeric"
    assert payload["success_rate"] is None


def test_evaluate_job_treats_success_rate_as_missing_for_unsupported_workflow(monkeypatch):
    """Verify an unmapped workflow fails the KPI instead of raising an exception."""
    payload = _evaluate_success_rate(monkeypatch, "sb3", "Metrics/success_rate", [0.3] * 20)

    assert payload["success"] is False
    assert payload["msg"] == "success_rate metric is missing or non-numeric"
    assert payload["success_rate"] is None


def test_evaluate_job_fails_when_success_rate_is_nonfinite(monkeypatch):
    """Verify a non-finite configured success-rate metric fails."""
    payload = _evaluate_success_rate(
        monkeypatch,
        "rsl_rl",
        "Metrics/success_rate",
        [0.3] * 19 + [math.nan],
    )

    assert payload["success"] is False
    assert payload["msg"] == "success_rate metric is missing or non-numeric"
    assert payload["success_rate"] is None


def test_evaluate_job_resets_sustained_success_streak(monkeypatch):
    """Verify a below-threshold sample resets the consecutive-success streak."""
    payload = _evaluate_success_rate(
        monkeypatch,
        "rsl_rl",
        "Metrics/success_rate",
        [0.3] * 19 + [0.29] + [0.3] * 19,
    )

    assert payload["success"] is False
    assert payload["msg"] == "success_rate below threshold: 0.29 < 0.3"
    assert payload["success_rate"] == 0.29


def test_scalar_threshold_behavior_is_preserved(monkeypatch):
    """Verify numeric thresholds retain the existing reward aggregation behavior."""
    log_data = {"Train/mean_reward": list(enumerate([1.0, 2.0, 3.0]))}
    monkeypatch.setattr(utils, "_retrieve_logs", lambda workflow, task: log_data)

    payload = utils.evaluate_job(
        "rsl_rl",
        "Isaac-Test",
        {"lower_thresholds": {"reward": 2.0}},
        duration=1.0,
    )

    assert payload["success"] is True
    assert payload["reward"] == 2.0
    assert payload["reward_threshold"] == 2.0
    assert "reward_consecutive_samples" not in payload
