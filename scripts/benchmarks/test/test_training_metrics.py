# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for benchmark training-metric logging helpers."""

from __future__ import annotations

import pytest

from scripts.benchmarks.utils import SUCCESS_RATE_LOG_TAGS, log_rl_training_metrics


class _FakeBenchmark:
    """Collect benchmark measurements without initializing benchmark backends."""

    def __init__(self):
        self.measurements: list[tuple[str, str, object, str]] = []

    def add_measurement(self, phase, measurement):
        self.measurements.append((phase, measurement.name, measurement.value, getattr(measurement, "unit", "")))

    def measurement_by_name(self, name: str):
        return next(m for m in self.measurements if m[1] == name)


@pytest.mark.parametrize(
    "workflow,reward_tag,episode_length_tag",
    [
        ("rl_games", "rewards/iter", "episode_lengths/iter"),
        ("rsl_rl", "Train/mean_reward", "Train/mean_episode_length"),
    ],
)
def test_log_rl_training_metrics_skips_missing_short_run_scalars(
    workflow: str, reward_tag: str, episode_length_tag: str, capsys: pytest.CaptureFixture[str]
):
    """Short benchmark runs may finish before reward and episode-length scalars are emitted."""
    benchmark = _FakeBenchmark()

    log_rl_training_metrics(
        benchmark,
        log_data={},
        reward_tag=reward_tag,
        episode_length_tag=episode_length_tag,
        task="Isaac-Ant-v0",
        workflow=workflow,
        should_check_convergence=True,
    )

    assert benchmark.measurements == []
    output = capsys.readouterr().out
    assert f"TensorBoard log is missing '{reward_tag}'" in output
    assert f"TensorBoard log is missing '{episode_length_tag}'" in output
    assert f"Cannot check convergence because '{reward_tag}' was not logged" in output


@pytest.mark.parametrize(
    "workflow,reward_tag,episode_length_tag",
    [
        ("rl_games", "rewards/iter", "episode_lengths/iter"),
        ("rsl_rl", "Train/mean_reward", "Train/mean_episode_length"),
    ],
)
def test_log_rl_training_metrics_logs_present_normal_run_scalars(
    workflow: str, reward_tag: str, episode_length_tag: str, capsys: pytest.CaptureFixture[str]
):
    """Normal runs with reward and episode-length scalars should log train metrics."""
    benchmark = _FakeBenchmark()

    log_rl_training_metrics(
        benchmark,
        log_data={
            reward_tag: [1.0, 2.0, 3.0],
            episode_length_tag: [10.0, 11.0],
            SUCCESS_RATE_LOG_TAGS[0]: [0.25, 0.5],
        },
        reward_tag=reward_tag,
        episode_length_tag=episode_length_tag,
        task="Isaac-Ant-v0",
        workflow=workflow,
    )

    assert benchmark.measurement_by_name("Rewards")[2] == [1.0, 2.0, 3.0]
    assert benchmark.measurement_by_name("Max Rewards")[2] == 3.0
    assert benchmark.measurement_by_name("Episode Lengths")[2] == [10.0, 11.0]
    assert benchmark.measurement_by_name("Max Episode Lengths")[2] == 11.0
    assert benchmark.measurement_by_name("Success Rates")[2] == [0.25, 0.5]
    assert benchmark.measurement_by_name("success_rate")[2] == 0.5
    assert "TensorBoard log is missing" not in capsys.readouterr().out
