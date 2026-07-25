# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for utilities shared by core tasks."""

import pytest
import torch

import isaaclab_tasks.core.utils as core_utils


def test_sample_joint_positions_within_limits_interpolates_endpoints(monkeypatch):
    """Verify reset-noise scales map samples between defaults and limit endpoints."""
    default_position = torch.tensor([[0.1, -0.2]])
    limits = torch.tensor([[[-1.0, 1.0], [-2.0, 2.0]]])
    monkeypatch.setattr(
        core_utils.math_utils,
        "sample_uniform",
        lambda *args, **kwargs: torch.tensor([[-1.0, 1.0]]),
    )

    assert torch.equal(core_utils.sample_joint_positions_within_limits(default_position, limits, 0.0), default_position)
    assert torch.allclose(
        core_utils.sample_joint_positions_within_limits(default_position, limits, 0.2),
        torch.tensor([[-0.12, 0.24]]),
    )
    assert torch.equal(
        core_utils.sample_joint_positions_within_limits(default_position, limits, 1.0),
        torch.tensor([[-1.0, 2.0]]),
    )


@pytest.mark.parametrize("noise_scale", (-1.0e-6, 1.000001))
def test_sample_joint_positions_within_limits_rejects_invalid_scale(noise_scale):
    """Verify interpolation scales outside the supported interval are rejected."""
    with pytest.raises(ValueError, match="Expected noise_scale in"):
        core_utils.sample_joint_positions_within_limits(
            torch.zeros(1, 1),
            torch.tensor([[[-1.0, 1.0]]]),
            noise_scale,
        )


def test_episode_error_recorder_reports_threshold_independent_statistics():
    """Verify episode error summaries contain no success-threshold policy."""
    recorder = core_utils.EpisodeErrorRecorder(num_envs=3, device="cpu")
    recorder.update(torch.tensor([0.3, 0.2, 0.4]))
    recorder.update(torch.tensor([0.1, 0.25, 0.35]))

    statistics = recorder.reset(torch.tensor([0, 1, 2]))

    # values are 0-dim device tensors (sync-free logging); compare via item()
    assert all(isinstance(v, torch.Tensor) and v.ndim == 0 for v in statistics.values())
    assert {k: v.item() for k, v in statistics.items()} == pytest.approx(
        {"mean": 0.21666667, "median": 0.2, "p90": 0.32}
    )
    assert torch.isinf(recorder.minimum_error).all()


def test_episode_error_recorder_skips_episodes_without_samples():
    """Verify initial resets do not emit non-finite diagnostic values."""
    recorder = core_utils.EpisodeErrorRecorder(num_envs=2, device="cpu")

    assert recorder.reset(torch.tensor([0, 1])) == {}


def test_episode_error_recorder_update_matches_masked_indexing_reference():
    """Verify the sync-free update equals boolean-mask indexing on NaN/inf/finite mixes."""

    def reference_update(minimum_error, has_sample, error):
        finite = torch.isfinite(error)
        minimum_error[finite] = torch.minimum(minimum_error[finite], error[finite])
        has_sample |= finite

    num_envs = 6
    recorder = core_utils.EpisodeErrorRecorder(num_envs=num_envs, device="cpu")
    reference_minimum = torch.full((num_envs,), torch.inf)
    reference_has_sample = torch.zeros(num_envs, dtype=torch.bool)
    nan, inf = float("nan"), float("inf")
    samples = [
        torch.tensor([0.5, nan, inf, -inf, 0.4, nan]),
        torch.tensor([nan, 0.3, inf, 0.2, 0.6, nan]),
        torch.tensor([0.1, inf, -inf, 0.7, nan, nan]),
    ]

    for error in samples:
        recorder.update(error)
        reference_update(reference_minimum, reference_has_sample, error)

    assert torch.equal(recorder.minimum_error, reference_minimum)
    assert torch.equal(recorder._has_sample, reference_has_sample)
    # env 5 never received a finite sample and must stay excluded from the statistics
    assert recorder.minimum_error[5] == torch.inf
    assert not recorder._has_sample[5]
