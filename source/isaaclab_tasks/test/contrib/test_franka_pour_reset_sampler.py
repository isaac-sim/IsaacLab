# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU tests for the Franka Pour reset-dataset sampler."""

import pytest
import torch

from isaaclab_tasks.contrib.franka_pour.reset_sampler import ResetDatasetSampler, ResetDatasetSamplerCfg

pytestmark = pytest.mark.kitless


def test_adaptive_weights_preserve_pour_configuration_and_recent_outcomes():
    """The adapter passes the target, concentration, floor, and history window to shared utilities."""
    sampler = ResetDatasetSampler(
        3,
        "cpu",
        ResetDatasetSamplerCfg(monitored_history_len=4, target_success_rate=0.25, kappa=4.0, epsilon=0.01),
    )
    torch.testing.assert_close(sampler._probabilities(), torch.full((3,), 1.0 / 3.0))
    sampler._record_validated(
        torch.tensor([0] * 4 + [1] * 6 + [2] * 4),
        torch.tensor([False] * 4 + [True, True, False, False, False, True] + [True] * 4),
    )
    # Recent rates are 0, 1/4, and 1. Their normalized Beta weights including the floor are:
    torch.testing.assert_close(sampler._probabilities(), torch.tensor([64.0 / 867, 739.0 / 867, 64.0 / 867]))
    sampler._record_validated(torch.ones(4, dtype=torch.long), torch.ones(4, dtype=torch.bool))
    torch.testing.assert_close(sampler._probabilities(), torch.full((3,), 1.0 / 3.0))


def test_uniform_replay_covers_every_row_before_repeating(monkeypatch: pytest.MonkeyPatch):
    """The configured replay fraction draws one complete shuffled cycle without replacement."""
    sampler = ResetDatasetSampler(
        5,
        "cpu",
        ResetDatasetSamplerCfg(uniform_fraction=0.5),
    )
    cyclic_rows: list[int] = []
    take_uniform_rows = sampler._take_uniform_rows

    def record_uniform_rows(count: int, generator: torch.Generator | None) -> torch.Tensor:
        rows = take_uniform_rows(count, generator)
        cyclic_rows.extend(rows.tolist())
        return rows

    monkeypatch.setattr(sampler, "_take_uniform_rows", record_uniform_rows)
    sampler._sample_with_uniform_replay(10, generator=torch.Generator().manual_seed(17))

    metrics = sampler.metrics()
    assert metrics["sampler/uniform_replay_fraction"] == pytest.approx(0.5)
    assert len(cyclic_rows) == sampler.row_count
    assert set(cyclic_rows) == set(range(sampler.row_count))
