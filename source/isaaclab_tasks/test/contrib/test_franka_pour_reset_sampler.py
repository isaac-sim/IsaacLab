# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU tests for the Franka Pour reset-dataset sampler."""

from collections import deque

import pytest
import torch

from isaaclab_tasks.contrib.franka_pour.reset_sampler import ResetDatasetSamplerCfg, _ResetDatasetSampler


def _ring_values(sampler: _ResetDatasetSampler, row: int) -> list[bool]:
    """Read one sampler ring in oldest-to-newest order."""
    size = int(sampler._history_sizes[row])
    pointer = int(sampler._history_pointers[row])
    capacity = sampler._success_history.shape[1]
    start = pointer if size == capacity else 0
    return [bool(sampler._success_history[row, (start + index) % capacity]) for index in range(size)]


def test_rolling_history_matches_per_row_deques():
    """Duplicate IDs, wraparound, and oversized batches match sequential deque updates."""
    capacity = 5
    row_count = 4
    sampler = _ResetDatasetSampler(
        row_count,
        "cpu",
        ResetDatasetSamplerCfg(monitored_history_len=capacity),
    )
    reference = [deque(maxlen=capacity) for _ in range(row_count)]
    generator = torch.Generator().manual_seed(31)

    for batch_size in (4, 11, 2, 17, 7, 23):
        rows = torch.randint(row_count, (batch_size,), generator=generator)
        success = torch.rand(batch_size, generator=generator) > 0.45
        sampler._record_validated(rows, success)
        for row, outcome in zip(rows.tolist(), success.tolist(), strict=True):
            reference[row].append(outcome)

        expected_sizes = torch.tensor([len(history) for history in reference], dtype=torch.int32)
        expected_counts = torch.tensor([sum(history) for history in reference], dtype=torch.int32)
        expected_rates = expected_counts.float() / expected_sizes.clamp_min(1)
        torch.testing.assert_close(sampler._history_sizes, expected_sizes)
        torch.testing.assert_close(sampler._history_success_counts, expected_counts)
        torch.testing.assert_close(sampler._success_rates, expected_rates)
        for row, history in enumerate(reference):
            assert _ring_values(sampler, row) == list(history)


def test_uniform_replay_covers_every_row_before_repeating(monkeypatch: pytest.MonkeyPatch):
    """The configured replay fraction draws one complete shuffled cycle without replacement."""
    sampler = _ResetDatasetSampler(
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
