# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Outcome attribution and selective resets at the shared sampler boundary."""

from collections import deque

import pytest
import torch

from isaaclab_tasks.utils import ResetStateSampler, ResetStateSamplerCfg, SuccessMonitor, SuccessMonitorCfg

pytestmark = [pytest.mark.unit, pytest.mark.kitless, pytest.mark.windows_ci]


def test_rolling_history_matches_per_row_deques():
    """Duplicate IDs, wraparound, and oversized batches retain the latest outcomes in input order."""
    capacity = 5
    row_count = 4
    monitor = SuccessMonitor(SuccessMonitorCfg(monitored_history_len=capacity), 1, row_count, "cpu")
    reference = [deque(maxlen=capacity) for _ in range(row_count)]
    generator = torch.Generator().manual_seed(31)

    for batch_size in (4, 11, 2, 17, 7, 23):
        rows = torch.randint(row_count, (batch_size,), generator=generator)
        success = torch.rand(batch_size, generator=generator) > 0.45
        monitor.success_update(rows, success)
        for row, outcome in zip(rows.tolist(), success.tolist(), strict=True):
            reference[row].append(outcome)

        expected_rates = torch.tensor([sum(history) / max(1, len(history)) for history in reference])
        torch.testing.assert_close(monitor.get_success_rate(), expected_rates)


def test_reset_records_previous_rows_and_preserves_unselected_environments() -> None:
    """Startup masks and partial resets never credit a replacement row or an untouched episode."""
    sampler = ResetStateSampler(
        ResetStateSamplerCfg(success_monitor=SuccessMonitorCfg(monitored_history_len=2)), 4, 3, "cpu"
    )
    sampler.row_ids[:] = torch.tensor([-1, -1, -1, 1])
    selected = torch.tensor([2, 0, 1])
    # A zero-step startup has no outcome to record, even when success flags contain old values.
    rows = sampler.reset(selected, torch.ones(3, dtype=torch.bool), torch.zeros(3, dtype=torch.bool))
    torch.testing.assert_close(sampler.row_ids[selected], rows)
    assert ((rows >= 0) & (rows < 3)).all()
    assert sampler.row_ids[3] == 1
    assert sampler.success_monitor.success_size.tolist() == [0, 0, 0]
    sampler.row_ids[selected] = torch.tensor([1, 0, 0])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        sampler.reset(selected, torch.tensor([False, True, False]))
    assert sampler.row_ids[3] == 1
    assert sampler.success_monitor.success_size.tolist() == [2, 1, 0]
    assert sampler.success_monitor.success_rate.tolist() == [0.5, 0.0, 0.0]
    # Record only one completed episode on the next reset; the masked one contributes nothing.
    sampler.row_ids[torch.tensor([0, 2])] = 2
    untouched = sampler.row_ids[torch.tensor([1, 3])].clone()
    sampler.reset(
        torch.tensor([0, 2]),
        torch.tensor([True, False]),
        valid=torch.tensor([True, False]),
    )
    assert sampler.success_monitor.success_size.tolist() == [2, 1, 1]
    assert sampler.success_monitor.success_rate.tolist() == [0.5, 0.0, 1.0]
    torch.testing.assert_close(sampler.row_ids[torch.tensor([1, 3])], untouched)
    assigned = sampler.row_ids.clone()
    sampler.reset(torch.empty(0, dtype=torch.long))
    assert sampler.success_monitor.success_size.tolist() == [2, 1, 1]
    torch.testing.assert_close(sampler.row_ids, assigned)


def test_sampling_tracks_the_current_success_frontier_without_starving_other_rows() -> None:
    """The sampler observes live rolling outcomes and retains positive weight for easy and hard starts."""
    with pytest.raises(ValueError, match="through sampling"):
        ResetStateSampler(ResetStateSamplerCfg(success_monitor=SuccessMonitorCfg(target_success_rate=0.8)), 3, 3, "cpu")
    sampler = ResetStateSampler(ResetStateSamplerCfg(), 3, 3, "cpu")
    env_ids = torch.arange(3)
    for success in ([False, True, False], [False, True, True]):
        sampler.row_ids[:] = env_ids
        sampler.reset(env_ids, torch.tensor(success))
    assert sampler.success_monitor.success_rate.tolist() == [0.0, 1.0, 0.5]
    exported = sampler.probabilities
    exported.zero_()
    probabilities = sampler.probabilities
    assert (probabilities > 0).all()
    torch.testing.assert_close(probabilities.sum(), torch.tensor(1.0))
    assert probabilities[2] > probabilities[0]
    assert probabilities[2] > probabilities[1]
