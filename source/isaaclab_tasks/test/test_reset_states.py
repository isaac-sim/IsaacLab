# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Outcome attribution and selective resets at the shared sampler boundary."""

import torch

from isaaclab_tasks.utils import ResetStateSampler, ResetStateSamplerCfg, SuccessMonitorCfg


def test_reset_records_previous_rows_and_preserves_unselected_environments():
    """Startup masks and partial resets never credit a replacement row or an untouched episode."""
    sampler = ResetStateSampler(
        ResetStateSamplerCfg(success_monitor=SuccessMonitorCfg(monitored_history_len=2)), 4, 3, "cpu"
    )
    sampler.row_ids[:] = torch.tensor([0, 0, 1, 1])
    selected = torch.tensor([2, 0, 1])
    only_row_two = torch.tensor([False, False, True])
    # A zero-step startup has no outcome to record, even when success flags contain old values.
    rows = sampler.reset(selected, torch.ones(3, dtype=torch.bool), torch.zeros(3, dtype=torch.bool), only_row_two)
    assert rows.tolist() == [2, 2, 2]
    assert sampler.success_monitor.success_size.tolist() == [0, 0, 0]
    sampler.row_ids[selected] = torch.tensor([1, 0, 0])
    sampler.reset(selected, torch.tensor([False, True, False]), row_mask=only_row_two)
    assert sampler.row_ids.tolist() == [2, 2, 2, 1]
    assert sampler.success_monitor.success_size.tolist() == [2, 1, 0]
    assert sampler.success_monitor.success_rate.tolist() == [0.5, 0.0, 0.0]
    # Record only one completed episode on the next reset; the masked one contributes nothing.
    sampler.reset(
        torch.tensor([0, 2]),
        torch.tensor([True, False]),
        valid=torch.tensor([True, False]),
        row_mask=torch.tensor([False, True, False]),
    )
    assert sampler.success_monitor.success_size.tolist() == [2, 1, 1]
    assert sampler.success_monitor.success_rate.tolist() == [0.5, 0.0, 1.0]
    assert sampler.row_ids.tolist() == [1, 2, 1, 1]
    sampler.reset(torch.empty(0, dtype=torch.long))
    assert sampler.success_monitor.success_size.tolist() == [2, 1, 1]
    assert sampler.row_ids.tolist() == [1, 2, 1, 1]


def test_sampling_tracks_the_current_success_frontier_without_starving_other_rows():
    """The sampler observes live rolling outcomes and retains positive weight for easy and hard starts."""
    sampler = ResetStateSampler(ResetStateSamplerCfg(), 3, 3, "cpu")
    env_ids = torch.arange(3)
    for success in ([False, True, False], [False, True, True]):
        sampler.row_ids[:] = env_ids
        sampler.reset(env_ids, torch.tensor(success))
    assert sampler.success_monitor.success_rate.tolist() == [0.0, 1.0, 0.5]
    assert (sampler.probabilities > 0).all()
    torch.testing.assert_close(sampler.probabilities.sum(), torch.tensor(1.0))
    assert sampler.probabilities[2] > sampler.probabilities[0]
    assert sampler.probabilities[2] > sampler.probabilities[1]
