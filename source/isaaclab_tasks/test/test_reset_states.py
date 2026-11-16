# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Outcome attribution and selective resets at the shared sampler boundary."""

import pytest
import torch

from isaaclab_tasks.utils import ResetStateSampler, ResetStateSamplerCfg
from isaaclab_tasks.utils.sampling import TargetSuccessScoreCfg, UniformScoreCfg

pytestmark = [pytest.mark.unit, pytest.mark.kitless, pytest.mark.windows_ci]


def test_reset_records_previous_rows_and_preserves_unselected_environments() -> None:
    """Opaque IDs, startup masks, and partial resets preserve previous-outcome attribution."""
    catalog_ids = torch.tensor([42, 7, 101])
    sampler = ResetStateSampler(ResetStateSamplerCfg(monitored_history_len=2), 4, 3, "cpu", state_ids=catalog_ids)
    all_envs = torch.arange(4)
    sampler.reset(all_envs, state_ids=torch.tensor([42, 42, 42, 7]))
    selected = torch.tensor([2, 0, 1])
    # A zero-step startup has no outcome to record, even when success flags contain old values.
    ids = sampler.reset(
        selected,
        torch.ones(3, dtype=torch.bool),
        torch.zeros(3, dtype=torch.bool),
        state_ids=torch.tensor([7, 42, 42]),
    )
    torch.testing.assert_close(sampler.get_state_ids(selected), ids)
    assert sampler.get_state_ids(torch.tensor([3])).item() == 7
    assert sampler.success_monitor.success_size.tolist() == [0, 0, 0]
    with pytest.raises(ValueError, match="not in the catalog"):
        sampler.reset(torch.tensor([0]), torch.tensor([True]), state_ids=torch.tensor([999]))
    assert sampler.success_monitor.success_size.tolist() == [0, 0, 0]
    sampler.reset(selected, torch.tensor([False, True, False]), state_ids=torch.tensor([101, 101, 101]))
    assert sampler.get_state_ids(torch.tensor([3])).item() == 7
    assert sampler.success_monitor.success_size.tolist() == [2, 1, 0]
    assert sampler.success_monitor.success_rate.tolist() == [0.5, 0.0, 0.0]
    # Record only one completed episode on the next reset; the masked one contributes nothing.
    untouched = sampler.get_state_ids(torch.tensor([1, 3]))
    sampler.reset(
        torch.tensor([0, 2]),
        torch.tensor([True, False]),
        valid=torch.tensor([True, False]),
        state_ids=torch.tensor([42, 42]),
    )
    assert sampler.success_monitor.success_size.tolist() == [2, 1, 1]
    assert sampler.success_monitor.success_rate.tolist() == [0.5, 0.0, 1.0]
    torch.testing.assert_close(sampler.get_state_ids(torch.tensor([1, 3])), untouched)
    assigned = sampler.get_state_ids(all_envs)
    sampler.reset(torch.empty(0, dtype=torch.long))
    assert sampler.success_monitor.success_size.tolist() == [2, 1, 1]
    torch.testing.assert_close(sampler.get_state_ids(all_envs), assigned)
    exported = sampler.get_state_ids(all_envs)
    exported.zero_()
    catalog_ids.zero_()
    torch.testing.assert_close(sampler.get_state_ids(all_envs), assigned)
    with pytest.raises(ValueError, match="unique"):
        ResetStateSampler(ResetStateSamplerCfg(), 1, 2, "cpu", state_ids=torch.tensor([7, 7]))


def test_sampling_tracks_the_current_success_frontier_without_starving_other_rows() -> None:
    """The sampler observes live rolling outcomes and retains positive weight for easy and hard starts."""
    sampler = ResetStateSampler(ResetStateSamplerCfg(), 5, 5, "cpu")
    env_ids = torch.arange(5)
    for success in (
        [False, True, True, True, True],
        [False, False, True, True, True],
        [False, False, False, True, True],
        [False, False, False, False, True],
    ):
        sampler.reset(env_ids, state_ids=env_ids)
        sampler.reset(env_ids, torch.tensor(success))
    assert sampler.success_monitor.success_rate.tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]
    exported = sampler.probabilities
    exported.zero_()
    probabilities = sampler.probabilities
    assert (probabilities > 0).all()
    torch.testing.assert_close(probabilities.sum(), torch.tensor(1.0))
    assert (probabilities[2] > probabilities[[0, 1, 3, 4]]).all()
    torch.testing.assert_close(
        probabilities, torch.tensor([0.05722444, 0.27138778, 0.34277556, 0.27138778, 0.05722444])
    )


def test_reset_scores_follow_additional_live_metrics() -> None:
    """A caller's score tensor can drive selection independently of measured episode successes."""
    readiness = torch.tensor([0.0, 0.5, 1.0])
    cfg = ResetStateSamplerCfg()
    cfg.sampling.strategies = [TargetSuccessScoreCfg(metric="readiness"), UniformScoreCfg(weight=0.05)]
    sampler = ResetStateSampler(cfg, 2, 3, "cpu", metrics={"readiness": readiness})
    assert sampler.probabilities.argmax().item() == 1
    readiness.copy_(torch.tensor([0.5, 0.0, 1.0]))
    sampler.reset(torch.empty(0, dtype=torch.long))
    assert sampler.probabilities.argmax().item() == 0
    assert sampler.success_monitor.success_size.tolist() == [0, 0, 0]
