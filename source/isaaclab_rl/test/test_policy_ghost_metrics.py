# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Episode statistics of each Policy Ghost slot."""

import pytest
import torch

from isaaclab_rl.entrypoints.policy_ghost_metrics import PolicyGhostMetricsTracker

pytestmark = pytest.mark.unit


def _tracker(pool=8):
    tracker = PolicyGhostMetricsTracker(slice_size=2, num_slots=3, num_envs=pool, step_dt=0.5, device="cpu")
    for slot in range(3):
        tracker.set_active(slot, True)
    return tracker


def _step(tracker, reward, done):
    tracker.update(torch.tensor(reward, dtype=torch.float32), torch.tensor(done, dtype=torch.bool))


def test_a_terminated_episode_is_counted_once_even_while_the_slot_waits_for_the_reference():
    tracker = _tracker()
    _step(tracker, [1, 1, 2, 2, 3, 3, 0, 0], [0] * 8)
    _step(tracker, [1, 1, 2, 2, 3, 3, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0])  # The ghost of env 0 falls.
    _step(
        tracker, [1, 1, 2, 2, 3, 3, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0]
    )  # It stays terminated until the reference resets.

    ghost = tracker.metrics(1)
    assert ghost.episodes == 1 and ghost.mean_return == pytest.approx(4.0) and ghost.mean_length == pytest.approx(1.0)
    assert tracker.metrics(0) is None


def test_a_reference_reset_ends_a_ghost_that_is_still_running_and_restarts_every_counterpart():
    tracker = _tracker()
    _step(tracker, [1, 1, 2, 2, 3, 3, 9, 9], [0] * 8)
    _step(tracker, [1, 1, 2, 2, 3, 3, 9, 9], [1, 0, 0, 0, 0, 0, 0, 0])  # The reference's env 0 times out.

    assert tracker.metrics(0).mean_return == pytest.approx(2.0)
    assert tracker.metrics(1).mean_return == pytest.approx(4.0) and tracker.metrics(2).mean_return == pytest.approx(6.0)
    _step(tracker, [1, 1, 2, 2, 3, 3, 9, 9], [0] * 8)
    _step(tracker, [1, 1, 2, 2, 3, 3, 9, 9], [1, 0, 0, 0, 0, 0, 0, 0])
    assert tracker.metrics(0).episodes == 2 and tracker.metrics(0).mean_return == pytest.approx(2.0)


def test_loading_or_removing_a_checkpoint_clears_the_slots_statistics():
    tracker = _tracker()
    _step(tracker, [1] * 8, [1] * 8)
    assert tracker.metrics(1) is not None

    tracker.set_active(1, False)

    assert tracker.metrics(1) is None and tracker.metrics(0) is not None
