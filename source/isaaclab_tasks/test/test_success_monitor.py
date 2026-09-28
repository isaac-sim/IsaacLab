# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Success monitors keep each slot's latest outcomes without synchronizing the device."""

from collections import deque

import pytest
import torch

from isaaclab.test.utils import DeviceScope, test_devices

from isaaclab_tasks.utils.success_monitor import SuccessMonitor, SuccessMonitorCfg

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("device", test_devices())
def test_updates_match_per_slot_history(device):
    """Duplicate slots, overflow beyond the history, and masked outcomes match a per-slot deque."""
    history, num_slots = 4, 6
    monitor = SuccessMonitor(SuccessMonitorCfg(monitored_history_len=history), 2, num_slots // 2, device)
    expected = [deque(maxlen=history) for _ in range(num_slots)]
    generator = torch.Generator().manual_seed(0)
    for _ in range(5):
        # repeat slots within a batch so some exceed the history length
        slot_ids = torch.randint(0, num_slots, (15,), generator=generator)
        success = torch.rand(15, generator=generator) < 0.5
        valid = torch.rand(15, generator=generator) < 0.8
        for slot, outcome, keep in zip(slot_ids.tolist(), success.tolist(), valid.tolist()):
            if keep:
                expected[slot].append(float(outcome))
        monitor.success_update(slot_ids.to(device), success.to(device), valid=valid.to(device))

        sizes = torch.tensor([len(outcomes) for outcomes in expected], dtype=torch.long)
        rates = torch.tensor([sum(outcomes) / max(len(outcomes), 1) for outcomes in expected])
        torch.testing.assert_close(monitor.success_size.cpu(), sizes)
        torch.testing.assert_close(monitor.success_rate.cpu(), rates)
        measured = sizes > 0
        torch.testing.assert_close(monitor.get_mean_success_rate().cpu(), rates[measured].mean())


def test_mean_success_rate_is_zero_before_outcomes():
    monitor = SuccessMonitor(SuccessMonitorCfg(), 1, 3, "cpu")
    assert monitor.get_mean_success_rate().item() == 0.0


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_update_and_mean_rate_do_not_synchronize(device):
    monitor = SuccessMonitor(SuccessMonitorCfg(monitored_history_len=3), 2, 4, device)
    slot_ids = torch.tensor([1, 5, 1, 1, 1, -1], device=device)
    success = torch.tensor([True, False, True, False, True, True], device=device)
    valid = slot_ids >= 0
    torch.cuda.synchronize(device)
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        monitor.success_update(slot_ids, success, valid=valid)
        mean = monitor.get_mean_success_rate()
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    # slot 1 keeps its latest three outcomes (True, False, True); slot 5 records one failure
    torch.testing.assert_close(monitor.success_rate[[1, 5]].cpu(), torch.tensor([2 / 3, 0.0]))
    torch.testing.assert_close(mean.cpu(), torch.tensor(1 / 3))
