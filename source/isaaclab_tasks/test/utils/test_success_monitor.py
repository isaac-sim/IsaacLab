# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the shared success-rate monitor."""

import pytest
import torch

from isaaclab_tasks.utils.success_monitor import SuccessMonitor, SuccessMonitorCfg


def test_success_monitor_state_round_trip() -> None:
    """Rolling histories can be restored without reaching into monitor internals."""
    cfg = SuccessMonitorCfg(monitored_history_len=3)
    monitor = SuccessMonitor(cfg, num_partitions=1, partition_size=3, device="cpu")
    monitor.success_update(
        torch.tensor([0, 0, 1, 0]),
        torch.tensor([True, False, True, True]),
    )

    restored = SuccessMonitor(cfg, num_partitions=1, partition_size=3, device="cpu")
    restored.set_state(monitor.get_state())

    torch.testing.assert_close(restored.success_buf, monitor.success_buf)
    torch.testing.assert_close(restored.success_pointer, monitor.success_pointer)
    torch.testing.assert_close(restored.success_size, monitor.success_size)
    torch.testing.assert_close(restored.success_rate, monitor.success_rate)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    (
        ("success_history", torch.zeros(2, 3), "expected"),
        ("history_pointer", torch.tensor([0, 1, 3]), "pointer"),
        ("history_size", torch.tensor([0, 1, 4]), "size"),
    ),
)
def test_success_monitor_rejects_invalid_state(key: str, value: torch.Tensor, message: str) -> None:
    """Checkpoint restoration validates every tensor before mutating state."""
    monitor = SuccessMonitor(SuccessMonitorCfg(monitored_history_len=3), 1, 3, "cpu")
    state = monitor.get_state()
    state[key] = value

    with pytest.raises(ValueError, match=message):
        monitor.set_state(state)


def test_success_monitor_rejects_missing_state() -> None:
    """Checkpoint restoration reports a missing required tensor."""
    monitor = SuccessMonitor(SuccessMonitorCfg(), 1, 1, "cpu")
    state = monitor.get_state()
    del state["history_size"]

    with pytest.raises(KeyError, match="history_size"):
        monitor.set_state(state)
