# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the shared success-rate monitor."""

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
