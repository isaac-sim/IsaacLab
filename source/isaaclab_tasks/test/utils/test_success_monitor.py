# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the shared success-rate monitor."""

import torch

from isaaclab_tasks.utils.success_monitor import SuccessMonitor, SuccessMonitorCfg


def test_success_monitor_resumes_rolling_outcomes_from_a_snapshot() -> None:
    """Restoring a snapshot preserves rates and the next ring-buffer update."""
    cfg = SuccessMonitorCfg(monitored_history_len=3)
    monitor = SuccessMonitor(cfg, num_partitions=1, partition_size=3, device="cpu")
    monitor.success_update(
        torch.tensor([0, 0, 1, 0]),
        torch.tensor([True, False, True, True]),
    )

    snapshot = monitor.get_state()
    monitor.success_update(torch.tensor([0]), torch.tensor([False]))

    restored = SuccessMonitor(cfg, num_partitions=1, partition_size=3, device="cpu")
    restored.set_state(snapshot)
    torch.testing.assert_close(restored.get_success_rate(), torch.tensor([2.0 / 3.0, 1.0, 0.0]))
    restored.success_update(torch.tensor([0, 1]), torch.tensor([False, False]))
    torch.testing.assert_close(restored.get_success_rate(), torch.tensor([1.0 / 3.0, 0.5, 0.0]))
