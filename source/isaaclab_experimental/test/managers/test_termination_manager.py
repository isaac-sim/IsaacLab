# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Warp-first termination manager."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import warp as wp
from isaaclab_experimental.managers.termination_manager import TerminationManager

from isaaclab.utils.warp import WarpLaunchCache


class TestTerminationManager:
    """Tests for :class:`TerminationManager`."""

    def test_reset_statistics_only_include_selected_environments(self):
        """Partial reset statistics should be averaged over selected environments only."""
        manager = TerminationManager.__new__(TerminationManager)
        manager._env = SimpleNamespace(num_envs=4, device="cpu")
        manager._env._warp_launch = WarpLaunchCache(device=manager._env.device)
        manager._term_names = ["fall", "timeout"]
        manager._last_episode_dones_wp = wp.array(
            [[True, False], [False, True], [True, True], [False, False]], dtype=wp.bool, device="cpu"
        )
        manager._term_done_avg_wp = wp.zeros(2, dtype=wp.float32, device="cpu")
        manager._reset_count_wp = wp.zeros(1, dtype=wp.int32, device="cpu")
        manager._reset_scale_wp = wp.zeros(1, dtype=wp.float32, device="cpu")
        manager._class_term_cfgs = []
        manager._reset_extras = {}
        env_mask = wp.array([True, False, True, False], dtype=wp.bool, device="cpu")

        manager.reset(env_mask=env_mask)

        np.testing.assert_allclose(manager._term_done_avg_wp.numpy(), [1.0, 0.5])
