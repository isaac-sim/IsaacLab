# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Value parity between the Direct and manager-based reorientation configurations.

Covers the values that define the task rather than how it is solved: timing, the
success tolerance, and the termination thresholds. Reward weights are deliberately
excluded, since each workflow tunes them against its own RL agent configuration.

This module is the check that the configuration comments refer to, so drift on
either side fails here rather than silently changing what a manager task trains on.
"""

import pytest

from isaaclab_tasks.core.reorient.config.allegro_hand.allegro_hand_direct_env_cfg import AllegroHandEnvCfg
from isaaclab_tasks.core.reorient.config.allegro_hand.allegro_hand_manager_env_cfg import AllegroHandManagerEnvCfg
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_direct_env_cfg import ShadowHandEnvCfg
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_manager_env_cfg import ShadowHandManagerEnvCfg


@pytest.mark.parametrize(
    "direct_cls, manager_cls",
    [
        pytest.param(AllegroHandEnvCfg, AllegroHandManagerEnvCfg, id="allegro"),
        pytest.param(ShadowHandEnvCfg, ShadowHandManagerEnvCfg, id="shadow"),
    ],
)
def test_manager_config_matches_direct_values(direct_cls, manager_cls):
    direct, manager = direct_cls(), manager_cls()

    assert (manager.decimation, manager.episode_length_s, manager.sim.dt) == (
        direct.decimation,
        direct.episode_length_s,
        direct.sim.dt,
    )
    assert manager.commands.object_pose.orientation_success_threshold == pytest.approx(direct.success_tolerance)
    assert manager.terminations.object_out_of_reach.params["threshold"] == pytest.approx(direct.fall_dist)
    # The Direct tasks fold the streak cap into their time-out signal.
    streak_cap = getattr(manager.terminations, "max_consecutive_success", None)
    assert (0 if streak_cap is None else streak_cap.params["num_success"]) == direct.max_consecutive_success
    assert streak_cap is None or streak_cap.time_out
