# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import pytest

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectMARLEnv
from isaaclab.test.env_cfgs import make_empty_direct_marl_env_cfg

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_lifecycle(device):
    """DirectMARLEnv owns the simulation context, republishes renderer scene state on reset and releases it on close."""
    sim_utils.create_new_stage()
    env = DirectMARLEnv(cfg=make_empty_direct_marl_env_cfg(device=device))
    try:
        assert not env._is_closed
        assert sim_utils.SimulationContext.instance() is env.sim
        assert env.num_agents == env.max_num_agents == 2

        # a reset must invalidate the renderer scene-state cadence
        env._get_observations = lambda: {}
        env.sim.render_context._last_scene_state_step = 7
        env.reset()
        assert env.sim.render_context._last_scene_state_step is None
    finally:
        env.close()

    assert env._is_closed
    assert sim_utils.SimulationContext.instance() is None
