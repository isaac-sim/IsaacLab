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
def test_initialization_and_close(device):
    """DirectMARLEnv initializes its spaces and releases its simulation context."""
    sim_utils.create_new_stage()
    env = DirectMARLEnv(cfg=make_empty_direct_marl_env_cfg(device=device))

    try:
        assert not env._is_closed
        assert sim_utils.SimulationContext.instance() is env.sim
        assert env.num_agents == 2
        assert env.max_num_agents == 2
        assert len(env.observation_spaces) == 2
        assert len(env.action_spaces) == 2
        assert env.state_space.shape == (7,)
    finally:
        env.close()

    assert env._is_closed
    assert sim_utils.SimulationContext.instance() is None


def test_reset_invalidates_renderer_scene_state_cadence():
    """A same-step multi-agent reset must republish renderer scene state."""
    env = None
    try:
        sim_utils.create_new_stage()
        env = DirectMARLEnv(cfg=make_empty_direct_marl_env_cfg())
        env._get_observations = lambda: {}
        env.sim.render_context._last_scene_state_step = 7

        env.reset()

        assert env.sim.render_context._last_scene_state_step is None
    finally:
        if env is not None:
            env.close()
