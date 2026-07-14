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
def test_initialization(device):
    """Test initialization of DirectMARLEnv."""
    # create a new stage
    sim_utils.create_new_stage()
    try:
        # create environment
        env = DirectMARLEnv(cfg=make_empty_direct_marl_env_cfg(device=device))
    except Exception as e:
        if "env" in locals() and hasattr(env, "_is_closed"):
            env.close()
        else:
            if hasattr(e, "obj") and hasattr(e.obj, "_is_closed"):
                e.obj.close()
        pytest.fail(f"Failed to set-up the DirectMARLEnv environment. Error: {e}")

    # check multi-agent config
    assert env.num_agents == 2
    assert env.max_num_agents == 2
    # check spaces
    assert env.state_space.shape == (7,)
    assert len(env.observation_spaces) == 2
    assert len(env.action_spaces) == 2
    # close the environment
    env.close()
