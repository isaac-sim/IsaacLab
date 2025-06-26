# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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

import omni.usd
import pytest

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass


@configclass
class EmptySceneCfg(InteractiveSceneCfg):
    """Configuration for an empty scene."""

    pass


def get_empty_base_env_cfg(device: str = "cuda:0", num_envs: int = 1, env_spacing: float = 1.0):
    """Generate base environment config based on device"""

    @configclass
    class EmptyEnvCfg(DirectMARLEnvCfg):
        """Configuration for the empty test environment."""

        # Scene settings
        scene: EmptySceneCfg = EmptySceneCfg(num_envs=num_envs, env_spacing=env_spacing)
        # Basic settings
        decimation = 1
        possible_agents = ["agent_0", "agent_1"]
        action_spaces = {"agent_0": 1, "agent_1": 2}
        observation_spaces = {"agent_0": 3, "agent_1": 4}
        state_space = -1
        episode_length_s = 100.0

    return EmptyEnvCfg()


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_initialization(device):
    """Test initialization of DirectMARLEnv."""
    # create a new stage
    omni.usd.get_context().new_stage()
    try:
        # create environment
        env = DirectMARLEnv(cfg=get_empty_base_env_cfg(device=device))
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
