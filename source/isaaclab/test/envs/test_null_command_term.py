# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

from collections import namedtuple

import pytest

from isaaclab.envs.mdp import NullCommandCfg


@pytest.fixture
def env():
    """Create a dummy environment."""
    return namedtuple("ManagerBasedRLEnv", ["num_envs", "dt", "device"])(20, 0.1, "cpu")


def test_str(env):
    """Test the string representation of the command manager."""
    cfg = NullCommandCfg()
    command_term = cfg.class_type(cfg, env)
    # print the expected string
    print()
    print(command_term)


def test_compute(env):
    """Test the compute function. For null command generator, it does nothing."""
    cfg = NullCommandCfg()
    command_term = cfg.class_type(cfg, env)

    # test the reset function
    command_term.reset()
    # test the compute function
    command_term.compute(dt=env.dt)
    # expect error
    with pytest.raises(RuntimeError):
        command_term.command
