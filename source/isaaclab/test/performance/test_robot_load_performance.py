# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

from __future__ import annotations

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

import torch

import pytest

from isaaclab_assets import ANYMAL_D_CFG, CARTPOLE_CFG

from isaaclab.assets import Articulation
from isaaclab.cloner import grid_transforms, usd_replicate
from isaaclab.sim import build_simulation_context
from isaaclab.sim.utils.prims import create_prim
from isaaclab.utils.timer import Timer


@pytest.mark.parametrize(
    "test_config,device",
    [
        ({"name": "Cartpole", "robot_cfg": CARTPOLE_CFG, "expected_load_time": 10.0}, "cuda:0"),
        ({"name": "Anymal_D", "robot_cfg": ANYMAL_D_CFG, "expected_load_time": 40.0}, "cuda:0"),
    ],
)
def test_robot_load_performance_physics_clone(test_config, device):
    """Test robot load time."""

    with build_simulation_context(device=device) as sim:
        sim._app_control_on_stop_handle = None
        num_articulations = 4096

        env_fmt = "/World/Robots_{}"
        create_prim(env_fmt.format(0))
        env_indices = torch.arange(num_articulations, dtype=torch.long, device=device)
        _default_env_origins, _ = grid_transforms(num_articulations, 1.5, device=device)
        usd_replicate(sim.stage, [env_fmt.format(0)], [env_fmt], env_indices, positions=_default_env_origins)

        with Timer(f"{test_config['name']} load time for device {device}") as timer:
            robot = Articulation(test_config["robot_cfg"].replace(prim_path="/World/Robots_.*/Robot"))  # noqa: F841
            sim.reset()
            elapsed_time = timer.time_elapsed
        assert elapsed_time <= test_config["expected_load_time"]
