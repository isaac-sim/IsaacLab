# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

from __future__ import annotations

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

import pytest

from isaacsim.core.cloner import GridCloner

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import build_simulation_context
from isaaclab.utils.timer import Timer

from isaaclab_assets import ANYMAL_D_CFG, CARTPOLE_CFG


@pytest.mark.parametrize(
    "test_config,device",
    [
        ({"name": "Cartpole", "robot_cfg": CARTPOLE_CFG, "expected_load_time": 10.0}, "cuda:0"),
        ({"name": "Cartpole", "robot_cfg": CARTPOLE_CFG, "expected_load_time": 10.0}, "cpu"),
        # TODO: regression - this used to be 40
        ({"name": "Anymal_D", "robot_cfg": ANYMAL_D_CFG, "expected_load_time": 50.0}, "cuda:0"),
        ({"name": "Anymal_D", "robot_cfg": ANYMAL_D_CFG, "expected_load_time": 50.0}, "cpu"),
    ],
)
def test_robot_load_performance(test_config, device):
    """Test robot load time."""
    with build_simulation_context(device=device) as sim:
        sim._app_control_on_stop_handle = None
        cloner = GridCloner(spacing=2, stage=sim.stage)
        target_paths = cloner.generate_paths("/World/Robots", 4096)
        sim_utils.get_current_stage().DefinePrim(target_paths[0], "Xform")
        _ = cloner.clone(
            source_prim_path=target_paths[0],
            prim_paths=target_paths,
            replicate_physics=False,
            copy_from_source=True,
        )
        with Timer(f"{test_config['name']} load time for device {device}") as timer:
            robot = Articulation(test_config["robot_cfg"].replace(prim_path="/World/Robots_.*/Robot"))  # noqa: F841
            sim.reset()
            elapsed_time = timer.time_elapsed
        assert elapsed_time <= test_config["expected_load_time"]
