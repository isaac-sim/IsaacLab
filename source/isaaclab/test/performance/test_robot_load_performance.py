# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

from __future__ import annotations

from isaaclab.test.utils import launch_test_simulation
from isaaclab.utils import replace

launch_test_simulation()

import pytest

from isaaclab import cloner
from isaaclab.assets import Articulation
from isaaclab.sim import build_simulation_context
from isaaclab.test.utils.devices import DeviceScope, test_devices
from isaaclab.utils.timer import Timer

from isaaclab_assets import ANYMAL_D_CFG, CARTPOLE_CFG

pytestmark = pytest.mark.integration

NUM_ENVS = 4096
SPACING = 2.0


# PhysX CPU and GPU load through distinct pipelines, so each asset is timed on both.
@pytest.mark.parametrize("device", test_devices(DeviceScope.CPU_AND_DEFAULT_CUDA))
@pytest.mark.parametrize(
    "test_config",
    [
        # TODO: regression - this used to be 10
        {"name": "Cartpole", "robot_cfg": CARTPOLE_CFG, "expected_load_time": 15.0},
        # TODO: regression - this used to be 40
        {"name": "Anymal_D", "robot_cfg": ANYMAL_D_CFG, "expected_load_time": 60.0},
    ],
)
def test_robot_load_performance(test_config, device):
    """Test robot load time."""
    with build_simulation_context(device=device) as sim:
        sim._app_control_on_stop_handle = None

        with Timer(f"{test_config['name']} load time for device {device}") as timer:
            cfg = replace(test_config["robot_cfg"], prim_path="{ENV_REGEX_NS}/Robot")
            plan = cloner.clone_plan_from_env_0(cloner.CloneCfg(), (cfg,), NUM_ENVS, SPACING)
            robot = Articulation(cfg)
            cloner.replicate(plan)
            sim.reset()
            elapsed_time = timer.time_elapsed
        assert robot.num_instances == NUM_ENVS
        assert elapsed_time <= test_config["expected_load_time"]
