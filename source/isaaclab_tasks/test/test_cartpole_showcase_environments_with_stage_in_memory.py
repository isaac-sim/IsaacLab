# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch the simulator
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

from isaaclab.utils.version import get_isaac_sim_version

"""Rest everything follows."""

import pytest

import isaaclab_tasks  # noqa: F401

# Local imports should be imported last
from env_test_utils import _run_environments, setup_environment  # isort: skip


@pytest.mark.parametrize("num_envs, device", [(2, "cuda")])
@pytest.mark.parametrize(
    "task_name",
    setup_environment(
        include_play=False, factory_envs=False, multi_agent=False, teleop_envs=False, cartpole_showcase_envs=True
    ),
)
def test_cartpole_showcase_environments_with_stage_in_memory_and_clone_in_fabric_disabled(task_name, num_envs, device):
    # skip test if stage in memory is not supported
    if get_isaac_sim_version().major < 5:
        pytest.skip("Stage in memory is not supported in this version of Isaac Sim")

    # run cartpole showcase environments with stage in memory
    _run_environments(task_name, device, num_envs, create_stage_in_memory=True, disable_clone_in_fabric=True)
