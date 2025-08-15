# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

import sys

# Import pinocchio in the main script to force the use of the dependencies installed by IsaacLab and not the one installed by Isaac Sim
# pinocchio is required by the Pink IK controller
if sys.platform != "win32":
    import pinocchio  # noqa: F401

from isaaclab.app import AppLauncher

# launch the simulator
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

from isaacsim.core.version import get_version

"""Rest everything follows."""

import pytest
from env_test_utils import _run_environments, setup_environment

import isaaclab_tasks  # noqa: F401

# note, running an env test without stage in memory then
# running an env test with stage in memory causes IsaacLab to hang.
# so, here we run all envs with stage in memory separately

# TODO(mtrepte): re-enable with fabric cloning fix
# @pytest.mark.parametrize("num_envs, device", [(2, "cuda")])
# @pytest.mark.parametrize("task_name", setup_environment(include_play=False,factory_envs=False, multi_agent=False))
# def test_environments_with_stage_in_memory_and_clone_in_fabric_disabled(task_name, num_envs, device):
# # skip test if stage in memory is not supported
# isaac_sim_version = float(".".join(get_version()[2]))
# if isaac_sim_version < 5:
#     pytest.skip("Stage in memory is not supported in this version of Isaac Sim")

# # run environments with stage in memory
# _run_environments(task_name, device, num_envs, create_stage_in_memory=True)


@pytest.mark.parametrize("num_envs, device", [(2, "cuda")])
@pytest.mark.parametrize("task_name", setup_environment(include_play=False, factory_envs=False, multi_agent=False))
def test_environments_with_stage_in_memory_and_clone_in_fabric_disabled(task_name, num_envs, device):
    # skip test if stage in memory is not supported
    isaac_sim_version = float(".".join(get_version()[2]))
    if isaac_sim_version < 5:
        pytest.skip("Stage in memory is not supported in this version of Isaac Sim")

    # run environments with stage in memory
    _run_environments(task_name, device, num_envs, create_stage_in_memory=True, disable_clone_in_fabric=True)
