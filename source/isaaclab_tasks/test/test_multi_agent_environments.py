# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch the simulator
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import pytest
from env_test_utils import _check_random_actions, setup_environment

import isaaclab_tasks  # noqa: F401


@pytest.mark.parametrize("num_envs, device", [(32, "cuda"), (1, "cuda")])
@pytest.mark.parametrize("task_name", setup_environment(multi_agent=True))
def test_environments(task_name, num_envs, device):
    """Run all environments with given parameters and check environments return valid signals."""
    print(f">>> Running test for environment: {task_name} with num_envs={num_envs} and device={device}")
    # check environment
    _check_random_actions(task_name, device, num_envs, multi_agent=True)
    # close the environment
    print(f">>> Closing environment: {task_name}")
    print("-" * 80)
