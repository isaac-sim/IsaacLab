# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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

import isaaclab_tasks  # noqa: F401

# Local imports should be imported last
from env_test_utils import _run_environments, setup_environment  # isort: skip


@pytest.mark.parametrize("num_envs, device", [(2, "cuda"), (1, "cuda")])
@pytest.mark.parametrize(
    "task_name",
    setup_environment(
        include_play=False,
        multi_agent=False,
        newton_mjwarp_envs=True,
        tier="core",
    ),
)
@pytest.mark.newton_ci
def test_environments_newton(task_name, num_envs, device):
    # run environments with MJWarp physics preset
    _run_environments(task_name, device, num_envs, physics_preset_name="newton_mjwarp", create_stage_in_memory=False)
