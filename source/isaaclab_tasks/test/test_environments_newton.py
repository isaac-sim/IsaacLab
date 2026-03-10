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
        factory_envs=False,
        multi_agent=False,
        teleop_envs=False,
        cartpole_showcase_envs=False,
        pickplace_stack_envs=False,
        newton_envs=True,
    ),
)
@pytest.mark.newton_ci
def test_environments_newton(task_name, num_envs, device):
    # run environments with newton physics preset
    _run_environments(task_name, device, num_envs, physics_preset_name="newton", create_stage_in_memory=False)
