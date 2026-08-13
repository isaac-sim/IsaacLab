# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch the simulator
app_launcher = AppLauncher(headless=True, enable_cameras=True, limit_cpu_threads=1)
simulation_app = app_launcher.app


"""Rest everything follows."""

import pytest

import isaaclab_tasks  # noqa: F401

# Local imports should be imported last
from env_test_utils import _run_environments, setup_environment  # isort: skip

@pytest.mark.parametrize("physics_preset_name", ["newton_mjwarp", "physx", "isaacsim_physx"])

@pytest.mark.parametrize("num_envs, device", [(2, "cuda"), (1, "cuda")])
@pytest.mark.parametrize(
    "task_name",
    setup_environment(
        multi_agent=False,
        tier="core",
    ),
)
@pytest.mark.isaacsim_ci
def test_environments(task_name, physics_preset_name, num_envs, device):
    # run environments without stage in memory
    _run_environments(
        task_name, device, num_envs, create_stage_in_memory=False, physics_preset_name=physics_preset_name
    )


@pytest.mark.parametrize("num_envs, device", [(2, "cuda")])
@pytest.mark.parametrize(
    "task_name",
    setup_environment(
        multi_agent=False,
        factory_envs=False,
        cartpole_showcase_envs=False,
        pickplace_stack_envs=False,
        teleop_envs=False,
        tier="contrib",
    ),
)
@pytest.mark.isaacsim_ci
def test_contrib_environments(task_name, num_envs, device):
