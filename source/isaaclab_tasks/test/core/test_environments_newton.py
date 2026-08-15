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
from env_test_utils import SINGLE_ENVIRONMENT_TASKS, _run_environments, setup_environment  # isort: skip


_COVERED_TASKS = [
    "Isaac-Cartpole",  # Already covered by test_environment_determinism.py
    "Isaac-Cartpole-Camera-Direct",  # Already covered by test_rendering_cartpole.py
    "Isaac-Lift-KukaAllegro-Camera",  # Already covered by test_rendering_lift_kuka_homo_kitless.py
    "Isaac-Reorient-Cube-Shadow-Camera-Direct",  # Already covered by test_rendering_shadow_hand_kitless.py
]

_ENVIRONMENT_TASKS = setup_environment(
    multi_agent=False,
    physics_preset_name="newton_mjwarp",
    tier="core",
    exclude_task_names=_COVERED_TASKS,
)


@pytest.mark.parametrize(
    "task_name",
    _ENVIRONMENT_TASKS,
)
@pytest.mark.parametrize("num_envs, device", [(2, "cuda")])
@pytest.mark.isaacsim_ci
def test_environments_newton(task_name, num_envs, device):
    _run_environments(task_name, device, num_envs, physics_preset_name="newton_mjwarp")


@pytest.mark.parametrize("task_name", [task for task in _ENVIRONMENT_TASKS if task in SINGLE_ENVIRONMENT_TASKS])
@pytest.mark.isaacsim_ci
def test_single_environment_newton(task_name):
    _run_environments(task_name, "cuda", 1, physics_preset_name="newton_mjwarp")
