# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke tests for core environments using the optional OvPhysX runtime."""

import pytest

# OvPhysX runs kitless and its runtime wheel is optional. CI jobs that exercise
# this backend install the wheel explicitly.
pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

import isaaclab_tasks  # noqa: E402, F401

# Local imports should be imported last
from env_test_utils import SINGLE_ENVIRONMENT_TASKS, _run_environments, setup_environment  # isort: skip


# Keep this physics smoke suite renderer-free. Matching state tasks cover the dynamics,
# while the dedicated kitless rendering tests cover the camera pipelines.
_COVERED_TASKS = [
    "Isaac-Cartpole-Camera",
    "Isaac-Cartpole-Camera-Direct",
    "Isaac-Lift-KukaAllegro-Camera",
    "Isaac-Reorient-Cube-Shadow-Camera",
    "Isaac-Reorient-Cube-Shadow-Camera-Direct",
    "Isaac-Reorient-KukaAllegro-Camera",
]

_ENVIRONMENT_TASKS = setup_environment(
    multi_agent=False,
    physics_preset_name="ovphysx",
    tier="core",
    exclude_task_names=_COVERED_TASKS,
)


@pytest.mark.parametrize(
    "task_name",
    _ENVIRONMENT_TASKS,
)
@pytest.mark.parametrize("num_envs, device", [(2, "cuda")])
@pytest.mark.isaacsim_ci
def test_environments_ovphysx(task_name, num_envs, device):
    _run_environments(task_name, device, num_envs, physics_preset_name="ovphysx")


@pytest.mark.parametrize("task_name", [task for task in _ENVIRONMENT_TASKS if task in SINGLE_ENVIRONMENT_TASKS])
@pytest.mark.isaacsim_ci
def test_single_environment_ovphysx(task_name):
    _run_environments(task_name, "cuda", 1, physics_preset_name="ovphysx")
