# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import pytest

import isaaclab_tasks  # noqa: F401

# Local imports should be imported last
from env_test_utils import _run_environments, setup_environment  # isort: skip


def _contrib_environment_params() -> list:
    """Return each contributed environment with its supported test device."""
    params = []
    for task_param in setup_environment(multi_agent=False, tier="contrib"):
        task_name = getattr(task_param, "values", (task_param,))[0]
        marks = getattr(task_param, "marks", ())
        device = "cpu" if "Suction" in task_name else "cuda"
        params.append(pytest.param(task_name, device, id=task_name, marks=marks))
    return params


@pytest.mark.parametrize("task_name, device", _contrib_environment_params())
def test_contrib_environments(task_name, device):
    _run_environments(task_name, device, num_envs=2)
