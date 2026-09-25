# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke tests for contributed environments that render with Isaac Sim's RTX renderer."""

import sys

# Import pinocchio before AppLauncher so Isaac Lab's dependency wins over Isaac Sim's bundled copy.
if sys.platform != "win32":
    import pinocchio  # noqa: F401

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import pytest

import isaaclab_tasks  # noqa: F401

# Local imports should be imported last
from contrib_env_test_utils import contrib_environment_params, num_envs  # isort: skip
from env_test_utils import _run_environments  # isort: skip


@pytest.mark.parametrize("task_name", contrib_environment_params("kit_cameras"))
def test_contrib_environments_kit_cameras(task_name):
    _run_environments(task_name, device="cuda", num_envs=num_envs(task_name))
