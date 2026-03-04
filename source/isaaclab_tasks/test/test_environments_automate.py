# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

import sys

# Import pinocchio in the main script to force the use of the dependencies
# installed by IsaacLab and not the one installed by Isaac Sim.
# pinocchio is required by the Pink IK controller
if sys.platform != "win32":
    import pinocchio  # noqa: F401

from isaaclab.app import AppLauncher

# launch the simulator
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import pytest

import isaaclab_tasks  # noqa: F401

# Local imports should be imported last
# Note: _check_random_actions is used directly here (instead of _run_environments) so that the
# skip guards in _run_environments for AutoMate environments are intentionally bypassed.
from env_test_utils import _check_random_actions  # isort: skip

# AutoMate environments require a CUDA installation that is present in the cuRobo Docker image
# but not in the base image. This test is intentionally excluded from the base-image CI jobs via
# CUROBO_TESTS / TESTS_TO_SKIP in test_settings.py and is only executed by the dedicated
# ``test-curobo`` CI job which uses the cuRobo Docker image.
AUTOMATE_ENVS = [
    "Isaac-AutoMate-Assembly-Direct-v0",
    "Isaac-AutoMate-Disassembly-Direct-v0",
]


@pytest.mark.parametrize("num_envs, device", [(32, "cuda"), (1, "cuda")])
@pytest.mark.parametrize("task_name", AUTOMATE_ENVS)
@pytest.mark.isaacsim_ci
def test_automate_environments(task_name, num_envs, device):
    _check_random_actions(task_name, device, num_envs, create_stage_in_memory=False)
