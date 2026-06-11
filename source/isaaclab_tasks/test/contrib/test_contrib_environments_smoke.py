# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Basic smoke test for contributed environments.

Instantiates each contributed task (registered under ``isaaclab_tasks.contrib``) and steps
it with random actions to verify it loads and runs without error. Environment families that
require dedicated handling are covered by their own test files in this directory and are
excluded here to avoid duplication:

- Factory / Forge: ``test_factory_environments.py``
- Cartpole Showcase: ``test_cartpole_showcase_environments.py``
- PickPlace / Stack / Place: ``test_pickplace_stack_environments.py``
- Teleop: ``test_teleop_environments.py``
- AutoMate: ``test_environments_automate.py``
- Skillgen: ``test_environments_skillgen.py``
"""

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


@pytest.mark.parametrize("num_envs, device", [(2, "cuda")])
@pytest.mark.parametrize(
    "task_name",
    setup_environment(
        include_play=False,
        multi_agent=False,
        factory_envs=False,
        cartpole_showcase_envs=False,
        pickplace_stack_envs=False,
        teleop_envs=False,
        tier="contrib",
    ),
)
def test_contrib_environments_smoke(task_name, num_envs, device):
    # run a short rollout with random actions to verify the environment loads and steps
    _run_environments(task_name, device, num_envs, create_stage_in_memory=False)
