# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(enable_cameras=True)

import pytest

import isaaclab_tasks  # noqa: F401

# Local imports should be imported last
from env_test_utils import _check_random_actions, setup_environment  # isort: skip


@pytest.mark.parametrize("num_envs, device", [(2, "cuda"), (1, "cuda")])
@pytest.mark.parametrize("task_name", setup_environment(multi_agent=True, tier="core"))
def test_environments(task_name, num_envs, device):
    """Run all multi-agent environments with random actions and check that they return valid signals."""
    _check_random_actions(task_name, device, num_envs, multi_agent=True)
