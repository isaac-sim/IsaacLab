# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This file contains the settings for the tests.
"""

import os

ISAACLAB_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
"""Path to the root directory of the Isaac Lab repository."""

DEFAULT_TIMEOUT = 300
"""The default timeout for each test in seconds."""

PER_TEST_TIMEOUTS = {
    "test_articulation.py": 500,
    "test_stage_in_memory.py": 500,
    "test_environments.py": 2000,  # This test runs through all the environments for 100 steps each
    "test_environments_with_stage_in_memory.py": (
        2000
    ),  # Like the above, with stage in memory and with and without fabric cloning
    "test_environment_determinism.py": 500,  # This test runs through many the environments for 100 steps each
    "test_factory_environments.py": 1000,  # This test runs through Factory environments for 100 steps each
    "test_multi_agent_environments.py": 800,  # This test runs through multi-agent environments for 100 steps each
    "test_generate_dataset.py": 500,  # This test runs annotation for 10 demos and generation until one succeeds
    "test_environments_training.py": 6000,
    "test_simulation_render_config.py": 500,
    "test_operational_space.py": 500,
}
"""A dictionary of tests and their timeouts in seconds.

Note: Any tests not listed here will use the default timeout.
"""

TESTS_TO_SKIP = [
    # lab
    "test_argparser_launch.py",  # app.close issue
    "test_build_simulation_context_nonheadless.py",  # headless
    "test_env_var_launch.py",  # app.close issue
    "test_kwarg_launch.py",  # app.close issue
    "test_differential_ik.py",  # Failing
    # lab_tasks
    "test_record_video.py",  # Failing
    "test_tiled_camera_env.py",  # Need to improve the logic
]
"""A list of tests to skip by run_tests.py"""

TEST_RL_ENVS = [
    # classic control
    "Isaac-Ant-v0",
    "Isaac-Cartpole-v0",
    # manipulation
    "Isaac-Lift-Cube-Franka-v0",
    "Isaac-Open-Drawer-Franka-v0",
    # dexterous manipulation
    "Isaac-Repose-Cube-Allegro-v0",
    # locomotion
    "Isaac-Velocity-Flat-Unitree-Go2-v0",
    "Isaac-Velocity-Rough-Anymal-D-v0",
    "Isaac-Velocity-Rough-G1-v0",
]
"""A list of RL environments to test training on by run_train_envs.py"""
