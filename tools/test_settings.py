# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This file contains the settings for the tests.
"""

import os

ISAACLAB_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
"""Path to the root directory of the Isaac Lab repository."""

DEFAULT_TIMEOUT = 1000
"""The default timeout for each test in seconds."""


PER_TEST_TIMEOUTS = {
    "test_articulation.py": 1500,
    "test_stage_in_memory.py": 1000,
    "test_imu.py": 1000,
    "test_environments.py": 10000,  # This test runs through all the environments for 100 steps each
    "test_environments_with_stage_in_memory.py": (
        10000
    ),  # Like the above, with stage in memory and with and without fabric cloning
    "test_environment_determinism.py": 1000,  # This test runs through many the environments for 100 steps each
    "test_pickplace_stack_environments.py": 10000,  # This test runs through PickPlace and Stack environments
    "test_factory_environments.py": 1000,  # This test runs through Factory environments for 100 steps each
    "test_multi_agent_environments.py": 800,  # This test runs through multi-agent environments for 100 steps each
    "test_generate_dataset_franka_state.py": 10000,  # This test runs annotation for 10 demos and generation for 1 demo
    "test_generate_dataset_franka_visuomotor.py": 10000,  # This test runs generation until one succeeds
    "test_generate_dataset_gr1t2_nutpour.py": 10000,  # This test runs generation until one succeeds
    "test_generate_dataset_gr1t2_pickplace.py": 10000,  # This test runs generation until one succeeds
    "test_generate_dataset_skillgen.py": 10000,  # This test runs generation for skillgen
    "test_pink_ik.py": 1000,  # This test runs through all the pink IK environments through various motions
    "test_environments_training.py": (
        10000
    ),  # This test runs through training for several environments and compares thresholds
    "test_environments_skillgen.py": 1000,
    "test_environments_automate.py": 2500,
    "test_teleop_environments.py": 5000,
    "test_teleop_environments_with_stage_in_memory.py": 5000,
    "test_cartpole_showcase_environments.py": 5000,
    "test_cartpole_showcase_environments_with_stage_in_memory.py": 5000,
    "test_simulation_render_config.py": 1000,
    "test_operational_space.py": 1000,
    "test_non_headless_launch.py": 1000,  # This test launches the app in non-headless mode and starts simulation
    "test_rl_games_wrapper.py": 1000,
    "test_rsl_rl_wrapper.py": 1000,
    "test_sb3_wrapper.py": 1000,
    "test_skrl_wrapper.py": 1000,
    "test_action_state_recorder_term.py": 1000,
    "test_manager_based_rl_env_obs_spaces.py": 1000,
    "test_visuotactile_sensor.py": 1000,
    "test_visuotactile_render.py": 1000,
    "test_rigid_object_collection.py": 1500,
    "test_outdated_sensor.py": 1000,
    "test_multi_tiled_camera.py": 1000,
    "test_multirotor.py": 1000,
    "test_shadow_hand_vision_presets.py": 5000,
    "test_environments_newton.py": 5000,
    "test_surface_gripper.py": 3000,
}
"""A dictionary of tests and their timeouts in seconds.

Note: Any tests not listed here will use the default timeout.
"""

CUROBO_PLANNER_TESTS = [
    "test_curobo_planner_franka.py",
    "test_curobo_planner_cube_stack.py",
    "test_pink_ik.py",
]
"""Tests for the cuRobo motion planner and Pink IK controller.

These tests are skipped in the base image CI jobs and run in the dedicated
``test-curobo`` CI job which uses the cuRobo Docker image.
"""

SKILLGEN_TESTS = [
    "test_generate_dataset_skillgen.py",
    "test_environments_skillgen.py",
    "test_environments_automate.py",
]
"""SkillGen and AutoMate environment tests.

These tests are skipped in the base image CI jobs and run in the dedicated
``test-skillgen`` CI job which uses the cuRobo Docker image.
"""

CUROBO_TESTS = [
    *CUROBO_PLANNER_TESTS,
    *SKILLGEN_TESTS,
]
"""A list of tests that require cuRobo installation.

These tests are skipped in the base image CI jobs and run separately in the
dedicated ``test-curobo`` and ``test-skillgen`` CI jobs which use the cuRobo
Docker image.
"""

QUARANTINED_TESTS: list[str] = []
"""A list of tests that are quarantined due to known instability.

These tests are skipped in normal CI runs. When the ``test-quarantined``
CI job is enabled (gated by the ``RUN_QUARANTINED_TESTS`` repository
variable), they run in a dedicated job where failures do not block PR
merges. The job is currently disabled. Add test filenames here to
quarantine them from regular CI.
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
    # curobo / skillgen - require cuRobo installation; run via test-curobo and test-skillgen CI jobs
    *CUROBO_TESTS,
    # quarantined tests - run in dedicated CI job that does not block PR merges
    *QUARANTINED_TESTS,
    "test_environments_training.py",  # Long-running RL training test; runs in dedicated CI job
]
"""A list of tests to skip in CI (see conftest.py)."""

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
