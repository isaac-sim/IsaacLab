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
    "test_articulation.py": 1000,
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
    "test_generate_dataset.py": 1000,  # This test runs annotation for 10 demos and generation until one succeeds
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
    "test_manager_based_rl_env_obs_spaces.py": 500,
    "test_visuotactile_sensor.py": 1000,
    "test_visuotactile_render.py": 1000,
    "test_rigid_object_collection.py": 1000,
    "test_outdated_sensor.py": 1000,
    "test_multi_tiled_camera.py": 1000,
    "test_multirotor.py": 1000,
    "test_shadow_hand_vision_presets.py": 5000,
    "test_environments_newton.py": 5000,
}
"""A dictionary of tests and their timeouts in seconds.

Note: Any tests not listed here will use the default timeout.
"""

CUROBO_TESTS = [
    "test_curobo_planner_franka.py",
    "test_curobo_planner_cube_stack.py",
    "test_generate_dataset_skillgen.py",
    "test_environments_skillgen.py",
    "test_environments_automate.py",
    "test_pink_ik.py",
]
"""A list of tests that require cuRobo installation.

These tests are skipped in the base image CI jobs and run separately in the
dedicated ``test-curobo`` CI job which uses the cuRobo Docker image.
"""

FLAKY_TESTS = [
    # Consistently failing or highly flaky (<60% pass rate across recent CI runs)
    "test_rigid_object_collection_iface.py",  # 0%
    "test_environments_newton.py",  # 5.9%
    "test_articulation_iface.py",  # 7.7%
    # "test_articulation.py",  # 10.5%
    "test_rigid_object_iface.py",  # 23.1%
    # "test_rigid_object_collection.py",  # 26.7%
    "test_physx_scene_data_provider_visualizer_contract.py",  # 30.8%
    "test_shadow_hand_vision_presets.py",  # 47.4%
    "test_mock_data_properties.py",  # 50.0%
    # "test_rigid_object.py",  # 57.7%
    # "test_contact_sensor.py",  # 60.0%
    "test_robot_load_performance.py",  # 60.0%
    # Failing in recent CI runs
    # "test_camera.py",
    "test_logger.py",
    "test_multirotor.py",
    "test_null_command_term.py",
    "test_sb3_wrapper.py",
    "test_selection_strategy.py",
    "test_spawn_lights.py",
]
"""A list of tests that are known to be flaky (< 60% pass rate).

These tests are skipped in normal CI runs and executed in the dedicated
``test-flaky`` CI job where failures do not block PR merges.
"""

SLIGHTLY_FLAKY_TESTS = [
    # Mildly intermittent (96%+ pass rate across recent CI runs)
    # "test_multi_mesh_ray_caster_camera.py",  # 96.2%
    # "test_ray_caster_camera.py",  # 96.2%
    "test_surface_gripper.py",  # 96.2%
    "test_env_cfg_no_forbidden_imports.py",  # 96.4%
]
"""A list of tests that are mildly flaky (96%+ pass rate).

These tests are skipped in normal CI runs and executed in the dedicated
``test-slightly-flaky`` CI job where failures do not block PR merges.
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
    # curobo / skillgen - require cuRobo installation; run via the test-curobo CI job
    *CUROBO_TESTS,
    # flaky tests - run in dedicated CI jobs that do not block PR merges
    *FLAKY_TESTS,
    *SLIGHTLY_FLAKY_TESTS,
    "test_environments_training.py",  # Long-running RL training test; runs in dedicated CI job
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
