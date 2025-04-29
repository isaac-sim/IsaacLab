# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import subprocess
import sys

# Tests that require a new instance of the AppLauncher
FIRST_RUN_TESTS = {
    "/app/test_argparser_launch.py",
    "/app/test_env_var_launch.py",
    "/app/test_kwarg_launch.py",
    "/performance/test_kit_startup_performance.py",
}

# Tests that require headless and camera enabled
SECOND_RUN_TESTS = {
    "/envs/test_env_rendering_logic.py",
    "/envs/test_manager_based_rl_env_ui.py",
    "/sensors/test_multi_tiled_camera.py",
    "/sensors/test_tiled_camera_env.py",
    "/sensors/test_outdated_sensor.py",
    "/sensors/test_imu.py",
    "/sensors/test_tiled_camera.py",
    "/sensors/test_ray_caster_camera.py",
    "/sensors/test_camera.py",
    "/sim/test_simulation_render_config.py",
}


def pytest_ignore_collect(path, config):
    """Ignore collecting tests that are not part of the current test stage."""
    if os.path.isdir(str(path)):
        return False

    # Check if the path 'source/IsaacLab/test' is in config.args
    if any(arg == "source/isaaclab/test" for arg in config.args):
        test_name = str(path).removeprefix(str(config.rootdir) + "/test")
        stage = os.getenv("PYTEST_EXEC_STAGE", "main")

        if stage == "first":
            return test_name not in FIRST_RUN_TESTS
        if stage == "second":
            return test_name not in SECOND_RUN_TESTS
        if stage == "main":
            return test_name in FIRST_RUN_TESTS or test_name in SECOND_RUN_TESTS

    # Default behavior if path is not 'source/IsaacLab/test'
    return False  # Default: collect everything if no stage is set


def run_test_group(test_files, stage_name, config_args):
    """Run a specific group of tests as a separate pytest session."""
    if test_files:
        print(f"\n🚀 Running {stage_name} tests...\n")
        env = os.environ.copy()
        env["PYTEST_EXEC_STAGE"] = stage_name

        # Run subprocess for the test group and capture the return code
        process = subprocess.run([sys.executable, "-m", "pytest", *config_args], env=env)
        return process.returncode  # Return the status of the test run

    return 0  # No tests in the group


def run_individual_tests(test_files):
    """Run each test file separately, ensuring one finishes before starting the next."""
    failed_tests = False  # Track if any test fails

    for test_file in test_files:
        print(f"\n\n🚀 Running {test_file} independently...\n")
        env = os.environ.copy()
        env["PYTEST_EXEC_STAGE"] = "first"

        # Run each test file independently in a subprocess and capture the return code
        process = subprocess.run([sys.executable, "-m", "pytest", str(test_file)], env=env)
        if process.returncode != 0:
            failed_tests = True  # Mark that at least one test failed

    return failed_tests  # Return True if any test failed


def pytest_sessionstart(session):
    """Intercept pytest startup to execute tests in the correct order."""
    if os.getenv("PYTEST_EXEC_STAGE"):
        return  # Prevent infinite loop in subprocesses

    # Check if the path 'source/IsaacLab/test' is in session.config.args
    if any(arg == "source/isaaclab/test" for arg in session.config.args):
        rootdir = str(session.config.rootpath) + "/test"
        failed = False  # Track failures

        # Step 1: Run first batch of tests separately (sequential execution)
        first_tests = [rootdir + test for test in FIRST_RUN_TESTS if os.path.exists(rootdir + test)]
        failed |= run_individual_tests(first_tests)

        # Step 2: Run second batch together (ensures completion before moving forward)
        failed |= run_test_group(SECOND_RUN_TESTS, "second", session.config.args)

        # Step 3: Run all remaining tests
        failed |= run_test_group([], "main", session.config.args)

        # Exit only at the end with failure code if any test failed
        if failed:
            sys.exit(1)

    # Default behavior if path is not 'source/IsaacLab/test'
    # Add any default session start logic here if needed
