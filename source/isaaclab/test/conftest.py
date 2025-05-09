# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import subprocess
import sys

import pytest

# Tests that should be skipped (if any)
SKIP_TESTS = {
    # lab
    "test_argparser_launch.py",  # app.close issue
    "test_build_simulation_context_nonheadless.py",  # headless
    "test_env_var_launch.py",  # app.close issue
    "test_kwarg_launch.py",  # app.close issue
    "test_differential_ik.py",  # Failing
    # lab_tasks
    "test_record_video.py",  # Failing
    "test_tiled_camera_env.py",  # Need to improve the logic
}


def pytest_ignore_collect(path, config):
    """Ignore collecting tests that are in the skip list."""
    if os.path.isdir(str(path)):
        return False

    # Get just the filename from the path
    test_name = os.path.basename(str(path))
    return test_name in SKIP_TESTS  # Skip tests in the skip list


def run_individual_tests(test_files):
    """Run each test file separately, ensuring one finishes before starting the next."""
    failed_tests = []  # Track failed tests

    for test_file in test_files:
        print(f"\n\n🚀 Running {test_file} independently...\n")
        env = os.environ.copy()

        # Run each test file independently in a subprocess
        process = subprocess.run([sys.executable, "-m", "pytest", str(test_file), "-v"], env=env)

        if process.returncode != 0:
            failed_tests.append(test_file)

    return failed_tests  # Return list of failed tests


def pytest_sessionstart(session):
    """Intercept pytest startup to execute tests in the correct order."""
    # Check if the path 'source/IsaacLab/test' is in session.config.args
    if any(arg == "source/isaaclab/test" for arg in session.config.args):
        rootdir = str(session.config.rootpath) + "/test"

        # Get all test files in the directory, but only within isaaclab
        test_files = []
        for root, _, files in os.walk(rootdir):
            # Skip if we're outside the isaaclab directory
            if "isaaclab" not in root:
                continue

            for file in files:
                if file.endswith("_test.py") or file.startswith("test_"):
                    # Skip if the file is in SKIP_TESTS
                    if file in SKIP_TESTS:
                        print(f"Skipping {file} as it's in the skip list")
                        continue

                    full_path = os.path.join(root, file)
                    test_files.append(full_path)

        # Run all tests individually
        failed_tests = run_individual_tests(test_files)

        # If any tests failed, mark the session as failed
        if failed_tests:
            print("\nFailed tests:")
            for test in failed_tests:
                print(f"  - {test}")
            pytest.exit("Test failures occurred", returncode=1)
