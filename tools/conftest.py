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


def pytest_ignore_collect(collection_path, config):
    # Skip collection and run each test script individually
    return True
    # # Only ignore collection for test files, not directories
    # if os.path.isdir(str(path)):
    #     return False

    # # Get just the filename from the path
    # test_name = os.path.basename(str(path))
    # return test_name in SKIP_TESTS  # Skip tests in the skip list


def run_individual_tests(test_files, workspace_root):
    """Run each test file separately, ensuring one finishes before starting the next."""
    failed_tests = []

    for test_file in test_files:
        print(f"\n\n🚀 Running {test_file} independently...\n")
        # get file name from path
        file_name = os.path.basename(test_file)
        env = os.environ.copy()

        # Run each test file with pytest but skip collection
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--no-header",
                f"--junitxml=tests/test-reports-{str(file_name)}.xml",
                str(test_file),
                "-v",
            ],
            env=env,
        )

        if process.returncode != 0:
            failed_tests.append(test_file)

    return failed_tests


def pytest_sessionstart(session):
    """Intercept pytest startup to execute tests in the correct order."""
    # Get the workspace root directory (one level up from tools)
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_dir = os.path.join(workspace_root, "source")

    if not os.path.exists(source_dir):
        print(f"Error: source directory not found at {source_dir}")
        pytest.exit("Source directory not found", returncode=1)

    # Get all test files in the source directory
    test_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.startswith("test_") and file.endswith(".py"):
                # Skip if the file is in SKIP_TESTS
                if file in SKIP_TESTS:
                    print(f"Skipping {file} as it's in the skip list")
                    continue

                full_path = os.path.join(root, file)
                test_files.append(full_path)

    if not test_files:
        print("No test files found in source directory")
        pytest.exit("No test files found", returncode=1)

    # Run all tests individually
    failed_tests = run_individual_tests(test_files, workspace_root)

    # Collect reports
    from junitparser import JUnitXml

    # create new full report
    full_report = JUnitXml()
    # read all reports and merge them
    for report in os.listdir("tests"):
        if report.endswith(".xml"):
            report_file = JUnitXml.fromfile(f"tests/{report}")
            full_report += report_file
    # write content to full report
    full_report_path = "tests/full_report.xml"
    full_report.write(full_report_path)

    # If any tests failed, mark the session as failed
    if failed_tests:
        print("\nFailed tests:")
        for test in failed_tests:
            print(f"  - {test}")
        pytest.exit("Test failures occurred", returncode=1)
