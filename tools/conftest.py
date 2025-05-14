# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import subprocess
import sys
from prettytable import PrettyTable

import pytest
from junitparser import JUnitXml

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


def run_individual_tests(test_files, workspace_root):
    """Run each test file separately, ensuring one finishes before starting the next."""
    failed_tests = []
    test_status = dict()

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

        # check report for any failures
        report = JUnitXml.fromfile(f"tests/test-reports-{str(file_name)}.xml")

        # Parse the integer values
        errors = int(report.errors)
        failures = int(report.failures)
        skipped = int(report.skipped)
        tests = int(report.tests)
        time_elapsed = float(report.time)
        # Check if there were any failures
        if errors > 0 or failures > 0:
            failed_tests.append(test_file)

        test_status[test_file] = {
            "errors": errors,
            "failures": failures,
            "skipped": skipped,
            "tests": tests,
            "result": "FAILED" if errors > 0 or failures > 0 else "passed",
            "time_elapsed": time_elapsed,
        }

    return failed_tests, test_status


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
    failed_tests, test_status = run_individual_tests(test_files, workspace_root)

    # Collect reports

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

    # print test status in a nice table
    # Calculate the number and percentage of passing tests
    num_tests = len(test_status)
    num_passing = len([test_path for test_path in test_files if test_status[test_path]["result"] == "passed"])
    num_failing = len([test_path for test_path in test_files if test_status[test_path]["result"] == "FAILED"])

    if num_tests == 0:
        passing_percentage = 100
    else:
        passing_percentage = num_passing / num_tests * 100

    # Print summaries of test results
    summary_str = "\n\n"
    summary_str += "===================\n"
    summary_str += "Test Result Summary\n"
    summary_str += "===================\n"

    summary_str += f"Total: {num_tests}\n"
    summary_str += f"Passing: {num_passing}\n"
    summary_str += f"Failing: {num_failing}\n"

    summary_str += f"Passing Percentage: {passing_percentage:.2f}%\n"

    # Print time elapsed in hours, minutes, seconds
    total_time = sum([test_status[test_path]["time_elapsed"] for test_path in test_files])

    summary_str += f"Total Time Elapsed: {total_time // 3600}h"
    summary_str += f"{total_time // 60 % 60}m"
    summary_str += f"{total_time % 60:.2f}s"

    summary_str += "\n\n=======================\n"
    summary_str += "Per Test Result Summary\n"
    summary_str += "=======================\n"

    # Construct table of results per test
    per_test_result_table = PrettyTable(field_names=["Test Path", "Result", "Time (s)", "# Tests"])
    per_test_result_table.align["Test Path"] = "l"
    per_test_result_table.align["Time (s)"] = "r"
    for test_path in test_files:
        num_tests_passed = (
            test_status[test_path]["tests"]
            - test_status[test_path]["failures"]
            - test_status[test_path]["errors"]
            - test_status[test_path]["skipped"]
        )
        per_test_result_table.add_row([
            test_path,
            test_status[test_path]["result"],
            f"{test_status[test_path]['time_elapsed']:0.2f}",
            f"{num_tests_passed}/{test_status[test_path]['tests']}",
        ])

    summary_str += per_test_result_table.get_string()

    # Print summary to console and log file
    print(summary_str)

    # If any tests failed, mark the session as failed
    if failed_tests:
        print("\nFailed tests:")
        for test in failed_tests:
            print(f"  - {test}")
        pytest.exit("Test failures occurred", returncode=1)
