# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import os
import select
import subprocess
import sys
import time

import pytest
from junitparser import Error, JUnitXml, TestCase, TestSuite
from prettytable import PrettyTable

# Local imports
import test_settings as test_settings  # isort: skip


def pytest_ignore_collect(collection_path, config):
    # Skip collection and run each test script individually
    return True


def capture_test_output_with_timeout(cmd, timeout, env):
    """Run a command with timeout and capture all output while streaming in real-time."""
    stdout_data = b""
    stderr_data = b""

    try:
        # Use Popen to capture output in real-time
        process = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0, universal_newlines=False
        )

        # Set up file descriptors for non-blocking reads
        stdout_fd = process.stdout.fileno()
        stderr_fd = process.stderr.fileno()

        # Set non-blocking mode (Unix systems only)
        try:
            import fcntl

            for fd in [stdout_fd, stderr_fd]:
                flags = fcntl.fcntl(fd, fcntl.F_GETFL)
                fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        except ImportError:
            # fcntl not available on Windows, use a simpler approach
            pass

        start_time = time.time()

        while process.poll() is None:
            # Check for timeout
            if time.time() - start_time > timeout:
                process.kill()
                try:
                    remaining_stdout, remaining_stderr = process.communicate(timeout=5)
                    stdout_data += remaining_stdout
                    stderr_data += remaining_stderr
                except subprocess.TimeoutExpired:
                    process.terminate()
                    remaining_stdout, remaining_stderr = process.communicate(timeout=1)
                    stdout_data += remaining_stdout
                    stderr_data += remaining_stderr
                return -1, stdout_data, stderr_data, True  # -1 indicates timeout

            # Check for available output
            try:
                ready_fds, _, _ = select.select([stdout_fd, stderr_fd], [], [], 0.1)

                for fd in ready_fds:
                    with contextlib.suppress(OSError):
                        if fd == stdout_fd:
                            chunk = process.stdout.read(1024)
                            if chunk:
                                stdout_data += chunk
                                # Print to stdout in real-time
                                sys.stdout.buffer.write(chunk)
                                sys.stdout.buffer.flush()
                        elif fd == stderr_fd:
                            chunk = process.stderr.read(1024)
                            if chunk:
                                stderr_data += chunk
                                # Print to stderr in real-time
                                sys.stderr.buffer.write(chunk)
                                sys.stderr.buffer.flush()
            except OSError:
                # select failed, fall back to simple polling
                time.sleep(0.1)
                continue

        # Get any remaining output
        remaining_stdout, remaining_stderr = process.communicate()
        stdout_data += remaining_stdout
        stderr_data += remaining_stderr

        return process.returncode, stdout_data, stderr_data, False

    except Exception as e:
        return -1, str(e).encode(), b"", False


def create_timeout_test_case(test_file, timeout, stdout_data, stderr_data):
    """Create a test case entry for a timeout test with captured logs."""
    test_suite = TestSuite(name=f"timeout_{os.path.splitext(os.path.basename(test_file))[0]}")
    test_case = TestCase(name="test_execution", classname=os.path.splitext(os.path.basename(test_file))[0])

    # Create error message with timeout info and captured logs
    error_msg = f"Test timed out after {timeout} seconds"

    # Add captured output to error details
    details = f"Timeout after {timeout} seconds\n\n"

    if stdout_data:
        details += "=== STDOUT ===\n"
        details += stdout_data.decode("utf-8", errors="replace") + "\n"

    if stderr_data:
        details += "=== STDERR ===\n"
        details += stderr_data.decode("utf-8", errors="replace") + "\n"

    error = Error(message=error_msg)
    error.text = details
    test_case.result = error

    test_suite.add_testcase(test_case)
    return test_suite


def run_individual_tests(test_files, workspace_root, isaacsim_ci):
    """Run each test file separately, ensuring one finishes before starting the next."""
    failed_tests = []
    test_status = {}

    for test_file in test_files:
        print(f"\n\n🚀 Running {test_file} independently...\n")
        # get file name from path
        file_name = os.path.basename(test_file)
        env = os.environ.copy()

        # Determine timeout for this test
        timeout = test_settings.PER_TEST_TIMEOUTS.get(file_name, test_settings.DEFAULT_TIMEOUT)

        # Prepare command
        # Note: Command options matter as they are used for cleanups inside AppLauncher
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "--no-header",
            f"--config-file={workspace_root}/pyproject.toml",
            f"--junitxml=tests/test-reports-{str(file_name)}.xml",
            "--tb=short",
        ]

        if isaacsim_ci:
            cmd.append("-m")
            cmd.append("isaacsim_ci")

        # Add the test file path last
        cmd.append(str(test_file))

        # Run test with timeout and capture output
        returncode, stdout_data, stderr_data, timed_out = capture_test_output_with_timeout(cmd, timeout, env)

        if timed_out:
            print(f"Test {test_file} timed out after {timeout} seconds...")
            failed_tests.append(test_file)

            # Create a special XML report for timeout tests with captured logs
            timeout_suite = create_timeout_test_case(test_file, timeout, stdout_data, stderr_data)
            timeout_report = JUnitXml()
            timeout_report.add_testsuite(timeout_suite)

            # Write timeout report
            report_file = f"tests/test-reports-{str(file_name)}.xml"
            timeout_report.write(report_file)

            test_status[test_file] = {
                "errors": 1,
                "failures": 0,
                "skipped": 0,
                "tests": 1,
                "result": "TIMEOUT",
                "time_elapsed": timeout,
            }
            continue

        if returncode != 0:
            failed_tests.append(test_file)

        # check report for any failures
        report_file = f"tests/test-reports-{str(file_name)}.xml"
        if not os.path.exists(report_file):
            print(f"Warning: Test report not found at {report_file}")
            failed_tests.append(test_file)
            test_status[test_file] = {
                "errors": 1,  # Assume error since we can't read the report
                "failures": 0,
                "skipped": 0,
                "tests": 0,
                "result": "FAILED",
                "time_elapsed": 0.0,
            }
            continue

        try:
            report = JUnitXml.fromfile(report_file)

            # Rename test suites to be more descriptive
            for suite in report:
                if suite.name == "pytest":
                    # Remove .py extension and use the filename as the test suite name
                    suite_name = os.path.splitext(file_name)[0]
                    suite.name = suite_name

            # Write the updated report back
            report.write(report_file)

            # Parse the integer values with None handling
            errors = int(report.errors) if report.errors is not None else 0
            failures = int(report.failures) if report.failures is not None else 0
            skipped = int(report.skipped) if report.skipped is not None else 0
            tests = int(report.tests) if report.tests is not None else 0
            time_elapsed = float(report.time) if report.time is not None else 0.0
        except Exception as e:
            print(f"Error reading test report {report_file}: {e}")
            failed_tests.append(test_file)
            test_status[test_file] = {
                "errors": 1,
                "failures": 0,
                "skipped": 0,
                "tests": 0,
                "result": "FAILED",
                "time_elapsed": 0.0,
            }
            continue

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

    print("~~~~~~~~~~~~ Finished running all tests")

    return failed_tests, test_status


def pytest_sessionstart(session):
    """Intercept pytest startup to execute tests in the correct order."""
    # Get the workspace root directory (one level up from tools)
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_dirs = [
        os.path.join(workspace_root, "scripts"),
        os.path.join(workspace_root, "source"),
    ]

    # Get filter pattern from environment variable or command line
    filter_pattern = os.environ.get("TEST_FILTER_PATTERN", "")
    exclude_pattern = os.environ.get("TEST_EXCLUDE_PATTERN", "")
    curobo_only = os.environ.get("TEST_CUROBO_ONLY", "false") == "true"

    isaacsim_ci = os.environ.get("ISAACSIM_CI_SHORT", "false") == "true"

    # Also try to get from pytest config
    if hasattr(session.config, "option") and hasattr(session.config.option, "filter_pattern"):
        filter_pattern = filter_pattern or getattr(session.config.option, "filter_pattern", "")
    if hasattr(session.config, "option") and hasattr(session.config.option, "exclude_pattern"):
        exclude_pattern = exclude_pattern or getattr(session.config.option, "exclude_pattern", "")

    print("=" * 50)
    print("CONFTEST.PY DEBUG INFO")
    print("=" * 50)
    print(f"Filter pattern: '{filter_pattern}'")
    print(f"Exclude pattern: '{exclude_pattern}'")
    print(f"Curobo-only mode: {curobo_only}")
    print(f"TEST_FILTER_PATTERN env var: '{os.environ.get('TEST_FILTER_PATTERN', 'NOT_SET')}'")
    print(f"TEST_EXCLUDE_PATTERN env var: '{os.environ.get('TEST_EXCLUDE_PATTERN', 'NOT_SET')}'")
    print(f"TEST_CUROBO_ONLY env var: '{os.environ.get('TEST_CUROBO_ONLY', 'NOT_SET')}'")
    print("=" * 50)

    # Get all test files in the source directories
    test_files = []

    for source_dir in source_dirs:
        if not os.path.exists(source_dir):
            print(f"Error: source directory not found at {source_dir}")
            pytest.exit("Source directory not found", returncode=1)

        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.startswith("test_") and file.endswith(".py"):
                    if curobo_only:
                        # In curobo-only mode, run exclusively the cuRobo and SkillGen tests.
                        # The normal TESTS_TO_SKIP list is intentionally bypassed here so that
                        # these tests (which are skipped in base-image jobs) can execute.
                        if file not in test_settings.CUROBO_TESTS:
                            continue
                    else:
                        # Skip if the file is in TESTS_TO_SKIP
                        if file in test_settings.TESTS_TO_SKIP:
                            print(f"Skipping {file} as it's in the skip list")
                            continue

                    full_path = os.path.join(root, file)

                    # Apply include filter
                    if filter_pattern and filter_pattern not in full_path:
                        print(f"Skipping {full_path} (does not match include pattern: {filter_pattern})")
                        continue

                    # Apply exclude filter
                    if exclude_pattern and exclude_pattern in full_path:
                        print(f"Skipping {full_path} (matches exclude pattern: {exclude_pattern})")
                        continue

                    test_files.append(full_path)

    if isaacsim_ci:
        new_test_files = []
        for test_file in test_files:
            with open(test_file) as f:
                if "@pytest.mark.isaacsim_ci" in f.read():
                    new_test_files.append(test_file)
        test_files = new_test_files

    if not test_files:
        print("No test files found in source directory")
        pytest.exit("No test files found", returncode=1)

    print(f"Found {len(test_files)} test files after filtering:")
    for test_file in test_files:
        print(f"  - {test_file}")

    # Run all tests individually
    failed_tests, test_status = run_individual_tests(test_files, workspace_root, isaacsim_ci)

    print("failed tests:", failed_tests)

    # Collect reports
    print("~~~~~~~~~ Collecting final report...")

    # create new full report
    full_report = JUnitXml()
    # read all reports and merge them
    for report in os.listdir("tests"):
        if report.endswith(".xml"):
            print(report)
            report_file = JUnitXml.fromfile(f"tests/{report}")
            full_report += report_file
    print("~~~~~~~~~~~~ Writing final report...")
    # write content to full report
    result_file = os.environ.get("TEST_RESULT_FILE", "full_report.xml")
    full_report_path = f"tests/{result_file}"
    print(f"Using result file: {result_file}")
    full_report.write(full_report_path)
    print("~~~~~~~~~~~~ Report written to", full_report_path)

    # print test status in a nice table
    # Calculate the number and percentage of passing tests
    num_tests = len(test_status)
    num_passing = len([test_path for test_path in test_files if test_status[test_path]["result"] == "passed"])
    num_failing = len([test_path for test_path in test_files if test_status[test_path]["result"] == "FAILED"])
    num_timeout = len([test_path for test_path in test_files if test_status[test_path]["result"] == "TIMEOUT"])

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
    summary_str += f"Timeout: {num_timeout}\n"
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
        per_test_result_table.add_row(
            [
                test_path,
                test_status[test_path]["result"],
                f"{test_status[test_path]['time_elapsed']:0.2f}",
                f"{num_tests_passed}/{test_status[test_path]['tests']}",
            ]
        )

    summary_str += per_test_result_table.get_string()

    # Print summary to console and log file
    print(summary_str)

    # Exit pytest after custom execution to prevent normal pytest from overwriting our report
    pytest.exit("Custom test execution completed", returncode=0 if num_failing == 0 else 1)
