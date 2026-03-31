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


# TODO: SimulationApp.close() can hang indefinitely in some tests (especially those using
# cameras or render products), causing CI timeouts. A fix that detects test completion via
# the JUnit XML report file and kills the process after a grace period was prototyped in
# https://github.com/isaac-sim/IsaacLab/pull/5097 — see commits:
#   242315b3722 Handle hanging subprocesses causing timeouts
#   bd4953019ab Add comment explaining intentional return 0 in shutdown-hang path
# A per-test-file conftest.py using atexit.register(os._exit) was also tried:
#   6840e5a3aeb Force stalled test subprocesses that hang after SimulationApp.close()
def capture_test_output_with_timeout(cmd, timeout, env):
    """Run a command with timeout and capture all output while streaming in real-time."""
    stdout_data = b""
    stderr_data = b""
    process = None

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

        # Drain any output the process wrote before or just after exiting.
        # Wrapped in try/except so a pipe error doesn't discard what was already captured.
        try:
            remaining_stdout, remaining_stderr = process.communicate(timeout=10)
            stdout_data += remaining_stdout
            stderr_data += remaining_stderr
        except Exception:
            pass

        return process.returncode, stdout_data, stderr_data, False

    except Exception as e:
        # Kill the process if it is still alive, then drain whatever it wrote.
        if process is not None and process.poll() is None:
            process.kill()
            with contextlib.suppress(Exception):
                rem_out, rem_err = process.communicate(timeout=5)
                stdout_data += rem_out
                stderr_data += rem_err
        # Append the exception message so the caller can see what went wrong,
        # but preserve any output already captured.
        stdout_data += f"\n[capture error: {e}]\n".encode()
        return -1, stdout_data, stderr_data, False


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
    xml_reports = []  # in-memory JUnitXml objects, used to build the merged report

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
            xml_reports.append(timeout_report)

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
            if returncode < 0:
                sig = -returncode
                reason = f"Process killed by signal {sig}"
                if sig == 9:
                    reason += " (SIGKILL — likely OOM killed)"
                elif sig == 6:
                    reason += " (SIGABRT)"
                print(f"⚠️  {test_file}: {reason}")
            else:
                reason = f"Process exited with code {returncode} but produced no report"
                print(f"Warning: Test report not found at {report_file}")

            crash_suite = TestSuite(name=f"crash_{os.path.splitext(file_name)[0]}")
            crash_case = TestCase(
                name="test_execution",
                classname=os.path.splitext(file_name)[0],
            )
            details = f"{reason}\n\n"
            if stdout_data:
                details += "=== STDOUT (last 2000 chars) ===\n"
                details += stdout_data.decode("utf-8", errors="replace")[-2000:] + "\n"
            if stderr_data:
                details += "=== STDERR (last 2000 chars) ===\n"
                details += stderr_data.decode("utf-8", errors="replace")[-2000:] + "\n"
            error = Error(message=reason)
            error.text = details
            crash_case.result = error
            crash_suite.add_testcase(crash_case)
            crash_report = JUnitXml()
            crash_report.add_testsuite(crash_suite)
            crash_report.write(report_file)
            xml_reports.append(crash_report)

            failed_tests.append(test_file)
            test_status[test_file] = {
                "errors": 1,
                "failures": 0,
                "skipped": 0,
                "tests": 1,
                "result": "CRASHED",
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
            xml_reports.append(report)

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

    return failed_tests, test_status, xml_reports


def _collect_test_files(
    source_dirs,
    filter_pattern,
    exclude_pattern,
    include_files,
    quarantined_only,
    curobo_only,
):
    """Collect test files from source directories, applying all active filters."""
    test_files = []
    for source_dir in source_dirs:
        if not os.path.exists(source_dir):
            print(f"Error: source directory not found at {source_dir}")
            pytest.exit("Source directory not found", returncode=1)

        for root, _, files in os.walk(source_dir):
            for file in files:
                if not (file.startswith("test_") and file.endswith(".py")):
                    continue

                # Mode-exclusive filters (each bypasses TESTS_TO_SKIP)
                if quarantined_only:
                    if file not in test_settings.QUARANTINED_TESTS:
                        continue
                elif curobo_only:
                    if file not in test_settings.CUROBO_TESTS:
                        continue
                else:
                    # An explicit include_files entry overrides TESTS_TO_SKIP, allowing
                    # dedicated jobs (e.g. test-environments-training) to run tests that
                    # are otherwise excluded from general CI runs.
                    if file in test_settings.TESTS_TO_SKIP and file not in include_files:
                        print(f"Skipping {file} as it's in the skip list")
                        continue

                full_path = os.path.join(root, file)

                if filter_pattern and filter_pattern not in full_path:
                    print(f"Skipping {full_path} (does not match include pattern: {filter_pattern})")
                    continue
                if exclude_pattern and any(p.strip() in full_path for p in exclude_pattern.split(",")):
                    print(f"Skipping {full_path} (matches exclude pattern: {exclude_pattern})")
                    continue
                if include_files and file not in include_files:
                    print(f"Skipping {full_path} (not in include files list)")
                    continue

                test_files.append(full_path)

    # Apply file-level sharding: sort deterministically, then select every Nth file.
    # Skip when include_files is set — in that case the test's own conftest handles
    # sharding at the test-item level (e.g. parametrized test cases).
    shard_index = os.environ.get("TEST_SHARD_INDEX", "")
    shard_count = os.environ.get("TEST_SHARD_COUNT", "")
    if shard_index and shard_count and not include_files:
        shard_index = int(shard_index)
        shard_count = int(shard_count)
        test_files.sort()
        test_files = [f for i, f in enumerate(test_files) if i % shard_count == shard_index]
        print(f"Shard {shard_index}/{shard_count}: selected {len(test_files)} test files")

    return test_files


def _write_empty_report():
    """Write an empty JUnit XML report so downstream CI steps find a valid file."""
    os.makedirs("tests", exist_ok=True)
    result_file = os.environ.get("TEST_RESULT_FILE", "full_report.xml")
    report = JUnitXml()
    report.write(f"tests/{result_file}")
    print(f"Wrote empty report to tests/{result_file}")


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
    include_files_str = os.environ.get("TEST_INCLUDE_FILES", "")
    quarantined_only = os.environ.get("TEST_QUARANTINED_ONLY", "false") == "true"
    curobo_only = os.environ.get("TEST_CUROBO_ONLY", "false") == "true"

    isaacsim_ci = os.environ.get("ISAACSIM_CI_SHORT", "false") == "true"

    # Parse include files list (comma-separated paths)
    include_files = set()
    if include_files_str:
        for f in include_files_str.split(","):
            f = f.strip()
            if f:
                include_files.add(os.path.basename(f))

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
    print(f"Include files: {include_files if include_files else 'none'}")
    print(f"Quarantined-only mode: {quarantined_only}")
    print(f"Curobo-only mode: {curobo_only}")
    print(f"TEST_FILTER_PATTERN env var: '{os.environ.get('TEST_FILTER_PATTERN', 'NOT_SET')}'")
    print(f"TEST_EXCLUDE_PATTERN env var: '{os.environ.get('TEST_EXCLUDE_PATTERN', 'NOT_SET')}'")
    print(f"TEST_INCLUDE_FILES env var: '{os.environ.get('TEST_INCLUDE_FILES', 'NOT_SET')}'")
    print(f"TEST_QUARANTINED_ONLY env var: '{os.environ.get('TEST_QUARANTINED_ONLY', 'NOT_SET')}'")
    print(f"TEST_CUROBO_ONLY env var: '{os.environ.get('TEST_CUROBO_ONLY', 'NOT_SET')}'")
    print("=" * 50)

    # Get all test files in the source directories
    test_files = _collect_test_files(
        source_dirs,
        filter_pattern,
        exclude_pattern,
        include_files,
        quarantined_only,
        curobo_only,
    )

    if isaacsim_ci:
        new_test_files = []
        for test_file in test_files:
            with open(test_file) as f:
                if "@pytest.mark.isaacsim_ci" in f.read():
                    new_test_files.append(test_file)
        test_files = new_test_files

    if not test_files:
        if quarantined_only:
            print("No quarantined tests configured — nothing to run.")
            _write_empty_report()
            pytest.exit("No quarantined tests configured", returncode=0)
        if filter_pattern:
            print(f"No test files found matching filter pattern '{filter_pattern}' — nothing to run.")
            _write_empty_report()
            pytest.exit("No test files found for filter", returncode=0)
        print("No test files found in source directory")
        pytest.exit("No test files found", returncode=1)

    print(f"Found {len(test_files)} test files after filtering:")
    for test_file in test_files:
        print(f"  - {test_file}")

    # Run all tests individually
    failed_tests, test_status, xml_reports = run_individual_tests(test_files, workspace_root, isaacsim_ci)

    print("failed tests:", failed_tests)

    # Collect reports
    print("~~~~~~~~~ Collecting final report...")

    # Merge in-memory report objects collected during the test run.  Reading the
    # on-disk files again risks losing <failure> elements if the junitparser
    # read/write round-trip does not preserve them faithfully.
    full_report = JUnitXml()
    for xml_report in xml_reports:
        print(xml_report)
        full_report += xml_report
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
    num_crashed = len([test_path for test_path in test_files if test_status[test_path]["result"] == "CRASHED"])

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
    summary_str += f"Crashed: {num_crashed}\n"
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
    pytest.exit(
        "Custom test execution completed",
        returncode=0 if (num_failing == 0 and num_timeout == 0 and num_crashed == 0) else 1,
    )
