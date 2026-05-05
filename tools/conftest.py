# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import os
import select
import signal
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


COLD_CACHE_BUFFER = 700
"""Extra seconds added to the first camera-enabled test's hard timeout.

The first test that uses ``enable_cameras=True`` may compile shaders during its
run (~600 s).  This buffer prevents that from being misreported as a test
timeout.  Only the first such test gets the extension — after it runs, the
on-disk cache is populated.
"""

STARTUP_DEADLINE = 45
"""Seconds to wait for AppLauncher init or pytest collection before declaring a
startup hang.

AppLauncher prints ``[ISAACLAB] AppLauncher initialization complete`` to
``sys.__stderr__`` (never suppressed) when Kit finishes initializing, and pytest
prints ``collected N items`` to stdout after collection.  If neither appears
within this deadline the process is treated as hung.  45 s is above any
legitimate Kit startup (typically 30--60 s) while still catching real hangs
without wasting the full hard timeout.
"""

STARTUP_HANG_RETRIES = 2
"""Number of times to retry a test that hangs during startup before giving up."""

TIMEOUT_RETRIES = 2
"""Number of times to retry a test that reaches its hard timeout before giving up."""

SHUTDOWN_GRACE_PERIOD = 30
"""Seconds to wait for clean exit after the JUnit XML report file appears.

When a test completes and writes its JUnit report, the subprocess may hang
during ``SimulationApp.close()`` or Kit shutdown.  Rather than wasting the
full hard timeout, we give the process a short grace period to exit, then
kill it.  The test results are taken from the report file (pass/fail), not
from the kill.
"""


def capture_test_output_with_timeout(cmd, timeout, env, startup_deadline=0, report_file=""):
    """Run a command with timeout and capture all output while streaming in real-time.

    Args:
        cmd: Command to execute.
        timeout: Maximum wall-clock seconds before the process is killed.
        env: Environment variables for the subprocess.
        startup_deadline: If > 0, the process is killed early when neither
            ``AppLauncher initialization complete`` (stderr) nor ``collected``
            (stdout) appears within this many seconds.
        report_file: Path to the JUnit XML report file.  When set, the process
            is given only :data:`SHUTDOWN_GRACE_PERIOD` seconds to exit after
            the file appears on disk.

    Returns:
        Tuple of ``(returncode, stdout_bytes, stderr_bytes, kill_reason,
        wall_time, pre_kill_diag)``.  *kill_reason* is ``""`` for normal exits,
        ``"timeout"`` for hard timeouts, ``"startup_hang"`` when the process
        did not reach pytest collection in time, or ``"shutdown_hang"`` when
        the test completed but the process hung during shutdown.
    """
    stdout_data = b""
    stderr_data = b""
    process = None

    try:
        # Each test gets its own session so orphaned Kit/Isaac Sim child
        # processes cannot send SIGHUP to the next test's process group.
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            universal_newlines=False,
            start_new_session=True,
        )
        pgid = os.getpgid(process.pid)

        stdout_fd = process.stdout.fileno()
        stderr_fd = process.stderr.fileno()

        try:
            import fcntl

            for fd in [stdout_fd, stderr_fd]:
                flags = fcntl.fcntl(fd, fcntl.F_GETFL)
                fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        except ImportError:
            pass

        start_time = time.time()
        startup_done = startup_deadline <= 0
        shutdown_deadline = 0.0

        while process.poll() is None:
            elapsed = time.time() - start_time

            if not startup_done:
                if b"AppLauncher initialization complete" in stderr_data or b"collected " in stdout_data:
                    startup_done = True

            if report_file and not shutdown_deadline and os.path.exists(report_file):
                shutdown_deadline = time.time() + SHUTDOWN_GRACE_PERIOD

            kill_reason = None
            if not startup_done and elapsed > startup_deadline:
                kill_reason = "startup_hang"
            elif shutdown_deadline and time.time() > shutdown_deadline:
                kill_reason = "shutdown_hang"
            elif elapsed > timeout:
                kill_reason = "timeout"

            if kill_reason:
                pre_kill_diag = _capture_system_diagnostics()

                # Kill the entire process group (test + any Kit children).
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    process.kill()
                try:
                    remaining_stdout, remaining_stderr = process.communicate(timeout=5)
                    stdout_data += remaining_stdout
                    stderr_data += remaining_stderr
                except subprocess.TimeoutExpired:
                    pass
                wall_time = time.time() - start_time
                return -1, stdout_data, stderr_data, kill_reason, wall_time, pre_kill_diag

            try:
                ready_fds, _, _ = select.select([stdout_fd, stderr_fd], [], [], 0.1)

                for fd in ready_fds:
                    with contextlib.suppress(OSError):
                        if fd == stdout_fd:
                            chunk = process.stdout.read(1024)
                            if chunk:
                                stdout_data += chunk
                                sys.stdout.buffer.write(chunk)
                                sys.stdout.buffer.flush()
                        elif fd == stderr_fd:
                            chunk = process.stderr.read(1024)
                            if chunk:
                                stderr_data += chunk
                                sys.stderr.buffer.write(chunk)
                                sys.stderr.buffer.flush()
            except OSError:
                time.sleep(0.1)
                continue

        # Drain any output the process wrote before or just after exiting.
        try:
            remaining_stdout, remaining_stderr = process.communicate(timeout=10)
            stdout_data += remaining_stdout
            stderr_data += remaining_stderr
        except Exception:
            pass

        # Kill any orphaned child processes (Kit, Isaac Sim) left by the test.
        try:
            os.killpg(pgid, signal.SIGKILL)
            time.sleep(1)
        except (ProcessLookupError, PermissionError, OSError):
            pass

        wall_time = time.time() - start_time
        return process.returncode, stdout_data, stderr_data, "", wall_time, ""

    except Exception as e:
        if process is not None and process.poll() is None:
            process.kill()
            with contextlib.suppress(Exception):
                rem_out, rem_err = process.communicate(timeout=5)
                stdout_data += rem_out
                stderr_data += rem_err
        stdout_data += f"\n[capture error: {e}]\n".encode()
        return -1, stdout_data, stderr_data, "", 0.0, ""


_SIGNAL_DESCRIPTIONS = {
    1: "SIGHUP — session leader exit or orphaned process cleanup",
    6: "SIGABRT",
    9: "SIGKILL — likely OOM killed",
    11: "SIGSEGV — segmentation fault",
    15: "SIGTERM",
}


def _signal_description(sig):
    """Return a human-readable description for a process killed by a signal."""
    base = f"Process killed by signal {sig}"
    desc = _SIGNAL_DESCRIPTIONS.get(sig)
    return f"{base} ({desc})" if desc else base


def _create_error_report(prefix, file_name, message, details):
    """Create a JUnit XML error report for a test that failed to produce its own.

    Returns a :class:`JUnitXml` object ready to be written to disk.
    """
    suite_name = os.path.splitext(file_name)[0]
    suite = TestSuite(name=f"{prefix}_{suite_name}")
    case = TestCase(name="test_execution", classname=suite_name)
    error = Error(message=message)
    error.text = details
    case.result = error
    suite.add_testcase(case)
    report = JUnitXml()
    report.add_testsuite(suite)
    return report


def _get_diagnostics(pre_kill_diag=""):
    """Return system diagnostics, truncated to 10 000 chars."""
    diag = pre_kill_diag or _capture_system_diagnostics()
    if len(diag) > 10000:
        diag = diag[:10000] + "\n... (truncated)"
    return diag


def _capture_system_diagnostics():
    """Capture system diagnostics (GPU, memory, processes) for crash investigation.

    All errors are caught and reported inline so this never raises.
    """
    sections = []

    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        if r.stdout:
            sections.append(f"--- nvidia-smi ---\n{r.stdout.strip()}")
    except Exception as e:
        sections.append(f"--- nvidia-smi --- FAILED: {e}")

    try:
        with open("/proc/meminfo") as f:
            lines = f.readlines()
        keys = ("MemTotal", "MemFree", "MemAvailable", "Committed_AS", "SwapTotal", "SwapFree")
        relevant = [line.strip() for line in lines if any(line.startswith(k) for k in keys)]
        if relevant:
            sections.append("--- /proc/meminfo ---\n" + "\n".join(relevant))
    except Exception as e:
        sections.append(f"--- /proc/meminfo --- FAILED: {e}")

    cgroup_lines = []
    for path in (
        "/sys/fs/cgroup/memory.current",
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory.events",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        "/sys/fs/cgroup/memory/memory.oom_control",
    ):
        try:
            with open(path) as f:
                cgroup_lines.append(f"{path}: {f.read().strip()}")
        except FileNotFoundError:
            pass
        except Exception as e:
            cgroup_lines.append(f"{path}: FAILED ({e})")
    if cgroup_lines:
        sections.append("--- cgroup memory ---\n" + "\n".join(cgroup_lines))

    try:
        r = subprocess.run(["ps", "auxf"], capture_output=True, text=True, timeout=5)
        if r.stdout:
            sections.append(f"--- process tree (ps auxf) ---\n{r.stdout.strip()}")
    except Exception as e:
        sections.append(f"--- process tree --- FAILED: {e}")

    try:
        r = subprocess.run(["dmesg", "-T"], capture_output=True, text=True, timeout=5)
        if r.stdout:
            lines = r.stdout.strip().split("\n")
            sections.append("--- dmesg (last 30 lines) ---\n" + "\n".join(lines[-30:]))
    except Exception:
        pass

    return "\n\n".join(sections)


def run_individual_tests(test_files, workspace_root, isaacsim_ci):
    """Run each test file separately, ensuring one finishes before starting the next."""
    failed_tests = []
    test_status = {}
    xml_reports = []
    cold_cache_applied = False

    for test_file in test_files:
        print(f"\n\n🚀 Running {test_file} independently...\n")
        file_name = os.path.basename(test_file)
        env = os.environ.copy()
        env["PYTHONFAULTHANDLER"] = "1"

        timeout = test_settings.PER_TEST_TIMEOUTS.get(file_name, test_settings.DEFAULT_TIMEOUT)

        # Read the test file once for cold-cache check.
        try:
            with open(test_file) as fh:
                test_content = fh.read()
        except OSError:
            test_content = ""

        # The first camera-enabled test in a fresh container compiles shaders
        # (~600 s).  Give it extra time so that doesn't look like a test timeout.
        is_cold_cache_test = not cold_cache_applied and "enable_cameras=True" in test_content
        if is_cold_cache_test:
            timeout += COLD_CACHE_BUFFER
            cold_cache_applied = True
            print(f"⏱️  Adding {COLD_CACHE_BUFFER}s cold-cache buffer (timeout now {timeout}s)")

        extra = COLD_CACHE_BUFFER if is_cold_cache_test else 0
        startup_deadline = min(timeout, STARTUP_DEADLINE + extra)

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

        cmd.append(str(test_file))

        report_file = f"tests/test-reports-{str(file_name)}.xml"

        # -- Run with retry on startup hang or hard timeout -----------------
        returncode, stdout_data, stderr_data, kill_reason = -1, b"", b"", ""
        wall_time, pre_kill_diag = 0.0, ""
        startup_hang_attempts = 0
        timeout_attempts = 0
        while True:
            with contextlib.suppress(FileNotFoundError):
                os.remove(report_file)

            returncode, stdout_data, stderr_data, kill_reason, wall_time, pre_kill_diag = (
                capture_test_output_with_timeout(
                    cmd, timeout, env, startup_deadline=startup_deadline, report_file=report_file
                )
            )

            has_report = os.path.exists(report_file)

            if kill_reason == "startup_hang" and startup_hang_attempts < STARTUP_HANG_RETRIES:
                startup_hang_attempts += 1
                print(
                    f"⚠️  {test_file}: startup hang detected after {startup_deadline}s"
                    f" (attempt {startup_hang_attempts}/{STARTUP_HANG_RETRIES + 1}), retrying..."
                )
                if stderr_data:
                    print("=== STDERR (last 5000 chars) ===")
                    print(stderr_data.decode("utf-8", errors="replace")[-5000:])
                diag = pre_kill_diag or _capture_system_diagnostics()
                if len(diag) > 10000:
                    diag = diag[:10000] + "\n... (truncated)"
                print(diag)
                continue

            if kill_reason == "timeout" and not has_report and timeout_attempts < TIMEOUT_RETRIES:
                timeout_attempts += 1
                print(
                    f"⚠️  {test_file}: timeout detected after {timeout}s"
                    f" (attempt {timeout_attempts}/{TIMEOUT_RETRIES + 1}), retrying..."
                )
                if stdout_data:
                    print("=== STDOUT (last 5000 chars) ===")
                    print(stdout_data.decode("utf-8", errors="replace")[-5000:])
                if stderr_data:
                    print("=== STDERR (last 5000 chars) ===")
                    print(stderr_data.decode("utf-8", errors="replace")[-5000:])
                diag = pre_kill_diag or _capture_system_diagnostics()
                if len(diag) > 10000:
                    diag = diag[:10000] + "\n... (truncated)"
                print(diag)
                continue
            break

        # -- Resolve result from kill_reason and report file ----------------
        has_report = os.path.exists(report_file)

        if kill_reason == "startup_hang":
            diag = _get_diagnostics(pre_kill_diag)
            print(f"⚠️  {test_file}: startup hang after {STARTUP_HANG_RETRIES + 1} attempt(s)")
            print(diag)

            msg = f"Startup hang after {startup_deadline}s (retried {STARTUP_HANG_RETRIES} time(s))"
            details = f"{msg}\n\n=== SYSTEM DIAGNOSTICS ===\n{diag}\n\n"
            if stderr_data:
                details += "=== STDERR (last 5000 chars) ===\n"
                details += stderr_data.decode("utf-8", errors="replace")[-5000:] + "\n"
            if stdout_data:
                details += "=== STDOUT (last 2000 chars) ===\n"
                details += stdout_data.decode("utf-8", errors="replace")[-2000:] + "\n"

            error_report = _create_error_report("startup_hang", file_name, msg, details)
            error_report.write(report_file)
            xml_reports.append(error_report)
            failed_tests.append(test_file)
            test_status[test_file] = {
                "errors": 1,
                "failures": 0,
                "skipped": 0,
                "tests": 1,
                "result": "STARTUP_HANG",
                "time_elapsed": 0.0,
                "wall_time": wall_time,
            }
            continue

        if kill_reason == "timeout" and not has_report:
            diag = _get_diagnostics(pre_kill_diag)
            print(f"Test {test_file} timed out after {timeout} seconds...")
            print(diag)

            msg = f"Timeout after {timeout} seconds (retried {timeout_attempts} time(s))"
            details = f"{msg}\n\n=== SYSTEM DIAGNOSTICS ===\n{diag}\n\n"
            if stdout_data:
                details += "=== STDOUT (last 5000 chars) ===\n"
                details += stdout_data.decode("utf-8", errors="replace")[-5000:] + "\n"
            if stderr_data:
                details += "=== STDERR (last 5000 chars) ===\n"
                details += stderr_data.decode("utf-8", errors="replace")[-5000:] + "\n"

            error_report = _create_error_report("timeout", file_name, msg, details)
            error_report.write(report_file)
            xml_reports.append(error_report)
            failed_tests.append(test_file)
            test_status[test_file] = {
                "errors": 1,
                "failures": 0,
                "skipped": 0,
                "tests": 1,
                "result": "TIMEOUT",
                "time_elapsed": timeout,
                "wall_time": wall_time,
            }
            continue

        if not has_report:
            reason = (
                _signal_description(-returncode)
                if returncode < 0
                else f"Process exited with code {returncode} but produced no report"
            )
            diag = _get_diagnostics()
            print(f"⚠️  {test_file}: {reason}")
            print(diag)

            details = f"{reason}\n\n=== SYSTEM DIAGNOSTICS ===\n{diag}\n\n"
            if stdout_data:
                details += "=== STDOUT (last 2000 chars) ===\n"
                details += stdout_data.decode("utf-8", errors="replace")[-2000:] + "\n"
            if stderr_data:
                details += "=== STDERR (last 2000 chars) ===\n"
                details += stderr_data.decode("utf-8", errors="replace")[-2000:] + "\n"

            error_report = _create_error_report("crash", file_name, reason, details)
            error_report.write(report_file)
            xml_reports.append(error_report)
            failed_tests.append(test_file)
            test_status[test_file] = {
                "errors": 1,
                "failures": 0,
                "skipped": 0,
                "tests": 1,
                "result": "CRASHED",
                "time_elapsed": 0.0,
                "wall_time": wall_time,
            }
            continue

        # -- Report file exists: parse actual test results -----------------
        if kill_reason in ("shutdown_hang", "timeout"):
            print(f"⚠️  {test_file}: shutdown hanged (killed after {wall_time:.0f}s, test had completed)")

        try:
            report = JUnitXml.fromfile(report_file)
            for suite in report:
                if suite.name == "pytest":
                    suite.name = os.path.splitext(file_name)[0]
            report.write(report_file)
            xml_reports.append(report)

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
                "wall_time": wall_time,
            }
            continue

        has_test_failures = errors > 0 or failures > 0
        shutdown_hanged = kill_reason in ("shutdown_hang", "timeout") and not has_test_failures

        if has_test_failures or (returncode != 0 and not shutdown_hanged):
            failed_tests.append(test_file)

        if shutdown_hanged:
            result = "passed (shutdown hanged)"
        elif has_test_failures:
            result = "FAILED"
        else:
            result = "passed"

        test_status[test_file] = {
            "errors": errors,
            "failures": failures,
            "skipped": skipped,
            "tests": tests,
            "result": result,
            "time_elapsed": time_elapsed,
            "wall_time": wall_time,
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
    num_passing = len([p for p in test_files if test_status[p]["result"].startswith("passed")])
    num_failing = len([p for p in test_files if test_status[p]["result"] == "FAILED"])
    num_timeout = len([p for p in test_files if test_status[p]["result"] == "TIMEOUT"])
    num_crashed = len([p for p in test_files if test_status[p]["result"] == "CRASHED"])
    num_startup_hang = len([p for p in test_files if test_status[p]["result"] == "STARTUP_HANG"])

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
    summary_str += f"Startup Hang: {num_startup_hang}\n"
    summary_str += f"Timeout: {num_timeout}\n"
    summary_str += f"Passing Percentage: {passing_percentage:.2f}%\n"

    total_wall = sum(test_status[test_path]["wall_time"] for test_path in test_files)
    total_test = sum(test_status[test_path]["time_elapsed"] for test_path in test_files)

    summary_str += f"Total Wall Time: {total_wall // 3600:.0f}h{total_wall // 60 % 60:.0f}m{total_wall % 60:.2f}s\n"
    summary_str += f"Total Test Time: {total_test // 3600:.0f}h{total_test // 60 % 60:.0f}m{total_test % 60:.2f}s"

    summary_str += "\n\n=======================\n"
    summary_str += "Per Test Result Summary\n"
    summary_str += "=======================\n"

    per_test_result_table = PrettyTable(field_names=["Test Path", "Result", "Test (s)", "Wall (s)", "# Tests"])
    per_test_result_table.align["Test Path"] = "l"
    per_test_result_table.align["Test (s)"] = "r"
    per_test_result_table.align["Wall (s)"] = "r"
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
                f"{test_status[test_path]['wall_time']:0.2f}",
                f"{num_tests_passed}/{test_status[test_path]['tests']}",
            ]
        )

    summary_str += per_test_result_table.get_string()

    # Print summary to console and log file
    print(summary_str)

    # Exit pytest after custom execution to prevent normal pytest from overwriting our report
    pytest.exit(
        "Custom test execution completed",
        returncode=0 if (num_failing == 0 and num_timeout == 0 and num_crashed == 0 and num_startup_hang == 0) else 1,
    )
