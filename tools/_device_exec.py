# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Executor for the per-file CI test runner.

Runs one :class:`~_device_plan.Unit` as its own pytest subprocess and owns the
subprocess lifecycle: streaming capture, startup-hang / hard-timeout /
shutdown-hang detection, fresh-process retries, and JUnit report parsing.

Device selection is the planner's job, expressed as the unit's mask: the
executor sets ``ISAACLAB_TEST_DEVICES`` to that mask (``test_devices()`` reads
it to pick the device variants) and, for a unit whose mask lacks cpu — an mgpu
shard or the cuda unit of a can't-mix split — injects the ``_agnostic_select``
plugin so device-agnostic tests do not run there. The executor never reads a
marker or a ``-k`` selector.
"""

from __future__ import annotations

import contextlib
import os
import select
import signal
import subprocess
import sys
import time
from dataclasses import dataclass

from junitparser import Error, JUnitXml, TestCase, TestSuite

from _device_plan import Unit  # isort: skip

COLD_CACHE_BUFFER = 700
"""Extra seconds added to the first camera-enabled test's hard timeout.

The first test that uses ``enable_cameras=True`` may compile shaders during its
run (~600 s).  This buffer prevents that from being misreported as a test
timeout.  Only the first such test gets the extension — after it runs, the
on-disk cache is populated.
"""

STARTUP_DEADLINE = 120
"""Seconds to wait for AppLauncher init or pytest collection before declaring a
startup hang.

AppLauncher prints ``[ISAACLAB] AppLauncher initialization complete`` to
``sys.__stderr__`` (never suppressed) when Kit finishes initializing, and pytest
prints ``collected N items`` to stdout after collection.  If neither appears
within this deadline the process is treated as hung.  Kit startup can exceed
60 s on cold CI workers, so this catches real startup hangs without killing
legitimate slow launches.
"""

STARTUP_HANG_RETRIES = 2
"""Number of times to retry a test that hangs during startup before giving up."""

TIMEOUT_RETRIES = 0
"""Number of times to retry a test that reaches its hard timeout before giving up."""

PROCESS_FAILURE_RETRIES_BY_FILE = {
    "test_visualizer_integration_physx.py": 4,
    "test_visualizer_integration_newton.py": 4,
    "test_visualizer_tiled_integration_physx.py": 4,
    "test_visualizer_tiled_integration_newton.py": 4,
}
"""Extra fresh-process attempts for visualizer tests that can enter stale render states."""

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


def _read_test_report(report_file, file_name):
    """Read a pytest JUnit report and return its summary fields."""
    report = JUnitXml.fromfile(report_file)
    for suite in report:
        if suite.name == "pytest":
            suite.name = os.path.splitext(file_name)[0]
    report.write(report_file)

    errors = int(report.errors) if report.errors is not None else 0
    failures = int(report.failures) if report.failures is not None else 0
    skipped = int(report.skipped) if report.skipped is not None else 0
    tests = int(report.tests) if report.tests is not None else 0
    time_elapsed = float(report.time) if report.time is not None else 0.0
    return report, errors, failures, skipped, tests, time_elapsed


def _retry_failed_test_in_fresh_process(
    *,
    test_file,
    file_name,
    cmd,
    timeout,
    env,
    startup_deadline,
    report_file,
    report,
    errors,
    failures,
    skipped,
    tests,
    time_elapsed,
    returncode,
    stdout_data,
    stderr_data,
    kill_reason,
    wall_time,
    pre_kill_diag,
):
    """Retry selected failed test files in a fresh subprocess."""
    has_test_failures = errors > 0 or failures > 0
    process_failure_attempts = 0
    max_process_failure_retries = PROCESS_FAILURE_RETRIES_BY_FILE.get(file_name, 0)

    while has_test_failures and process_failure_attempts < max_process_failure_retries:
        process_failure_attempts += 1
        print(
            f"⚠️  {test_file}: failed in subprocess"
            f" (attempt {process_failure_attempts}/{max_process_failure_retries + 1}), retrying in fresh process..."
        )
        with contextlib.suppress(FileNotFoundError):
            os.remove(report_file)

        returncode, stdout_data, stderr_data, kill_reason, wall_time, pre_kill_diag = capture_test_output_with_timeout(
            cmd, timeout, env, startup_deadline=startup_deadline, report_file=report_file
        )
        if not os.path.exists(report_file):
            break

        try:
            report, errors, failures, skipped, tests, time_elapsed = _read_test_report(report_file, file_name)
            has_test_failures = errors > 0 or failures > 0
        except Exception as e:
            print(f"Error reading retry test report {report_file}: {e}")
            has_test_failures = True
            errors = 1
            failures = 0
            skipped = 0
            tests = 0
            time_elapsed = 0.0
            break

    return (
        report,
        errors,
        failures,
        skipped,
        tests,
        time_elapsed,
        returncode,
        stdout_data,
        stderr_data,
        kill_reason,
        wall_time,
        pre_kill_diag,
        has_test_failures,
    )


@dataclass
class _UnitContext:
    """Per-file inputs shared across the units the runner drives for one file.

    Attributes:
        test_file: Path to the test file being driven.
        file_name: Basename of ``test_file`` (used for JUnit naming).
        workspace_root: Repository root; passed to pytest's ``--config-file`` and
            used to locate the ``_agnostic_select`` plugin.
        isaacsim_ci: Whether ``ISAACSIM_CI_SHORT`` is active; toggles the
            ``-m isaacsim_ci`` selector.
        timeout: Per-unit hard timeout in seconds.
        startup_deadline: Per-unit startup-hang deadline in seconds.
        env: Base environment for the pytest subprocess; :func:`run_unit` copies
            it and adds the per-unit ``ISAACLAB_TEST_DEVICES`` mask.
    """

    test_file: str
    file_name: str
    workspace_root: str
    isaacsim_ci: bool
    timeout: int
    startup_deadline: int
    env: dict


_RESULT_PRIORITY = {
    "STARTUP_HANG": 5,
    "CRASHED": 4,
    "TIMEOUT": 3,
    "FAILED": 2,
    "passed (shutdown hanged)": 1,
    "passed": 0,
}


def merge_unit_status(prev: dict | None, new: dict) -> dict:
    """Merge per-unit status dicts into a single per-file entry.

    Counters (``errors``, ``failures``, ``skipped``, ``tests``,
    ``time_elapsed``, ``wall_time``) are summed. ``result`` becomes the more
    severe of the two via :data:`_RESULT_PRIORITY`.
    """
    if prev is None:
        return new
    return {
        "errors": prev["errors"] + new["errors"],
        "failures": prev["failures"] + new["failures"],
        "skipped": prev["skipped"] + new["skipped"],
        "tests": prev["tests"] + new["tests"],
        "time_elapsed": prev["time_elapsed"] + new["time_elapsed"],
        "wall_time": prev["wall_time"] + new["wall_time"],
        "result": prev["result"]
        if _RESULT_PRIORITY.get(prev["result"], 0) >= _RESULT_PRIORITY.get(new["result"], 0)
        else new["result"],
    }


def _build_unit_cmd(ctx: _UnitContext, unit: Unit) -> tuple[list[str], dict, str, str]:
    """Build the pytest invocation for one unit (no subprocess).

    Device-variant selection is the unit's mask, set as ``ISAACLAB_TEST_DEVICES``
    for ``test_devices()`` to read — there is no ``-k``. A mask without cpu
    (position 0) additionally injects the ``_agnostic_select`` plugin so
    device-agnostic tests do not run there.

    Args:
        ctx: Static per-file context (paths, isaacsim_ci flag, base env).
        unit: The work unit to run.

    Returns:
        ``(cmd, env, report_file, pass_file_label)`` — the argv, the per-unit
        environment, the JUnit report path, and the label used in error reports.
    """
    # Split units of one file share a slug, so suffix the report with the mask to
    # keep both XMLs; a non-split unit (one per file) keeps the suffix-free name
    # the mgpu aggregator unslugs back to a path.
    report_suffix = f"-{unit.mask}" if unit.split else ""
    pass_file_label = f"{ctx.file_name}{report_suffix}"
    # Slug the full test path (not just the basename) into the report filename so
    # two concurrent shards running same-basename files (e.g.
    # ``isaaclab_newton/.../test_articulation.py`` vs
    # ``isaaclab_physx/.../test_articulation.py``) don't write to the same path
    # inside the shared ``/workspace/isaaclab`` mount and trigger false
    # shutdown_hang detections in sibling shards via the report-file existence check.
    report_slug = str(ctx.test_file).replace("/", "__").replace("\\", "__")
    report_file = f"tests/test-reports-{report_slug}{report_suffix}.xml"

    env = dict(ctx.env)
    env["ISAACLAB_TEST_DEVICES"] = unit.mask

    inject_agnostic = unit.mask[:1] == "0"
    if inject_agnostic:
        tools_dir = os.path.join(ctx.workspace_root, "tools")
        env["PYTHONPATH"] = tools_dir + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-s",
        "-v",  # per-test names in the log: if a file hangs, the last name pinpoints the culprit
        "--no-header",
        f"--config-file={ctx.workspace_root}/pyproject.toml",
        f"--junitxml={report_file}",
        "--tb=short",
    ]
    if inject_agnostic:
        cmd += ["-p", "_agnostic_select"]
    if ctx.isaacsim_ci:
        cmd += ["-m", "isaacsim_ci"]
    cmd.append(str(ctx.test_file))
    return cmd, env, report_file, pass_file_label


def run_unit(ctx: _UnitContext, unit: Unit) -> tuple[JUnitXml | None, dict, bool]:
    """Drive one pytest subprocess for ``unit`` and return its results.

    Args:
        ctx: Static per-file context (paths, timeouts, base env).
        unit: The work unit to run; its mask becomes ``ISAACLAB_TEST_DEVICES``.

    Returns:
        A 3-tuple ``(xml_report, status_dict, was_failure)``:
            * ``xml_report``: parsed JUnit XML, or ``None`` if the unit produced
              no report (e.g. startup hang).
            * ``status_dict``: per-unit counters compatible with the entries
              appended to ``test_status``.
            * ``was_failure``: whether the unit should add ``ctx.test_file`` to
              the ``failed_tests`` list.
    """
    cmd, env, report_file, pass_file_label = _build_unit_cmd(ctx, unit)
    report_suffix = f"-{unit.mask}" if unit.split else ""  # display suffix for this unit's log lines

    # -- Run with retry on startup hang or hard timeout -----------------
    returncode, stdout_data, stderr_data, kill_reason = -1, b"", b"", ""
    wall_time, pre_kill_diag = 0.0, ""
    startup_hang_attempts = 0
    timeout_attempts = 0
    while True:
        with contextlib.suppress(FileNotFoundError):
            os.remove(report_file)

        returncode, stdout_data, stderr_data, kill_reason, wall_time, pre_kill_diag = capture_test_output_with_timeout(
            cmd, ctx.timeout, env, startup_deadline=ctx.startup_deadline, report_file=report_file
        )

        has_report = os.path.exists(report_file)

        if kill_reason == "startup_hang" and startup_hang_attempts < STARTUP_HANG_RETRIES:
            startup_hang_attempts += 1
            print(
                f"⚠️  {ctx.test_file}{report_suffix}: startup hang detected after {ctx.startup_deadline}s"
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
                f"⚠️  {ctx.test_file}{report_suffix}: timeout detected after {ctx.timeout}s"
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
        print(f"⚠️  {ctx.test_file}{report_suffix}: startup hang after {STARTUP_HANG_RETRIES + 1} attempt(s)")
        print(diag)

        msg = f"Startup hang after {ctx.startup_deadline}s (retried {STARTUP_HANG_RETRIES} time(s))"
        details = f"{msg}\n\n=== SYSTEM DIAGNOSTICS ===\n{diag}\n\n"
        if stderr_data:
            details += "=== STDERR (last 5000 chars) ===\n"
            details += stderr_data.decode("utf-8", errors="replace")[-5000:] + "\n"
        if stdout_data:
            details += "=== STDOUT (last 2000 chars) ===\n"
            details += stdout_data.decode("utf-8", errors="replace")[-2000:] + "\n"

        error_report = _create_error_report("startup_hang", pass_file_label, msg, details)
        error_report.write(report_file)
        return (
            error_report,
            {
                "errors": 1,
                "failures": 0,
                "skipped": 0,
                "tests": 1,
                "result": "STARTUP_HANG",
                "time_elapsed": 0.0,
                "wall_time": wall_time,
            },
            True,
        )

    if kill_reason == "timeout" and not has_report:
        diag = _get_diagnostics(pre_kill_diag)
        print(f"Test {ctx.test_file}{report_suffix} timed out after {ctx.timeout} seconds...")
        print(diag)

        msg = f"Timeout after {ctx.timeout} seconds (retried {timeout_attempts} time(s))"
        details = f"{msg}\n\n=== SYSTEM DIAGNOSTICS ===\n{diag}\n\n"
        if stdout_data:
            details += "=== STDOUT (last 5000 chars) ===\n"
            details += stdout_data.decode("utf-8", errors="replace")[-5000:] + "\n"
        if stderr_data:
            details += "=== STDERR (last 5000 chars) ===\n"
            details += stderr_data.decode("utf-8", errors="replace")[-5000:] + "\n"

        error_report = _create_error_report("timeout", pass_file_label, msg, details)
        error_report.write(report_file)
        return (
            error_report,
            {
                "errors": 1,
                "failures": 0,
                "skipped": 0,
                "tests": 1,
                "result": "TIMEOUT",
                "time_elapsed": ctx.timeout,
                "wall_time": wall_time,
            },
            True,
        )

    if not has_report:
        reason = (
            _signal_description(-returncode)
            if returncode < 0
            else f"Process exited with code {returncode} but produced no report"
        )
        diag = _get_diagnostics()
        print(f"⚠️  {ctx.test_file}{report_suffix}: {reason}")
        print(diag)

        details = f"{reason}\n\n=== SYSTEM DIAGNOSTICS ===\n{diag}\n\n"
        if stdout_data:
            details += "=== STDOUT (last 2000 chars) ===\n"
            details += stdout_data.decode("utf-8", errors="replace")[-2000:] + "\n"
        if stderr_data:
            details += "=== STDERR (last 2000 chars) ===\n"
            details += stderr_data.decode("utf-8", errors="replace")[-2000:] + "\n"

        error_report = _create_error_report("crash", pass_file_label, reason, details)
        error_report.write(report_file)
        return (
            error_report,
            {
                "errors": 1,
                "failures": 0,
                "skipped": 0,
                "tests": 1,
                "result": "CRASHED",
                "time_elapsed": 0.0,
                "wall_time": wall_time,
            },
            True,
        )

    # -- Report file exists: parse actual test results -----------------
    if kill_reason in ("shutdown_hang", "timeout"):
        print(f"⚠️  {ctx.test_file}{report_suffix}: shutdown hanged (killed after {wall_time:.0f}s, test had completed)")

    try:
        report, errors, failures, skipped, tests, time_elapsed = _read_test_report(report_file, pass_file_label)
    except Exception as e:
        print(f"Error reading test report {report_file}: {e}")
        return (
            None,
            {
                "errors": 1,
                "failures": 0,
                "skipped": 0,
                "tests": 0,
                "result": "FAILED",
                "time_elapsed": 0.0,
                "wall_time": wall_time,
            },
            True,
        )

    (
        report,
        errors,
        failures,
        skipped,
        tests,
        time_elapsed,
        returncode,
        stdout_data,
        stderr_data,
        kill_reason,
        wall_time,
        pre_kill_diag,
        has_test_failures,
    ) = _retry_failed_test_in_fresh_process(
        test_file=ctx.test_file,
        file_name=ctx.file_name,
        cmd=cmd,
        timeout=ctx.timeout,
        env=env,
        startup_deadline=ctx.startup_deadline,
        report_file=report_file,
        report=report,
        errors=errors,
        failures=failures,
        skipped=skipped,
        tests=tests,
        time_elapsed=time_elapsed,
        returncode=returncode,
        stdout_data=stdout_data,
        stderr_data=stderr_data,
        kill_reason=kill_reason,
        wall_time=wall_time,
        pre_kill_diag=pre_kill_diag,
    )

    shutdown_hanged = kill_reason in ("shutdown_hang", "timeout") and not has_test_failures
    was_failure = has_test_failures or (returncode != 0 and not shutdown_hanged)

    if shutdown_hanged:
        result = "passed (shutdown hanged)"
    elif has_test_failures:
        result = "FAILED"
    else:
        result = "passed"

    return (
        report,
        {
            "errors": errors,
            "failures": failures,
            "skipped": skipped,
            "tests": tests,
            "result": result,
            "time_elapsed": time_elapsed,
            "wall_time": wall_time,
        },
        was_failure,
    )
