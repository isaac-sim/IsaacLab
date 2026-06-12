# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Low-level subprocess execution and JUnit report I/O for one pytest run.

:class:`ProcessCapture` launches a command, streams its output live, and kills it
on a startup hang, hard timeout, or shutdown hang, returning enough to classify
the outcome. :class:`JUnitReport` reads a produced report or synthesizes an error
report when the process produced none. Both are device- and unit-agnostic; the
unit-level orchestration that uses them is :class:`~test_runner.execution.UnitRunner`.
"""

from __future__ import annotations

import contextlib
import os
import select
import signal
import subprocess
import sys
import time

from junitparser import Error, JUnitXml, TestCase, TestSuite

_SIGNAL_DESCRIPTIONS = {
    1: "SIGHUP — session leader exit or orphaned process cleanup",
    6: "SIGABRT",
    9: "SIGKILL — likely OOM killed",
    11: "SIGSEGV — segmentation fault",
    15: "SIGTERM",
}
_DIAG_LIMIT = 10000  # chars; truncate diagnostics so a crash dump can't flood the log


class ProcessCapture:
    """Run a pytest subprocess with hang/timeout detection and live output streaming."""

    def __init__(self, shutdown_grace_period: int):
        # Seconds to allow for clean exit after the report file appears, before
        # killing a process that hung in SimulationApp.close() / Kit shutdown.
        self._grace = shutdown_grace_period

    @staticmethod
    def signal_description(sig: int) -> str:
        """Human-readable reason a process was killed by signal ``sig``."""
        base = f"Process killed by signal {sig}"
        desc = _SIGNAL_DESCRIPTIONS.get(sig)
        return f"{base} ({desc})" if desc else base

    @staticmethod
    def _system_diagnostics() -> str:
        """GPU/memory/process snapshot for crash investigation. Never raises."""
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

    def diagnostics(self, pre_kill: str = "") -> str:
        """Return diagnostics (a pre-kill snapshot if given, else a fresh one), truncated."""
        diag = pre_kill or self._system_diagnostics()
        return diag[:_DIAG_LIMIT] + "\n... (truncated)" if len(diag) > _DIAG_LIMIT else diag

    def run(self, cmd, timeout, env, startup_deadline=0, report_file=""):
        """Run ``cmd``, streaming output, and kill it on a hang or timeout.

        Args:
            cmd: argv to execute.
            timeout: hard wall-clock limit [s].
            env: subprocess environment.
            startup_deadline: if > 0, kill when neither AppLauncher-init nor
                pytest-collection markers appear within this many seconds.
            report_file: when set, the process gets only the configured grace
                period to exit after this file appears on disk.

        Returns:
            ``(returncode, stdout_bytes, stderr_bytes, kill_reason, wall_time,
            pre_kill_diag)``; ``kill_reason`` is ``""`` for a clean exit, else
            ``"startup_hang"`` / ``"timeout"`` / ``"shutdown_hang"``.
        """
        stdout_data = b""
        stderr_data = b""
        process = None
        try:
            # Own session so orphaned Kit children can't SIGHUP the next test's group.
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
                    shutdown_deadline = time.time() + self._grace

                kill_reason = None
                if not startup_done and elapsed > startup_deadline:
                    kill_reason = "startup_hang"
                elif shutdown_deadline and time.time() > shutdown_deadline:
                    kill_reason = "shutdown_hang"
                elif elapsed > timeout:
                    kill_reason = "timeout"

                if kill_reason:
                    pre_kill_diag = self._system_diagnostics()
                    try:  # kill the whole group (test + any Kit children)
                        os.killpg(pgid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError, OSError):
                        process.kill()
                    try:
                        rem_out, rem_err = process.communicate(timeout=5)
                        stdout_data += rem_out
                        stderr_data += rem_err
                    except subprocess.TimeoutExpired:
                        pass
                    return -1, stdout_data, stderr_data, kill_reason, time.time() - start_time, pre_kill_diag

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

            try:  # drain output written just before/after exit
                rem_out, rem_err = process.communicate(timeout=10)
                stdout_data += rem_out
                stderr_data += rem_err
            except Exception:
                pass
            try:  # reap any orphaned Kit/Isaac Sim children
                os.killpg(pgid, signal.SIGKILL)
                time.sleep(1)
            except (ProcessLookupError, PermissionError, OSError):
                pass
            return process.returncode, stdout_data, stderr_data, "", time.time() - start_time, ""
        except Exception as e:
            if process is not None and process.poll() is None:
                process.kill()
                with contextlib.suppress(Exception):
                    rem_out, rem_err = process.communicate(timeout=5)
                    stdout_data += rem_out
                    stderr_data += rem_err
            stdout_data += f"\n[capture error: {e}]\n".encode()
            return -1, stdout_data, stderr_data, "", 0.0, ""


class JUnitReport:
    """Read a unit's JUnit report, or synthesize one when the process produced none."""

    @staticmethod
    def error(prefix: str, file_name: str, message: str, details: str) -> JUnitXml:
        """Build a one-case JUnit report standing in for a run that wrote no report."""
        suite_name = os.path.splitext(file_name)[0]
        suite = TestSuite(name=f"{prefix}_{suite_name}")
        case = TestCase(name="test_execution", classname=suite_name)
        result = Error(message=message)
        result.text = details
        case.result = result
        suite.add_testcase(case)
        report = JUnitXml()
        report.add_testsuite(suite)
        return report

    @staticmethod
    def read(report_file: str, file_name: str):
        """Parse a JUnit report; rename its synthetic ``pytest`` suite to the file.

        Returns:
            ``(report, errors, failures, skipped, tests, time_elapsed)``.
        """
        report = JUnitXml.fromfile(report_file)
        for suite in report:
            if suite.name == "pytest":
                suite.name = os.path.splitext(file_name)[0]
        report.write(report_file)
        return (
            report,
            int(report.errors) if report.errors is not None else 0,
            int(report.failures) if report.failures is not None else 0,
            int(report.skipped) if report.skipped is not None else 0,
            int(report.tests) if report.tests is not None else 0,
            float(report.time) if report.time is not None else 0.0,
        )
