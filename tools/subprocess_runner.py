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
    # Import here to avoid circular dependency; SHUTDOWN_GRACE_PERIOD is defined in conftest.
    # We define a local default that matches conftest's constant.
    _SHUTDOWN_GRACE_PERIOD = 30

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
                shutdown_deadline = time.time() + _SHUTDOWN_GRACE_PERIOD

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


def classify_failure_phase(
    stdout: str, stderr: str, exit_code: int, wall_time_s: float, timeout_s: float
) -> str | None:
    """Classify the failure phase of a benchmark run.

    Priority order (highest to lowest):
    1. oom: exit_code == 137 or "oom-kill" in stderr
    2. hang: wall_time_s >= timeout_s * 0.95
    3. import: "Traceback" in stdout/stderr AND no "AppLauncher" in stdout
    4. driver: "CudaError" or "CUDA_ERROR_" in stdout/stderr (case-sensitive)
    5. init: "AppLauncher initialization complete" in stdout but no "Step Frametimes" in stdout
    6. runtime: exit_code != 0 and "Step Frametimes" in stdout
    7. null: exit_code == 0

    Args:
        stdout: Captured standard output as a string.
        stderr: Captured standard error as a string.
        exit_code: Process exit code.
        wall_time_s: Measured wall-clock time in seconds.
        timeout_s: Configured timeout in seconds.

    Returns:
        A failure phase string or None for a clean exit.
    """
    combined = stdout + stderr

    # 1. OOM
    if exit_code == 137 or "oom-kill" in stderr:
        return "oom"

    # 2. Hang
    if wall_time_s >= timeout_s * 0.95:
        return "hang"

    # 3. Import error
    if "Traceback" in combined and "AppLauncher" not in stdout:
        return "import"

    # 4. Driver error
    if "CudaError" in combined or "CUDA_ERROR_" in combined:
        return "driver"

    # 5. Init failure
    if exit_code != 0 and "AppLauncher initialization complete" in stdout and "Step Frametimes" not in stdout:
        return "init"

    # 6. Runtime failure (partial run then crash)
    if exit_code != 0 and "Step Frametimes" in stdout:
        return "runtime"

    # 7. Clean exit
    return None


def run_benchmark(cmd: list, timeout_s: float) -> dict:
    """Run a benchmark command and return structured result.

    Args:
        cmd: Command list to execute.
        timeout_s: Hard timeout in seconds.

    Returns:
        Dict with keys:
            - exit_code (int): Process exit code.
            - stdout_tail (str): Last 2000 characters of combined stdout.
            - wall_time_s (float): Measured wall-clock time in seconds.
            - startup_time_s (float): Startup time in seconds (0.0 stub for POC).
            - failure_phase (str | None): Classified failure phase.
    """
    env = os.environ.copy()

    returncode, stdout_bytes, stderr_bytes, kill_reason, wall_time, _ = capture_test_output_with_timeout(
        cmd,
        timeout=timeout_s,
        env=env,
        startup_deadline=0,
        report_file="",
    )

    stdout_str = stdout_bytes.decode("utf-8", errors="replace")
    stderr_str = stderr_bytes.decode("utf-8", errors="replace")

    # Use kill_reason to override exit_code for hang detection
    effective_exit_code = returncode
    if kill_reason in ("timeout", "startup_hang"):
        effective_exit_code = -1

    failure_phase = classify_failure_phase(
        stdout=stdout_str,
        stderr=stderr_str,
        exit_code=effective_exit_code,
        wall_time_s=wall_time,
        timeout_s=timeout_s,
    )

    combined_output = stdout_str
    stdout_tail = combined_output[-2000:] if len(combined_output) > 2000 else combined_output

    return {
        "exit_code": returncode,
        "stdout_tail": stdout_tail,
        "wall_time_s": wall_time,
        "startup_time_s": 0.0,
        "failure_phase": failure_phase,
    }
