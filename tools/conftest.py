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

import hang_dump

HANG_DUMP_PASSES = 2
HANG_DUMP_GRACE = 3


def _capture_system_diagnostics():
    sections = []
    try:
        result = subprocess.run(["ps", "auxf"], capture_output=True, text=True, timeout=5)
        if result.stdout:
            sections.append(f"--- process tree (ps auxf) ---\n{result.stdout.strip()}")
    except Exception as exc:
        sections.append(f"--- process tree --- FAILED: {exc}")
    return "\n\n".join(sections)


def _drain_ready_output(process, stdout_fd, stderr_fd, timeout=0.1):
    stdout_chunk = b""
    stderr_chunk = b""
    try:
        ready_fds, _, _ = select.select([stdout_fd, stderr_fd], [], [], timeout)
        for fd in ready_fds:
            with contextlib.suppress(OSError):
                if fd == stdout_fd:
                    chunk = process.stdout.read(1024)
                    if chunk:
                        stdout_chunk += chunk
                        sys.stdout.buffer.write(chunk)
                        sys.stdout.buffer.flush()
                elif fd == stderr_fd:
                    chunk = process.stderr.read(1024)
                    if chunk:
                        stderr_chunk += chunk
                        sys.stderr.buffer.write(chunk)
                        sys.stderr.buffer.flush()
    except OSError:
        time.sleep(timeout)
    return stdout_chunk, stderr_chunk


def _dump_hung_process_stacks(process, stdout_fd, stderr_fd, env):
    stdout_data = b""
    stderr_data = b""
    dumps = []
    dump_file = env.get(hang_dump.DUMP_PATH_ENV_VAR, "")
    if hang_dump.DUMP_SIGNAL is None or not dump_file:
        return "", stdout_data, stderr_data

    for _ in range(HANG_DUMP_PASSES):
        start = hang_dump.size(dump_file)
        try:
            os.kill(process.pid, hang_dump.DUMP_SIGNAL)
        except OSError:
            break
        deadline = time.time() + HANG_DUMP_GRACE
        while time.time() < deadline:
            stdout_chunk, stderr_chunk = _drain_ready_output(process, stdout_fd, stderr_fd)
            stdout_data += stdout_chunk
            stderr_data += stderr_chunk
        if dumped := hang_dump.read_since(dump_file, start):
            dumps.append(dumped)
        if process.poll() is not None:
            break

    if not dumps:
        return "", stdout_data, stderr_data
    body = "\n".join(f"----- dump {index} of {len(dumps)} -----\n{dump}" for index, dump in enumerate(dumps, 1))
    return f"=== HANG STACK DUMP (all threads) ===\n{body}", stdout_data, stderr_data


def capture_test_output_with_timeout(cmd, timeout, env, startup_deadline=0, report_file=""):
    stdout_data = b""
    stderr_data = b""
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

        for fd in (stdout_fd, stderr_fd):
            flags = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
    except ImportError:
        pass

    start_time = time.time()
    startup_done = startup_deadline <= 0
    shutdown_deadline = 0.0
    while process.poll() is None:
        elapsed = time.time() - start_time
        if not startup_done and (b"AppLauncher initialization complete" in stderr_data or b"collected " in stdout_data):
            startup_done = True
        if report_file and not shutdown_deadline and os.path.exists(report_file):
            shutdown_deadline = time.time() + 30

        kill_reason = ""
        if not startup_done and elapsed > startup_deadline:
            kill_reason = "startup_hang"
        elif shutdown_deadline and time.time() > shutdown_deadline:
            kill_reason = "shutdown_hang"
        elif elapsed > timeout:
            kill_reason = "timeout"

        if kill_reason:
            pre_kill_diag = _capture_system_diagnostics()
            hang_stacks, dump_stdout, dump_stderr = _dump_hung_process_stacks(process, stdout_fd, stderr_fd, env)
            stdout_data += dump_stdout
            stderr_data += dump_stderr
            if hang_stacks:
                pre_kill_diag = f"{hang_stacks}\n\n{pre_kill_diag}"
            try:
                os.killpg(pgid, signal.SIGKILL)
            except OSError:
                process.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                remaining_stdout, remaining_stderr = process.communicate(timeout=5)
                stdout_data += remaining_stdout
                stderr_data += remaining_stderr
            return -1, stdout_data, stderr_data, kill_reason, time.time() - start_time, pre_kill_diag

        stdout_chunk, stderr_chunk = _drain_ready_output(process, stdout_fd, stderr_fd)
        stdout_data += stdout_chunk
        stderr_data += stderr_chunk

    remaining_stdout, remaining_stderr = process.communicate()
    stdout_data += remaining_stdout
    stderr_data += remaining_stderr
    return process.returncode, stdout_data, stderr_data, "", time.time() - start_time, ""
