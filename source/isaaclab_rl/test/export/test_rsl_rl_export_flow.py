# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export pipeline integration tests.

Each test calls ``export.py`` as a subprocess so that Isaac Sim's AppLauncher
is fully isolated per task and the export logic is not duplicated here.
The export artifacts land in the default checkpoint directory; only the
per-task export subdirectory is removed after each test.
"""

import os
import shutil
import subprocess
import time

import pytest

# Root of the repository (three levels up from this file).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_EXPORT_SCRIPT = os.path.join("scripts", "reinforcement_learning", "leapp", "rsl_rl", "export.py")
_EXPORT_TIMEOUT = 600
_OUTPUT_TAIL_CHARS = 5000
_TIMING_PREFIXES = ("[EXPORT_TEST_TIMING]", "[LEAPP_EXPORT_TIMING]", "[INFO] LEAPP version:")


# Tasks with confirmed pretrained checkpoints (Direct and no-checkpoint tasks excluded).
TASKS = [
    # Classic
    "Isaac-Ant-v0",
    "Isaac-Cartpole-v0",
    # Navigation
    "Isaac-Navigation-Flat-Anymal-C-v0",
    "Isaac-Navigation-Flat-Anymal-C-Play-v0",
    # Locomotion Velocity
    "Isaac-Velocity-Flat-Anymal-B-v0",
    "Isaac-Velocity-Flat-Anymal-B-Play-v0",
    "Isaac-Velocity-Rough-Anymal-B-v0",
    "Isaac-Velocity-Rough-Anymal-B-Play-v0",
    "Isaac-Velocity-Flat-Anymal-C-v0",
    "Isaac-Velocity-Flat-Anymal-C-Play-v0",
    "Isaac-Velocity-Rough-Anymal-C-v0",
    "Isaac-Velocity-Rough-Anymal-C-Play-v0",
    "Isaac-Velocity-Flat-Anymal-D-v0",
    "Isaac-Velocity-Flat-Anymal-D-Play-v0",
    "Isaac-Velocity-Rough-Anymal-D-v0",
    "Isaac-Velocity-Rough-Anymal-D-Play-v0",
    "Isaac-Velocity-Flat-Cassie-v0",
    "Isaac-Velocity-Flat-Cassie-Play-v0",
    "Isaac-Velocity-Rough-Cassie-v0",
    "Isaac-Velocity-Rough-Cassie-Play-v0",
    "Isaac-Velocity-Flat-G1-v0",
    "Isaac-Velocity-Flat-G1-Play-v0",
    "Isaac-Velocity-Rough-G1-v0",
    "Isaac-Velocity-Rough-G1-Play-v0",
    "Isaac-Velocity-Flat-H1-v0",
    "Isaac-Velocity-Flat-H1-Play-v0",
    "Isaac-Velocity-Rough-H1-v0",
    "Isaac-Velocity-Rough-H1-Play-v0",
    "Isaac-Velocity-Flat-Spot-v0",
    "Isaac-Velocity-Flat-Spot-Play-v0",
    "Isaac-Velocity-Flat-Unitree-A1-v0",
    "Isaac-Velocity-Flat-Unitree-A1-Play-v0",
    "Isaac-Velocity-Rough-Unitree-A1-v0",
    "Isaac-Velocity-Rough-Unitree-A1-Play-v0",
    "Isaac-Velocity-Flat-Unitree-Go1-v0",
    "Isaac-Velocity-Flat-Unitree-Go1-Play-v0",
    "Isaac-Velocity-Rough-Unitree-Go1-v0",
    "Isaac-Velocity-Rough-Unitree-Go1-Play-v0",
    "Isaac-Velocity-Flat-Unitree-Go2-v0",
    "Isaac-Velocity-Flat-Unitree-Go2-Play-v0",
    "Isaac-Velocity-Rough-Unitree-Go2-v0",
    "Isaac-Velocity-Rough-Unitree-Go2-Play-v0",
    # Manipulation Reach
    "Isaac-Reach-Franka-v0",
    "Isaac-Reach-Franka-Play-v0",
    "Isaac-Reach-UR10-v0",
    "Isaac-Reach-UR10-Play-v0",
    # Manipulation Lift
    "Isaac-Lift-Cube-Franka-v0",
    "Isaac-Lift-Cube-Franka-Play-v0",
    # Manipulation Cabinet
    "Isaac-Open-Drawer-Franka-v0",
    "Isaac-Open-Drawer-Franka-Play-v0",
    # Dexsuite
    "Isaac-Dexsuite-Kuka-Allegro-Reorient-v0",
    "Isaac-Dexsuite-Kuka-Allegro-Reorient-Play-v0",
    "Isaac-Dexsuite-Kuka-Allegro-Lift-v0",
    "Isaac-Dexsuite-Kuka-Allegro-Lift-Play-v0",
]


def _export_dir(task_name: str) -> str:
    """Return the directory where export.py writes artifacts for *task_name*."""
    train_task = task_name.replace("-Play", "")
    return os.path.join(_REPO_ROOT, ".pretrained_checkpoints", "rsl_rl", train_task, task_name)


def _ensure_text(output: str | bytes | None) -> str:
    """Return subprocess output as text."""
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output


def _extract_timing_lines(*outputs: str | bytes | None) -> str:
    """Extract timing lines from subprocess output."""
    lines = []
    for output in outputs:
        for line in _ensure_text(output).splitlines():
            if line.startswith(_TIMING_PREFIXES):
                lines.append(line)
    return "\n".join(lines)


def _leapp_log_tail(export_dir: str) -> str:
    """Return the tail of the LEAPP log when it exists."""
    log_txt_path = os.path.join(export_dir, "log.txt")
    if not os.path.isfile(log_txt_path):
        return ""
    with open(log_txt_path) as f:
        last_lines = f.readlines()[-50:]
    return f"\n--- leapp log.txt (last 50 lines) ---\n{''.join(last_lines)}"


@pytest.mark.parametrize("task_name", TASKS)
def test_export_flow(task_name):
    """Run export.py for *task_name* and assert the expected artifacts are created."""
    export_dir = _export_dir(task_name)
    start_time = time.perf_counter()

    try:
        command = [
            "./isaaclab.sh",
            "-p",
            _EXPORT_SCRIPT,
            "--task",
            task_name,
            "--use_pretrained_checkpoint",
            "--disable_graph_visualization",
            "--headless",
        ]
        print(f"[EXPORT_TEST_TIMING] task={task_name} status=start timeout={_EXPORT_TIMEOUT}s", flush=True)
        try:
            result = subprocess.run(
                command,
                cwd=_REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=_EXPORT_TIMEOUT,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = _ensure_text(exc.stdout)
            stderr = _ensure_text(exc.stderr)
            timing_lines = _extract_timing_lines(stdout, stderr)
            elapsed = time.perf_counter() - start_time
            pytest.fail(
                f"export.py timed out after {_EXPORT_TIMEOUT}s for {task_name} (elapsed {elapsed:.2f}s).\n"
                f"--- timing lines ---\n{timing_lines or '<none>'}\n"
                f"--- stdout tail ---\n{stdout[-_OUTPUT_TAIL_CHARS:]}\n"
                f"--- stderr tail ---\n{stderr[-_OUTPUT_TAIL_CHARS:]}"
                f"{_leapp_log_tail(export_dir)}"
            )

        elapsed = time.perf_counter() - start_time
        timing_lines = _extract_timing_lines(result.stdout, result.stderr)
        print(
            f"[EXPORT_TEST_TIMING] task={task_name} status=finished "
            f"returncode={result.returncode} elapsed={elapsed:.2f}s",
            flush=True,
        )
        if timing_lines:
            print(f"--- timing lines for {task_name} ---\n{timing_lines}", flush=True)

        # Gracefully skip tasks whose checkpoint isn't published yet
        if "pre-trained checkpoint is currently unavailable" in result.stdout:
            pytest.skip(f"No pretrained checkpoint available for {task_name.replace('-Play', '')}")

        # Skip tasks whose checkpoint was saved with an older rsl_rl architecture
        # that does not use the 'actor_state_dict' key expected by the current runner
        if "actor_state_dict" in result.stderr:
            pytest.skip(
                f"{task_name} checkpoint uses an older network architecture incompatible with the current rsl_rl runner"
            )

        # Surface stdout/stderr on failure for easier debugging
        if result.returncode != 0:
            pytest.fail(
                f"export.py exited with code {result.returncode}.\n"
                f"--- timing lines ---\n{timing_lines or '<none>'}\n"
                f"--- stdout tail ---\n{result.stdout[-_OUTPUT_TAIL_CHARS:]}\n"
                f"--- stderr tail ---\n{result.stderr[-_OUTPUT_TAIL_CHARS:]}"
                f"{_leapp_log_tail(export_dir)}"
            )

        assert os.path.isfile(os.path.join(export_dir, f"{task_name}.onnx")), "Missing .onnx export"
        assert os.path.isfile(os.path.join(export_dir, f"{task_name}.yaml")), "Missing .yaml export"
        assert os.path.isfile(os.path.join(export_dir, "log.txt")), "Missing log.txt"

    finally:
        shutil.rmtree(export_dir, ignore_errors=True)
