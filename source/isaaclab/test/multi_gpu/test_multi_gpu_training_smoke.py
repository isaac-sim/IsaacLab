# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-GPU training smoke tests.

Setup:
    - none; each test launches a real multi-rank training run as a subprocess
Tests:
    - physics-only task on 2 GPUs -> verify training completes
    - each of the seven runnable backend stacks on 4 GPUs, in the host's device
      order and exposed as ``3,1,2,0`` -> verify training completes

``CUDA_VISIBLE_DEVICES`` renumbers devices for CUDA but not for the graphics stack, so only a
reordered mask makes the two indices disagree. That case is what exercises renderer device
selection; the default ordering passes even when selection is wrong and exists as its control.
"""

from __future__ import annotations

import contextlib
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Small on purpose: Kit boot dominates the runtime at this size, and the defect these guard
# reproduces at any env count. Sized for the CI pool's 23 GiB A10G rather than the 48 GiB L40S
# it was first measured on -- at 1024 a single rank reached 20.72 GiB and the run died allocating
# 864 MiB more.
_NUM_ENVS = "512"
_MAX_ITERATIONS = "3"

# A hung run goes silent while a slow one keeps logging, so silence is the signal -- nothing here
# gates on how long a run takes, only on it having stopped. Startup and steady state have very
# different silence profiles, so each gets its own budget.
#
# Startup is legitimately silent for long stretches: Kit boot, Warp kernel compilation, and
# OVRTX ray-tracing pipeline compilation all report little or nothing to stdout. Deliberately
# generous, because the cost of being wrong is a false failure on slower hardware -- the CI pool
# is 4x A10G, well below the L40S these were first measured on.
_STARTUP_IDLE_TIMEOUT_S = 600
# Once iterations are logging, gaps are small (~10 s first, ~6 s after), so silence here is a
# real hang and worth catching quickly.
_STEADY_IDLE_TIMEOUT_S = 120
# Backstop for a run that dribbles output forever, generous for the same reason as the startup
# budget: it exists to bound a stuck runner, not to time a healthy run.
_HARD_TIMEOUT_S = 1800
# rsl_rl logs this once per iteration; its first appearance is what ends the startup phase.
_ITERATION_MARKER = "Learning iteration"
# How often to record which process holds memory on which GPU while the run is alive.
_GPU_SAMPLE_INTERVAL_S = 15
# Tail of the child's output kept in the failure message. Sized against torchrun rather than
# guessed: when a rank dies, its ChildFailedError block alone runs past 2000 characters, so a
# smaller budget reports only that torchrun noticed a failure and drops the rank's own error --
# the one line that says what actually went wrong.
_FAILURE_OUTPUT_CHARS = 8000

_PHYSICS_ONLY_TASK = "Isaac-Cartpole-Direct"
_CAMERA_TASK = "Isaac-Cartpole-Camera-Direct"

# Four ranks rather than two: with two, a wrong device assignment can still land on a visible GPU
# by chance.
_CAMERA_RANKS = 4

# Not sorted and not contiguous-from-zero, so no rank resolves by assuming the list is ordered.
_UNORDERED_DEVICES = (3, 1, 2, 0)

# The backend grid is 3 physics x 3 renderers; two of the nine cells cannot run at all, rejected
# before launch by ``sim_launcher._validate_runtime`` because OVRTX and OvPhysX are kitless and
# cannot share a process with Kit: ``isaacsim_physx,ovrtx`` and ``ovphysx,isaacsim_rtx``. The
# Newton renderer pairs with every physics backend -- ``NewtonManager.get_model`` builds a shadow
# model when the active backend is not Newton -- so it appears three times here. The ``kitless``
# marker routes each stack to the CI image that carries its runtime. All seven run: ``ovphysx,ovrtx``
# did not complete under ovphysx 0.5.10 and was xfailed, which 0.5.11 fixed.
_STACKS = [
    pytest.param("isaacsim_physx", id="isaacsim_physx-kit_rtx"),
    pytest.param("newton_mjwarp,isaacsim_rtx", id="newton-kit_rtx"),
    # Kit physics with a kitless renderer: still a Kit process, so it stays in the Kit lane.
    pytest.param("isaacsim_physx,newton_renderer", id="isaacsim_physx-newton_renderer"),
    pytest.param("newton_mjwarp,ovrtx", id="newton-ovrtx", marks=pytest.mark.kitless),
    pytest.param("newton_mjwarp,newton_renderer", id="newton-newton_renderer", marks=pytest.mark.kitless),
    pytest.param("ovphysx,newton_renderer", id="ovphysx-newton_renderer", marks=pytest.mark.kitless),
    pytest.param("ovphysx,ovrtx", id="ovphysx-ovrtx", marks=pytest.mark.kitless),
]

_DEVICE_ORDERS = [
    pytest.param(None, id="default_order"),
    pytest.param(_UNORDERED_DEVICES, id="unordered_devices"),
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _free_port() -> int:
    """Return a port that is free right now, for this run's torchrun rendezvous.

    Cases run sequentially in one CI job and a killed rank releases the port asynchronously,
    so reusing torchrun's default 29500 makes the next case abort with "The server socket has
    failed to listen on any local network address".
    """
    with socket.socket() as probe:
        probe.bind(("", 0))
        return probe.getsockname()[1]


def _gpu_state() -> str:
    """Return a one-line per-GPU snapshot: index, model, used and total memory.

    A failing run reports only what it could not allocate, which is not enough to tell a
    device that is genuinely full from one that is too small for the workload. Captured on
    failure so the report carries the hardware it ran on.
    """
    probe = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        return "nvidia-smi unavailable"
    return " | ".join(line.strip() for line in probe.stdout.splitlines() if line.strip())


def _gpu_processes() -> str:
    """Return which process holds how much memory on which GPU, as ``pid@bus=used``.

    Sampled while the child is alive: a totals-only snapshot taken after cleanup shows empty
    devices, so it cannot separate a workload too large for the card from ranks allocating on a
    device that is not theirs.
    """
    probe = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_bus_id,used_memory", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        return "nvidia-smi unavailable"
    entries = []
    for line in probe.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 3:
            entries.append(f"{fields[0]}@{fields[1][-7:]}={fields[2]}")
    return " | ".join(entries) or "no compute processes"


def _run_training(
    devices: tuple[int, ...] | None,
    task: str,
    presets: str,
    num_gpus: int,
) -> tuple[str, str]:
    """Launch a multi-rank training run and wait for it to settle.

    Streams the child's output so a stalled run is killed after a period of silence rather than
    occupying a CI runner until the hard timeout. The budget is wide until the first logged
    iteration and narrow afterwards, so a slow start is not mistaken for a hang.

    Args:
        devices: GPU indices to expose, in the order given, or ``None`` to leave the inherited
            visibility untouched.
        task: Gym task id to train.
        presets: Value for the ``presets=`` selector (physics and/or renderer).
        num_gpus: Number of ranks to launch.

    Returns:
        ``(outcome, output, gpu_processes)`` where outcome is ``"passed"``, ``"failed"`` or
        ``"hung"``, output is the combined stdout/stderr captured so far, and gpu_processes is the
        last per-process GPU placement sampled while the child was alive.
    """
    env = dict(os.environ)
    # Docker's --gpus flag is unavailable from inside the test process, so devices are selected
    # with CUDA_VISIBLE_DEVICES instead.
    if devices is None:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in devices)
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        sys.executable,
        "scripts/reinforcement_learning/train_multigpu.py",
        "--num_gpus",
        str(num_gpus),
        # Without this only rank 0 is printed; when another rank dies the log carries no evidence.
        "--log_all_ranks",
        # Never torchrun's default 29500: see :func:`_free_port`.
        "--master_port",
        str(_free_port()),
        "--rl_library",
        "rsl_rl",
        "--task",
        task,
        f"presets={presets}",
        "--num_envs",
        _NUM_ENVS,
        "--max_iterations",
        _MAX_ITERATIONS,
    ]

    started = time.monotonic()
    last_output = started
    lines: list[str] = []
    iterating = False
    last_sampled = started
    gpu_processes = "not sampled"
    # Own process group: train_multigpu.py spawns torchrun, which spawns the rank workers.
    # Killing only the wrapper leaves Kit ranks alive holding GPU memory and the rendezvous port.
    process = subprocess.Popen(
        cmd,
        cwd=_repo_root(),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    try:
        os.set_blocking(process.stdout.fileno(), False)  # type: ignore[union-attr]
        while True:
            line = process.stdout.readline()  # type: ignore[union-attr]
            now = time.monotonic()
            if now - last_sampled > _GPU_SAMPLE_INTERVAL_S:
                gpu_processes = _gpu_processes()
                last_sampled = now
            # Checked every iteration, not only when the pipe is quiet: a child that floods stdout
            # would otherwise never reach the backstop and could grow ``lines`` without bound.
            idle_budget = _STEADY_IDLE_TIMEOUT_S if iterating else _STARTUP_IDLE_TIMEOUT_S
            if now - started > _HARD_TIMEOUT_S or now - last_output > idle_budget:
                _kill_process_group(process)
                return "hung", "".join(lines), gpu_processes
            if line:
                lines.append(line)
                last_output = now
                if not iterating and _ITERATION_MARKER in line:
                    iterating = True
            elif process.poll() is not None:
                break
            else:
                time.sleep(0.2)
        lines.append(process.stdout.read() or "")  # type: ignore[union-attr]
    finally:
        _kill_process_group(process)

    output = "".join(lines)
    passed = (
        process.returncode == 0
        and "Traceback (most recent call last):" not in output
        # Load-bearing: a run that OOMs or exits early can still return 0.
        and "Training time:" in output
    )
    return ("passed" if passed else "failed"), output, gpu_processes


def _kill_process_group(process: subprocess.Popen) -> None:
    """Kill the child's whole process group and reap it; a no-op once it has exited."""
    if process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        process.kill()
    # Best-effort: the group is already SIGKILLed, and blocking on an unreapable child would
    # trade one hang for another.
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=30)


def _assert_training_passed(
    outcome: str, output: str, gpu_processes: str, devices: tuple[int, ...] | None = None
) -> None:
    """Assert a training subprocess actually trained, not merely exited cleanly."""
    where = f" on CUDA_VISIBLE_DEVICES={devices}" if devices is not None else " with no device mask"
    assert outcome == "passed", (
        f"outcome={outcome}{where}\ngpus: {_gpu_state()}\nlast live placement: {gpu_processes}\n"
        f"{output[-_FAILURE_OUTPUT_CHARS:]}"
    )


def _require_devices(count: int) -> None:
    """Skip unless the host can address ``count`` CUDA devices."""
    # Local import so collecting this module does not pull torch in before Kit.
    import torch

    available = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if available < count:
        pytest.skip(f"needs {count} visible CUDA devices, host has {available}")


@pytest.mark.smoke
@pytest.mark.integration
class TestMultiGpuTrainingSmoke:
    """Multi-rank training coverage across the supported backend stacks."""

    def test_physics_only_trains(self) -> None:
        """Physics-only multi-GPU training completes with no device mask.

        The cheapest signal that the launcher and NCCL are healthy before any renderer is
        involved. Needs only two devices, so it still runs on hosts too small for the rest.
        """
        _require_devices(2)
        _assert_training_passed(*_run_training(None, _PHYSICS_ONLY_TASK, "isaacsim_physx", num_gpus=2))

    @pytest.mark.rendering
    @pytest.mark.parametrize("devices", _DEVICE_ORDERS)
    @pytest.mark.parametrize("stack", _STACKS)
    def test_camera_training(self, stack: str, devices: tuple[int, ...] | None) -> None:
        """Camera-rendered training on four GPUs for one backend stack and device order."""
        _require_devices(_CAMERA_RANKS)
        _assert_training_passed(
            *_run_training(devices, _CAMERA_TASK, stack, num_gpus=_CAMERA_RANKS),
            devices=devices,
        )
