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
from dataclasses import dataclass
from pathlib import Path

import pytest

# Small on purpose: Kit boot dominates the runtime at this size, and the defect reproduces at
# 1024 envs exactly as it does at 2048.
_NUM_ENVS = "1024"
_MAX_ITERATIONS = "3"

# A hung run goes silent while a slow one keeps logging, so silence is the signal. 90 s sits far
# above the observed inter-line gap (~10 s first iteration, ~6 s after) and above any gap during
# Kit boot, which logs continuously.
_IDLE_TIMEOUT_S = 90
# Kitless renderers break that assumption: on a cold shader cache OVRTX compiles ray-tracing
# pipeline objects for minutes, reporting progress only to its own log. Measured on 8x L40: a
# cold run was still compiling when the 90 s timer killed it.
_KITLESS_IDLE_TIMEOUT_S = 300
# Backstop for a run that dribbles output forever; ~2x a passing run.
_HARD_TIMEOUT_S = 600

_PHYSICS_ONLY_TASK = "Isaac-Cartpole-Direct"
_CAMERA_TASK = "Isaac-Cartpole-Camera-Direct"

# Four ranks rather than two: with two, a wrong device assignment can still land on a visible GPU
# by chance.
_CAMERA_RANKS = 4

# Not sorted and not contiguous-from-zero, so no rank resolves by assuming the list is ordered.
_UNORDERED_DEVICES = (3, 1, 2, 0)

_OVPHYSX_OVRTX_XFAIL_REASON = (
    "``ovphysx,ovrtx`` does not complete a multi-GPU run. The limitation is the pairing, not"
    " either backend alone: measured on 8x L40 (2026-08-26, four ranks) ``ovphysx,newton_renderer``"
    " trains in 11.4 s and ``newton_mjwarp,ovrtx`` in 27.3 s, while this combination still had not"
    " reached the first iteration at 900 s -- with the shader cache already warmed by the preceding"
    " ovrtx run, so it is not cold-compile latency."
)


@dataclass(frozen=True)
class _Stack:
    """A physics/renderer backend combination and the silence budget it needs."""

    presets: str
    idle_timeout_s: int


# The backend grid is 3 physics x 3 renderers; two of the nine cells cannot run at all, rejected
# before launch by ``sim_launcher._validate_runtime`` because OVRTX and OvPhysX are kitless and
# cannot share a process with Kit: ``isaacsim_physx,ovrtx`` and ``ovphysx,isaacsim_rtx``. The
# Newton renderer pairs with every physics backend -- ``NewtonManager.get_model`` builds a shadow
# model when the active backend is not Newton -- so it appears three times here. The ``kitless``
# marker routes each stack to the CI image that carries its runtime.
_STACKS = [
    pytest.param(_Stack("isaacsim_physx", _IDLE_TIMEOUT_S), id="isaacsim_physx-kit_rtx"),
    pytest.param(_Stack("newton_mjwarp,isaacsim_rtx", _IDLE_TIMEOUT_S), id="newton-kit_rtx"),
    # Kit physics with a kitless renderer: still a Kit process, so it stays in the Kit lane, but
    # Warp kernel compilation is silent on stdout and needs the wider silence budget.
    pytest.param(
        _Stack("isaacsim_physx,newton_renderer", _KITLESS_IDLE_TIMEOUT_S),
        id="isaacsim_physx-newton_renderer",
    ),
    pytest.param(
        _Stack("newton_mjwarp,ovrtx", _KITLESS_IDLE_TIMEOUT_S),
        id="newton-ovrtx",
        marks=pytest.mark.kitless,
    ),
    pytest.param(
        _Stack("newton_mjwarp,newton_renderer", _KITLESS_IDLE_TIMEOUT_S),
        id="newton-newton_renderer",
        marks=pytest.mark.kitless,
    ),
    pytest.param(
        _Stack("ovphysx,newton_renderer", _KITLESS_IDLE_TIMEOUT_S),
        id="ovphysx-newton_renderer",
        marks=pytest.mark.kitless,
    ),
    # Xfailed rather than omitted so the gap is reported by every run. ``run=False`` because the
    # failure mode is a non-terminating run: executing it would spend the full idle timeout to
    # re-derive a known result. Flip to ``run=True`` when the pairing is fixed, so an XPASS
    # reports it.
    pytest.param(
        _Stack("ovphysx,ovrtx", _KITLESS_IDLE_TIMEOUT_S),
        id="ovphysx-ovrtx",
        marks=[
            pytest.mark.kitless,
            pytest.mark.xfail(reason=_OVPHYSX_OVRTX_XFAIL_REASON, run=False),
        ],
    ),
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


def _run_training(
    devices: tuple[int, ...] | None,
    task: str,
    presets: str,
    num_gpus: int,
    idle_timeout_s: int = _IDLE_TIMEOUT_S,
) -> tuple[str, str]:
    """Launch a multi-rank training run and wait for it to settle.

    Streams the child's output so a stalled run is killed after ``idle_timeout_s`` of silence
    rather than occupying a CI runner until the hard timeout.

    Args:
        devices: GPU indices to expose, in the order given, or ``None`` to leave the inherited
            visibility untouched.
        task: Gym task id to train.
        presets: Value for the ``presets=`` selector (physics and/or renderer).
        num_gpus: Number of ranks to launch.
        idle_timeout_s: Seconds of stdout silence [s] treated as a hang.

    Returns:
        ``(outcome, output)`` where outcome is ``"passed"``, ``"failed"`` or ``"hung"``, and
        output is the combined stdout/stderr captured so far.
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
            # Checked every iteration, not only when the pipe is quiet: a child that floods stdout
            # would otherwise never reach the backstop and could grow ``lines`` without bound.
            if now - started > _HARD_TIMEOUT_S or now - last_output > idle_timeout_s:
                _kill_process_group(process)
                return "hung", "".join(lines)
            if line:
                lines.append(line)
                last_output = now
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
    return ("passed" if passed else "failed"), output


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


def _assert_training_passed(outcome: str, output: str, devices: tuple[int, ...] | None = None) -> None:
    """Assert a training subprocess actually trained, not merely exited cleanly."""
    where = f" on CUDA_VISIBLE_DEVICES={devices}" if devices is not None else " with no device mask"
    assert outcome == "passed", f"outcome={outcome}{where}\ngpus: {_gpu_state()}\n{output[-2000:]}"


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
    def test_camera_training(self, stack: _Stack, devices: tuple[int, ...] | None) -> None:
        """Camera-rendered training on four GPUs for one backend stack and device order."""
        _require_devices(_CAMERA_RANKS)
        _assert_training_passed(
            *_run_training(
                devices,
                _CAMERA_TASK,
                stack.presets,
                num_gpus=_CAMERA_RANKS,
                idle_timeout_s=stack.idle_timeout_s,
            ),
            devices=devices,
        )
