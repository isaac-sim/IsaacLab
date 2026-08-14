# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-GPU training smoke tests.

Setup:
    - none; each test launches a real multi-rank training run as a subprocess
Tests:
    - physics-only task on 2 GPUs -> verify training completes
    - each stack on 4 GPUs with no device mask -> verify training completes
    - each stack on 4 GPUs exposed as ``3,1,2,0`` -> verify training completes

Unlike the rest of the suite these are not parametrized over ``device``: a
multi-GPU run owns several devices at once, so the per-shard single-device
parametrization the multi-GPU workflow applies elsewhere does not model it.
They are driven by dedicated workflow steps instead, one per renderer, selected
by the ``kitless`` marker.

Cases are split by whether the visible devices are exposed in their natural
order, because ``CUDA_VISIBLE_DEVICES`` renumbers devices for CUDA but not for
the graphics stack. Only a reordered mask makes the two indices disagree, which
is what exercises renderer device selection; the default ordering tests the case
where they coincide and cannot catch a selection defect at all.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Small on purpose: Kit boot dominates the runtime at this size, and the defect
# reproduces at 1024 envs exactly as it does at 2048.
_NUM_ENVS = "1024"
_MAX_ITERATIONS = "3"

# A hung run goes silent with both GPUs at 0% utilisation, while a slow one keeps
# logging -- so silence, not elapsed time, is the signal. 90 s sits far above the
# observed inter-line gap for this task (~10 s for the first iteration, ~6 s
# after) and above any gap during Kit boot, which logs continuously.
_IDLE_TIMEOUT_S = 90
# Backstop for a run that dribbles output forever. A passing run here is ~5 min
# (Kit boot ~4 min + ~30 s training), so this is ~2x headroom and caps the worst
# case a single parametrization can cost CI.
_HARD_TIMEOUT_S = 600

_PHYSICS_ONLY_TASK = "Isaac-Cartpole-Direct"
_CAMERA_TASK = "Isaac-Cartpole-Camera-Direct"

# Four ranks rather than two: with two, a single wrong device assignment can still land on a
# visible GPU by chance, and the rank-to-device mapping is too small to be wrong in an
# interesting way.
_CAMERA_RANKS = 4

# Deliberately not sorted and not contiguous-from-zero. Every rank's CUDA index differs from its
# graphics index, so no rank can be resolved by assuming the visible list is ordered.
_UNORDERED_DEVICES = (3, 1, 2, 0)

# (id, presets) for each physics/renderer stack worth covering, split by renderer so a fault in
# one cannot mask the other: the Kit-renderer cases are what guard device selection. The
# ``kitless`` marker selects between them.
#
# ``isaacsim_physx,ovrtx`` is absent by design: IsaacLab rejects it, since ovrtx
# is a kitless renderer and cannot pair with Kit physics.
#
# TODO: add ``ovphysx,ovrtx`` once OvPhysX supports multi-GPU. It currently hangs
# at the first parameter sync on *any* GPU pair, so it is a separate defect and would only cost
# CI a deliberate timeout while asserting something already known. See the process-global
# device-mode lock in ``isaaclab_ovphysx.physics.ovphysx_manager``.
_KIT_RENDERER_STACKS = [
    pytest.param("isaacsim_physx", id="isaacsim_physx-kit_rtx"),
    pytest.param("newton_mjwarp,isaacsim_rtx", id="newton-kit_rtx"),
]

_KITLESS_STACKS = [
    pytest.param("newton_mjwarp,ovrtx", id="newton-ovrtx"),
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _run_training(devices: tuple[int, ...] | None, task: str, presets: str, num_gpus: int) -> tuple[str, str]:
    """Launch a multi-rank training run and wait for it to settle.

    Streams the child's output so a stalled run is killed after
    :data:`_IDLE_TIMEOUT_S` of silence rather than occupying a CI runner until the
    hard timeout.

    Args:
        devices: GPU indices to expose, in the order given, or ``None`` to leave the inherited
            visibility untouched.
        task: Gym task id to train.
        presets: Value for the ``presets=`` selector (physics and/or renderer).
        num_gpus: Number of ranks to launch.

    Returns:
        ``(outcome, output)`` where outcome is ``"passed"``, ``"failed"`` or
        ``"hung"``, and output is the combined stdout/stderr captured so far.
    """
    env = dict(os.environ)
    # Docker's --gpus flag is not available from inside the test process, so the
    # pair is selected with CUDA_VISIBLE_DEVICES. Verified to reproduce the
    # canonical signature (exit 139 with cudainterop frames) rather than masking
    # it behind a device-enumeration artifact.
    if devices is None:
        # The naive case: whatever the host exposes, which is what a user gets by default.
        env.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in devices)
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        sys.executable,
        "scripts/reinforcement_learning/train_multigpu.py",
        "--num_gpus",
        str(num_gpus),
        # Without this only rank 0 is printed; when rank 1 is the one that dies
        # the log carries no evidence of why.
        "--log_all_ranks",
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
    # Own process group: train_multigpu.py spawns torchrun, which spawns the rank
    # workers. Killing only the wrapper skips its signal-forwarding handler and
    # leaves Kit ranks alive holding GPU memory and the rendezvous port.
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
            # Checked on every iteration, not only when the pipe is quiet: a child
            # that floods stdout would otherwise never reach the backstop and
            # could grow ``lines`` without bound.
            if now - started > _HARD_TIMEOUT_S or now - last_output > _IDLE_TIMEOUT_S:
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
        # The load-bearing check: a run that OOMs or exits early can still return 0.
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
    # Reaping is best-effort: the group is already SIGKILLed, and blocking the
    # test run on an unreapable child would trade one hang for another.
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=30)


def _assert_training_passed(outcome: str, output: str, devices: tuple[int, ...] | None = None) -> None:
    """Assert a training subprocess actually trained, not merely exited cleanly."""
    where = f" on CUDA_VISIBLE_DEVICES={devices}" if devices is not None else " with no device mask"
    assert outcome == "passed", f"outcome={outcome}{where}\n{output[-2000:]}"


def _visible_cuda_device_count() -> int:
    """Return how many CUDA devices this process can address.

    Counts what CUDA exposes rather than what the host physically has, so a
    ``CUDA_VISIBLE_DEVICES``-restricted runner or a MIG layout is reported as the
    caller will actually see it.
    """
    # Local import so collecting this module does not pull torch in before Kit.
    import torch

    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def _require_devices(count: int) -> None:
    """Skip unless the host can address ``count`` CUDA devices."""
    available = _visible_cuda_device_count()
    if available < count:
        pytest.skip(f"needs {count} visible CUDA devices, host has {available}")


@pytest.mark.smoke
@pytest.mark.integration
class TestMultiGpuTrainingSmoke:
    """Four-rank training smoke coverage across the physics and renderer stacks."""

    def test_physics_only_trains(self) -> None:
        """Physics-only multi-GPU training completes with no device mask.

        The guard that always runs, and the cheapest signal that the launcher and NCCL are healthy
        before any renderer is involved. Requires only two devices so it still runs on hosts too
        small for the camera cases.
        """
        _require_devices(2)
        _assert_training_passed(*_run_training(None, _PHYSICS_ONLY_TASK, "isaacsim_physx", num_gpus=2))

    @pytest.mark.rendering
    @pytest.mark.parametrize("presets", _KIT_RENDERER_STACKS)
    def test_kit_renderer_camera_trains_without_device_mask(self, presets: str) -> None:
        """Kit-rendered training on four GPUs with no ``CUDA_VISIBLE_DEVICES`` set.

        What a user gets by default. CUDA and graphics device indices coincide here, so this passes
        even when device selection is wrong -- it is the baseline the masked case is read against,
        not a test of selection.
        """
        _require_devices(_CAMERA_RANKS)
        _assert_training_passed(*_run_training(None, _CAMERA_TASK, presets, num_gpus=_CAMERA_RANKS))

    @pytest.mark.rendering
    @pytest.mark.parametrize("presets", _KIT_RENDERER_STACKS)
    def test_kit_renderer_camera_trains_on_unordered_devices(self, presets: str) -> None:
        """Kit-rendered training on four GPUs exposed in a non-monotonic order.

        The regression guard for renderer device selection. ``CUDA_VISIBLE_DEVICES`` renumbers
        devices for CUDA but not for the graphics stack, so with ``3,1,2,0`` every rank's CUDA index
        differs from its graphics index and none of them can be recovered by assuming the list is
        sorted or contiguous. Passing a CUDA index straight to ``/renderer/activeGpu`` selects a
        device outside the visible set and the run dies with ``CUDA error 700``.
        """
        _require_devices(_CAMERA_RANKS)
        _assert_training_passed(
            *_run_training(_UNORDERED_DEVICES, _CAMERA_TASK, presets, num_gpus=_CAMERA_RANKS),
            devices=_UNORDERED_DEVICES,
        )

    @pytest.mark.rendering
    @pytest.mark.kitless
    @pytest.mark.parametrize("presets", _KITLESS_STACKS)
    def test_kitless_camera_trains_without_device_mask(self, presets: str) -> None:
        """Kitless-rendered training on four GPUs with no ``CUDA_VISIBLE_DEVICES`` set."""
        _require_devices(_CAMERA_RANKS)
        _assert_training_passed(*_run_training(None, _CAMERA_TASK, presets, num_gpus=_CAMERA_RANKS))

    @pytest.mark.rendering
    @pytest.mark.kitless
    @pytest.mark.parametrize("presets", _KITLESS_STACKS)
    def test_kitless_camera_trains_on_unordered_devices(self, presets: str) -> None:
        """Kitless-rendered training on four GPUs exposed in a non-monotonic order.

        ``ovrtx`` selects its device through CUDA, which ``CUDA_VISIBLE_DEVICES`` renumbers
        consistently, so this stack was never affected by the graphics-index defect the Kit case
        covers. It runs strict as the control: a failure here is a different defect.
        """
        _require_devices(_CAMERA_RANKS)
        _assert_training_passed(
            *_run_training(_UNORDERED_DEVICES, _CAMERA_TASK, presets, num_gpus=_CAMERA_RANKS),
            devices=_UNORDERED_DEVICES,
        )
