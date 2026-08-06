# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-GPU training smoke tests.

Setup:
    - none; each test launches a real two-rank training run as a subprocess
Tests:
    - physics-only task on any 2 GPUs -> verify training completes
    - each physics/renderer stack on a same-switch GPU pair -> verify training completes
    - each physics/renderer stack on a cross-socket GPU pair -> Kit-renderer stacks
      are expected to fail, NVBUG#6565122

Unlike the rest of the suite these are not parametrized over ``device``: a
multi-GPU run owns two devices at once, so the per-shard single-device
parametrization the multi-GPU workflow applies elsewhere does not model it.
They are driven by a dedicated workflow step instead.

Which GPU pair a case uses is resolved from the host at runtime rather than
hardcoded. On an 8-GPU two-socket box the default ``cuda:0,cuda:1`` pick is a
*same-switch* pair and does not exercise the cross-socket path at all, so a
fixed pick would quietly stop testing the thing this file exists for.
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

from isaaclab.test.utils import gpu_pairs_by_topology

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

# (id, presets) for each physics/renderer stack worth covering.
#
# ``isaacsim_physx,ovrtx`` is absent by design: IsaacLab rejects it, since ovrtx
# is a kitless renderer and cannot pair with Kit physics.
#
# TODO: add ``ovphysx,ovrtx`` once OvPhysX supports multi-GPU. It currently hangs
# at the first parameter sync on *any* GPU pair -- same-switch included -- so it
# is a separate defect from NVBUG#6565122 and would only cost CI a deliberate
# timeout while asserting something already known. See the process-global
# device-mode lock in ``isaaclab_ovphysx.physics.ovphysx_manager``.
_CAMERA_STACKS = [
    pytest.param("isaacsim_physx", id="isaacsim_physx-kit_rtx"),
    pytest.param("newton_mjwarp,isaacsim_rtx", id="newton-kit_rtx"),
    pytest.param("newton_mjwarp,ovrtx", id="newton-ovrtx"),
]

# Stacks that drive Kit's Isaac Sim RTX renderer. These are the ones that fail on
# a cross-socket pair; the kitless ``ovrtx`` stack passes there.
_KIT_RENDERER_STACKS = frozenset({"isaacsim_physx", "newton_mjwarp,isaacsim_rtx"})

_XFAIL_REASON = (
    "Kit's Isaac Sim RTX renderer corrupts the host heap when a multi-GPU rendering job spans a"
    " cross-socket (SYS) GPU pair, surfacing as SIGSEGV inside libcarb.cudainterop.plugin.so."
    " Entered between Kit 110.0.0 and Kit 110.1.2. The same run passes on a same-switch pair, and"
    " with presets=newton_mjwarp,ovrtx on the same pair. NVBUG#6565122"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _run_training(pair: tuple[int, int], task: str, presets: str) -> tuple[str, str]:
    """Launch a two-rank training run pinned to ``pair`` and wait for it to settle.

    Streams the child's output so a stalled run is killed after
    :data:`_IDLE_TIMEOUT_S` of silence rather than occupying a CI runner until the
    hard timeout.

    Args:
        pair: GPU indices to pin the two ranks to.
        task: Gym task id to train.
        presets: Value for the ``presets=`` selector (physics and/or renderer).

    Returns:
        ``(outcome, output)`` where outcome is ``"passed"``, ``"failed"`` or
        ``"hung"``, and output is the combined stdout/stderr captured so far.
    """
    env = dict(os.environ)
    # Docker's --gpus flag is not available from inside the test process, so the
    # pair is selected with CUDA_VISIBLE_DEVICES. Verified to reproduce the
    # canonical signature (exit 139 with cudainterop frames) rather than masking
    # it behind a device-enumeration artifact.
    env["CUDA_VISIBLE_DEVICES"] = f"{pair[0]},{pair[1]}"
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        sys.executable,
        "scripts/reinforcement_learning/train_multigpu.py",
        "--num_gpus",
        "2",
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


def _matches_known_crash(output: str) -> bool:
    """Whether a failure carries the NVBUG#6565122 signature rather than some other fault.

    The crash is a SIGSEGV inside ``libcarb.cudainterop.plugin.so``. Without this
    check an OOM, an argument error, or an unrelated hang on a cross-socket pair
    would all be absorbed by the expected-failure marker.
    """
    return "cudainterop" in output and ("exitcode  : 139" in output or "Signal 11" in output)


def _assert_training_passed(outcome: str, output: str) -> None:
    """Assert a training subprocess actually trained, not merely exited cleanly."""
    assert outcome == "passed", f"outcome={outcome}\n{output[-2000:]}"


def _visible_cuda_device_count() -> int:
    """Return how many CUDA devices this process can address.

    Counts what CUDA exposes rather than what the host physically has, so a
    ``CUDA_VISIBLE_DEVICES``-restricted runner or a MIG layout is reported as the
    caller will actually see it.
    """
    # Local import so collecting this module does not pull torch in before Kit.
    import torch

    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def _pair_or_skip(kind: str) -> tuple[int, int]:
    """Return a GPU pair of ``kind``, or skip with the reason it is unavailable."""
    pairs = gpu_pairs_by_topology()
    if not pairs:
        pytest.skip("GPU topology could not be determined from nvidia-smi topo -m")
    if kind not in pairs:
        pytest.skip(f"host has no {kind} GPU pair (available: {sorted(pairs)})")
    return pairs[kind]


@pytest.mark.smoke
@pytest.mark.integration
class TestMultiGpuTrainingSmoke:
    """Two-rank training smoke coverage across the stacks and interconnect classes the host offers."""

    def test_physics_only_trains_on_any_pair(self) -> None:
        """Physics-only multi-GPU training completes on the first two visible GPUs.

        This is the guard that always runs. It deliberately does NOT consult the
        topology: the camera cases are gated on the host offering a pair of the
        right interconnect class, so a host with an unreadable topology, a MIG
        layout, or only unmeasured link classes skips all six of them. Were this
        case gated too, the step would exit 0 having launched no training at all.
        """
        count = _visible_cuda_device_count()
        if count < 2:
            pytest.skip(f"multi-GPU smoke needs 2 visible CUDA devices, host has {count}")
        _assert_training_passed(*_run_training((0, 1), _PHYSICS_ONLY_TASK, "isaacsim_physx"))

    @pytest.mark.rendering
    @pytest.mark.parametrize("presets", _CAMERA_STACKS)
    def test_camera_trains_on_same_switch_pair(self, presets: str) -> None:
        """Camera-based multi-GPU training completes when both GPUs share a PCIe switch.

        Strict for every stack. This is the regression guard: it is the
        configuration NVBUG#6565122 does *not* affect, so a failure here is a new
        defect rather than the known one.
        """
        pair = _pair_or_skip("SAME_SWITCH")
        _assert_training_passed(*_run_training(pair, _CAMERA_TASK, presets))

    @pytest.mark.rendering
    @pytest.mark.parametrize("presets", _CAMERA_STACKS)
    def test_camera_trains_on_cross_socket_pair(self, presets: str, request) -> None:
        """Camera-based multi-GPU training across a cross-socket GPU pair.

        Stacks driving Kit's RTX renderer are expected to fail while
        NVBUG#6565122 is open; the kitless ``ovrtx`` stack must still pass, which
        is what pins the defect to the renderer rather than to multi-GPU
        rendering in general. The failing cases run rather than skip so a Kit fix
        surfaces as XPASS instead of going unnoticed.
        """
        pair = _pair_or_skip("CROSS_SOCKET")
        outcome, output = _run_training(pair, _CAMERA_TASK, presets)
        if presets in _KIT_RENDERER_STACKS and outcome == "failed" and _matches_known_crash(output):
            # Marked only once the documented signature is confirmed, and applied
            # after the run rather than as a decorator. A blanket marker would
            # absorb an OOM, an argument error, or an unrelated hang on this pair
            # as though it were NVBUG#6565122, and would also cover the ovrtx
            # stack, which must stay strict.
            pytest.xfail(_XFAIL_REASON)
        _assert_training_passed(outcome, output)
