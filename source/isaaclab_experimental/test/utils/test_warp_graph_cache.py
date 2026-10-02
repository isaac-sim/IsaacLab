# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the WarpGraphCache capture lifecycle."""

from __future__ import annotations

import pytest
import torch
import warp as wp
from isaaclab_experimental.utils.warp_graph_cache import CAPTURE_ENV_VAR, SYNC_DEBUG_ENV_VAR, WarpGraphCache

wp.init()
pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA device required")

DEVICE = "cuda:0"


@wp.kernel
def _increment(values: wp.array(dtype=wp.int32)):
    i = wp.tid()
    values[i] = values[i] + 1


class _CountingStage:
    """Stage that increments a device counter and counts its Python invocations."""

    def __init__(self):
        self.calls = 0

    def __call__(self, values: wp.array) -> wp.array:
        self.calls += 1
        wp.launch(_increment, dim=values.shape[0], inputs=[values], device=DEVICE)
        return values


def _read(values: wp.array) -> list[int]:
    wp.synchronize_device(DEVICE)
    return values.numpy().tolist()


@pytest.fixture
def cache(monkeypatch):
    monkeypatch.delenv(CAPTURE_ENV_VAR, raising=False)
    monkeypatch.delenv(SYNC_DEBUG_ENV_VAR, raising=False)
    cache = WarpGraphCache(DEVICE)
    yield cache
    cache.close()


def test_first_armed_call_records_and_runs_the_stage_once(cache):
    """Recording has no eager warm-up: a stateful stage advances once per call, including the first."""
    stage = _CountingStage()
    values = wp.zeros(2, dtype=wp.int32, device=DEVICE)
    cache.arm()

    result = cache.call("Group_stage", stage, values)
    assert _read(values) == [1, 1]
    assert result is values

    cache.call("Group_stage", stage, values)
    assert _read(values) == [2, 2]
    assert stage.calls == 1, "the second call must replay the recorded graph"
    assert cache.captured_stages == ("Group_stage",)


def test_stages_run_eagerly_until_armed(cache):
    stage = _CountingStage()
    values = wp.zeros(1, dtype=wp.int32, device=DEVICE)

    cache.call("Group_stage", stage, values)
    cache.call("Group_stage", stage, values)
    assert (stage.calls, _read(values), cache.captured_stages) == (2, [2], ())

    cache.arm()
    cache.call("Group_stage", stage, values)
    cache.call("Group_stage", stage, values)
    assert (stage.calls, _read(values), cache.captured_stages) == (3, [4], ("Group_stage",))


def test_reallocated_argument_records_the_stage_again(cache):
    """A graph replays recorded pointers, so a new array must not reuse the old graph."""
    stage = _CountingStage()
    old = wp.zeros(1, dtype=wp.int32, device=DEVICE)
    cache.arm()
    cache.call("Group_stage", stage, old)

    new = wp.zeros(1, dtype=wp.int32, device=DEVICE)
    cache.call("Group_stage", stage, new)
    cache.call("Group_stage", stage, new)

    assert (_read(old), _read(new), stage.calls) == ([1], [2], 2)


def test_changed_scalar_argument_records_the_stage_again(cache):
    values = wp.zeros(1, dtype=wp.float32, device=DEVICE)

    def add(values: wp.array, amount: float) -> None:
        wp.launch(_add_scalar, dim=1, inputs=[values, amount], device=DEVICE)

    cache.arm()
    cache.call("Group_stage", add, values, 1.0)
    cache.call("Group_stage", add, values, 10.0)
    assert _read(values) == [11.0]


@wp.kernel
def _add_scalar(values: wp.array(dtype=wp.float32), amount: wp.float32):
    values[wp.tid()] = values[wp.tid()] + amount


def test_invalidate_drops_graphs_and_waits_for_the_next_arm(cache):
    stage = _CountingStage()
    values = wp.zeros(1, dtype=wp.int32, device=DEVICE)
    cache.arm()
    cache.call("Group_stage", stage, values)

    cache.invalidate()
    cache.call("Group_stage", stage, values)
    assert (stage.calls, cache.captured_stages) == (2, ())

    cache.arm()
    cache.call("Group_stage", stage, values)
    cache.call("Group_stage", stage, values)
    assert (stage.calls, _read(values), cache.captured_stages) == (3, [4], ("Group_stage",))


def test_group_invalidation_keeps_other_groups_recorded(cache):
    values = wp.zeros(1, dtype=wp.int32, device=DEVICE)
    cache.arm()
    cache.call("First_stage", _CountingStage(), values)
    cache.call("Second_stage", _CountingStage(), values)

    cache.invalidate("First")

    assert cache.captured_stages == ("Second_stage",)
    stage = _CountingStage()
    cache.call("First_stage", stage, values)
    assert (stage.calls, cache.captured_stages) == (1, ("First_stage", "Second_stage"))


def test_non_capturable_group_stays_eager(cache):
    stage = _CountingStage()
    values = wp.zeros(1, dtype=wp.int32, device=DEVICE)
    cache.register_capturability("Group", True)
    cache.register_capturability("Group", False)
    cache.register_capturability("Group", True)
    cache.arm()

    cache.call("Group_stage", stage, values)
    cache.call("Group_stage", stage, values)

    assert (stage.calls, _read(values), cache.captured_stages) == (2, [2], ())


def test_capture_env_var_forces_eager_execution(monkeypatch):
    monkeypatch.setenv(CAPTURE_ENV_VAR, "0")
    cache = WarpGraphCache(DEVICE)
    stage = _CountingStage()
    values = wp.zeros(1, dtype=wp.int32, device=DEVICE)
    cache.arm()

    cache.call("Group_stage", stage, values)
    cache.call("Group_stage", stage, values)

    assert (stage.calls, _read(values), cache.captured_stages) == (2, [2], ())


def test_sync_debug_trap_raises_on_a_hidden_host_sync(monkeypatch):
    monkeypatch.setenv(CAPTURE_ENV_VAR, "0")
    monkeypatch.setenv(SYNC_DEBUG_ENV_VAR, "1")
    cache = WarpGraphCache(DEVICE)
    flags = torch.ones(4, dtype=torch.bool, device=DEVICE)

    with pytest.raises(RuntimeError, match="synchroniz"):
        cache.call("Group_stage", lambda: flags.any().item())
    assert torch.cuda.get_sync_debug_mode() == 0, "the trap must be released after the stage"


@wp.kernel
def _affine(values: wp.array(dtype=wp.int32), scale: wp.int32, shift: wp.int32):
    values[wp.tid()] = values[wp.tid()] * scale + shift


class _AffineStep:
    """Stage step that applies ``x * scale + shift`` on the device and counts its Python invocations."""

    def __init__(self, scale: int, shift: int):
        self.scale, self.shift, self.calls = scale, shift, 0

    def __call__(self, values: wp.array) -> wp.array:
        self.calls += 1
        wp.launch(_affine, dim=values.shape[0], inputs=[values, self.scale, self.shift], device=DEVICE)
        return values


def test_steps_record_capturable_runs_and_run_the_rest_eagerly_in_order(cache):
    """Each run of capturable steps becomes one graph; an eager step runs between them on every call."""
    double, increment, triple = _AffineStep(2, 0), _AffineStep(1, 1), _AffineStep(3, 0)
    steps = ((True, double), (False, increment), (True, triple))
    values = wp.ones(1, dtype=wp.int32, device=DEVICE)
    cache.arm()

    cache.call_steps("Group_stage", steps, values)
    cache.call_steps("Group_stage", steps, values)

    # the non-commuting steps only give 57 in order: ((1 * 2 + 1) * 3 = 9) -> ((9 * 2 + 1) * 3 = 57)
    assert _read(values) == [57]
    assert (double.calls, increment.calls, triple.calls) == (1, 2, 1)
    assert cache.captured_stages == ("Group_stage[0]", "Group_stage[2]")
