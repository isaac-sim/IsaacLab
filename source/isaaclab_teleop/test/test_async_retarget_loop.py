# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Tests for :class:`AsyncRetargetLoop` (pure Python, no Omniverse required)."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
import torch
from isaaclab_teleop.async_retarget_loop import AsyncRetargetLoop, EmaTimingEstimatorCfg

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_EXPECTED = torch.ones(3)


@pytest.fixture
def make_loop():
    """Factory fixture: creates a loop, yields it, and guarantees stop() on teardown."""
    loops: list[AsyncRetargetLoop] = []

    def _factory(step_fn, **kwargs) -> AsyncRetargetLoop:
        loop = AsyncRetargetLoop(step_fn, **kwargs)
        loops.append(loop)
        return loop

    yield _factory

    for loop in loops:
        loop.stop()


def _constant_step_fn(
    _anchor: np.ndarray,
    _target: np.ndarray | torch.Tensor | None,
) -> torch.Tensor:
    return torch.ones(3)


def _sleeping_step_fn(sleep_s: float = 0.005):
    """Return a step_fn that sleeps for a fixed duration before returning."""

    def _fn(
        _anchor: np.ndarray,
        _target: np.ndarray | torch.Tensor | None,
    ) -> torch.Tensor:
        time.sleep(sleep_s)
        return torch.ones(3)

    return _fn


# ---------------------------------------------------------------------------
# TestBlockingConsume
# ---------------------------------------------------------------------------


class TestBlockingConsume:
    def test_consume_returns_fresh_result(self, make_loop):
        loop = make_loop(_constant_step_fn)
        loop.start()

        action, fresh = loop.consume(block=True)

        assert fresh
        assert action is not None
        torch.testing.assert_close(action, _EXPECTED)

    def test_second_consume_waits_for_new_generation(self, make_loop):
        loop = make_loop(_constant_step_fn)
        loop.start()

        _, fresh1 = loop.consume(block=True)
        assert fresh1

        gen_before = loop._consumed_gen
        _, fresh2 = loop.consume(block=True)
        assert fresh2
        assert loop._consumed_gen > gen_before


# ---------------------------------------------------------------------------
# TestNonBlockingConsume
# ---------------------------------------------------------------------------


class TestNonBlockingConsume:
    def test_non_blocking_returns_immediately(self, make_loop):
        loop = make_loop(_sleeping_step_fn(sleep_s=0.5))
        loop.start()

        t0 = time.monotonic()
        _action, _fresh = loop.consume(block=False)
        elapsed = time.monotonic() - t0

        assert elapsed < 0.1

    def test_non_blocking_stale_flag(self, make_loop):
        loop = make_loop(_sleeping_step_fn(sleep_s=0.02))
        loop.start()

        loop.consume(block=True)

        _action, fresh = loop.consume(block=False)
        assert not fresh


# ---------------------------------------------------------------------------
# TestEMAConvergence
# ---------------------------------------------------------------------------


class TestEMAConvergence:
    def test_step_period_converges(self, make_loop):
        cadence_s = 0.020
        loop = make_loop(_constant_step_fn, timing_cfg=EmaTimingEstimatorCfg(ema_alpha=0.3))
        loop.start()

        for _ in range(20):
            loop.consume(block=True)
            time.sleep(cadence_s)

        assert loop._timing.step_period == pytest.approx(cadence_s, rel=0.5)

    def test_retarget_dur_converges(self, make_loop):
        sleep_s = 0.005
        loop = make_loop(_sleeping_step_fn(sleep_s=sleep_s), timing_cfg=EmaTimingEstimatorCfg(ema_alpha=0.3))
        loop.start()

        for _ in range(20):
            loop.consume(block=True)

        assert loop._timing.retarget_dur == pytest.approx(sleep_s, rel=0.5)


# ---------------------------------------------------------------------------
# TestErrorPropagation
# ---------------------------------------------------------------------------


class TestErrorPropagation:
    def test_persistent_failure_surfaces_after_max(self, make_loop):
        def _always_fail(_a, _t):
            raise ValueError("boom")

        loop = make_loop(_always_fail)
        loop.start()

        with pytest.raises(RuntimeError, match="Background retarget thread failed") as exc_info:
            for _ in range(20):
                loop.consume(block=True)
                time.sleep(0.05)

        assert isinstance(exc_info.value.__cause__, ValueError)

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    def test_dead_thread_raises_instead_of_deadlock(self, make_loop):
        def _raise_base_exception(_a, _t):
            raise KeyboardInterrupt("simulated")

        loop = make_loop(_raise_base_exception)
        loop.start()

        # Give the thread time to die.
        time.sleep(0.3)
        assert not loop._thread.is_alive()

        with pytest.raises(RuntimeError, match="died unexpectedly"):
            loop.consume(block=True)

    def test_transient_failures_recover(self, make_loop):
        call_count = 0
        lock = threading.Lock()

        def _fail_then_succeed(_a, _t):
            nonlocal call_count
            with lock:
                call_count += 1
                n = call_count
            if n <= 2:
                raise ValueError(f"transient failure {n}")
            return torch.ones(3)

        loop = make_loop(_fail_then_succeed)
        loop.start()

        deadline = time.monotonic() + 5.0
        action = None
        while time.monotonic() < deadline:
            action, fresh = loop.consume(block=True)
            if fresh and action is not None:
                break

        assert action is not None
        torch.testing.assert_close(action, _EXPECTED)


# ---------------------------------------------------------------------------
# TestInputIsolation
# ---------------------------------------------------------------------------


class TestInputIsolation:
    def test_update_inputs_copies_arrays(self, make_loop):
        loop = make_loop(_constant_step_fn)

        original = np.eye(4, dtype=np.float32)
        loop.update_inputs(original)

        original[:3, 3] = [99.0, 99.0, 99.0]

        np.testing.assert_array_equal(
            loop._anchor_matrix[:3, 3],
            [0.0, 0.0, 0.0],
        )

    def test_update_inputs_clones_tensor(self, make_loop):
        loop = make_loop(_constant_step_fn)

        original = torch.eye(4)
        loop.update_inputs(np.eye(4, dtype=np.float32), target_T_world=original)

        original[0, 3] = 99.0

        stored = loop._target_T_world
        assert isinstance(stored, torch.Tensor)
        assert stored[0, 3].item() == 0.0


# ---------------------------------------------------------------------------
# TestLifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_stop_is_idempotent(self, make_loop):
        loop = make_loop(_constant_step_fn)
        loop.start()
        loop.consume(block=True)

        loop.stop()
        loop.stop()

    def test_start_after_stop(self, make_loop):
        loop = make_loop(_constant_step_fn)

        loop.start()
        action1, _ = loop.consume(block=True)
        loop.stop()

        loop.start()
        action2, _ = loop.consume(block=True)
        loop.stop()

        assert action1 is not None
        assert action2 is not None
        torch.testing.assert_close(action1, _EXPECTED)
        torch.testing.assert_close(action2, _EXPECTED)

    def test_action_is_cloned(self, make_loop):
        shared_tensor = torch.ones(3)

        def _return_shared(_a, _t):
            return shared_tensor

        loop = make_loop(_return_shared)
        loop.start()

        action, _ = loop.consume(block=True)

        assert action is not shared_tensor
        assert action.data_ptr() != shared_tensor.data_ptr()
        torch.testing.assert_close(action, shared_tensor)
