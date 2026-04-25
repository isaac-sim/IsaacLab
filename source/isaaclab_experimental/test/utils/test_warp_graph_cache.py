# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for WarpGraphCache capture/replay and warm-up behavior."""

from __future__ import annotations

import unittest

import warp as wp
from isaaclab_experimental.utils.warp_graph_cache import WarpGraphCache


@wp.kernel
def _add_one(a: wp.array(dtype=wp.float32), b: wp.array(dtype=wp.float32)):
    """Simple warp kernel: b[i] = a[i] + 1."""
    i = wp.tid()
    b[i] = a[i] + 1.0


class TestWarpGraphCache(unittest.TestCase):
    """Tests for :class:`WarpGraphCache`."""

    def setUp(self):
        self.device = "cuda:0"
        self.cache = WarpGraphCache()

    # ------------------------------------------------------------------
    # Warm-up
    # ------------------------------------------------------------------

    def test_warmup_runs_before_capture(self):
        """The function should be called eagerly (warm-up) before graph capture.

        We verify this by counting total invocations on the first call.
        Warm-up = 1, capture = 1, so fn should be called exactly 2 times.
        """
        call_count = [0]
        src = wp.zeros(4, dtype=wp.float32, device=self.device)
        dst = wp.zeros(4, dtype=wp.float32, device=self.device)

        def counted_launch():
            call_count[0] += 1
            wp.launch(_add_one, dim=4, inputs=[src, dst], device=self.device)
            return dst

        # First call: warm-up + capture
        self.cache.capture_or_replay("stage_a", counted_launch)
        self.assertEqual(call_count[0], 2, "First call should invoke fn twice (warm-up + capture)")

        # Second call: replay only
        self.cache.capture_or_replay("stage_a", counted_launch)
        self.assertEqual(call_count[0], 2, "Replay should NOT invoke fn again")

    def test_warmup_flushes_first_call_allocations(self):
        """Warm-up should handle first-call allocations so capture is clean.

        Simulates a hasattr guard pattern: allocate a buffer on first call only.
        Without warm-up, the allocation would be recorded in the graph.
        """
        holder = {}
        src = wp.ones(8, dtype=wp.float32, device=self.device)

        def fn_with_hasattr_guard():
            if "buf" not in holder:
                holder["buf"] = wp.zeros(8, dtype=wp.float32, device=self.device)
            wp.launch(_add_one, dim=8, inputs=[src, holder["buf"]], device=self.device)
            return holder["buf"]

        # Should not raise — warm-up handles the allocation outside capture
        result = self.cache.capture_or_replay("guarded", fn_with_hasattr_guard)
        self.assertIsNotNone(result)

        # Verify the kernel produced correct output
        result_np = result.numpy()
        for val in result_np:
            self.assertAlmostEqual(val, 2.0, places=5)

    # ------------------------------------------------------------------
    # Capture / replay correctness
    # ------------------------------------------------------------------

    def test_capture_produces_correct_output(self):
        """After capture, replaying the graph should produce correct results."""
        src = wp.full(4, value=3.0, dtype=wp.float32, device=self.device)
        dst = wp.zeros(4, dtype=wp.float32, device=self.device)

        def my_fn():
            wp.launch(_add_one, dim=4, inputs=[src, dst], device=self.device)
            return dst

        result = self.cache.capture_or_replay("compute", my_fn)
        result_np = result.numpy()
        for val in result_np:
            self.assertAlmostEqual(val, 4.0, places=5)

    def test_replay_uses_updated_input(self):
        """Replay should re-read from the same input buffer (pointer-stable).

        CUDA graph replay re-executes the same kernel on the same memory
        addresses. If we update the input buffer in-place, the output
        should reflect the new values.
        """
        src = wp.full(4, value=1.0, dtype=wp.float32, device=self.device)
        dst = wp.zeros(4, dtype=wp.float32, device=self.device)

        def my_fn():
            wp.launch(_add_one, dim=4, inputs=[src, dst], device=self.device)
            return dst

        # Capture
        self.cache.capture_or_replay("replay_test", my_fn)

        # Update input in-place
        wp.copy(src, wp.full(4, value=10.0, dtype=wp.float32, device=self.device))

        # Replay — should see updated input
        result = self.cache.capture_or_replay("replay_test", my_fn)
        result_np = result.numpy()
        for val in result_np:
            self.assertAlmostEqual(val, 11.0, places=5)

    def test_cached_result_is_same_reference(self):
        """Replay should return the exact same object reference as capture."""
        src = wp.zeros(4, dtype=wp.float32, device=self.device)
        dst = wp.zeros(4, dtype=wp.float32, device=self.device)

        def my_fn():
            wp.launch(_add_one, dim=4, inputs=[src, dst], device=self.device)
            return dst

        result1 = self.cache.capture_or_replay("ref_test", my_fn)
        result2 = self.cache.capture_or_replay("ref_test", my_fn)
        self.assertIs(result1, result2, "Replay must return the same object reference")

    def test_multiple_stages_independent(self):
        """Different stages should be captured and replayed independently."""
        src_a = wp.full(4, value=1.0, dtype=wp.float32, device=self.device)
        dst_a = wp.zeros(4, dtype=wp.float32, device=self.device)
        src_b = wp.full(4, value=5.0, dtype=wp.float32, device=self.device)
        dst_b = wp.zeros(4, dtype=wp.float32, device=self.device)

        def fn_a():
            wp.launch(_add_one, dim=4, inputs=[src_a, dst_a], device=self.device)
            return dst_a

        def fn_b():
            wp.launch(_add_one, dim=4, inputs=[src_b, dst_b], device=self.device)
            return dst_b

        result_a = self.cache.capture_or_replay("stage_a", fn_a)
        result_b = self.cache.capture_or_replay("stage_b", fn_b)

        self.assertAlmostEqual(result_a.numpy()[0], 2.0, places=5)
        self.assertAlmostEqual(result_b.numpy()[0], 6.0, places=5)

        # Replay both
        result_a2 = self.cache.capture_or_replay("stage_a", fn_a)
        result_b2 = self.cache.capture_or_replay("stage_b", fn_b)
        self.assertIs(result_a, result_a2)
        self.assertIs(result_b, result_b2)

    # ------------------------------------------------------------------
    # Invalidation
    # ------------------------------------------------------------------

    def test_invalidate_all(self):
        """invalidate() with no args should drop all cached graphs."""
        src = wp.zeros(4, dtype=wp.float32, device=self.device)
        dst = wp.zeros(4, dtype=wp.float32, device=self.device)

        call_count = [0]

        def my_fn():
            call_count[0] += 1
            wp.launch(_add_one, dim=4, inputs=[src, dst], device=self.device)
            return dst

        self.cache.capture_or_replay("s1", my_fn)
        self.assertEqual(call_count[0], 2)  # warm-up + capture

        self.cache.invalidate()

        # After invalidation, next call should re-warm-up and re-capture
        self.cache.capture_or_replay("s1", my_fn)
        self.assertEqual(call_count[0], 4)  # 2 more (warm-up + capture)

    def test_invalidate_single_stage(self):
        """invalidate(stage) should only drop the named stage."""
        src = wp.zeros(4, dtype=wp.float32, device=self.device)
        dst_a = wp.zeros(4, dtype=wp.float32, device=self.device)
        dst_b = wp.zeros(4, dtype=wp.float32, device=self.device)

        count_a = [0]
        count_b = [0]

        def fn_a():
            count_a[0] += 1
            wp.launch(_add_one, dim=4, inputs=[src, dst_a], device=self.device)
            return dst_a

        def fn_b():
            count_b[0] += 1
            wp.launch(_add_one, dim=4, inputs=[src, dst_b], device=self.device)
            return dst_b

        self.cache.capture_or_replay("a", fn_a)
        self.cache.capture_or_replay("b", fn_b)
        self.assertEqual(count_a[0], 2)
        self.assertEqual(count_b[0], 2)

        # Invalidate only "a"
        self.cache.invalidate("a")

        self.cache.capture_or_replay("a", fn_a)
        self.cache.capture_or_replay("b", fn_b)
        self.assertEqual(count_a[0], 4, "Stage 'a' should re-capture after invalidation")
        self.assertEqual(count_b[0], 2, "Stage 'b' should replay (not re-capture)")

    def test_invalidate_nonexistent_stage_is_noop(self):
        """Invalidating a stage that was never captured should not raise."""
        self.cache.invalidate("nonexistent")  # should not raise

    # ------------------------------------------------------------------
    # Args / kwargs forwarding
    # ------------------------------------------------------------------

    def test_args_and_kwargs_forwarded(self):
        """capture_or_replay should forward args and kwargs to fn."""
        src = wp.full(4, value=2.0, dtype=wp.float32, device=self.device)
        dst = wp.zeros(4, dtype=wp.float32, device=self.device)

        def my_fn(a, b, device="cuda:0"):
            wp.launch(_add_one, dim=4, inputs=[a, b], device=device)
            return b

        result = self.cache.capture_or_replay(
            "args_test",
            my_fn,
            args=(src, dst),
            kwargs={"device": self.device},
        )
        self.assertAlmostEqual(result.numpy()[0], 3.0, places=5)


if __name__ == "__main__":
    unittest.main()
