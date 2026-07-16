# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for WarpLaunchCache record-and-replay behavior."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import warp as wp

from isaaclab.utils.warp import ProxyArray, WarpLaunchCache


@wp.kernel
def _affine(
    source: wp.array(dtype=wp.float32),
    scale: float,
    output: wp.array(dtype=wp.float32),
):
    """Apply a scalar multiplier to an array."""
    index = wp.tid()
    output[index] = source[index] * scale


@wp.kernel
def _increment(output: wp.array(dtype=wp.float32)):
    """Increment every launched output element."""
    index = wp.tid()
    output[index] = output[index] + 1.0


@wp.kernel
def _write_index(output: wp.array(dtype=wp.float32)):
    """Write one-based thread indices to an output array."""
    index = wp.tid()
    output[index] = wp.float32(index + 1)


class TestWarpLaunchCache(unittest.TestCase):
    """Tests for :class:`WarpLaunchCache`."""

    @classmethod
    def setUpClass(cls):
        wp.init()
        if not wp.is_cuda_available():
            raise unittest.SkipTest("CUDA is required for Warp launch replay tests.")

    def setUp(self):
        self.device = wp.get_device("cuda:0")
        self.cache = WarpLaunchCache(device=self.device)

    def tearDown(self):
        wp.synchronize_device(self.device)

    def test_static_launch_records_once_then_replays(self):
        """The first call should record and execute, then later calls should replay."""
        output = wp.zeros(8, dtype=wp.float32, device=self.device)

        self.cache.launch(_increment, dim=8, inputs=[output])
        self.cache.launch(_increment, dim=8, inputs=[output])

        self.assertEqual(len(self.cache._entries), 1)
        np.testing.assert_array_equal(output.numpy(), np.full(8, 2.0, dtype=np.float32))

    def test_fixed_device_cache_is_callable_from_its_owner(self):
        """An owner should retain and call one fixed-device launcher."""
        owner = SimpleNamespace(_warp_launch=WarpLaunchCache(device=self.device))
        output = wp.zeros(4, dtype=wp.float32, device=self.device)

        owner._warp_launch(_increment, dim=4, inputs=[output])
        owner._warp_launch(_increment, dim=4, inputs=[output])

        np.testing.assert_array_equal(output.numpy(), np.full(4, 2.0, dtype=np.float32))

    def test_sites_distinguish_persistent_argument_sets(self):
        """One kernel should own independent commands for distinct stable sites."""
        output_a = wp.zeros(4, dtype=wp.float32, device=self.device)
        output_b = wp.zeros(4, dtype=wp.float32, device=self.device)

        self.cache(_increment, dim=4, inputs=[output_a], site=output_a)
        self.cache(_increment, dim=4, inputs=[output_b], site=output_b)
        self.cache(_increment, dim=4, inputs=[output_a], site=output_a)
        self.cache(_increment, dim=4, inputs=[output_b], site=output_b)

        self.assertEqual(len(self.cache._entries), 2)
        np.testing.assert_array_equal(output_a.numpy(), np.full(4, 2.0, dtype=np.float32))
        np.testing.assert_array_equal(output_b.numpy(), np.full(4, 2.0, dtype=np.float32))

    def test_dim_change_updates_recorded_bounds(self):
        """A changed launch dimension should update the cached command bounds."""
        output = wp.zeros(8, dtype=wp.float32, device=self.device)

        self.cache(_write_index, dim=4, inputs=[output])
        self.cache(_write_index, dim=8, inputs=[output])

        np.testing.assert_array_equal(output.numpy(), np.arange(1, 9, dtype=np.float32))

    def test_eager_environment_mode_uses_current_arguments(self):
        """The eager switch should bypass recording and use each call's arguments."""
        source = wp.full(4, value=2.0, dtype=wp.float32, device=self.device)
        output = wp.zeros(4, dtype=wp.float32, device=self.device)

        with patch.dict(os.environ, {"ISAACLAB_WARP_LAUNCH_MODE": "eager"}):
            cache = WarpLaunchCache(device=self.device)
            cache(_affine, dim=4, inputs=[source, 2.0], outputs=[output])
            cache(_affine, dim=4, inputs=[source, 5.0], outputs=[output])

        self.assertFalse(cache._entries)
        np.testing.assert_array_equal(output.numpy(), np.full(4, 10.0, dtype=np.float32))

    def test_disabled_cache_launches_eagerly(self):
        """A disabled owner cache should remain a direct-launch fallback."""
        cache = WarpLaunchCache(device=self.device, enabled=False)
        source = wp.full(4, value=2.0, dtype=wp.float32, device=self.device)
        output = wp.zeros(4, dtype=wp.float32, device=self.device)

        cache(_affine, dim=4, inputs=[source, 2.0], outputs=[output])
        cache(_affine, dim=4, inputs=[source, 3.0], outputs=[output])

        self.assertFalse(cache._entries)
        np.testing.assert_array_equal(output.numpy(), np.full(4, 6.0, dtype=np.float32))

    def test_debug_mode_rejects_changed_static_array(self):
        """Debug mode should detect replacement of persistent array storage."""
        source_a = wp.ones(4, dtype=wp.float32, device=self.device)
        source_b = wp.ones(4, dtype=wp.float32, device=self.device)
        output = wp.zeros(4, dtype=wp.float32, device=self.device)
        cache = WarpLaunchCache(device=self.device, debug=True)

        cache(_affine, dim=4, inputs=[source_a, 1.0], outputs=[output])
        with self.assertRaisesRegex(RuntimeError, "static argument 0 changed"):
            cache(_affine, dim=4, inputs=[source_b, 1.0], outputs=[output])

    def test_debug_mode_rejects_repointed_torch_tensor(self):
        """Debug mode should detect a tensor object repointed to new storage."""
        source = torch.ones(4, dtype=torch.float32, device=str(self.device))
        replacement = torch.full_like(source, 3.0)
        output = wp.zeros(4, dtype=wp.float32, device=self.device)
        cache = WarpLaunchCache(device=self.device, debug=True)

        cache(_affine, dim=4, inputs=[source, 2.0], outputs=[output])
        source.set_(replacement)

        with self.assertRaisesRegex(RuntimeError, "static argument 0 changed"):
            cache(_affine, dim=4, inputs=[source, 2.0], outputs=[output])

    def test_debug_mode_rejects_changed_static_scalar(self):
        """Debug mode should reject a changed recorded scalar value."""
        source = wp.ones(4, dtype=wp.float32, device=self.device)
        output = wp.zeros(4, dtype=wp.float32, device=self.device)
        cache = WarpLaunchCache(device=self.device, debug=True)

        cache(_affine, dim=4, inputs=[source, 1.0], outputs=[output])
        with self.assertRaisesRegex(RuntimeError, "static argument 1 changed"):
            cache(_affine, dim=4, inputs=[source, 2.0], outputs=[output])

    def test_debug_mode_accepts_new_proxy_for_same_array(self):
        """Debug mode should compare ProxyArray storage instead of wrapper identity."""
        source = wp.ones(4, dtype=wp.float32, device=self.device)
        output = wp.zeros(4, dtype=wp.float32, device=self.device)
        cache = WarpLaunchCache(device=self.device, debug=True)

        cache(_affine, dim=4, inputs=[ProxyArray(source), 2.0], outputs=[output])
        cache(_affine, dim=4, inputs=[ProxyArray(source), 2.0], outputs=[output])

        np.testing.assert_array_equal(output.numpy(), np.full(4, 2.0, dtype=np.float32))

    def test_zero_dim_is_noop_and_does_not_populate_cache(self):
        """A zero-sized launch should neither execute nor reserve a command."""
        output = wp.zeros(4, dtype=wp.float32, device=self.device)

        self.cache(_increment, dim=(4, 0), inputs=[output])

        self.assertFalse(self.cache._entries)
        np.testing.assert_array_equal(output.numpy(), np.zeros(4, dtype=np.float32))

    def test_fixed_device_fast_path_skips_device_lookup(self):
        """A fixed-device cache should not resolve its device after construction."""
        output = wp.zeros(4, dtype=wp.float32, device=self.device)
        self.cache(_increment, dim=4, inputs=[output])

        with patch.object(wp, "get_device", side_effect=AssertionError("unexpected device lookup")):
            self.cache(_increment, dim=4, inputs=[output])

        np.testing.assert_array_equal(output.numpy(), np.full(4, 2.0, dtype=np.float32))

    def test_invalidate_site_and_reset(self):
        """Invalidation should drop one stable site or every recorded command."""
        output_a = wp.zeros(1, dtype=wp.float32, device=self.device)
        output_b = wp.zeros(1, dtype=wp.float32, device=self.device)
        self.cache(_increment, dim=1, inputs=[output_a], site="a")
        self.cache(_increment, dim=1, inputs=[output_b], site="b")

        self.cache.invalidate(site="a")
        self.assertEqual(len(self.cache._entries), 1)
        self.cache.reset()
        self.assertFalse(self.cache._entries)

    def test_invalidate_accepts_tensor_site_identity(self):
        """Invalidation should accept a tensor site without evaluating elementwise equality."""
        output = wp.zeros(1, dtype=wp.float32, device=self.device)
        site = torch.arange(2, device=str(self.device))
        self.cache(_increment, dim=1, inputs=[output], site=site)

        self.cache.invalidate(site=site)

        self.assertFalse(self.cache._entries)

    def test_reset_synchronizes_only_nonempty_cache_device(self):
        """Reset should drain recorded commands without syncing an empty cache."""
        empty_cache = WarpLaunchCache(device=self.device, enabled=False)
        with patch.object(wp, "synchronize_device", wraps=wp.synchronize_device) as synchronize:
            empty_cache.reset()
            synchronize.assert_not_called()

            output = wp.zeros(1, dtype=wp.float32, device=self.device)
            self.cache(_increment, dim=1, inputs=[output])
            self.cache.reset()
            synchronize.assert_called_once_with(self.device)

    def test_replay_can_be_recorded_inside_cuda_capture(self):
        """A warmed command replay should compose with CUDA graph capture."""
        output = wp.zeros(4, dtype=wp.float32, device=self.device)
        self.cache(_increment, dim=4, inputs=[output])

        with wp.ScopedCapture(device=self.device) as capture:
            self.cache(_increment, dim=4, inputs=[output])

        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(output.numpy(), np.full(4, 2.0, dtype=np.float32))

    def test_captured_graph_survives_launch_cache_reset(self):
        """A graph should retain a captured replay after its Python command is reset."""
        output = wp.zeros(4, dtype=wp.float32, device=self.device)
        self.cache(_increment, dim=4, inputs=[output])

        with wp.ScopedCapture(device=self.device) as capture:
            self.cache(_increment, dim=4, inputs=[output])

        self.cache.reset()
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(output.numpy(), np.full(4, 2.0, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
