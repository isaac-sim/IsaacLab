# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for ManagerCallSwitch config loading, mode resolution, and dispatch."""

from __future__ import annotations

import json
import os
import unittest

import warp as wp
from isaaclab_experimental.utils.manager_call_switch import ManagerCallMode, ManagerCallSwitch


@wp.kernel
def _add_one(a: wp.array(dtype=wp.float32), b: wp.array(dtype=wp.float32)):
    i = wp.tid()
    b[i] = a[i] + 1.0


class TestManagerCallMode(unittest.TestCase):
    """Tests for the ManagerCallMode enum."""

    def test_enum_values(self):
        self.assertEqual(ManagerCallMode.STABLE, 0)
        self.assertEqual(ManagerCallMode.WARP_NOT_CAPTURED, 1)
        self.assertEqual(ManagerCallMode.WARP_CAPTURED, 2)

    def test_ordering(self):
        self.assertLess(ManagerCallMode.STABLE, ManagerCallMode.WARP_NOT_CAPTURED)
        self.assertLess(ManagerCallMode.WARP_NOT_CAPTURED, ManagerCallMode.WARP_CAPTURED)


# ======================================================================
# Config loading
# ======================================================================


class TestConfigLoading(unittest.TestCase):
    """Tests for ManagerCallSwitch config parsing from dict, JSON, env var, and None."""

    def test_none_uses_default(self):
        switch = ManagerCallSwitch(cfg_source=None)
        self.assertEqual(switch.get_mode_for_manager("RewardManager"), ManagerCallMode.WARP_CAPTURED)

    def test_dict_config(self):
        switch = ManagerCallSwitch(cfg_source={"default": 1})
        self.assertEqual(switch.get_mode_for_manager("RewardManager"), ManagerCallMode.WARP_NOT_CAPTURED)

    def test_dict_per_manager_override(self):
        switch = ManagerCallSwitch(cfg_source={"default": 2, "RewardManager": 0})
        self.assertEqual(switch.get_mode_for_manager("RewardManager"), ManagerCallMode.STABLE)
        self.assertEqual(switch.get_mode_for_manager("ActionManager"), ManagerCallMode.WARP_CAPTURED)

    def test_dict_without_default_key(self):
        """A dict missing 'default' should inherit from DEFAULT_CONFIG."""
        switch = ManagerCallSwitch(cfg_source={"RewardManager": 0})
        # default should be 2 (from DEFAULT_CONFIG)
        self.assertEqual(switch.get_mode_for_manager("ActionManager"), ManagerCallMode.WARP_CAPTURED)
        self.assertEqual(switch.get_mode_for_manager("RewardManager"), ManagerCallMode.STABLE)

    def test_json_string_config(self):
        cfg_str = json.dumps({"default": 1, "ObservationManager": 0})
        switch = ManagerCallSwitch(cfg_source=cfg_str)
        self.assertEqual(switch.get_mode_for_manager("ObservationManager"), ManagerCallMode.STABLE)
        self.assertEqual(switch.get_mode_for_manager("RewardManager"), ManagerCallMode.WARP_NOT_CAPTURED)

    def test_empty_string_uses_default(self):
        switch = ManagerCallSwitch(cfg_source="")
        self.assertEqual(switch.get_mode_for_manager("RewardManager"), ManagerCallMode.WARP_CAPTURED)

    def test_env_var_fallback(self):
        """When cfg_source is None, should read from MANAGER_CALL_CONFIG env var."""
        old = os.environ.get(ManagerCallSwitch.ENV_VAR)
        try:
            os.environ[ManagerCallSwitch.ENV_VAR] = json.dumps({"default": 0})
            switch = ManagerCallSwitch(cfg_source=None)
            self.assertEqual(switch.get_mode_for_manager("RewardManager"), ManagerCallMode.STABLE)
        finally:
            if old is None:
                os.environ.pop(ManagerCallSwitch.ENV_VAR, None)
            else:
                os.environ[ManagerCallSwitch.ENV_VAR] = old

    def test_invalid_json_raises(self):
        with self.assertRaises(json.JSONDecodeError):
            ManagerCallSwitch(cfg_source="not valid json")

    def test_invalid_mode_value_raises(self):
        with self.assertRaises(ValueError):
            ManagerCallSwitch(cfg_source={"default": 99})

    def test_non_int_mode_raises(self):
        with self.assertRaises(TypeError):
            ManagerCallSwitch(cfg_source={"default": "fast"})

    def test_invalid_cfg_type_raises(self):
        with self.assertRaises(TypeError):
            ManagerCallSwitch(cfg_source=42)


# ======================================================================
# Mode resolution and capping
# ======================================================================


class TestModeResolution(unittest.TestCase):
    """Tests for mode resolution, MAX_MODE_OVERRIDES, and dynamic capturability."""

    def test_max_mode_override_caps_default(self):
        """Scene is capped to WARP_NOT_CAPTURED by MAX_MODE_OVERRIDES."""
        switch = ManagerCallSwitch(cfg_source={"default": 2})
        # Scene has a static cap of WARP_NOT_CAPTURED
        self.assertEqual(
            switch.get_mode_for_manager("Scene"),
            ManagerCallMode.WARP_NOT_CAPTURED,
        )
        # Other managers should not be capped
        self.assertEqual(
            switch.get_mode_for_manager("RewardManager"),
            ManagerCallMode.WARP_CAPTURED,
        )

    def test_max_modes_kwarg(self):
        """max_modes kwarg should cap managers beyond MAX_MODE_OVERRIDES."""
        switch = ManagerCallSwitch(
            cfg_source={"default": 2},
            max_modes={"RewardManager": ManagerCallMode.WARP_NOT_CAPTURED},
        )
        self.assertEqual(
            switch.get_mode_for_manager("RewardManager"),
            ManagerCallMode.WARP_NOT_CAPTURED,
        )

    def test_register_manager_capturability_downgrades(self):
        """register_manager_capturability(False) should cap a manager to WARP_NOT_CAPTURED."""
        switch = ManagerCallSwitch(cfg_source={"default": 2})
        self.assertEqual(switch.get_mode_for_manager("RewardManager"), ManagerCallMode.WARP_CAPTURED)

        switch.register_manager_capturability("RewardManager", capturable=False)
        self.assertEqual(
            switch.get_mode_for_manager("RewardManager"),
            ManagerCallMode.WARP_NOT_CAPTURED,
        )

    def test_register_capturability_true_is_noop(self):
        """register_manager_capturability(True) should not change anything."""
        switch = ManagerCallSwitch(cfg_source={"default": 2})
        switch.register_manager_capturability("RewardManager", capturable=True)
        self.assertEqual(switch.get_mode_for_manager("RewardManager"), ManagerCallMode.WARP_CAPTURED)

    def test_register_capturability_does_not_upgrade(self):
        """If a manager is already capped to NOT_CAPTURED, registering capturable=False again shouldn't change it."""
        switch = ManagerCallSwitch(cfg_source={"default": 2})
        switch.register_manager_capturability("RewardManager", capturable=False)
        switch.register_manager_capturability("RewardManager", capturable=False)
        self.assertEqual(
            switch.get_mode_for_manager("RewardManager"),
            ManagerCallMode.WARP_NOT_CAPTURED,
        )

    def test_capturability_interacts_with_static_cap(self):
        """Dynamic capturability should respect existing static caps."""
        switch = ManagerCallSwitch(
            cfg_source={"default": 2},
            max_modes={"RewardManager": ManagerCallMode.WARP_NOT_CAPTURED},
        )
        # Already capped, register_capturability(False) should be harmless
        switch.register_manager_capturability("RewardManager", capturable=False)
        self.assertEqual(
            switch.get_mode_for_manager("RewardManager"),
            ManagerCallMode.WARP_NOT_CAPTURED,
        )

    def test_mode_0_not_affected_by_cap(self):
        """A manager explicitly set to STABLE (0) should stay at 0 even with cap at 1."""
        switch = ManagerCallSwitch(cfg_source={"default": 2, "RewardManager": 0})
        switch.register_manager_capturability("RewardManager", capturable=False)
        self.assertEqual(switch.get_mode_for_manager("RewardManager"), ManagerCallMode.STABLE)


# ======================================================================
# Stage dispatch
# ======================================================================


class TestStageDispatch(unittest.TestCase):
    """Tests for call_stage routing through stable / warp-eager / warp-captured paths."""

    def test_stable_mode_calls_stable_fn(self):
        switch = ManagerCallSwitch(cfg_source={"default": 0})
        called = {"stable": False, "warp": False}

        def stable_fn():
            called["stable"] = True
            return "stable_result"

        def warp_fn():
            called["warp"] = True
            return "warp_result"

        result = switch.call_stage(
            stage="RewardManager_compute",
            warp_call={"fn": warp_fn},
            stable_call={"fn": stable_fn},
        )
        self.assertTrue(called["stable"])
        self.assertFalse(called["warp"])
        self.assertEqual(result, "stable_result")

    def test_stable_mode_without_stable_call_raises(self):
        switch = ManagerCallSwitch(cfg_source={"default": 0})
        with self.assertRaises(ValueError):
            switch.call_stage(
                stage="RewardManager_compute",
                warp_call={"fn": lambda: None},
                stable_call=None,
            )

    def test_warp_eager_mode_calls_warp_fn(self):
        switch = ManagerCallSwitch(cfg_source={"default": 1})
        called = {"stable": False, "warp": False}

        def stable_fn():
            called["stable"] = True
            return "stable_result"

        def warp_fn():
            called["warp"] = True
            return "warp_result"

        result = switch.call_stage(
            stage="RewardManager_compute",
            warp_call={"fn": warp_fn},
            stable_call={"fn": stable_fn},
        )
        self.assertFalse(called["stable"])
        self.assertTrue(called["warp"])
        self.assertEqual(result, "warp_result")

    def test_output_transform(self):
        """The 'output' key in call spec should transform the return value."""
        switch = ManagerCallSwitch(cfg_source={"default": 1})

        def warp_fn():
            return [1, 2, 3]

        result = switch.call_stage(
            stage="RewardManager_compute",
            warp_call={"fn": warp_fn, "output": len},
        )
        self.assertEqual(result, 3)

    def test_args_and_kwargs_forwarded(self):
        """call_stage should forward args and kwargs from the call spec."""
        switch = ManagerCallSwitch(cfg_source={"default": 1})
        received = {}

        def warp_fn(a, b, key=None):
            received["a"] = a
            received["b"] = b
            received["key"] = key

        switch.call_stage(
            stage="RewardManager_compute",
            warp_call={"fn": warp_fn, "args": (1, 2), "kwargs": {"key": "val"}},
        )
        self.assertEqual(received, {"a": 1, "b": 2, "key": "val"})


class TestStageDispatchCaptured(unittest.TestCase):
    """Tests for WARP_CAPTURED mode (requires GPU)."""

    def setUp(self):
        self.device = "cuda:0"

    def test_captured_mode_produces_correct_output(self):
        """WARP_CAPTURED should capture and replay a warp kernel correctly."""
        switch = ManagerCallSwitch(cfg_source={"default": 2})
        src = wp.full(4, value=5.0, dtype=wp.float32, device=self.device)
        dst = wp.zeros(4, dtype=wp.float32, device=self.device)

        def warp_fn():
            wp.launch(_add_one, dim=4, inputs=[src, dst], device=self.device)
            return dst

        # First call: warm-up + capture
        result = switch.call_stage(
            stage="RewardManager_compute",
            warp_call={"fn": warp_fn},
        )
        self.assertAlmostEqual(result.numpy()[0], 6.0, places=5)

        # Replay
        wp.copy(src, wp.full(4, value=10.0, dtype=wp.float32, device=self.device))
        result2 = switch.call_stage(
            stage="RewardManager_compute",
            warp_call={"fn": warp_fn},
        )
        self.assertIs(result, result2, "Replay must return same reference")
        self.assertAlmostEqual(result2.numpy()[0], 11.0, places=5)

    def test_captured_warmup_call_count(self):
        """WARP_CAPTURED first call should invoke fn exactly 2 times (warm-up + capture)."""
        switch = ManagerCallSwitch(cfg_source={"default": 2})
        src = wp.zeros(4, dtype=wp.float32, device=self.device)
        dst = wp.zeros(4, dtype=wp.float32, device=self.device)
        call_count = [0]

        def warp_fn():
            call_count[0] += 1
            wp.launch(_add_one, dim=4, inputs=[src, dst], device=self.device)
            return dst

        # First call_stage: warm-up (1) + capture (2)
        switch.call_stage(stage="RewardManager_compute", warp_call={"fn": warp_fn})
        self.assertEqual(call_count[0], 2, "First captured call should invoke fn twice (warm-up + capture)")

        # Second call_stage: replay only — no new fn invocation
        switch.call_stage(stage="RewardManager_compute", warp_call={"fn": warp_fn})
        self.assertEqual(call_count[0], 2, "Replay should not invoke fn again")

        # Third call_stage: still replay
        switch.call_stage(stage="RewardManager_compute", warp_call={"fn": warp_fn})
        self.assertEqual(call_count[0], 2)

    def test_captured_warmup_handles_hasattr_guard(self):
        """Warm-up should flush first-call allocations so capture doesn't record them.

        This simulates the real-world pattern where MDP terms allocate scratch
        buffers on first call using hasattr guards.
        """
        switch = ManagerCallSwitch(cfg_source={"default": 2})
        src = wp.ones(8, dtype=wp.float32, device=self.device)
        holder = {}

        def fn_with_guard():
            if "buf" not in holder:
                # First-call allocation — must happen during warm-up, not capture
                holder["buf"] = wp.zeros(8, dtype=wp.float32, device=self.device)
            wp.launch(_add_one, dim=8, inputs=[src, holder["buf"]], device=self.device)
            return holder["buf"]

        # Should not raise — warm-up handles the allocation outside capture context
        result = switch.call_stage(
            stage="RewardManager_compute",
            warp_call={"fn": fn_with_guard},
        )
        result_np = result.numpy()
        for val in result_np:
            self.assertAlmostEqual(val, 2.0, places=5)

        # Replay should also work (allocation already done, only kernel replays)
        wp.copy(src, wp.full(8, value=5.0, dtype=wp.float32, device=self.device))
        result2 = switch.call_stage(
            stage="RewardManager_compute",
            warp_call={"fn": fn_with_guard},
        )
        for val in result2.numpy():
            self.assertAlmostEqual(val, 6.0, places=5)

    def test_warp_eager_no_warmup(self):
        """WARP_NOT_CAPTURED mode should call fn exactly once per call (no warm-up)."""
        switch = ManagerCallSwitch(cfg_source={"default": 1})
        call_count = [0]

        def warp_fn():
            call_count[0] += 1

        switch.call_stage(stage="RewardManager_compute", warp_call={"fn": warp_fn})
        self.assertEqual(call_count[0], 1, "Eager mode should call fn exactly once")

        switch.call_stage(stage="RewardManager_compute", warp_call={"fn": warp_fn})
        self.assertEqual(call_count[0], 2, "Eager mode should call fn again each time")


# ======================================================================
# Graph invalidation
# ======================================================================


class TestGraphInvalidation(unittest.TestCase):
    """Tests for invalidate_graphs on ManagerCallSwitch."""

    def setUp(self):
        self.device = "cuda:0"

    def test_invalidate_forces_recapture(self):
        switch = ManagerCallSwitch(cfg_source={"default": 2})
        src = wp.full(4, value=1.0, dtype=wp.float32, device=self.device)
        dst = wp.zeros(4, dtype=wp.float32, device=self.device)

        call_count = [0]

        def warp_fn():
            call_count[0] += 1
            wp.launch(_add_one, dim=4, inputs=[src, dst], device=self.device)
            return dst

        # First capture
        switch.call_stage(stage="RewardManager_compute", warp_call={"fn": warp_fn})
        self.assertEqual(call_count[0], 2)  # warm-up + capture

        # Replay — no new calls
        switch.call_stage(stage="RewardManager_compute", warp_call={"fn": warp_fn})
        self.assertEqual(call_count[0], 2)

        # Invalidate
        switch.invalidate_graphs()

        # Should re-warm-up + re-capture
        switch.call_stage(stage="RewardManager_compute", warp_call={"fn": warp_fn})
        self.assertEqual(call_count[0], 4)


# ======================================================================
# resolve_manager_class
# ======================================================================


class TestResolveManagerClass(unittest.TestCase):
    """Tests for resolve_manager_class."""

    def test_stable_resolves_to_isaaclab_managers(self):
        switch = ManagerCallSwitch(cfg_source={"default": 0})
        cls = switch.resolve_manager_class("RewardManager")
        self.assertTrue(cls.__module__.startswith("isaaclab.managers"))

    def test_warp_resolves_to_experimental_managers(self):
        switch = ManagerCallSwitch(cfg_source={"default": 1})
        cls = switch.resolve_manager_class("RewardManager")
        self.assertTrue(cls.__module__.startswith("isaaclab_experimental"))

    def test_mode_override(self):
        """mode_override should bypass the config for class resolution."""
        switch = ManagerCallSwitch(cfg_source={"default": 2})
        cls = switch.resolve_manager_class("RewardManager", mode_override=ManagerCallMode.STABLE)
        self.assertTrue(cls.__module__.startswith("isaaclab.managers"))

    def test_invalid_manager_raises(self):
        switch = ManagerCallSwitch(cfg_source={"default": 0})
        with self.assertRaises(AttributeError):
            switch.resolve_manager_class("NonexistentManager")


# ======================================================================
# Manager name parsing
# ======================================================================


class TestManagerNameParsing(unittest.TestCase):
    """Tests for stage name → manager name extraction."""

    def test_valid_stage_name(self):
        switch = ManagerCallSwitch(cfg_source={"default": 1})
        # Dispatch a stage with valid name format
        called = [False]

        def fn():
            called[0] = True

        switch.call_stage(stage="RewardManager_compute", warp_call={"fn": fn})
        self.assertTrue(called[0])

    def test_invalid_stage_name_raises(self):
        switch = ManagerCallSwitch(cfg_source={"default": 1})
        with self.assertRaises(ValueError):
            switch.call_stage(stage="nounderscore", warp_call={"fn": lambda: None})


if __name__ == "__main__":
    unittest.main()
