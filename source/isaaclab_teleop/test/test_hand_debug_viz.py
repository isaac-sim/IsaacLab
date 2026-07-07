# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Tests for hand joint debug visualization plumbing.

Covers :meth:`TeleopSessionLifecycle._chain_hand_debug_outputs` (HandsSource
auto-discovery), :meth:`TeleopSessionLifecycle.enable_hand_debug` (runtime
lazy-enable guards and in-place restart), and
:func:`~isaaclab_teleop.visualizers.hand_joint_visualizer._extract_world_joint_positions`
(anchor-to-world transform math, including the regression for markers staying
world-frame correct under a ``target_T_world`` rebase).

All heavy dependencies (isaacteleop, carb, omni.kit, isaacsim) are stubbed
out via ``sys.modules`` so these tests run in a plain Python environment.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Stub out isaacteleop and Kit modules before any isaaclab_teleop imports.
# ---------------------------------------------------------------------------

_MODULES_TO_STUB = [
    "isaacteleop",
    "isaacteleop.cloudxr",
    "isaacteleop.oxr",
    "isaacteleop.retargeting_engine",
    "isaacteleop.retargeting_engine.interface",
    "isaacteleop.retargeting_engine.interface.execution_events",
    "isaacteleop.retargeting_engine.interface.retargeter_core_types",
    "isaacteleop.retargeting_engine.interface.tensor_group_type",
    "isaacteleop.retargeting_engine.deviceio_source_nodes",
    "isaacteleop.retargeting_engine.deviceio_source_nodes.deviceio_tensor_types",
    "isaacteleop.retargeting_engine.tensor_types",
    "isaacteleop.retargeting_engine_ui",
    "isaacteleop.teleop_session_manager",
    "isaacteleop.teleop_session_manager.teleop_state_manager_retargeter",
    "isaacteleop.teleop_session_manager.teleop_state_manager_types",
    "isaacsim",
    "isaacsim.kit",
    "isaacsim.kit.xr",
    "isaacsim.kit.xr.teleop",
    "isaacsim.kit.xr.teleop.bridge",
    "carb",
    "carb.settings",
    "carb.eventdispatcher",
    "omni",
    "omni.kit",
    "omni.kit.app",
    "omni.kit.xr",
    "omni.kit.xr.system",
    "omni.kit.xr.system.openxr",
]

_stubs_installed: dict[str, ModuleType | MagicMock] = {}


class _FakeHandsSource:
    """Stand-in for ``isaacteleop...HandsSource`` (class attrs + output())."""

    LEFT = "hand_left"
    RIGHT = "hand_right"

    def __init__(self, name: str) -> None:
        self.name = name

    def output(self, key: str) -> str:
        return f"selector:{self.name}:{key}"


class _FakeOutputCombiner:
    """Captures the output mapping passed to ``OutputCombiner``."""

    def __init__(self, output_mapping: dict) -> None:
        self.output_mapping = output_mapping


class _FakeHandInputIndex:
    """Stand-in for the generated ``HandInputIndex`` enum."""

    JOINT_POSITIONS = 0


class _FakeControllerInputIndex:
    """Stand-in for the generated ``ControllerInputIndex`` enum (real index values)."""

    AIM_POSITION = 3
    AIM_ORIENTATION = 4
    AIM_IS_VALID = 5


class _FakeControllersSource:
    """Stand-in for ``isaacteleop...ControllersSource`` (class attrs + output())."""

    LEFT = "controller_left"
    RIGHT = "controller_right"

    def __init__(self, name: str) -> None:
        self.name = name

    def output(self, key: str) -> str:
        return f"selector:{self.name}:{key}"


class _FakeBaseRetargeter:
    """Stand-in base class so modules subclassing ``BaseRetargeter`` import cleanly."""

    def __init__(self, name: str) -> None:
        self.name = name


def _install_stubs():
    """Insert MagicMock modules for all heavy dependencies."""
    for name in _MODULES_TO_STUB:
        if name not in sys.modules:
            sys.modules[name] = _stubs_installed.setdefault(name, MagicMock())
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            setattr(sys.modules[parent_name], child_name, sys.modules[name])

    dsn_mod = sys.modules["isaacteleop.retargeting_engine.deviceio_source_nodes"]
    dsn_mod.HandsSource = _FakeHandsSource  # type: ignore[attr-defined]
    dsn_mod.ControllersSource = _FakeControllersSource  # type: ignore[attr-defined]

    iface_mod = sys.modules["isaacteleop.retargeting_engine.interface"]
    iface_mod.OutputCombiner = _FakeOutputCombiner  # type: ignore[attr-defined]
    iface_mod.BaseRetargeter = _FakeBaseRetargeter  # type: ignore[attr-defined]
    iface_mod.RetargeterIOType = dict  # type: ignore[attr-defined]

    tt_mod = sys.modules["isaacteleop.retargeting_engine.tensor_types"]
    tt_mod.HandInputIndex = _FakeHandInputIndex  # type: ignore[attr-defined]
    tt_mod.ControllerInputIndex = _FakeControllerInputIndex  # type: ignore[attr-defined]


def _restore_stubs():
    """Remove stubs installed for this test module from ``sys.modules``."""
    for name in reversed(_MODULES_TO_STUB):
        stub = _stubs_installed.get(name)
        if stub is None:
            continue
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = sys.modules.get(parent_name)
            if parent is not None and getattr(parent, child_name, None) is stub:
                delattr(parent, child_name)
        if sys.modules.get(name) is stub:
            del sys.modules[name]


_install_stubs()

from isaaclab_teleop.isaac_teleop_cfg import IsaacTeleopCfg  # noqa: E402
from isaaclab_teleop.session_lifecycle import TeleopSessionLifecycle  # noqa: E402
from isaaclab_teleop.visualizers.controller_aim_visualizer import _aim_world_pose  # noqa: E402
from isaaclab_teleop.visualizers.hand_joint_visualizer import _extract_world_joint_positions  # noqa: E402

_restore_stubs()


@pytest.fixture(autouse=True)
def _stub_heavy_dependencies():
    """Keep these tests isolated from modules collected later in the suite."""
    _install_stubs()
    yield
    _restore_stubs()


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeLeaf:
    """A generic non-HandsSource leaf node."""

    def __init__(self, name: str) -> None:
        self.name = name


class _FakePipeline:
    """Stand-in for a user pipeline (leaf nodes + action output)."""

    def __init__(self, leaves: list) -> None:
        self._leaves = leaves

    def get_leaf_nodes(self) -> list:
        return self._leaves

    def output(self, key: str) -> str:
        return f"selector:user_pipeline:{key}"


class _FakeTensorGroup:
    """Stand-in for an ``OptionalTensorGroup`` carrying joint positions."""

    def __init__(self, positions, is_none: bool = False) -> None:
        self._positions = positions
        self.is_none = is_none

    def __getitem__(self, idx):
        assert idx == _FakeHandInputIndex.JOINT_POSITIONS
        return self._positions


def _make_lifecycle(**kwargs) -> TeleopSessionLifecycle:
    """Build a lifecycle with a minimal config and no control channel."""
    cfg = IsaacTeleopCfg(pipeline_builder=lambda: None, control_channel_uuid=None)
    for key, value in kwargs.pop("cfg_overrides", {}).items():
        setattr(cfg, key, value)
    return TeleopSessionLifecycle(cfg, **kwargs)


# ===========================================================================
# _chain_hand_debug_outputs: HandsSource auto-discovery
# ===========================================================================


class TestChainHandDebugOutputs:
    def test_chains_when_hands_source_present(self):
        pipeline = _FakePipeline([_FakeLeaf("controllers"), _FakeHandsSource("hands")])
        mapping: dict = {}
        chained = TeleopSessionLifecycle._chain_hand_debug_outputs(pipeline, mapping)
        assert chained is True
        assert mapping[TeleopSessionLifecycle.HAND_LEFT_KEY] == "selector:hands:hand_left"
        assert mapping[TeleopSessionLifecycle.HAND_RIGHT_KEY] == "selector:hands:hand_right"

    def test_mapping_untouched_without_hands_source(self):
        pipeline = _FakePipeline([_FakeLeaf("controllers"), _FakeLeaf("head")])
        mapping: dict = {"action": "selector:user_pipeline:action"}
        chained = TeleopSessionLifecycle._chain_hand_debug_outputs(pipeline, mapping)
        assert chained is False
        assert mapping == {"action": "selector:user_pipeline:action"}


# ===========================================================================
# start(): chaining gated by the enable_debug_visualization lifecycle arg
# ===========================================================================


class TestStartChaining:
    def _start(self, enable_debug_visualization: bool) -> TeleopSessionLifecycle:
        pipeline = _FakePipeline([_FakeHandsSource("hands")])
        cfg = IsaacTeleopCfg(pipeline_builder=lambda: pipeline, control_channel_uuid=None)
        lifecycle = TeleopSessionLifecycle(cfg, enable_debug_visualization=enable_debug_visualization)
        with patch.object(lifecycle, "_try_start_session", return_value=True):
            lifecycle.start()
        return lifecycle

    def test_start_chains_when_enabled(self):
        lifecycle = self._start(enable_debug_visualization=True)
        assert lifecycle.debug_viz_chained is True
        assert TeleopSessionLifecycle.HAND_LEFT_KEY in lifecycle.pipeline.output_mapping
        assert TeleopSessionLifecycle.HAND_RIGHT_KEY in lifecycle.pipeline.output_mapping
        assert TeleopSessionLifecycle._CONTROLLER_LEFT_KEY in lifecycle.pipeline.output_mapping

    def test_start_does_not_chain_when_disabled(self):
        lifecycle = self._start(enable_debug_visualization=False)
        assert lifecycle.debug_viz_chained is False
        assert TeleopSessionLifecycle.HAND_LEFT_KEY not in lifecycle.pipeline.output_mapping
        assert TeleopSessionLifecycle.HAND_RIGHT_KEY not in lifecycle.pipeline.output_mapping
        assert TeleopSessionLifecycle._CONTROLLER_LEFT_KEY not in lifecycle.pipeline.output_mapping
        # The right controller is always chained for button polling.
        assert TeleopSessionLifecycle._CONTROLLER_RIGHT_KEY in lifecycle.pipeline.output_mapping


# ===========================================================================
# enable_debug_viz: guards and in-place restart
# ===========================================================================


class TestEnableDebugViz:
    def test_refused_before_start(self):
        lifecycle = _make_lifecycle()
        assert lifecycle.enable_debug_viz() is False

    def test_noop_when_already_chained(self):
        lifecycle = _make_lifecycle()
        lifecycle._debug_viz_chained = True
        assert lifecycle.enable_debug_viz() is True

    def test_refused_while_recording(self):
        lifecycle = _make_lifecycle(mcap_record_path="session.mcap")
        lifecycle._pipeline = object()
        assert lifecycle.enable_debug_viz() is False

    def test_refused_during_replay(self):
        lifecycle = _make_lifecycle(mcap_replay_path="session.mcap")
        lifecycle._pipeline = object()
        assert lifecycle.enable_debug_viz() is False

    def test_enables_without_hands_source(self):
        """Controller aim viz works without a HandsSource; only hand keys are absent."""
        lifecycle = _make_lifecycle()
        lifecycle._pipeline = _FakeOutputCombiner({})
        lifecycle._user_pipeline = _FakePipeline([_FakeLeaf("head")])
        with patch.object(lifecycle, "_try_start_session", return_value=True):
            assert lifecycle.enable_debug_viz() is True
        assert lifecycle.debug_viz_chained is True
        assert TeleopSessionLifecycle._CONTROLLER_LEFT_KEY in lifecycle.pipeline.output_mapping
        assert TeleopSessionLifecycle.HAND_LEFT_KEY not in lifecycle.pipeline.output_mapping

    def test_restarts_session_with_chained_pipeline(self):
        lifecycle = _make_lifecycle()
        old_pipeline = _FakeOutputCombiner({})
        old_session = MagicMock()
        lifecycle._pipeline = old_pipeline
        lifecycle._user_pipeline = _FakePipeline([_FakeHandsSource("hands")])
        lifecycle._session = old_session
        lifecycle._last_step_result = {"action": "stale"}
        with patch.object(lifecycle, "_try_start_session", return_value=True) as try_start:
            assert lifecycle.enable_debug_viz() is True
        old_session.__exit__.assert_called_once()
        try_start.assert_called_once()
        assert lifecycle.debug_viz_chained is True
        assert lifecycle.pipeline is not old_pipeline
        assert TeleopSessionLifecycle.HAND_LEFT_KEY in lifecycle.pipeline.output_mapping
        assert TeleopSessionLifecycle._CONTROLLER_LEFT_KEY in lifecycle.pipeline.output_mapping
        assert lifecycle.last_step_result is None

    def test_chained_even_when_restart_deferred(self):
        lifecycle = _make_lifecycle()
        lifecycle._pipeline = _FakeOutputCombiner({})
        lifecycle._user_pipeline = _FakePipeline([_FakeHandsSource("hands")])
        with patch.object(lifecycle, "_try_start_session", return_value=False):
            assert lifecycle.enable_debug_viz() is True
        assert lifecycle.debug_viz_chained is True


# ===========================================================================
# consume_visualization_toggle delegation
# ===========================================================================


class TestConsumeVisualizationToggle:
    def test_false_without_control_channel(self):
        lifecycle = _make_lifecycle()
        assert lifecycle.consume_visualization_toggle() is False

    def test_delegates_to_message_processor(self):
        lifecycle = _make_lifecycle()
        processor = MagicMock()
        processor.consume_visualization_toggle.return_value = True
        lifecycle._message_processor = processor
        assert lifecycle.consume_visualization_toggle() is True
        processor.consume_visualization_toggle.assert_called_once()


# ===========================================================================
# _extract_world_joint_positions: anchor-to-world transform math
# ===========================================================================


class TestExtractWorldJointPositions:
    def test_none_without_hand_data(self):
        assert _extract_world_joint_positions({}, np.eye(4)) is None

    def test_none_when_hands_inactive(self):
        result = {
            TeleopSessionLifecycle.HAND_LEFT_KEY: _FakeTensorGroup(None, is_none=True),
            TeleopSessionLifecycle.HAND_RIGHT_KEY: _FakeTensorGroup(None, is_none=True),
        }
        assert _extract_world_joint_positions(result, np.eye(4)) is None

    def test_identity_transform_passes_through(self):
        joints = np.arange(78, dtype=np.float32).reshape(26, 3)
        result = {TeleopSessionLifecycle.HAND_LEFT_KEY: _FakeTensorGroup(joints)}
        positions = _extract_world_joint_positions(result, np.eye(4))
        assert positions.shape == (26, 3)
        torch.testing.assert_close(positions, torch.as_tensor(joints))

    def test_both_hands_are_concatenated(self):
        left = np.zeros((26, 3), dtype=np.float32)
        right = np.ones((26, 3), dtype=np.float32)
        result = {
            TeleopSessionLifecycle.HAND_LEFT_KEY: _FakeTensorGroup(left),
            TeleopSessionLifecycle.HAND_RIGHT_KEY: _FakeTensorGroup(right),
        }
        positions = _extract_world_joint_positions(result, np.eye(4))
        assert positions.shape == (52, 3)
        torch.testing.assert_close(positions[:26], torch.zeros(26, 3))
        torch.testing.assert_close(positions[26:], torch.ones(26, 3))

    def test_world_transform_applied(self):
        """Regression: markers must be placed via world_T_anchor, not the
        (possibly rebased) pipeline transform.

        90 deg rotation about +z plus translation: anchor-frame (1, 0, 0)
        must land at world (t_x + 0, t_y + 1, t_z + 0).
        """
        world_T_anchor = np.array(
            [
                [0.0, -1.0, 0.0, 10.0],
                [1.0, 0.0, 0.0, 20.0],
                [0.0, 0.0, 1.0, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        joints = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        result = {TeleopSessionLifecycle.HAND_RIGHT_KEY: _FakeTensorGroup(joints)}
        positions = _extract_world_joint_positions(result, world_T_anchor)
        torch.testing.assert_close(positions, torch.tensor([[10.0, 21.0, 30.0]]))

    def test_accepts_torch_tensor_input(self):
        joints = torch.rand(26, 3)
        result = {TeleopSessionLifecycle.HAND_LEFT_KEY: _FakeTensorGroup(joints)}
        positions = _extract_world_joint_positions(result, np.eye(4))
        torch.testing.assert_close(positions, joints)


# ===========================================================================
# _aim_world_pose: controller aim transform math
# ===========================================================================


class _FakeControllerGroup:
    """Stand-in for a controller ``OptionalTensorGroup``."""

    def __init__(self, position, quat_xyzw, aim_valid: bool = True, is_none: bool = False) -> None:
        self._fields = {
            _FakeControllerInputIndex.AIM_POSITION: position,
            _FakeControllerInputIndex.AIM_ORIENTATION: quat_xyzw,
            _FakeControllerInputIndex.AIM_IS_VALID: aim_valid,
        }
        self.is_none = is_none

    def __getitem__(self, idx):
        return self._fields[idx]


_IDENTITY_XYZW = [0.0, 0.0, 0.0, 1.0]


class TestAimWorldPose:
    def test_none_controller(self):
        assert _aim_world_pose(None, np.eye(4)) is None

    def test_inactive_controller(self):
        group = _FakeControllerGroup([0, 0, 0], _IDENTITY_XYZW, is_none=True)
        assert _aim_world_pose(group, np.eye(4)) is None

    def test_invalid_aim(self):
        group = _FakeControllerGroup([0, 0, 0], _IDENTITY_XYZW, aim_valid=False)
        assert _aim_world_pose(group, np.eye(4)) is None

    def test_identity_transform_passes_through_xyzw(self):
        """Aim orientations stay XYZW end-to-end: OpenXR and
        :meth:`VisualizationMarkers.visualize` share the convention, so an
        identity anchor transform must not reorder components."""
        # 90 deg about +x: xyzw = (sin45, 0, 0, cos45)
        s = np.sin(np.pi / 4)
        c = np.cos(np.pi / 4)
        group = _FakeControllerGroup([1.0, 2.0, 3.0], [s, 0.0, 0.0, c])
        position, quat_xyzw = _aim_world_pose(group, np.eye(4))
        np.testing.assert_allclose(position, [1.0, 2.0, 3.0], atol=1e-6)
        np.testing.assert_allclose(quat_xyzw, [s, 0.0, 0.0, c], atol=1e-6)

    def test_world_transform_applied(self):
        """Regression: aim axes must be placed via world_T_anchor.

        90 deg rotation about +z plus translation: anchor-frame position
        (1, 0, 0) with identity orientation must land at world
        (t_x, t_y + 1, t_z) oriented 90 deg about +z.
        """
        world_T_anchor = np.array(
            [
                [0.0, -1.0, 0.0, 10.0],
                [1.0, 0.0, 0.0, 20.0],
                [0.0, 0.0, 1.0, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        group = _FakeControllerGroup([1.0, 0.0, 0.0], _IDENTITY_XYZW)
        position, quat_xyzw = _aim_world_pose(group, world_T_anchor)
        np.testing.assert_allclose(position, [10.0, 21.0, 30.0], atol=1e-6)
        s = np.sin(np.pi / 4)
        c = np.cos(np.pi / 4)
        np.testing.assert_allclose(quat_xyzw, [0.0, 0.0, s, c], atol=1e-6)
