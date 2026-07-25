# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Tests for step-failure handling in :class:`TeleopSessionLifecycle`.

The async retarget worker dies permanently on any pipeline exception, so
session re-entry is the only recovery.  These tests cover the failure
diagnosis (pipeline error vs. external XR teardown) and the restart
cooldown that prevents a persistent pipeline error from churning
teardown/restart cycles every frame.

All heavy dependencies (isaacteleop, carb, omni.kit, isaacsim) are stubbed
out via ``sys.modules`` so these tests run in a plain Python environment.
"""

from __future__ import annotations

import sys
import time
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

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

    iface_mod = sys.modules["isaacteleop.retargeting_engine.interface"]
    iface_mod.BaseRetargeter = _FakeBaseRetargeter  # type: ignore[attr-defined]
    iface_mod.RetargeterIOType = dict  # type: ignore[attr-defined]


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

_restore_stubs()


@pytest.fixture(autouse=True)
def _stub_heavy_dependencies():
    """Keep these tests isolated from modules collected later in the suite."""
    _install_stubs()
    yield
    _restore_stubs()


def _make_lifecycle(**kwargs) -> TeleopSessionLifecycle:
    cfg = IsaacTeleopCfg(pipeline_builder=lambda: None, control_channel_uuid=None)
    return TeleopSessionLifecycle(cfg, **kwargs)


def _make_failing_lifecycle(error: Exception) -> TeleopSessionLifecycle:
    """Lifecycle with a live session mock whose step raises *error*."""
    lifecycle = _make_lifecycle()
    lifecycle._pipeline = object()
    session = MagicMock()
    session.step.side_effect = error
    lifecycle._session = session
    return lifecycle


class TestStepFailureDiagnosis:
    def test_pipeline_error_with_active_xr_sets_holdoff(self):
        lifecycle = _make_failing_lifecycle(ValueError("Found zero norm quaternions in `quat`."))
        session = lifecycle._session
        with (
            patch.object(TeleopSessionLifecycle, "_kit_xr_session_is_active", return_value=True),
            patch.object(lifecycle, "_build_external_inputs", return_value=None),
        ):
            assert lifecycle.step() is None
        session.__exit__.assert_called_once()
        assert lifecycle._session is None
        assert lifecycle._restart_holdoff_until > time.monotonic()

    def test_external_xr_teardown_has_no_holdoff(self):
        lifecycle = _make_failing_lifecycle(RuntimeError("XR_ERROR_INSTANCE_LOST"))
        session = lifecycle._session
        with (
            patch.object(TeleopSessionLifecycle, "_kit_xr_session_is_active", return_value=False),
            patch.object(lifecycle, "_build_external_inputs", return_value=None),
        ):
            assert lifecycle.step() is None
        session.__exit__.assert_called_once()
        assert lifecycle._session is None
        # Stop-AR -> Start-AR recovery latency must stay unchanged.
        assert lifecycle._restart_holdoff_until == 0.0


class TestRestartHoldoff:
    def test_holdoff_blocks_session_restart(self):
        lifecycle = _make_lifecycle()
        lifecycle._pipeline = object()
        lifecycle._restart_holdoff_until = time.monotonic() + 60.0
        with patch.object(lifecycle, "_try_start_session") as try_start:
            assert lifecycle.step() is None
        try_start.assert_not_called()

    def test_expired_holdoff_allows_restart(self):
        lifecycle = _make_lifecycle()
        lifecycle._pipeline = object()
        lifecycle._restart_holdoff_until = time.monotonic() - 1.0
        with patch.object(lifecycle, "_try_start_session", return_value=False) as try_start:
            assert lifecycle.step() is None
        try_start.assert_called_once()
