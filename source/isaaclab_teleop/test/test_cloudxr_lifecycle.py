# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Tests for CloudXR runtime auto-launch lifecycle.

These tests exercise the CloudXR auto-launch logic in
:class:`~isaaclab_teleop.session_lifecycle.TeleopSessionLifecycle` without
requiring the Omniverse/Isaac Sim stack or a real CloudXR installation.

All heavy dependencies (isaacteleop, carb, omni.kit, isaacsim) are
stubbed out via ``sys.modules`` and ``unittest.mock`` so these tests run
in a plain Python environment.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub out isaacteleop and Kit modules before any isaaclab_teleop imports.
# TeleopSessionLifecycle.__init__ tries to import several Kit/XR modules;
# we inject stubs so the constructor can complete without Omniverse.
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


def _install_stubs():
    """Insert MagicMock modules for all heavy dependencies."""
    for name in _MODULES_TO_STUB:
        if name not in sys.modules:
            _stubs_installed[name] = MagicMock()
            sys.modules[name] = _stubs_installed[name]


_install_stubs()

from isaaclab_teleop.isaac_teleop_cfg import (  # noqa: E402
    CLOUDXR_AVP_ENV,
    CLOUDXR_JS_ENV,
    IsaacTeleopCfg,
)
from isaaclab_teleop.session_lifecycle import TeleopSessionLifecycle  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg() -> IsaacTeleopCfg:
    """Build a minimal IsaacTeleopCfg with a dummy pipeline_builder."""
    return IsaacTeleopCfg(
        pipeline_builder=lambda: MagicMock(),
        control_channel_uuid=None,
    )


def _make_lifecycle(
    cloudxr_env_file: str | None = None,
    auto_launch_cloudxr: bool = True,
) -> TeleopSessionLifecycle:
    """Create a TeleopSessionLifecycle with Kit dependencies safely stubbed."""
    cfg = _make_cfg()
    return TeleopSessionLifecycle(
        cfg,
        cloudxr_env_file=cloudxr_env_file,
        auto_launch_cloudxr=auto_launch_cloudxr,
    )


# ============================================================================
# Shipped .env profile paths
# ============================================================================


class TestEnvProfilePaths:
    """Tests for the shipped .env profile path constants."""

    def test_avp_env_is_absolute_path(self):
        assert os.path.isabs(CLOUDXR_AVP_ENV)

    def test_js_env_is_absolute_path(self):
        assert os.path.isabs(CLOUDXR_JS_ENV)

    def test_avp_env_file_exists(self):
        assert Path(CLOUDXR_AVP_ENV).is_file(), f"Missing: {CLOUDXR_AVP_ENV}"

    def test_js_env_file_exists(self):
        assert Path(CLOUDXR_JS_ENV).is_file(), f"Missing: {CLOUDXR_JS_ENV}"

    def test_avp_env_filename(self):
        assert Path(CLOUDXR_AVP_ENV).name == "avp-cloudxr.env"

    def test_js_env_filename(self):
        assert Path(CLOUDXR_JS_ENV).name == "cloudxrjs-cloudxr.env"

    def test_profiles_are_in_same_directory(self):
        assert Path(CLOUDXR_AVP_ENV).parent == Path(CLOUDXR_JS_ENV).parent


# ============================================================================
# _ensure_cloudxr_runtime
# ============================================================================


class TestEnsureCloudXRRuntime:
    """Tests for the ``_ensure_cloudxr_runtime`` method on TeleopSessionLifecycle."""

    def test_skip_when_env_var_set(self):
        """ISAACLAB_CXR_SKIP_AUTOLAUNCH=1 skips the launch entirely."""
        lifecycle = _make_lifecycle(cloudxr_env_file="/tmp/test.env")

        with patch.dict(os.environ, {"ISAACLAB_CXR_SKIP_AUTOLAUNCH": "1"}):
            lifecycle._ensure_cloudxr_runtime()

        assert lifecycle._cloudxr_launcher is None

    def test_skip_when_env_var_set_with_whitespace(self):
        """Whitespace around the env var value is stripped before comparison."""
        lifecycle = _make_lifecycle(cloudxr_env_file="/tmp/test.env")

        with patch.dict(os.environ, {"ISAACLAB_CXR_SKIP_AUTOLAUNCH": " 1 "}):
            lifecycle._ensure_cloudxr_runtime()

        assert lifecycle._cloudxr_launcher is None

    def test_no_skip_when_env_var_zero(self):
        """ISAACLAB_CXR_SKIP_AUTOLAUNCH=0 does NOT skip the launch."""
        mock_cls = MagicMock()
        fake_module = MagicMock()
        fake_module.CloudXRLauncher = mock_cls

        lifecycle = _make_lifecycle(cloudxr_env_file="/tmp/test.env")

        with (
            patch.dict(os.environ, {"ISAACLAB_CXR_SKIP_AUTOLAUNCH": "0"}),
            patch.dict(sys.modules, {"isaacteleop.cloudxr": fake_module}),
        ):
            lifecycle._ensure_cloudxr_runtime()

        assert lifecycle._cloudxr_launcher is not None

    def test_skip_when_auto_launch_false(self):
        """auto_launch_cloudxr=False skips the launch."""
        lifecycle = _make_lifecycle(
            cloudxr_env_file="/tmp/test.env",
            auto_launch_cloudxr=False,
        )

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ISAACLAB_CXR_SKIP_AUTOLAUNCH", None)
            lifecycle._ensure_cloudxr_runtime()

        assert lifecycle._cloudxr_launcher is None

    def test_idempotency(self):
        """Calling _ensure_cloudxr_runtime twice does not create a second launcher."""
        sentinel = MagicMock()
        lifecycle = _make_lifecycle(cloudxr_env_file="/tmp/test.env")
        lifecycle._cloudxr_launcher = sentinel

        lifecycle._ensure_cloudxr_runtime()

        assert lifecycle._cloudxr_launcher is sentinel

    def test_launches_with_correct_args(self):
        """CloudXRLauncher is constructed with hardcoded install_dir/accept_eula and the env file."""
        mock_cls = MagicMock()
        lifecycle = _make_lifecycle(cloudxr_env_file="/etc/cxr.env")

        fake_module = MagicMock()
        fake_module.CloudXRLauncher = mock_cls

        with (
            patch.dict(os.environ, {}, clear=False),
            patch.dict(sys.modules, {"isaacteleop.cloudxr": fake_module}),
        ):
            os.environ.pop("ISAACLAB_CXR_SKIP_AUTOLAUNCH", None)
            lifecycle._ensure_cloudxr_runtime()

        mock_cls.assert_called_once_with(
            install_dir=str(Path.home() / ".cloudxr"),
            env_config="/etc/cxr.env",
            accept_eula=False,
        )
        assert lifecycle._cloudxr_launcher is mock_cls.return_value

    def test_env_var_takes_precedence_over_auto_launch(self):
        """ISAACLAB_CXR_SKIP_AUTOLAUNCH=1 overrides auto_launch_cloudxr=True."""
        lifecycle = _make_lifecycle(
            cloudxr_env_file="/tmp/test.env",
            auto_launch_cloudxr=True,
        )

        with patch.dict(os.environ, {"ISAACLAB_CXR_SKIP_AUTOLAUNCH": "1"}):
            lifecycle._ensure_cloudxr_runtime()

        assert lifecycle._cloudxr_launcher is None


# ============================================================================
# Lifecycle start/stop integration with CloudXR
# ============================================================================


class TestLifecycleCloudXRIntegration:
    """Tests for CloudXR launch/shutdown within the start()/stop() lifecycle."""

    def _make_started_lifecycle(self) -> tuple[TeleopSessionLifecycle, MagicMock]:
        """Build a lifecycle whose start() has been called with a mock launcher."""
        mock_cls = MagicMock()
        lifecycle = _make_lifecycle(cloudxr_env_file="/tmp/test.env")

        fake_cxr_module = MagicMock()
        fake_cxr_module.CloudXRLauncher = mock_cls

        fake_teleop_modules = {
            "isaacteleop.cloudxr": fake_cxr_module,
            "isaacteleop.retargeting_engine.deviceio_source_nodes": MagicMock(),
            "isaacteleop.retargeting_engine.interface": MagicMock(),
        }

        with (
            patch.dict(os.environ, {}, clear=False),
            patch.dict(sys.modules, fake_teleop_modules),
        ):
            os.environ.pop("ISAACLAB_CXR_SKIP_AUTOLAUNCH", None)
            lifecycle.start()

        return lifecycle, mock_cls.return_value

    def test_start_launches_runtime(self):
        """start() invokes _ensure_cloudxr_runtime when cloudxr_env_file is set."""
        lifecycle, mock_launcher = self._make_started_lifecycle()
        assert lifecycle._cloudxr_launcher is mock_launcher

    def test_stop_calls_launcher_stop(self):
        """stop() calls CloudXRLauncher.stop()."""
        lifecycle, mock_launcher = self._make_started_lifecycle()
        lifecycle.stop()
        mock_launcher.stop.assert_called_once()

    def test_stop_clears_launcher_on_success(self):
        """After a successful stop(), _cloudxr_launcher is set to None."""
        lifecycle, mock_launcher = self._make_started_lifecycle()
        lifecycle.stop()
        assert lifecycle._cloudxr_launcher is None

    def test_stop_retains_launcher_on_runtime_error(self):
        """When CloudXRLauncher.stop() raises RuntimeError, the launcher is retained for atexit."""
        lifecycle, mock_launcher = self._make_started_lifecycle()
        mock_launcher.stop.side_effect = RuntimeError("process not found")

        lifecycle.stop()

        assert lifecycle._cloudxr_launcher is mock_launcher

    def test_start_without_cloudxr_env_file(self):
        """start() works normally when no cloudxr_env_file is provided."""
        lifecycle = _make_lifecycle(cloudxr_env_file=None)

        fake_teleop_modules = {
            "isaacteleop.retargeting_engine.deviceio_source_nodes": MagicMock(),
            "isaacteleop.retargeting_engine.interface": MagicMock(),
        }

        with patch.dict(sys.modules, fake_teleop_modules):
            lifecycle.start()

        assert lifecycle._cloudxr_launcher is None

    def test_start_with_auto_launch_disabled(self):
        """start() skips CloudXR launch when auto_launch_cloudxr=False."""
        lifecycle = _make_lifecycle(
            cloudxr_env_file="/tmp/test.env",
            auto_launch_cloudxr=False,
        )

        fake_teleop_modules = {
            "isaacteleop.retargeting_engine.deviceio_source_nodes": MagicMock(),
            "isaacteleop.retargeting_engine.interface": MagicMock(),
        }

        with (
            patch.dict(os.environ, {}, clear=False),
            patch.dict(sys.modules, fake_teleop_modules),
        ):
            os.environ.pop("ISAACLAB_CXR_SKIP_AUTOLAUNCH", None)
            lifecycle.start()

        assert lifecycle._cloudxr_launcher is None

    def test_stop_without_cloudxr_env_file(self):
        """stop() does not error when no CloudXR launcher was created."""
        lifecycle = _make_lifecycle(cloudxr_env_file=None)
        lifecycle.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
