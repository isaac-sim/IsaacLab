# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for RTX streaming wait helpers.

Covers callback state updates, subscription behavior, and timeout-aware wait
logic in :mod:`isaaclab_physx.renderers.isaac_rtx_renderer_utils`.
"""

from __future__ import annotations

import sys
import time
import types
from unittest.mock import MagicMock, patch

# Stub ``omni`` / ``omni.usd`` in ``sys.modules`` before importing the module
# under test so its top-level ``import omni.usd`` succeeds outside a running
# Kit runtime. Per-test fixtures below still patch these with fresh mocks, so
# each test remains isolated.
if "omni" not in sys.modules:
    sys.modules["omni"] = types.ModuleType("omni")
if "omni.usd" not in sys.modules:
    _omni_usd_stub = MagicMock()
    sys.modules["omni.usd"] = _omni_usd_stub
    setattr(sys.modules["omni"], "usd", _omni_usd_stub)

import isaaclab_physx.renderers.isaac_rtx_renderer_utils as rtx_utils  # noqa: E402
import pytest  # noqa: E402

pytestmark = pytest.mark.isaacsim_ci

# test-specific timeout overrides for _STREAMING_WAIT_TIMEOUT_S
STREAMING_TIMEOUT_S = 0.1
STREAMING_TIMEOUT_SHORT_S = 0.01

# simulated per-update sleep to advance wall-clock time inside the wait loop
MOCK_UPDATE_SLEEP_S = 0.02

# how many app.update() iterations before the mock becomes idle
MOCK_ITERATIONS_BEFORE_IDLE = 3


@pytest.fixture(autouse=True)
def _reset_globals(monkeypatch):
    """Restore module-level state so tests are isolated."""
    monkeypatch.setattr(rtx_utils, "_last_render_update_key", (0, -1))


@pytest.fixture()
def mock_omni_usd():
    """Make ``omni.usd`` importable outside the Isaac Sim runtime.

    Both ``sys.modules`` and the ``omni`` namespace attribute must be set,
    because ``import omni.usd`` resolves the parent package first and then
    looks up ``.usd`` as an attribute.
    """
    import omni

    mock_module = MagicMock()
    with (
        patch.dict(sys.modules, {"omni.usd": mock_module}),
        patch.object(omni, "usd", mock_module, create=True),
    ):
        yield mock_module


@pytest.fixture()
def mock_omni_kit_app():
    """Make ``omni.kit.app`` importable outside the Isaac Sim runtime."""
    import omni

    mock_kit = MagicMock()
    mock_module = MagicMock()
    mock_kit.app = mock_module
    with (
        patch.dict(sys.modules, {"omni.kit": mock_kit, "omni.kit.app": mock_module}),
        patch.object(omni, "kit", mock_kit, create=True),
    ):
        yield mock_module


# ---------------------------------------------------------------------------
# _get_stage_streaming_busy
# ---------------------------------------------------------------------------


class TestGetStageStreamingBusy:
    """Synchronous streaming status query delegates to UsdContext."""

    def test_returns_true_when_busy(self, mock_omni_usd):
        mock_ctx = MagicMock()
        mock_ctx.get_stage_streaming_status.return_value = True
        mock_omni_usd.get_context.return_value = mock_ctx
        assert rtx_utils._get_stage_streaming_busy() is True

    def test_returns_false_when_idle(self, mock_omni_usd):
        mock_ctx = MagicMock()
        mock_ctx.get_stage_streaming_status.return_value = False
        mock_omni_usd.get_context.return_value = mock_ctx
        assert rtx_utils._get_stage_streaming_busy() is False

    def test_returns_false_when_no_context(self, mock_omni_usd):
        mock_omni_usd.get_context.return_value = None
        assert rtx_utils._get_stage_streaming_busy() is False


# ---------------------------------------------------------------------------
# _wait_for_streaming_complete
# ---------------------------------------------------------------------------


class TestWaitForStreamingComplete:
    """Blocking wait pumps app.update() while busy and respects timeout.

    These tests patch ``_get_stage_streaming_busy`` at the module level so
    they don't depend on ``omni.usd`` being importable.
    """

    def test_returns_immediately_when_not_busy(self, mock_omni_kit_app):
        """Skips loop and issues only the final update when idle."""
        mock_app = MagicMock()
        mock_omni_kit_app.get_app.return_value = mock_app

        with patch.object(rtx_utils, "_get_stage_streaming_busy", return_value=False):
            rtx_utils._wait_for_streaming_complete()

        mock_app.update.assert_called_once()

    def test_pumps_updates_until_idle(self, mock_omni_kit_app):
        """Pumps updates until streaming reports idle."""
        mock_app = MagicMock()
        mock_omni_kit_app.get_app.return_value = mock_app
        loop_calls = 0

        def _streaming_status():
            return loop_calls < MOCK_ITERATIONS_BEFORE_IDLE

        def _count_update():
            nonlocal loop_calls
            loop_calls += 1

        mock_app.update.side_effect = _count_update

        with patch.object(rtx_utils, "_get_stage_streaming_busy", side_effect=_streaming_status):
            rtx_utils._wait_for_streaming_complete()

        assert mock_app.update.call_count == MOCK_ITERATIONS_BEFORE_IDLE + 1

    def test_respects_timeout(self, monkeypatch, mock_omni_kit_app):
        """Exits wait loop on timeout if busy never clears."""
        monkeypatch.setattr(rtx_utils, "_STREAMING_WAIT_TIMEOUT_S", STREAMING_TIMEOUT_S)
        mock_app = MagicMock()
        mock_app.update.side_effect = lambda: time.sleep(MOCK_UPDATE_SLEEP_S)
        mock_omni_kit_app.get_app.return_value = mock_app

        with patch.object(rtx_utils, "_get_stage_streaming_busy", return_value=True):
            rtx_utils._wait_for_streaming_complete()

        assert mock_app.update.call_count > 0

    def test_timeout_logs_warning(self, monkeypatch, mock_omni_kit_app):
        """Logs warning when timeout is reached while still busy."""
        monkeypatch.setattr(rtx_utils, "_STREAMING_WAIT_TIMEOUT_S", STREAMING_TIMEOUT_SHORT_S)
        mock_app = MagicMock()
        mock_omni_kit_app.get_app.return_value = mock_app
        mock_logger = MagicMock()

        with (
            patch.object(rtx_utils, "_get_stage_streaming_busy", return_value=True),
            patch.object(rtx_utils, "logger", mock_logger),
        ):
            rtx_utils._wait_for_streaming_complete()

        mock_logger.warning.assert_called_once()
        assert "RTX streaming did not complete within" in mock_logger.warning.call_args[0][0]

    def test_logs_info_on_non_trivial_completion(self, mock_omni_kit_app):
        """Logs completion info when streaming finishes after delay."""
        mock_app = MagicMock()
        mock_omni_kit_app.get_app.return_value = mock_app
        mock_logger = MagicMock()
        call_count = 0

        def _streaming_status():
            return call_count < 1

        def _become_idle_after_delay():
            nonlocal call_count
            time.sleep(MOCK_UPDATE_SLEEP_S)
            call_count += 1

        mock_app.update.side_effect = _become_idle_after_delay

        with (
            patch.object(rtx_utils, "_get_stage_streaming_busy", side_effect=_streaming_status),
            patch.object(rtx_utils, "logger", mock_logger),
        ):
            rtx_utils._wait_for_streaming_complete()

        mock_logger.info.assert_called_once()
        assert "RTX streaming completed in" in mock_logger.info.call_args[0][0]
