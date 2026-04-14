# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for RTX streaming wait helpers.

Covers direct-polling status queries, timeout-aware wait logic,
and dedup stamping in
:mod:`isaaclab_physx.renderers.isaac_rtx_renderer_utils`.
"""

from __future__ import annotations

import sys
import time
from unittest.mock import MagicMock, patch

import isaaclab_physx.renderers.isaac_rtx_renderer_utils as rtx_utils
import pytest

STREAMING_TIMEOUT_S = 0.1
STREAMING_TIMEOUT_SHORT_S = 0.01

MOCK_UPDATE_SLEEP_S = 0.02

MOCK_QUERIES_BEFORE_IDLE = 3


@pytest.fixture()
def _mock_omni_usd():
    """Inject a mock ``omni.usd`` module so ``import omni.usd`` succeeds.

    ``import omni.usd`` requires *both* a ``sys.modules`` entry and an
    attribute on the parent ``omni`` package.  We patch both and restore
    the original state on teardown.
    """
    import omni

    mock_usd = MagicMock()
    had_usd = hasattr(omni, "usd")
    old_usd = getattr(omni, "usd", None)
    omni.usd = mock_usd
    with patch.dict(sys.modules, {"omni.usd": mock_usd}):
        yield mock_usd
    if had_usd:
        omni.usd = old_usd
    else:
        delattr(omni, "usd")


# -----------------------------------------------------------
# _is_stage_loading_or_streaming
# -----------------------------------------------------------


class TestIsStageLoadingOrStreaming:
    """Direct status query identifies load/stream/idle."""

    def test_idle(self, _mock_omni_usd):
        ctx = MagicMock()
        ctx.get_stage_loading_status.return_value = (
            "",
            0,
            0,
        )
        ctx.get_stage_streaming_status.return_value = False
        _mock_omni_usd.get_context.return_value = ctx

        assert rtx_utils._is_stage_loading_or_streaming() is False

    def test_loading(self, _mock_omni_usd):
        ctx = MagicMock()
        ctx.get_stage_loading_status.return_value = (
            "",
            5,
            10,
        )
        _mock_omni_usd.get_context.return_value = ctx

        assert rtx_utils._is_stage_loading_or_streaming() is True

    def test_streaming(self, _mock_omni_usd):
        ctx = MagicMock()
        ctx.get_stage_loading_status.return_value = (
            "",
            0,
            0,
        )
        ctx.get_stage_streaming_status.return_value = True
        _mock_omni_usd.get_context.return_value = ctx

        assert rtx_utils._is_stage_loading_or_streaming() is True

    def test_loading_skips_streaming_check(self, _mock_omni_usd):
        ctx = MagicMock()
        ctx.get_stage_loading_status.return_value = (
            "",
            2,
            5,
        )
        _mock_omni_usd.get_context.return_value = ctx

        rtx_utils._is_stage_loading_or_streaming()
        ctx.get_stage_streaming_status.assert_not_called()


# -----------------------------------------------------------
# _wait_for_streaming_complete
# -----------------------------------------------------------


def _patch_busy(return_values):
    """Mock ``_is_stage_loading_or_streaming`` with a sequence.

    Each call pops the next value. Once exhausted, returns the
    last value forever.
    """
    values = list(return_values)

    def _side_effect():
        if len(values) > 1:
            return values.pop(0)
        return values[0]

    return patch.object(
        rtx_utils,
        "_is_stage_loading_or_streaming",
        side_effect=_side_effect,
    )


class TestWaitForStreamingComplete:
    """Blocking wait pumps app.update() while busy."""

    def test_returns_immediately_when_idle(self):
        """No pumps at all when stage is already idle."""
        mock_app = MagicMock()
        with (
            patch(
                "omni.kit.app.get_app",
                return_value=mock_app,
            ),
            _patch_busy([False]),
        ):
            rtx_utils._wait_for_streaming_complete()

        mock_app.update.assert_not_called()

    def test_pumps_updates_until_idle(self):
        """Pumps updates until status flips to idle."""
        mock_app = MagicMock()
        # busy, busy (while-check), idle (while-check),
        # idle (post-loop check)
        with (
            patch(
                "omni.kit.app.get_app",
                return_value=mock_app,
            ),
            _patch_busy([True, True, False]),
        ):
            rtx_utils._wait_for_streaming_complete()

        # 1 loop pump + 1 final pump
        assert mock_app.update.call_count == 2

    def test_respects_timeout(self, monkeypatch):
        """Exits wait loop on timeout if stage never idles."""
        monkeypatch.setattr(
            rtx_utils,
            "_STREAMING_WAIT_TIMEOUT_S",
            STREAMING_TIMEOUT_S,
        )
        mock_app = MagicMock()
        mock_app.update.side_effect = lambda: time.sleep(MOCK_UPDATE_SLEEP_S)

        with (
            patch(
                "omni.kit.app.get_app",
                return_value=mock_app,
            ),
            _patch_busy([True]),
        ):
            rtx_utils._wait_for_streaming_complete()

        assert mock_app.update.call_count > 0

    def test_timeout_logs_warning(self, monkeypatch):
        """Logs warning when timeout reached while busy."""
        monkeypatch.setattr(
            rtx_utils,
            "_STREAMING_WAIT_TIMEOUT_S",
            STREAMING_TIMEOUT_SHORT_S,
        )
        mock_app = MagicMock()
        mock_logger = MagicMock()

        with (
            patch(
                "omni.kit.app.get_app",
                return_value=mock_app,
            ),
            _patch_busy([True]),
            patch.object(rtx_utils, "logger", mock_logger),
        ):
            rtx_utils._wait_for_streaming_complete()

        mock_logger.warning.assert_called_once()
        msg = mock_logger.warning.call_args[0][0]
        assert "did not complete within" in msg

    def test_logs_info_on_non_trivial_completion(self):
        """Logs info when streaming finishes after delay."""
        mock_app = MagicMock()
        mock_logger = MagicMock()

        def _slow_update():
            time.sleep(MOCK_UPDATE_SLEEP_S)

        mock_app.update.side_effect = _slow_update

        with (
            patch(
                "omni.kit.app.get_app",
                return_value=mock_app,
            ),
            _patch_busy([True, True, False]),
            patch.object(rtx_utils, "logger", mock_logger),
        ):
            rtx_utils._wait_for_streaming_complete()

        mock_logger.info.assert_called_once()
        msg = mock_logger.info.call_args[0][0]
        assert "RTX streaming completed in" in msg
