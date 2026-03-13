# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for RTX streaming wait helpers.

Covers callback state updates, subscription behavior, and timeout-aware wait
logic in :mod:`isaaclab_physx.renderers.isaac_rtx_renderer_utils`.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

import isaaclab_physx.renderers.isaac_rtx_renderer_utils as rtx_utils

# test-specific timeout overrides for _STREAMING_WAIT_TIMEOUT_S
STREAMING_TIMEOUT_S = 0.1
STREAMING_TIMEOUT_SHORT_S = 0.01

# simulated per-update sleep to advance wall-clock time inside the wait loop
MOCK_UPDATE_SLEEP_S = 0.02

# how many app.update() iterations before the mock becomes idle
MOCK_ITERATIONS_BEFORE_IDLE = 3


@pytest.fixture(autouse=True)
def _reset_streaming_globals(monkeypatch):
    """Restore module-level streaming state so tests are isolated."""
    monkeypatch.setattr(rtx_utils, "_streaming_is_busy", False)
    monkeypatch.setattr(rtx_utils, "_streaming_subscription", None)
    monkeypatch.setattr(rtx_utils, "_streaming_subscribed", False)


# ---------------------------------------------------------------------------
# _on_streaming_status_event
# ---------------------------------------------------------------------------


class TestOnStreamingStatusEvent:
    """Callback correctly translates RTX streaming events into the busy flag."""

    def test_sets_busy_true(self):
        """Sets busy flag to true when event reports busy."""
        rtx_utils._on_streaming_status_event({"isBusy": True})
        assert rtx_utils._streaming_is_busy is True

    def test_sets_busy_false(self):
        """Sets busy flag to false when event reports idle."""
        rtx_utils._streaming_is_busy = True
        rtx_utils._on_streaming_status_event({"isBusy": False})
        assert rtx_utils._streaming_is_busy is False

    def test_ignores_missing_key(self):
        """Leaves busy state unchanged when key is absent."""
        rtx_utils._streaming_is_busy = True
        rtx_utils._on_streaming_status_event({"otherField": 42})
        assert rtx_utils._streaming_is_busy is True

    def test_ignores_none_value(self):
        """Leaves busy state unchanged when value is None."""
        rtx_utils._streaming_is_busy = True
        rtx_utils._on_streaming_status_event({"isBusy": None})
        assert rtx_utils._streaming_is_busy is True

    def test_handles_non_subscriptable_event(self):
        """Ignores malformed events that are not subscriptable."""
        rtx_utils._on_streaming_status_event(None)
        assert rtx_utils._streaming_is_busy is False


# ---------------------------------------------------------------------------
# _ensure_streaming_subscription
# ---------------------------------------------------------------------------


class TestEnsureStreamingSubscription:
    """Subscription helper registers once and does not retry on first failure."""

    def test_subscribes_to_correct_event(self):
        """Subscribes to the expected event and stores its handle."""
        mock_dispatcher = MagicMock()
        mock_dispatcher.observe_event.return_value = "sub_handle"

        with patch("carb.eventdispatcher.get_eventdispatcher", return_value=mock_dispatcher):
            rtx_utils._ensure_streaming_subscription()

        mock_dispatcher.observe_event.assert_called_once_with(
            observer_name="isaaclab_rtx_streaming_wait",
            event_name=rtx_utils._RTX_STREAMING_STATUS_EVENT,
            on_event=rtx_utils._on_streaming_status_event,
        )
        assert rtx_utils._streaming_subscribed is True
        assert rtx_utils._streaming_subscription == "sub_handle"

    def test_idempotent_after_first_call(self):
        """Performs subscription at most once across repeated calls."""
        mock_dispatcher = MagicMock()
        with patch("carb.eventdispatcher.get_eventdispatcher", return_value=mock_dispatcher):
            rtx_utils._ensure_streaming_subscription()
            rtx_utils._ensure_streaming_subscription()

        assert mock_dispatcher.observe_event.call_count == 1

    def test_handles_missing_dispatcher(self):
        """Marks subscription as attempted even when dispatcher is unavailable."""
        with patch("carb.eventdispatcher.get_eventdispatcher", return_value=None):
            rtx_utils._ensure_streaming_subscription()

        assert rtx_utils._streaming_subscribed is True
        assert rtx_utils._streaming_subscription is None


# ---------------------------------------------------------------------------
# _wait_for_streaming_complete
# ---------------------------------------------------------------------------


class TestWaitForStreamingComplete:
    """Blocking wait pumps app.update() while busy and respects timeout."""

    def test_returns_immediately_when_not_busy(self):
        """Skips loop and issues only the final update when idle."""
        mock_app = MagicMock()
        with patch("omni.kit.app.get_app", return_value=mock_app):
            rtx_utils._wait_for_streaming_complete()

        # Only the final update call, no streaming loop iterations.
        mock_app.update.assert_called_once()

    def test_pumps_updates_until_idle(self):
        """Pumps updates until busy flips to false."""
        rtx_utils._streaming_is_busy = True
        mock_app = MagicMock()
        loop_calls = 0

        def _simulate_streaming_done():
            nonlocal loop_calls
            loop_calls += 1
            if loop_calls >= MOCK_ITERATIONS_BEFORE_IDLE:
                rtx_utils._streaming_is_busy = False

        mock_app.update.side_effect = _simulate_streaming_done

        with patch("omni.kit.app.get_app", return_value=mock_app):
            rtx_utils._wait_for_streaming_complete()

        assert mock_app.update.call_count == MOCK_ITERATIONS_BEFORE_IDLE + 1
        assert rtx_utils._streaming_is_busy is False

    def test_respects_timeout(self, monkeypatch):
        """Exits wait loop on timeout if busy never clears."""
        monkeypatch.setattr(rtx_utils, "_STREAMING_WAIT_TIMEOUT_S", STREAMING_TIMEOUT_S)
        rtx_utils._streaming_is_busy = True
        mock_app = MagicMock()
        mock_app.update.side_effect = lambda: time.sleep(MOCK_UPDATE_SLEEP_S)

        with patch("omni.kit.app.get_app", return_value=mock_app):
            rtx_utils._wait_for_streaming_complete()

        assert rtx_utils._streaming_is_busy is True
        assert mock_app.update.call_count > 0

    def test_timeout_logs_warning(self, monkeypatch):
        """Logs warning when timeout is reached while still busy."""
        monkeypatch.setattr(rtx_utils, "_STREAMING_WAIT_TIMEOUT_S", STREAMING_TIMEOUT_SHORT_S)
        rtx_utils._streaming_is_busy = True
        mock_app = MagicMock()
        mock_logger = MagicMock()

        with (
            patch("omni.kit.app.get_app", return_value=mock_app),
            patch.object(rtx_utils, "logger", mock_logger),
        ):
            rtx_utils._wait_for_streaming_complete()

        mock_logger.warning.assert_called_once()
        assert "RTX streaming did not complete within" in mock_logger.warning.call_args[0][0]

    def test_logs_info_on_non_trivial_completion(self):
        """Logs completion info when streaming finishes after delay."""
        rtx_utils._streaming_is_busy = True
        mock_app = MagicMock()
        mock_logger = MagicMock()

        def _become_idle_after_delay():
            time.sleep(MOCK_UPDATE_SLEEP_S)
            rtx_utils._streaming_is_busy = False

        mock_app.update.side_effect = _become_idle_after_delay

        with (
            patch("omni.kit.app.get_app", return_value=mock_app),
            patch.object(rtx_utils, "logger", mock_logger),
        ):
            rtx_utils._wait_for_streaming_complete()

        mock_logger.info.assert_called_once()
        assert "RTX streaming completed in" in mock_logger.info.call_args[0][0]
