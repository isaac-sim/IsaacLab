# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for RTX streaming waits and render-update cadence."""

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

from isaaclab.scene_data import SceneDataFormat  # noqa: E402

# test-specific timeout overrides for _STREAMING_WAIT_TIMEOUT_S
STREAMING_TIMEOUT_S = 0.1

# simulated per-update sleep to advance wall-clock time inside the wait loop
MOCK_UPDATE_SLEEP_S = 0.02

# how many app.update() iterations before the mock becomes idle
MOCK_ITERATIONS_BEFORE_IDLE = 3


@pytest.fixture(autouse=True)
def _reset_globals(monkeypatch):
    """Restore module-level state so tests are isolated."""
    monkeypatch.setattr(rtx_utils, "_last_render_update_key", (0, -1, -1))


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
# _wait_for_streaming_complete
# ---------------------------------------------------------------------------


class TestWaitForStreamingComplete:
    """Blocking wait pumps app.update() while busy and respects timeout."""

    @pytest.mark.parametrize("has_context", [False, True])
    def test_returns_immediately_when_not_busy(self, mock_omni_usd, mock_omni_kit_app, has_context):
        """Idle and absent stages need only the final update."""
        mock_app = MagicMock()
        mock_omni_kit_app.get_app.return_value = mock_app
        context = mock_omni_usd.get_context.return_value
        context.get_stage_streaming_status.return_value = False
        if not has_context:
            mock_omni_usd.get_context.return_value = None

        rtx_utils._wait_for_streaming_complete()

        mock_app.update.assert_called_once()

    def test_pumps_updates_until_idle(self, mock_omni_usd, mock_omni_kit_app):
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
        mock_omni_usd.get_context.return_value.get_stage_streaming_status.side_effect = _streaming_status

        rtx_utils._wait_for_streaming_complete()

        assert mock_app.update.call_count == MOCK_ITERATIONS_BEFORE_IDLE + 1

    def test_respects_timeout(self, monkeypatch, mock_omni_kit_app, caplog):
        """Exits wait loop on timeout if busy never clears."""
        monkeypatch.setattr(rtx_utils, "_STREAMING_WAIT_TIMEOUT_S", STREAMING_TIMEOUT_S)
        mock_app = MagicMock()
        mock_app.update.side_effect = lambda: time.sleep(MOCK_UPDATE_SLEEP_S)
        mock_omni_kit_app.get_app.return_value = mock_app

        with patch.object(rtx_utils, "_get_stage_streaming_busy", return_value=True):
            rtx_utils._wait_for_streaming_complete()

        assert mock_app.update.call_count > 0
        assert "RTX streaming did not complete within" in caplog.text


# ---------------------------------------------------------------------------
# ensure_isaac_rtx_render_update
# ---------------------------------------------------------------------------


class TestEnsureIsaacRtxRenderUpdate:
    """Tests for :func:`ensure_isaac_rtx_render_update`.

    Covers dedup logic, visualizer-skip behaviour, and the first-call-for-sim
    guard that prevents annotator buffers from never being populated.
    """

    @pytest.fixture()
    def mock_sim(self):
        """A minimal mock of :class:`SimulationContext`."""
        sim = MagicMock()
        sim._physics_step_count = 0
        sim.get_physics_step_count.side_effect = lambda: sim._physics_step_count
        sim._render_generation = 0
        sim.render_generation = 0
        sim.is_rendering = True
        sim.visualizers = []
        return sim

    @pytest.fixture()
    def pumping_visualizer(self):
        """A visualizer that claims to pump ``app.update()``."""
        viz = MagicMock()
        viz.pumps_app_update.return_value = True
        return viz

    @pytest.fixture()
    def mock_sim_context(self, monkeypatch):
        """Patch ``sim_utils`` without importing the real ``SimulationContext``."""
        sim_context = MagicMock()
        monkeypatch.setattr(rtx_utils, "sim_utils", types.SimpleNamespace(SimulationContext=sim_context))
        return sim_context

    def test_visualizer_pumps_only_after_initial_render_update(
        self, mock_sim, mock_sim_context, pumping_visualizer, mock_omni_kit_app
    ):
        """Publish the first frame before yielding app updates to an active visualizer."""
        mock_sim.visualizers = [pumping_visualizer]
        mock_app = MagicMock()
        mock_omni_kit_app.get_app.return_value = mock_app
        mock_sim_context.instance.return_value = mock_sim
        provider = mock_sim.get_scene_data_provider.return_value
        mock_app.update.side_effect = provider.get_transforms.assert_called_once

        with patch.object(rtx_utils, "_get_stage_streaming_busy", return_value=False):
            rtx_utils.ensure_isaac_rtx_render_update()
            mock_app.update.assert_called_once()
            mock_app.update.reset_mock()

            mock_sim._physics_step_count = 1
            rtx_utils.ensure_isaac_rtx_render_update()

        mock_app.update.assert_not_called()
        provider.get_transforms.assert_called_once()
        assert provider.get_transforms.call_args.args[0]._cls is SceneDataFormat.FabricMatrix44
        mock_sim.physics_manager.forward.assert_not_called()

    def test_no_sim_is_noop(self, mock_sim_context, mock_omni_kit_app):
        """No-op when SimulationContext.instance() returns None."""
        mock_app = MagicMock()
        mock_omni_kit_app.get_app.return_value = mock_app
        mock_sim_context.instance.return_value = None

        rtx_utils.ensure_isaac_rtx_render_update()

        mock_app.update.assert_not_called()

    def test_dedup_same_step(self, mock_sim, mock_sim_context, mock_omni_kit_app):
        """Second call in the same physics step is a no-op (dedup)."""
        mock_app = MagicMock()
        mock_omni_kit_app.get_app.return_value = mock_app
        mock_sim_context.instance.return_value = mock_sim

        with (
            patch.object(rtx_utils, "_get_stage_streaming_busy", return_value=False),
        ):
            rtx_utils.ensure_isaac_rtx_render_update()
            mock_app.update.assert_called_once()
            mock_app.update.reset_mock()

            rtx_utils.ensure_isaac_rtx_render_update()

        mock_app.update.assert_not_called()

    @pytest.mark.parametrize("force", [False, True])
    def test_not_rendering_pumps_only_when_forced(self, mock_sim, mock_sim_context, mock_omni_kit_app, force):
        """Offscreen capture publishes through SDP only when a frame is requested."""
        mock_sim.is_rendering = False
        mock_app = MagicMock()
        mock_omni_kit_app.get_app.return_value = mock_app
        mock_sim_context.instance.return_value = mock_sim

        with patch.object(rtx_utils, "_get_stage_streaming_busy", return_value=False):
            rtx_utils.ensure_isaac_rtx_render_update(force=force)

        assert mock_app.update.call_count == int(force)
        provider = mock_sim.get_scene_data_provider.return_value
        assert provider.get_transforms.call_count == int(force)
        mock_sim.physics_manager.forward.assert_not_called()
