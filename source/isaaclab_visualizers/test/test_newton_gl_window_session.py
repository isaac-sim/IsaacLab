# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native GL windows survive program switches and release scene resources."""

import sys
import time
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import isaaclab_visualizers.newton.newton_visualizer as newton_visualizer
import newton
import numpy as np
import pytest
import warp as wp
from isaaclab_visualizers.newton import NewtonGLBackendCfg, NewtonGLVisualizer

from isaaclab.app.loading_screen import report_activity
from isaaclab.visualizers import VisualizerCfg, WindowCfg

pytestmark = [pytest.mark.integration, pytest.mark.rendering, pytest.mark.kitless]


@pytest.mark.skipif(sys.platform != "linux", reason="EGL presentation test requires Linux.")
@pytest.mark.parametrize("failure", [False, True])
def test_window_session_reuses_native_context_and_releases_scene(monkeypatch, failure):
    """Switching releases the old scene and panels, preserving GL resources until normal or failed exit."""
    from newton._src.viewer.viewer_gl import ViewerGui

    # Keep real GL resources and UI callbacks while avoiding a desktop dependency.
    monkeypatch.setenv("PYOPENGL_PLATFORM", "egl")
    monkeypatch.setenv("DISPLAY", ":unused")
    monkeypatch.setattr(newton_visualizer, "write_desktop_entry", lambda *_args: None)
    initialize_gl = newton_visualizer.ViewerGL.__init__

    def initialize_egl(viewer, *args, **kwargs):
        kwargs["headless"] = True
        initialize_gl(viewer, *args, **kwargs)
        gui_options = {"workspace_enabled": True} if hasattr(newton_visualizer.ViewerGL, "set_solver") else {}
        viewer.gui = ViewerGui(viewer, viewer.renderer.window, **gui_options)

    monkeypatch.setattr(newton_visualizer.ViewerGL, "__init__", initialize_egl)
    cfg = VisualizerCfg(window=WindowCfg(size=(480, 320)), backend=NewtonGLBackendCfg())
    provider = SimpleNamespace(num_envs=1)
    sim = SimpleNamespace(
        get_scene_data_provider=lambda: provider,
        device="cpu",
        backend_manager=SimpleNamespace(backend_name="newton"),
        cfg=SimpleNamespace(gravity=(0.0, 0.0, -9.81)),
        vis_marker_registry=SimpleNamespace(get_groups=lambda: {}),
    )
    expected_exit = pytest.raises(RuntimeError, match="Next program failed") if failure else nullcontext()
    with expected_exit, wp.ScopedDevice("cpu"):
        with NewtonGLVisualizer.window_session() as session:
            first = NewtonGLVisualizer(cfg)
            # Initialize presentation only: physics owns an independent lifetime.
            newton_visualizer.NewtonVisualizerBase.initialize(first, sim, cameras=[])
            viewer = first._viewer
            first_keys = first.key_event_source
            assert first_keys is not None and not first_keys.closed
            window, context = viewer.renderer.window, viewer.renderer.window.context
            builder = newton.ModelBuilder()
            builder.add_particle(pos=(0.0, 0.0, 1.0), vel=(0.0, 0.0, 0.0), mass=1.0)
            model = builder.finalize(device="cpu")
            viewer.set_model(model)
            branding_callbacks = tuple(viewer.gui._ui_callbacks.get("free", []))
            assert viewer._render_training_controls in viewer.gui._ui_callbacks["side"]
            if viewer._workspace_enabled:
                assert viewer._render_branding in branding_callbacks
            viewer.log_points("program/particles", points=model.particle_q, radii=0.1)
            program_panel = Mock()
            first.register_ui_callback(program_panel, position="panel")
            assert viewer.objects

            first.close()

            assert first._viewer is first._sim is None
            assert first.key_event_source is None and first_keys.closed
            assert viewer.model is None
            assert not viewer.objects
            assert program_panel not in viewer.gui._ui_callbacks["panel"]
            assert window.context is context

            second = NewtonGLVisualizer(cfg)
            try:
                claimed = False
                if hasattr(viewer.gui, "show_loading_splash"):
                    loading_exit = (
                        pytest.raises(RuntimeError, match="Next program failed") if failure else nullcontext()
                    )
                    with loading_exit, session.loading("Loading next example"):
                        assert viewer.gui._loading_splash_active
                        first_frame = viewer.get_frame(render_ui=True).numpy().copy()
                        frame_count = viewer.gui.ui.imgui.get_frame_count()
                        time.sleep(0.31)
                        report_activity("Building next scene")
                        assert viewer.gui.ui.imgui.get_frame_count() > frame_count
                        frame = viewer.get_frame(render_ui=True).numpy()
                        # The upper center contains the animated robot, away from the elapsed clock.
                        height, width = first_frame.shape[:2]
                        logo = np.s_[height // 8 : height // 2, width // 3 : 2 * width // 3]
                        assert np.count_nonzero(frame[logo] != first_frame[logo]) > 10
                        report_activity(None)
                        if failure:
                            raise RuntimeError("Next program failed")
                        newton_visualizer.NewtonVisualizerBase.initialize(second, sim, cameras=[])
                        claimed = True
                        assert not viewer.gui._loading_splash_active
                    assert not viewer.gui._loading_splash_active
                if not claimed:
                    newton_visualizer.NewtonVisualizerBase.initialize(second, sim, cameras=[])
                assert second._viewer is viewer
                assert second.key_event_source is not None
                assert second.key_event_source is not first_keys
                assert not second.key_event_source.closed
                assert second._viewer.renderer.window is window
                assert window.context is context
                assert second._draw_streaming_view_controls in viewer.gui._ui_callbacks["side"]
                assert first._draw_streaming_view_controls not in viewer.gui._ui_callbacks["side"]
                # Exercise real model rebinding: window controls survive exactly once.
                provider.create_mapping = lambda _labels: None
                sim.get_or_create_backend = lambda _cfg: SimpleNamespace(model=model)
                second.newton_cfg = object()
                second._is_initialized = True
                second.reset()
                second.reset()
                assert viewer.gui._ui_callbacks["side"].count(viewer._render_training_controls) == 1
                assert viewer.gui._ui_callbacks["side"].count(second._draw_streaming_view_controls) == 1
                assert tuple(viewer.gui._ui_callbacks.get("free", [])) == branding_callbacks
            finally:
                second.close()
            if failure:
                raise RuntimeError("Next program failed")

    assert window.context is None
