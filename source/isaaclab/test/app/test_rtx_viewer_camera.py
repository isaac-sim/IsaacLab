# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch-time decisions for the OVRTX viewer camera that a Newton GL window can switch to."""

import importlib.util
import sys
import types

import pytest
from isaaclab_visualizers.newton import NewtonGLBackendCfg

import isaaclab.app.sim_launcher as sim_launcher
from isaaclab.visualizers import ImageViewCfg, VisualizerCfg, WindowCfg


def _fake_display(monkeypatch: pytest.MonkeyPatch, width: int = 1920, height: int = 1080) -> None:
    """Make the launcher see an X display of the given size, whatever the host."""
    screen = types.SimpleNamespace(width=width, height=height)
    display = types.SimpleNamespace(get_default_screen=lambda: screen)
    monkeypatch.setitem(
        sys.modules, "pyglet", types.SimpleNamespace(display=types.SimpleNamespace(get_display=lambda: display))
    )
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("DISPLAY", ":0")


def test_oversized_window_shrinks_to_fit_the_display_keeping_its_aspect(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_display(monkeypatch)

    width, height = WindowCfg(size=(1920, 1080)).fit_to_display()

    assert width <= 0.92 * 1920 and height <= 0.92 * 1080
    assert abs(width / height - 1920 / 1080) < 0.01


def test_window_that_already_fits_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_display(monkeypatch)

    assert WindowCfg(size=(1280, 720)).fit_to_display() == (1280, 720)


def test_window_size_is_unchanged_without_a_display_to_query(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)

    assert WindowCfg(size=(1920, 1080)).fit_to_display() == (1920, 1080)


@pytest.mark.parametrize(
    ("backend_kwargs", "headless", "expected"),
    [
        ({}, False, False),
        ({"rtx_camera": False}, False, False),
        ({"rtx_camera": True}, False, True),
        ({"rtx_camera": True}, True, False),
    ],
    ids=["default-off", "explicit-off", "explicit-on", "explicit-on-recording-only"],
)
def test_viewer_camera_is_declared_only_when_requested(backend_kwargs, headless, expected) -> None:
    visualizer = VisualizerCfg(backend=NewtonGLBackendCfg(**backend_kwargs), headless=headless)

    assert sim_launcher._wants_rtx_viewer_camera(visualizer) is expected


@pytest.mark.parametrize(
    ("headless", "view"),
    [(True, None), (False, ImageViewCfg(source="/World/envs/env_0/Camera"))],
    ids=["recording-only-window", "window-with-a-view"],
)
def test_starting_in_rtx_declares_the_viewer_camera_even_without_a_window_or_with_a_view(
    monkeypatch: pytest.MonkeyPatch, headless, view
) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    visualizer = VisualizerCfg(backend=NewtonGLBackendCfg(initial_renderer="rtx"), headless=headless, view=view)

    assert sim_launcher._wants_rtx_viewer_camera(visualizer) is True


def test_starting_in_rtx_without_the_viewer_camera_is_rejected() -> None:
    visualizer = VisualizerCfg(backend=NewtonGLBackendCfg(initial_renderer="rtx", rtx_camera=False))

    with pytest.raises(ValueError, match="requires rtx_camera"):
        sim_launcher._wants_rtx_viewer_camera(visualizer)


def test_the_declared_viewer_camera_renders_only_when_its_view_is_shown() -> None:
    from isaaclab.scene import InteractiveSceneCfg

    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=1.0)
    visualizer = VisualizerCfg(backend=NewtonGLBackendCfg())

    name, *_ = sim_launcher._declare_viewer_camera(scene_cfg, 0, visualizer)

    assert vars(scene_cfg)[name].render_on_demand
    assert visualizer.backend.rtx_camera_path == f"/World/{name}"


@pytest.mark.parametrize(("selection", "declared"), [("newton", False), ("newton_gl", False), ("newton_rtx", True)])
def test_only_the_newton_rtx_selection_adds_the_viewer_camera(selection, declared) -> None:
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg

    cfg = types.SimpleNamespace(
        sim=SimulationCfg(physics=sim_launcher.NewtonCfg()),
        scene=InteractiveSceneCfg(num_envs=2, env_spacing=4.0),
    )

    sim_launcher.validate_launch(cfg, {"visualizer": [selection]})

    assert bool([name for name in vars(cfg.scene) if name.startswith("viewer_camera_")]) is declared
