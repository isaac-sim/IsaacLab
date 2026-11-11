# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch-time decisions for the OVRTX viewer camera that a Newton GL window can switch to."""

import importlib.util
import sys
import types

import pytest
from isaaclab_physx.physics import PhysxCfg
from isaaclab_visualizers.newton import NewtonGLBackendCfg

import isaaclab.app.sim_launcher as sim_launcher
from isaaclab.visualizers import VisualizerCfg


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

    width, height = sim_launcher._fit_window_to_display((1920, 1080))

    assert width <= 0.92 * 1920 and height <= 0.92 * 1080
    assert abs(width / height - 1920 / 1080) < 0.01


def test_window_that_already_fits_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_display(monkeypatch)

    assert sim_launcher._fit_window_to_display((1280, 720)) == (1280, 720)


def test_window_size_is_unchanged_without_a_display_to_query(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)

    assert sim_launcher._fit_window_to_display((1920, 1080)) == (1920, 1080)


@pytest.mark.parametrize(
    ("rtx_camera", "headless", "physics", "ovrtx_installed", "expected"),
    [
        (True, False, PhysxCfg(), False, True),
        (False, False, None, True, False),
        (None, True, None, True, False),
        (None, False, PhysxCfg(), True, False),
        (None, False, None, False, False),
        (None, False, None, True, True),
    ],
    ids=["explicit-on", "explicit-off", "recording-only-window", "physx", "no-ovrtx-package", "auto"],
)
def test_viewer_camera_is_declared_only_where_it_can_be_used(
    monkeypatch: pytest.MonkeyPatch, rtx_camera, headless, physics, ovrtx_installed, expected
) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object() if ovrtx_installed else None)
    visualizer = VisualizerCfg(backend=NewtonGLBackendCfg(rtx_camera=rtx_camera), headless=headless)
    simulations = [types.SimpleNamespace(physics=physics)]

    assert sim_launcher._wants_rtx_viewer_camera(visualizer, simulations) is expected
