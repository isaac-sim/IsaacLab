# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the runtime appearance controls of the OVRTX view in the Newton GL window."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("ovrtx")
from isaaclab_visualizers.newton.newton_rtx_appearance import RtxAppearance, find_scene_lights


class _Attr:
    def __init__(self, value):
        self._value = value

    def __bool__(self):
        return self._value is not None

    def HasValue(self):
        return self._value is not None

    def Get(self):
        return self._value


class _Prim:
    def __init__(self, path, kind, intensity=None, color=None):
        self._path, self._kind = path, kind
        self._attrs = {"inputs:intensity": _Attr(intensity), "inputs:color": _Attr(color)}

    def GetPath(self):
        return self._path

    def GetTypeName(self):
        return self._kind

    def GetAttribute(self, name):
        return self._attrs.get(name, _Attr(None))


class _Stage:
    def __init__(self, prims):
        self._prims = prims

    def Traverse(self):
        return iter(self._prims)


class _Native:
    def __init__(self, fail: bool = False, fail_add: bool = False):
        self.writes, self.fail, self.fail_add = [], fail, fail_add
        self.added, self.removed = [], []

    def write_attribute(self, **kwargs):
        if self.fail:
            raise RuntimeError("write refused")
        self.writes.append(kwargs)

    def add_usd_reference_from_string(self, usd, path):
        if self.fail_add:
            raise RuntimeError("cannot add")
        self.added.append((usd, path))
        return f"reference:{path}"

    def remove_usd(self, reference):
        self.removed.append(reference)


def _appearance(native=None, background_color=None, ovstage=False, prims=None):
    prims = prims or [
        _Prim("/World/sun", "DistantLight", 3000.0, (0.75, 0.75, 0.75)),
        _Prim("/World/dome", "DomeLight", 500.0, (1.0, 0.5, 0.0)),
    ]
    renderer = SimpleNamespace(_use_ovstage=ovstage, backend=SimpleNamespace(renderer=native or _Native()))
    return RtxAppearance(_Stage(prims), renderer, "/RenderCamera_0/RenderProduct", background_color), renderer


def test_only_lights_with_an_intensity_and_a_color_are_found():
    stage = _Stage(
        [
            _Prim("/World/sun", "DistantLight", 3000.0, (0.75, 0.75, 0.75)),
            _Prim("/World/bare", "DomeLight"),
            _Prim("/World/cube", "Cube", 1.0, (1.0, 1.0, 1.0)),
        ]
    )

    lights = find_scene_lights(stage)

    assert [(light.path, light.intensity) for light in lights] == [("/World/sun", 3000.0)]


def test_lights_are_written_scaled_and_tinted_in_one_call_per_attribute():
    appearance, renderer = _appearance()
    appearance.light_scale, appearance.light_tint = 0.5, (1.0, 0.0, 0.0)

    assert appearance.apply_lights()

    visibility, intensity, color = renderer.backend.renderer.writes
    assert visibility["tensor"] == ["inherited", "inherited"]
    assert intensity["prim_paths"] == ["/World/sun", "/World/dome"]
    np.testing.assert_allclose(intensity["tensor"], [1500.0, 250.0])
    np.testing.assert_allclose(color["tensor"], [[0.75, 0.0, 0.0], [1.0, 0.0, 0.0]])


def test_background_is_the_scene_sky_or_one_color():
    appearance, renderer = _appearance()
    appearance.background, appearance.background_color = 1, (0.0, 1.0, 0.0)

    assert appearance.apply_background()
    kind, color = renderer.backend.renderer.writes
    assert kind["tensor"] == ["color"] and kind["prim_paths"] == ["/RenderCamera_0/RenderProduct"]
    np.testing.assert_allclose(color["tensor"], [[0.0, 1.0, 0.0]])

    renderer.backend.renderer.writes.clear()
    appearance.background = 0
    assert appearance.apply_background()
    assert [write["tensor"] for write in renderer.backend.renderer.writes] == [["domeLight"]]


def test_reset_returns_to_the_cameras_configured_background_and_the_scene_lights():
    appearance, renderer = _appearance(background_color=(0.1, 0.2, 0.3))
    assert appearance.background == 1  # The camera was configured with a solid color.
    appearance.background, appearance.light_scale = 0, 2.0

    assert appearance.reset()

    assert (appearance.background, appearance.background_color, appearance.light_scale) == (1, (0.1, 0.2, 0.3), 1.0)
    intensity = next(w for w in renderer.backend.renderer.writes if w["attribute_name"] == "inputs:intensity")
    np.testing.assert_allclose(intensity["tensor"], [3000.0, 500.0])


def test_a_refused_write_is_reported_instead_of_raised():
    appearance, _ = _appearance(native=_Native(fail=True))

    assert not appearance.apply_lights()
    assert "write refused" in appearance.error


def test_the_controls_are_unavailable_on_the_shared_stage_path():
    appearance, _ = _appearance(ovstage=True)

    assert not appearance.available


def _written(renderer, attribute):
    return [w for w in renderer.backend.renderer.writes if w["attribute_name"] == attribute][-1]


def test_a_dome_or_distant_lighting_adds_the_viewer_lights_once_and_hides_the_scene_lights():
    appearance, renderer = _appearance()
    native = renderer.backend.renderer
    assert native.added == []  # Nothing is added while the scene's own lights are used.

    appearance.lighting = 1  # Dome Light.
    assert appearance.apply_lights()
    appearance.lighting = 3  # Dome + Distant.
    assert appearance.apply_lights()

    assert [path for _, path in native.added] == ["/NewtonViewerLights"]  # Added once, on first use.
    visibility = _written(renderer, "visibility")
    assert dict(zip(visibility["prim_paths"], visibility["tensor"], strict=True)) == {
        "/World/sun": "invisible",
        "/World/dome": "invisible",
        "/NewtonViewerLights/Dome": "inherited",
        "/NewtonViewerLights/Sun": "inherited",
    }


def test_the_distant_lighting_shows_only_the_viewer_sun_and_scene_lights_return_with_the_scene_lighting():
    appearance, renderer = _appearance()
    appearance.lighting = 2
    appearance.apply_lights()
    visibility = _written(renderer, "visibility")
    shown = [
        path for path, state in zip(visibility["prim_paths"], visibility["tensor"], strict=True) if state == "inherited"
    ]
    assert shown == ["/NewtonViewerLights/Sun"]

    appearance.lighting = 0
    appearance.apply_lights()
    visibility = _written(renderer, "visibility")
    shown = [
        path for path, state in zip(visibility["prim_paths"], visibility["tensor"], strict=True) if state == "inherited"
    ]
    assert shown == ["/World/sun", "/World/dome"]


def test_a_lighting_that_cannot_add_its_lights_reports_it_and_closing_removes_them():
    refused, _ = _appearance(native=_Native(fail_add=True))
    refused.lighting = 1
    assert not refused.apply_lights()
    assert "cannot add" in refused.error

    appearance, renderer = _appearance()
    appearance.lighting = 1
    appearance.apply_lights()
    appearance.close()
    appearance.close()  # A second close is harmless.
    assert renderer.backend.renderer.removed == ["reference:/NewtonViewerLights"]


def test_reset_returns_to_the_scene_lighting():
    appearance, _ = _appearance()
    appearance.lighting = 3

    appearance.reset()

    assert appearance.lighting == 0
