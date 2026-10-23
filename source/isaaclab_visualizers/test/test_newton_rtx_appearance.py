# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the runtime appearance controls of the OVRTX view in the Newton GL window."""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("ovrtx")
from isaaclab_visualizers.newton import newton_rtx_appearance as module
from isaaclab_visualizers.newton.newton_rtx_appearance import RtxAppearance, find_scene_lights, sun_matrix


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
        return f"reference:{path}:{len(self.added)}"

    def remove_usd(self, reference):
        self.removed.append(reference)


def _appearance(native=None, background_color=None, ovstage=False):
    prims = [
        _Prim("/World/sun", "DistantLight", 3000.0, (0.75, 0.75, 0.75)),
        _Prim("/World/cube", "Cube", 1.0, (1, 1, 1)),
    ]
    renderer = SimpleNamespace(_use_ovstage=ovstage, backend=SimpleNamespace(renderer=native or _Native()))
    return RtxAppearance(
        _Stage(prims), renderer, "/RenderCamera_0/RenderProduct", background_color
    ), renderer.backend.renderer


def _last(native, attribute):
    return [w for w in native.writes if w["attribute_name"] == attribute][-1]


def _states(native):
    write = _last(native, "visibility")
    return dict(zip(write["prim_paths"], write["tensor"], strict=True))


def test_only_lights_with_an_intensity_and_a_color_are_found():
    stage = _Stage(
        [
            _Prim("/World/sun", "DistantLight", 3000.0, (0.75, 0.75, 0.75)),
            _Prim("/World/bare", "DomeLight"),
            _Prim("/World/cube", "Cube", 1.0, (1.0, 1.0, 1.0)),
        ]
    )

    assert [(light.path, light.intensity) for light in find_scene_lights(stage)] == [("/World/sun", 3000.0)]


def test_scene_lights_are_on_and_the_dome_and_distant_lights_are_off_by_default():
    appearance, native = _appearance()

    assert (appearance.scene_enabled, appearance.dome_enabled, appearance.sun_enabled) == (True, False, False)
    assert appearance.apply_lights()

    assert native.added == []  # The viewer's own lights are only added once enabled.
    assert _states(native) == {"/World/sun": "inherited"}


def test_scene_lights_take_an_intensity_scale_and_a_color_tint():
    appearance, native = _appearance()
    appearance.scene_scale, appearance.scene_tint = 0.5, (1.0, 0.0, 0.0)

    appearance.apply_lights()

    np.testing.assert_allclose(_last(native, "inputs:intensity")["tensor"], [1500.0])
    np.testing.assert_allclose(_last(native, "inputs:color")["tensor"], [[0.75, 0.0, 0.0]])


def test_the_dome_light_is_added_once_when_enabled_and_shows_with_its_own_settings():
    appearance, native = _appearance()
    appearance.dome_enabled, appearance.dome_intensity, appearance.dome_color = True, 2500.0, (0.2, 0.3, 0.4)

    assert appearance.apply_lights()
    appearance.dome_intensity = 100.0
    assert appearance.apply_lights()

    assert [path for _, path in native.added] == ["/NewtonViewerDome"]  # Added once.
    assert "asset inputs:texture:file = @@" in native.added[0][0]  # No texture yet.
    assert _states(native) == {"/World/sun": "inherited", "/NewtonViewerDome": "inherited"}
    assert _last(native, "inputs:intensity")["tensor"].tolist() == [3000.0, 100.0]


def test_each_light_can_be_turned_off_independently():
    appearance, native = _appearance()
    appearance.dome_enabled = appearance.sun_enabled = True
    appearance.scene_enabled = False

    appearance.apply_lights()

    assert _states(native) == {
        "/World/sun": "invisible",
        "/NewtonViewerDome": "inherited",
        "/NewtonViewerSun": "inherited",
    }
    appearance.sun_enabled = False
    appearance.apply_lights()
    assert _states(native)["/NewtonViewerSun"] == "invisible"


def test_the_distant_light_gets_its_angle_and_direction():
    appearance, native = _appearance()
    appearance.sun_enabled, appearance.sun_angle, appearance.sun_elevation, appearance.sun_azimuth = (
        True,
        4.0,
        90.0,
        0.0,
    )

    assert appearance.apply_lights()

    assert [path for _, path in native.added] == ["/NewtonViewerSun"]
    np.testing.assert_allclose(_last(native, "inputs:angle")["tensor"], [4.0])
    matrix = _last(native, "omni:xform")["tensor"]
    assert matrix.shape == (1, 4, 4)
    np.testing.assert_allclose(matrix[0][2][:3], [0.0, 0.0, 1.0], atol=1e-9)  # Straight down: local +Z is up.


@pytest.mark.parametrize(("elevation", "azimuth"), [(40.0, 0.0), (65.0, 35.0), (90.0, 0.0), (10.0, 270.0)])
def test_the_sun_transform_is_a_rotation_that_points_local_z_toward_the_light(elevation, azimuth):
    matrix = sun_matrix(elevation, azimuth)
    rotation = matrix[:3, :3]

    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    el, az = np.radians(elevation), np.radians(azimuth)
    np.testing.assert_allclose(matrix[2, :3], [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], atol=1e-9)


def test_a_sky_texture_is_loaded_in_the_background_and_adds_the_dome_light_again_with_it(monkeypatch):
    monkeypatch.setattr(module, "retrieve_file_path", lambda source: f"/cache/{source.rsplit('/', 1)[-1]}")
    appearance, native = _appearance()
    appearance.dome_enabled = True
    appearance.apply_lights()

    appearance.use_sky("https://assets/skies/lakeside.hdr")
    assert appearance.texture_status == "Loading sky..."
    deadline = time.time() + 5
    while not appearance.poll():
        assert time.time() < deadline
        time.sleep(0.01)

    assert native.removed == ["reference:/NewtonViewerDome:1"]  # The first dome light was replaced.
    assert "@/cache/lakeside.hdr@" in native.added[-1][0]
    assert appearance.dome_color == (1.0, 1.0, 1.0)  # A texture is not tinted blue.
    assert appearance.texture_status == ""


def test_removing_the_sky_texture_restores_the_sky_blue_dome_without_a_download():
    appearance, native = _appearance()
    appearance.dome_enabled = True
    appearance._dome_texture_file, appearance.dome_texture = "/cache/sky.hdr", "https://assets/sky.hdr"
    appearance.apply_lights()

    appearance.use_sky("")

    assert appearance.dome_texture == "" and appearance.dome_color == (0.53, 0.68, 0.9)
    assert "asset inputs:texture:file = @@" in native.added[-1][0]


def test_a_sky_that_cannot_be_loaded_is_reported(monkeypatch):
    def refuse(source):
        raise OSError("no network")

    monkeypatch.setattr(module, "retrieve_file_path", refuse)
    appearance, _ = _appearance()
    appearance.dome_enabled = True

    appearance.use_sky("https://assets/missing.hdr")
    deadline = time.time() + 5
    while appearance.texture_status == "Loading sky...":
        appearance.poll()
        assert time.time() < deadline
        time.sleep(0.01)

    assert appearance.texture_status.startswith("Could not load the sky") and appearance.dome_texture == ""


def test_background_is_the_scene_sky_or_one_color():
    appearance, native = _appearance()
    appearance.background, appearance.background_color = 1, (0.0, 1.0, 0.0)

    assert appearance.apply_background()
    kind, color = native.writes
    assert kind["tensor"] == ["color"] and kind["prim_paths"] == ["/RenderCamera_0/RenderProduct"]
    np.testing.assert_allclose(color["tensor"], [[0.0, 1.0, 0.0]])

    native.writes.clear()
    appearance.background = 0
    assert appearance.apply_background()
    assert [write["tensor"] for write in native.writes] == [["domeLight"]]


def test_reset_returns_to_the_scene_lights_and_the_cameras_configured_background():
    appearance, native = _appearance(background_color=(0.1, 0.2, 0.3))
    assert appearance.background == 1  # The camera was configured with a solid color.
    appearance.background, appearance.scene_enabled, appearance.sun_enabled = 0, False, True
    appearance.apply_lights()

    assert appearance.reset()

    assert (appearance.background, appearance.background_color) == (1, (0.1, 0.2, 0.3))
    assert (appearance.scene_enabled, appearance.sun_enabled) == (True, False)
    assert _states(native) == {"/World/sun": "inherited", "/NewtonViewerSun": "invisible"}


def test_a_refused_write_or_a_light_that_cannot_be_added_is_reported_instead_of_raised():
    refused, _ = _appearance(native=_Native(fail=True))
    assert not refused.apply_lights()
    assert "write refused" in refused.error

    cannot_add, _ = _appearance(native=_Native(fail_add=True))
    cannot_add.dome_enabled = True
    assert not cannot_add.apply_lights()
    assert "cannot add" in cannot_add.error


def test_closing_removes_the_viewer_lights_once():
    appearance, native = _appearance()
    appearance.dome_enabled = appearance.sun_enabled = True
    appearance.apply_lights()

    appearance.close()
    appearance.close()

    assert native.removed == ["reference:/NewtonViewerDome:1", "reference:/NewtonViewerSun:2"]


def test_the_controls_are_unavailable_on_the_shared_stage_path():
    appearance, _ = _appearance(ovstage=True)

    assert not appearance.available
