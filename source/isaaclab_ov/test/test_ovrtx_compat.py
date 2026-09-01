# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for OVRTX 0.4 / 0.5 render-var key compatibility."""

from __future__ import annotations

import importlib.metadata
import importlib.util

import pytest
from packaging.version import Version

_REQUIRED_MODULES = ("isaaclab_ov", "pxr")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]

pytestmark = [
    pytest.mark.isaacsim_ci,
    pytest.mark.skipif(
        bool(_MISSING_MODULES),
        reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}",
    ),
]

if not _MISSING_MODULES:
    from isaaclab_ov.renderers.ovrtx_compat import (  # noqa: E402
        RENDER_VAR_FRAME_KEYS,
        build_render_var_frame_keys,
        detect_ovrtx_version,
        uses_prim_path_render_vars,
    )
    from isaaclab_ov.renderers.ovrtx_usd import render_var_prim_paths_by_source  # noqa: E402
else:
    RENDER_VAR_FRAME_KEYS = None
    build_render_var_frame_keys = None
    detect_ovrtx_version = None
    render_var_prim_paths_by_source = None
    uses_prim_path_render_vars = None


def test_detect_ovrtx_version_reads_distribution_metadata(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.4.1.364340")
    assert detect_ovrtx_version() == Version("0.4.1.364340")


def test_detect_ovrtx_version_returns_none_when_uninstalled(monkeypatch: pytest.MonkeyPatch):
    def _missing(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", _missing)
    assert detect_ovrtx_version() is None


def test_detect_ovrtx_version_returns_none_for_unparseable_version(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "internal-build")
    assert detect_ovrtx_version() is None


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        (None, False),
        (Version("0.4.1.364340"), False),
        (Version("0.4.2"), False),
        (Version("0.5"), True),
        (Version("0.5.0.12345"), True),
        (Version("1.0"), True),
    ],
)
def test_uses_prim_path_render_vars_switches_at_ovrtx_05(version: Version | None, expected: bool):
    assert uses_prim_path_render_vars(version) is expected


@pytest.mark.parametrize("version", [None, Version("0.4.1.364340")])
def test_ovrtx_04_frame_keys_are_source_names(version: Version | None):
    keys = build_render_var_frame_keys(version)
    assert keys["LdrColor"] == "LdrColor"
    assert keys["DiffuseAlbedoSD"] == "DiffuseAlbedoSD"
    assert keys["SemanticIdMap"] == "SemanticIdMap"


def test_ovrtx_05_frame_keys_are_render_var_prim_paths():
    keys = build_render_var_frame_keys(Version("0.5"))
    # The prim path is not always the source name with a prefix, so it is read from the authored paths.
    assert keys["LdrColor"] == "/Render/Vars/LdrColor"
    assert keys["DiffuseAlbedoSD"] == "/Render/Vars/albedo"
    assert keys["DistanceToImagePlaneSD"] == "/Render/Vars/depth"
    assert keys["SemanticSegmentation"] == "/Render/Vars/semantic"
    assert keys["SemanticIdMap"] == "/Render/Vars/SemanticIdMap"


def test_frame_keys_cover_every_authored_render_var():
    """Every source the renderer can read has a key, so lookups cannot silently miss an AOV."""
    sources = set(render_var_prim_paths_by_source())
    assert set(build_render_var_frame_keys(Version("0.5"))) == sources
    assert set(build_render_var_frame_keys(None)) == sources


def test_installed_frame_keys_match_the_installed_version():
    """The published mapping is baked from the installed OVRTX version at import."""
    assert dict(RENDER_VAR_FRAME_KEYS) == dict(build_render_var_frame_keys(detect_ovrtx_version()))


def test_published_frame_keys_are_read_only():
    with pytest.raises(TypeError):
        RENDER_VAR_FRAME_KEYS["LdrColor"] = "mutated"  # type: ignore[index]
