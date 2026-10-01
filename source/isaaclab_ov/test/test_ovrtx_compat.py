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
        detect_ovrtx_version,
        uses_prim_path_render_vars,
    )
else:
    detect_ovrtx_version = None
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
