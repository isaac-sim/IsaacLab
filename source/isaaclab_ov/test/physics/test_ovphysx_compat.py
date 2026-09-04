# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for OVPhysX 0.5.11 / 0.6 lifecycle compatibility."""

from __future__ import annotations

import importlib.metadata
import importlib.util

import pytest
from packaging.version import Version

_REQUIRED_MODULES = ("isaaclab_ov",)
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]

pytestmark = [
    pytest.mark.isaacsim_ci,
    pytest.mark.skipif(
        bool(_MISSING_MODULES),
        reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}",
    ),
]

if not _MISSING_MODULES:
    from isaaclab_ov.physics.ovphysx_compat import (  # noqa: E402
        OVPHYSX_LIFECYCLE_ENTRY_POINTS,
        build_lifecycle_entry_points,
        detect_ovphysx_version,
    )
else:
    OVPHYSX_LIFECYCLE_ENTRY_POINTS = None
    build_lifecycle_entry_points = None
    detect_ovphysx_version = None

_LEGACY_ENTRY_POINTS = {"warmup": "warmup_gpu", "destroy": "release"}
_CURRENT_ENTRY_POINTS = {"warmup": "warmup", "destroy": "destroy"}


def test_detect_ovphysx_version_reads_distribution_metadata(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.5.11")
    assert detect_ovphysx_version() == Version("0.5.11")


def test_detect_ovphysx_version_returns_none_when_uninstalled(monkeypatch: pytest.MonkeyPatch):
    def _missing(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", _missing)
    assert detect_ovphysx_version() is None


def test_detect_ovphysx_version_returns_none_for_unparseable_version(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "internal-build")
    assert detect_ovphysx_version() is None


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        (None, _LEGACY_ENTRY_POINTS),
        (Version("0.5.11"), _LEGACY_ENTRY_POINTS),
        (Version("0.6.0.dev1+trunk.e15a64a2"), _CURRENT_ENTRY_POINTS),
        (Version("0.6"), _CURRENT_ENTRY_POINTS),
        (Version("1.0"), _CURRENT_ENTRY_POINTS),
    ],
)
def test_lifecycle_entry_points(version: Version | None, expected: dict[str, str]):
    entry_points = build_lifecycle_entry_points(version)
    assert dict(entry_points) == expected


def test_published_entry_points_are_read_only():
    with pytest.raises(TypeError):
        OVPHYSX_LIFECYCLE_ENTRY_POINTS["warmup"] = "mutated"  # type: ignore[index]
