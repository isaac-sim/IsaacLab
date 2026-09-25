# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for OVPhysX 0.5.11 / 0.6 lifecycle and dynamics compatibility."""

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
        requires_legacy_joint_sign_correction,
    )
else:
    OVPHYSX_LIFECYCLE_ENTRY_POINTS = None
    build_lifecycle_entry_points = None
    detect_ovphysx_version = None
    requires_legacy_joint_sign_correction = None

_LEGACY_ENTRY_POINTS = {"warmup": "warmup_gpu", "destroy": "release"}
_CURRENT_ENTRY_POINTS = {"warmup": "warmup", "destroy": "destroy"}


def _missing_distribution(name: str) -> str:
    raise importlib.metadata.PackageNotFoundError(name)


@pytest.mark.parametrize(
    ("metadata_version", "expected"),
    [
        (lambda name: "0.5.11", Version("0.5.11")),
        (_missing_distribution, None),
        (lambda name: "internal-build", None),
    ],
    ids=["installed", "uninstalled", "unparsable"],
)
def test_detect_ovphysx_version_reads_distribution_metadata(
    monkeypatch: pytest.MonkeyPatch, metadata_version, expected: Version | None
):
    monkeypatch.setattr(importlib.metadata, "version", metadata_version)
    assert detect_ovphysx_version() == expected


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


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        (None, True),
        (Version("0.5.11"), True),
        (Version("0.6.0.dev1+trunk.e15a64a2"), False),
        (Version("0.6"), False),
        (Version("1.0"), False),
    ],
)
def test_reversed_joint_sign_correction_version_boundary(version: Version | None, expected: bool):
    assert requires_legacy_joint_sign_correction(version) is expected


@pytest.mark.parametrize("version", [None, Version("0.5.11"), Version("0.6.2"), Version("0.6.3"), Version("0.7.0")])
def test_clone_dispatch_uses_supported_signature(monkeypatch, version):
    """Legacy homogeneous clones keep working without the new env_ids keyword."""
    from isaaclab_ov.physics import ovphysx_compat

    calls = []

    class LegacyPhysX:
        def clone(self, source, targets, transforms):
            calls.append((source, targets, transforms))
            return 7

    class CurrentPhysX:
        def clone(self, source, targets, transforms, *, env_ids):
            calls.append((source, targets, transforms, env_ids))
            return 7

    monkeypatch.setattr(ovphysx_compat, "OVPHYSX_VERSION", version)
    current = version is not None and version >= Version("0.6.3")
    physx = CurrentPhysX() if current else LegacyPhysX()
    assert ovphysx_compat.clone_physics(physx, "/env0", ["/env1"], None, [1]) == 7
    expected = ("/env0", ["/env1"], None, [1]) if current else ("/env0", ["/env1"], None)
    assert calls == [expected]
