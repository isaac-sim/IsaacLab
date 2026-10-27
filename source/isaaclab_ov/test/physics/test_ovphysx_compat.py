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


@pytest.mark.parametrize(
    ("version", "expected"),
    [(None, False), (Version("0.6.4"), False), (Version("0.7.0.dev1"), True), (Version("1.0"), True)],
)
def test_task_api_version_boundary(version: Version | None, expected: bool):
    from isaaclab_ov.physics.ovphysx_compat import uses_task_api

    assert uses_task_api(version) is expected


def test_external_schema_resources_are_checked_without_importing_provider(monkeypatch, tmp_path):
    from isaaclab_ov.physics import ovphysx_compat as compat

    monkeypatch.setattr(compat, "OVPHYSX_USES_TASKS", True)
    metadata_dir = tmp_path / "physx_usd_schemas-25.11.2.dist-info"
    metadata_dir.mkdir()
    (metadata_dir / "METADATA").write_text("Name: physx-usd-schemas\nVersion: 25.11.2\n")
    root = tmp_path / "physx_usd_schemas"
    root.mkdir()
    (root / "__init__.py").write_text("raise AssertionError('must not import USD provider')\n")
    paths = [root / name / "resources" for name in ("PhysxSchema", "OmniUsdPhysicsDeformableSchema")]
    for path in paths:
        path.mkdir(parents=True)
        (path / "plugInfo.json").write_text("{}")
        (path / "generatedSchema.usda").write_text("#usda 1.0\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert compat.physx_schema_paths(None) == paths
    (paths[1] / "generatedSchema.usda").unlink()
    with pytest.raises(FileNotFoundError, match="Incomplete physx-usd-schemas"):
        compat.physx_schema_paths(None)
    monkeypatch.setattr(importlib.metadata, "distribution", _missing_distribution)
    with pytest.raises(importlib.metadata.PackageNotFoundError):
        compat.physx_schema_paths(None)
