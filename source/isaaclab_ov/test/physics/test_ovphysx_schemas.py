# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for external PhysX schema resource discovery without importing USD plugins."""

import importlib.metadata

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


def _missing_distribution(name: str) -> str:
    raise importlib.metadata.PackageNotFoundError(name)


def test_external_schema_resources_are_checked_without_importing_provider(monkeypatch, tmp_path):
    from isaaclab_ov.physics.schemas import physx_schema_paths

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
    assert physx_schema_paths() == paths
    (paths[1] / "generatedSchema.usda").unlink()
    with pytest.raises(FileNotFoundError, match="Incomplete physx-usd-schemas"):
        physx_schema_paths()
    monkeypatch.setattr(importlib.metadata, "distribution", _missing_distribution)
    with pytest.raises(importlib.metadata.PackageNotFoundError):
        physx_schema_paths()
