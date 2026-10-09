# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for optional OvPhysX runtime imports."""

import importlib
import importlib.metadata
import sys

import pytest
from isaaclab_ov._runtime import _OVPHYSX_INSTALL_MESSAGE, import_ovphysx, physx_schema_paths


def test_import_ovphysx_reports_install_command_when_runtime_missing(monkeypatch):
    """Missing root ``ovphysx`` imports raise the Isaac Lab install hint."""

    def import_module_raises_missing_ovphysx(module_name: str):
        raise ModuleNotFoundError("No module named 'ovphysx'", name="ovphysx")

    monkeypatch.setattr(importlib, "import_module", import_module_raises_missing_ovphysx)

    with pytest.raises(ModuleNotFoundError) as exc_info:
        import_ovphysx("ovphysx.types")

    assert str(exc_info.value) == _OVPHYSX_INSTALL_MESSAGE
    assert "uv run --extra ovphysx" in str(exc_info.value)
    assert "./isaaclab.sh" not in str(exc_info.value)
    assert exc_info.value.name == "ovphysx"
    assert exc_info.value.__cause__.name == "ovphysx"


def test_import_ovphysx_preserves_nested_missing_dependency(monkeypatch):
    """Missing dependencies inside ``ovphysx`` are not rewritten as install hints."""

    def import_module_raises_missing_dependency(module_name: str):
        raise ModuleNotFoundError("No module named 'carb'", name="carb")

    monkeypatch.setattr(importlib, "import_module", import_module_raises_missing_dependency)

    with pytest.raises(ModuleNotFoundError) as exc_info:
        import_ovphysx()

    assert exc_info.value.name == "carb"
    assert "carb" in str(exc_info.value)


def test_physx_schema_discovery_validates_external_resources_without_importing(monkeypatch, tmp_path):
    """Discovery validates both modules and does not run the provider's USD import."""
    metadata_dir = tmp_path / "physx_usd_schemas-25.11.1.dist-info"
    metadata_dir.mkdir()
    (metadata_dir / "METADATA").write_text("Name: physx-usd-schemas\nVersion: 25.11.1\n")
    package_dir = tmp_path / "physx_usd_schemas"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("raise AssertionError('schema discovery must not import the provider')\n")
    expected_paths = [
        package_dir / "PhysxSchema" / "resources",
        package_dir / "OmniUsdPhysicsDeformableSchema" / "resources",
    ]
    for path in expected_paths:
        path.mkdir(parents=True)
        (path / "plugInfo.json").write_text("{}")
        (path / "generatedSchema.usda").write_text("#usda 1.0\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "physx_usd_schemas", raising=False)

    assert physx_schema_paths() == expected_paths
    assert "physx_usd_schemas" not in sys.modules

    missing = expected_paths[1] / "generatedSchema.usda"
    missing.unlink()
    with pytest.raises(FileNotFoundError, match="Incomplete physx-usd-schemas"):
        physx_schema_paths()


def test_physx_schema_discovery_reports_missing_provider(monkeypatch):
    def missing_distribution(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "distribution", missing_distribution)

    with pytest.raises(FileNotFoundError, match="Install the 'ovphysx' extra or physx-usd-schemas"):
        physx_schema_paths()
