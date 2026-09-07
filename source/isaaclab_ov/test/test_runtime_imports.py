# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for optional OvPhysX runtime imports."""

import importlib
import importlib.metadata
from pathlib import Path
from types import ModuleType

import pytest
from isaaclab_ov import _runtime
from isaaclab_ov._runtime import _OVPHYSX_INSTALL_MESSAGE, import_ovphysx


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


def test_required_omniverseclient_version_reads_ovphysx_marker(tmp_path: Path):
    """OvPhysX's full OVStage build marker resolves to the public OmniClient package version."""
    marker_path = tmp_path / "plugins" / "ovstage-omniclient.version"
    marker_path.parent.mkdir()
    marker_path.write_text("2.74.0-release.7316+gl.ec99a64b\n", encoding="utf-8")

    assert _runtime._required_omniverseclient_version(tmp_path) == "2.74.0"


def test_required_omniverseclient_version_rejects_invalid_marker(tmp_path: Path):
    """Malformed compatibility metadata fails before OvPhysX initializes native plugins."""
    marker_path = tmp_path / "plugins" / "ovstage-omniclient.version"
    marker_path.parent.mkdir()
    marker_path.write_text("release-unknown\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Invalid OvPhysX OmniClient compatibility marker"):
        _runtime._required_omniverseclient_version(tmp_path)


def test_import_ovphysx_rejects_mismatched_omniverseclient(monkeypatch: pytest.MonkeyPatch):
    """A mismatched OmniClient release reports the exact version OvPhysX requires."""
    monkeypatch.setattr(importlib, "import_module", lambda module_name: ModuleType(module_name))
    monkeypatch.setattr(_runtime, "_installed_ovphysx_omniverseclient_version", lambda: "2.74.0")
    monkeypatch.setattr(importlib.metadata, "version", lambda distribution_name: "2.72.3")

    with pytest.raises(RuntimeError, match=r"requires omniverseclient==2\.74\.0, but 2\.72\.3 is installed"):
        import_ovphysx()
