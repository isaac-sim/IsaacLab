# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for optional OvPhysX runtime imports."""

import importlib

import pytest
from isaaclab_ovphysx._runtime import _OVPHYSX_INSTALL_MESSAGE, import_ovphysx


def test_import_ovphysx_reports_install_command_when_runtime_missing(monkeypatch):
    """Missing root ``ovphysx`` imports raise the Isaac Lab install hint."""

    def import_module_raises_missing_ovphysx(module_name: str):
        raise ModuleNotFoundError("No module named 'ovphysx'", name="ovphysx")

    monkeypatch.setattr(importlib, "import_module", import_module_raises_missing_ovphysx)

    with pytest.raises(ModuleNotFoundError) as exc_info:
        import_ovphysx("ovphysx.types")

    assert str(exc_info.value) == _OVPHYSX_INSTALL_MESSAGE
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
