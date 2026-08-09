# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the deprecated :mod:`isaaclab_ovphysx` compatibility namespace."""

import importlib
import sys

import pytest


def test_importing_compatibility_namespace_warns(monkeypatch):
    """Importing the old package name emits a deprecation warning."""
    monkeypatch.delitem(sys.modules, "isaaclab_ovphysx", raising=False)

    with pytest.warns(DeprecationWarning, match="import OVPhysX APIs from 'isaaclab_ov'"):
        importlib.import_module("isaaclab_ovphysx")


def test_compatibility_namespace_reexports_canonical_classes():
    """Old package imports resolve to the exact canonical class objects."""
    canonical_package = importlib.import_module("isaaclab_ov.physics")
    deprecated_package = importlib.import_module("isaaclab_ovphysx.physics")
    canonical_module = importlib.import_module("isaaclab_ov.physics.ovphysx_manager_cfg")
    deprecated_module = importlib.import_module("isaaclab_ovphysx.physics.ovphysx_manager_cfg")

    assert deprecated_package.OvPhysxCfg is canonical_package.OvPhysxCfg
    assert deprecated_module.OvPhysxCfg is canonical_module.OvPhysxCfg
