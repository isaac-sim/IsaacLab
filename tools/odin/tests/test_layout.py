# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Guards the OdinV2 package layout against regressions to the old tree."""

import importlib
from pathlib import Path

import pytest

_ODIN_ROOT = Path(__file__).resolve().parents[1]

# Sub-packages that belonged to the fleet-and-OSMO harness OdinV2 replaced.
# Their reappearance means the fleet stack was resurrected by mistake; it lives
# on the antoiner/feat/odin branch and is deliberately absent here.
_REMOVED_SUBPACKAGES = ("asgard", "valhalla", "hugin", "munin", "common", "bifrost", "scripts")


@pytest.mark.parametrize("name", _REMOVED_SUBPACKAGES)
def test_removed_subpackage_is_absent(name: str) -> None:
    assert not (_ODIN_ROOT / name).exists(), f"tools/odin/{name} should not exist in OdinV2"


def test_package_imports() -> None:
    assert importlib.import_module("tools.odin") is not None
