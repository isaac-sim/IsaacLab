# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The deprecated ``isaaclab_physx`` alias keeps resolving to the renamed ``isaaclab_isaacsim`` package."""

import importlib
import sys

import pytest

pytestmark = pytest.mark.kitless


def test_isaaclab_physx_warns_and_resolves_to_the_renamed_modules():
    for name in [m for m in sys.modules if m == "isaaclab_physx" or m.startswith("isaaclab_physx.")]:
        del sys.modules[name]

    with pytest.warns(DeprecationWarning, match="'isaaclab_physx' package is deprecated"):
        importlib.import_module("isaaclab_physx")

    renamed = importlib.import_module("isaaclab_isaacsim.physics")
    assert importlib.import_module("isaaclab_physx.physics") is renamed
    assert renamed.__spec__.name == "isaaclab_isaacsim.physics"  # the alias import keeps the canonical metadata
    from isaaclab_isaacsim.renderers import IsaacRtxRendererCfg as renamed_cfg
    from isaaclab_physx.renderers import IsaacRtxRendererCfg

    assert IsaacRtxRendererCfg is renamed_cfg
