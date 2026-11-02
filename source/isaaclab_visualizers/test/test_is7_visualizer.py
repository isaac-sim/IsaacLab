# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Isaac Sim 7 viewer visualizer registration and scene export."""

from __future__ import annotations

import pytest
from isaaclab_visualizers.is7 import IS7BackendCfg
from isaaclab_visualizers.is7.is7_visualizer import IS7Visualizer

from pxr import Usd

from isaaclab.scene_data.scene_data_provider import REQUIRES_STAGE_AND_MODEL
from isaaclab.visualizers.visualizer_cfg import VISUALIZER_TYPES, VisualizerCfg

_SCENE = """#usda 1.0
def "World"
{
    def Xform "Proto" (instanceable = true)
    {
        def Mesh "mesh"
        {
            point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        }
    }
    def "env_0" (
        instanceable = true
        prepend references = </World/Proto>
    )
    {
    }
    def "env_1" (
        instanceable = true
        prepend references = </World/Proto>
    )
    {
    }
}
"""


def test_is7_is_registered():
    assert "is7" in VISUALIZER_TYPES
    assert REQUIRES_STAGE_AND_MODEL["is7"] == (True, False)
    assert IS7BackendCfg().visualizer_type == "is7"


@pytest.mark.parametrize(("max_points", "expanded"), [(9, True), (8, False)])
def test_export_types_containers_and_expands_instances_within_budget(max_points, expanded):
    """The three instanced meshes count 9 points when expanded, so a budget of 8 keeps the instancing."""
    visualizer = IS7Visualizer(VisualizerCfg(backend=IS7BackendCfg(deinstance_max_points=max_points)))
    stage = Usd.Stage.CreateInMemory()
    stage.GetRootLayer().ImportFromString(_SCENE)
    try:
        exported = Usd.Stage.Open(str(visualizer._export_scene(stage)))
        assert exported.GetPrimAtPath("/World").GetTypeName() == "Xform"
        assert exported.GetPrimAtPath("/World/env_0").GetTypeName() == "Xform"
        assert exported.GetPrimAtPath("/World/env_1").IsInstance() is not expanded
    finally:
        visualizer._resources.close()
