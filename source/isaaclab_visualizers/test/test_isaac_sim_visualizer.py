# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Isaac Sim 7 viewer visualizer registration and scene export."""

from __future__ import annotations

from types import SimpleNamespace

from isaaclab_visualizers.isaac_sim import IsaacSimBackendCfg
from isaaclab_visualizers.isaac_sim.isaac_sim_visualizer import IsaacSimVisualizer

from pxr import Usd

from isaaclab.scene_data.scene_data_provider import REQUIRES_STAGE_AND_MODEL
from isaaclab.visualizers.visualizer_cfg import VISUALIZER_TYPES, VisualizerCfg

_SCENE = """#usda 1.0
def "World"
{
    def "envs"
    {
        def Xform "env_0"
        {
            def Xform "Robot"
            {
                def Mesh "mesh"
                {
                    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
                }
            }
        }
        def Xform "env_1"
        {
            double3 xformOp:translate = (2, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
            def Xform "Robot"
            {
                def Mesh "mesh"
                {
                    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
                }
            }
        }
    }
}
"""


def _export(scene: str):
    """Export a scene the way the visualizer populates the shared stage, without a clone plan."""
    stage = Usd.Stage.CreateInMemory()
    stage.GetRootLayer().ImportFromString(scene)
    visualizer = IsaacSimVisualizer(VisualizerCfg(backend=IsaacSimBackendCfg()))
    try:
        path, replication = visualizer._export_prototypes(SimpleNamespace(stage=stage, get_clone_plan=lambda: None))
        return Usd.Stage.Open(str(path)), replication
    finally:
        visualizer._resources.close()


def test_isaac_sim_is_registered():
    assert "isaac_sim" in VISUALIZER_TYPES
    assert REQUIRES_STAGE_AND_MODEL["isaac_sim"] == (True, False)
    assert IsaacSimBackendCfg().visualizer_type == "isaac_sim"


def test_export_keeps_one_copy_and_clones_the_other_environments():
    exported, replication = _export(_SCENE)
    assert replication.copies == [("/World/envs/env_0/Robot", ["/World/envs/env_1/Robot"])]
    assert exported.GetPrimAtPath("/World/envs/env_0/Robot/mesh")
    # The other environment keeps its root and placement; its contents come from the clone.
    assert exported.GetPrimAtPath("/World/envs/env_1")
    copy = exported.GetPrimAtPath("/World/envs/env_1/Robot")
    assert not copy or not copy.IsActive()
    # The viewer's stage tree omits typeless containers, so they are exported as transforms.
    assert exported.GetPrimAtPath("/World").GetTypeName() == "Xform"


def test_export_without_environments_is_whole():
    exported, replication = _export('#usda 1.0\ndef "World"\n{\n    def Cube "box"\n    {\n    }\n}\n')
    assert replication is None
    assert exported.GetPrimAtPath("/World/box")
