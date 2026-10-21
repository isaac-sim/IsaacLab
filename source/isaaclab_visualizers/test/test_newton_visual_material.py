# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualizer material-writer contracts."""

from isaaclab_physx.renderers.visual_material import FabricVisualMaterialWriter
from isaaclab_visualizers.kit.kit_visualizer import KitVisualizer


def test_kit_visualizer_publishes_fabric_visual_material_writer() -> None:
    visualizer = object.__new__(KitVisualizer)
    assert visualizer.visual_material_writer is FabricVisualMaterialWriter
