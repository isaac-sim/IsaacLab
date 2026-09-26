# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualizer material-writer contracts."""

from isaaclab_newton.physics import NewtonManager
from isaaclab_physx.renderers.visual_material import FabricVisualMaterialWriter
from isaaclab_visualizers.kit.kit_visualizer import KitVisualizer
from isaaclab_visualizers.newton.newton_visualizer import NewtonVisualizer


def test_newton_visualizer_exposes_shared_writer_factory() -> None:
    visualizer = object.__new__(NewtonVisualizer)
    assert visualizer.visual_material_writer == NewtonManager.create_visual_material_writer


def test_kit_visualizer_publishes_fabric_visual_material_writer() -> None:
    visualizer = object.__new__(KitVisualizer)
    assert visualizer.visual_material_writer is FabricVisualMaterialWriter
