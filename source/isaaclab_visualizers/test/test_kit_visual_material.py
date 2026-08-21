# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Kit visualizer material-writer registration."""

import ast
from pathlib import Path


def test_kit_visualizer_publishes_fabric_visual_material_writer() -> None:
    source = Path(__file__).parents[1] / "isaaclab_visualizers" / "kit" / "kit_visualizer.py"
    tree = ast.parse(source.read_text())
    kit_visualizer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "KitVisualizer")
    writer_hook = next(
        node
        for node in kit_visualizer.body
        if isinstance(node, ast.FunctionDef) and node.name == "visual_material_writer"
    )
    assert any(
        isinstance(node, ast.Return) and ast.unparse(node.value) == "FabricVisualMaterialWriter"
        for node in ast.walk(writer_hook)
    )
