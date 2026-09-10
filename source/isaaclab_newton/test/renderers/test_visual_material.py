# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Newton's compiled visual-material writers."""

import torch
import warp as wp
from isaaclab_newton.renderers.newton_warp_renderer import NewtonWarpRenderer
from isaaclab_newton.renderers.visual_material import (
    VisualMaterialWriter,
    VisualShapeColorWriter,
    import_builder_visual_material_paths,
)
from newton import ModelBuilder
from newton.selection import ArticulationView

from pxr import Sdf, Usd, UsdGeom, UsdShade

from isaaclab.renderers.base_renderer import VisualMaterialBatch


def _srgb(colors: torch.Tensor) -> torch.Tensor:
    return torch.where(colors <= 0.0031308, 12.92 * colors, 1.055 * colors.pow(1.0 / 2.4) - 0.055)


def _material_model(material_paths: list[str]):
    stage = Usd.Stage.CreateInMemory()
    builder = ModelBuilder()
    for index, material_path in enumerate(material_paths):
        shape = UsdGeom.Cube.Define(stage, f"/World/shape_{index}")
        builder.add_shape_box(-1, label=str(shape.GetPath()))
        if material_path:
            material = UsdShade.Material.Define(stage, material_path)
            UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(material)
    import_builder_visual_material_paths(builder, stage)
    return builder.finalize(device="cpu")


def test_import_captures_effective_material_binding() -> None:
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/World/Robot")
    shape = UsdGeom.Cube.Define(stage, "/World/Robot/mesh")
    material = UsdShade.Material.Define(stage, "/World/Looks/robot")
    UsdShade.MaterialBindingAPI.Apply(root.GetPrim()).Bind(material)
    builder = ModelBuilder()
    builder.add_shape_box(-1, label=str(shape.GetPath()))

    colors = tuple(tuple(color) for color in builder.shape_color)
    import_builder_visual_material_paths(builder, stage)

    assert builder.finalize(device="cpu").isaaclab.visual_material_path == ["/World/Looks/robot"]
    assert tuple(tuple(color) for color in builder.shape_color) == colors


def test_import_preserves_binding_to_a_logical_clone_not_yet_on_stage() -> None:
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/World/envs/env_1/Robot")
    shape = UsdGeom.Cube.Define(stage, "/World/envs/env_1/Robot/mesh")
    relationship = UsdShade.MaterialBindingAPI.Apply(root.GetPrim()).GetDirectBindingRel()
    relationship.SetTargets([Sdf.Path("/World/envs/env_1/Robot/material")])
    builder = ModelBuilder()
    builder.add_shape_box(-1, label=str(shape.GetPath()))

    import_builder_visual_material_paths(builder, stage)

    assert builder.finalize(device="cpu").isaaclab.visual_material_path == ["/World/envs/env_1/Robot/material"]


def test_material_writer_scatters_only_dirty_material_rows(monkeypatch) -> None:
    model = _material_model(["/Looks/a", "/Looks/b", "/Looks/a", ""])
    values = torch.tensor([[0.1, 0.2, 0.3], [0.8, 0.6, 0.4]])
    batch = VisualMaterialBatch(
        "color", ("/Looks/a", "/Looks/b"), ("/Looks/a/Shader", "/Looks/b/Shader"), ("color", "color"), values
    )
    writer = VisualMaterialWriter(model, (batch,))
    before = wp.to_torch(model.shape_color).clone()
    launch = wp.launch
    launches = 0

    def count_launch(*args, **kwargs):
        nonlocal launches
        launches += 1
        return launch(*args, **kwargs)

    monkeypatch.setattr(wp, "launch", count_launch)
    writer(
        {"color": wp.array([1], dtype=wp.int32, device="cpu")},
        wp.array([0], dtype=wp.int32, device="cpu"),
    )
    wp.synchronize()

    assert launches == 1
    expected = _srgb(values)
    actual = wp.to_torch(model.shape_color)
    torch.testing.assert_close(actual[[0, 2]], before[[0, 2]])
    torch.testing.assert_close(actual[1], expected[1])


def _articulation_model(num_envs: int = 3):
    scene = ModelBuilder()
    paths = []
    for env_id in range(num_envs):
        path = f"/World/envs/env_{env_id}/Robot"
        robot = ModelBuilder()
        base = robot.add_link(label=f"{path}/base")
        link = robot.add_link(label=f"{path}/link")
        root_joint = robot.add_joint_free(base, label=f"{path}/root_joint")
        child_joint = robot.add_joint_fixed(base, link, label=f"{path}/fixed_joint")
        robot.add_articulation([root_joint, child_joint], label=path)
        robot.add_shape_box(base, label=f"{path}/base/visual_a")
        robot.add_shape_box(base, label=f"{path}/base/visual_b")
        robot.add_shape_box(link, label=f"{path}/link/visual")
        scene.add_world(robot)
        paths.append(path)
    return scene.finalize(device="cpu"), tuple(paths)


def test_shape_writer_samples_each_body_and_selected_environment_independently() -> None:
    model, paths = _articulation_model()
    original = wp.to_torch(model.shape_color).clone()
    writer = VisualShapeColorWriter(model, ArticulationView(model, list(paths), verbose=False), ("base", "link"))
    colors = torch.tensor(
        [
            [[0.1, 0.2, 0.3], [0.3, 0.4, 0.5]],
            [[0.5, 0.6, 0.7], [0.7, 0.8, 0.9]],
        ]
    )

    writer(colors, torch.tensor([0, 2], dtype=torch.int32))
    wp.synchronize()

    actual = wp.to_torch(model.shape_color).reshape(3, 3, 3)
    expected = _srgb(colors)
    torch.testing.assert_close(actual[0, :2], expected[0, 0].expand(2, 3))
    torch.testing.assert_close(actual[0, 2], expected[0, 1])
    torch.testing.assert_close(actual[1], original.reshape(3, 3, 3)[1])
    torch.testing.assert_close(actual[2, :2], expected[1, 0].expand(2, 3))
    torch.testing.assert_close(actual[2, 2], expected[1, 1])


def test_newton_renderer_exposes_shared_writer_factory() -> None:
    renderer = object.__new__(NewtonWarpRenderer)
    assert renderer.visual_material_writer.__func__.__name__ == "create_visual_material_writer"
