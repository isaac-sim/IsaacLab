# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for ancestor authoring in :func:`~isaaclab.cloner.usd_replicate`."""

import pytest
import torch

from pxr import Gf, Sdf, Usd, UsdGeom

from isaaclab.cloner import UsdReplicateContext, usd_replicate


def _make_stage_with_source(source_path: str) -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    for prefix in Sdf.Path(source_path).GetPrefixes():
        stage.DefinePrim(prefix, "Xform")
    return stage


def test_usd_replicate_defines_nested_destination_ancestors():
    """Copied prims under a nested scope compose as defined prims in target envs."""
    stage = _make_stage_with_source("/World/envs/env_0/Groceries/Object")
    stage.DefinePrim("/World/envs/env_1", "Xform")

    usd_replicate(
        stage,
        sources=["/World/envs/env_0/Groceries/Object"],
        destinations=["/World/envs/env_{}/Groceries/Object"],
        env_ids=torch.tensor([0, 1]),
    )

    copied_scope = stage.GetPrimAtPath("/World/envs/env_1/Groceries")
    copied_prim = stage.GetPrimAtPath("/World/envs/env_1/Groceries/Object")
    assert copied_scope.IsDefined(), "intermediate ancestor must compose as a defined prim"
    assert copied_prim.IsDefined(), "copied prim must compose as a defined prim"


def test_usd_replicate_keeps_existing_ancestor_specs():
    """Ancestors already defined in the target env are left untouched."""
    stage = _make_stage_with_source("/World/envs/env_0/Groceries/Object")
    for prefix in Sdf.Path("/World/envs/env_1/Groceries").GetPrefixes():
        stage.DefinePrim(prefix, "Xform")

    usd_replicate(
        stage,
        sources=["/World/envs/env_0/Groceries/Object"],
        destinations=["/World/envs/env_{}/Groceries/Object"],
        env_ids=torch.tensor([0, 1]),
    )

    scope = stage.GetPrimAtPath("/World/envs/env_1/Groceries")
    assert scope.IsDefined()
    assert scope.GetTypeName() == "Xform"
    assert stage.GetPrimAtPath("/World/envs/env_1/Groceries/Object").IsDefined()


def test_usd_replicate_context_drains_after_model_init_callback(monkeypatch):
    """A deferred context applies its queue once even when MODEL_INIT repeats."""
    stage = _make_stage_with_source("/World/envs/env_0/Robot")
    context = UsdReplicateContext(stage)
    context.queue(
        "/World/envs/env_0/Robot",
        "/World/envs/env_{}/Robot",
        torch.tensor([1]),
    )

    copied_paths = []
    copy_spec = Sdf.CopySpec

    def capture_copy(source_layer, source_path, destination_layer, destination_path):
        copied_paths.append(str(destination_path))
        return copy_spec(source_layer, source_path, destination_layer, destination_path)

    monkeypatch.setattr(Sdf, "CopySpec", capture_copy)
    context.replicate(None)
    context.replicate(None)

    assert copied_paths == ["/World/envs/env_1/Robot"]


def test_leaf_destination_positions_missing_instance_root_before_copy(monkeypatch):
    """A leaf-only row positions and orients its missing instance root before copying the child."""
    stage = _make_stage_with_source("/World/envs/env_0/Camera")
    camera_offset = Gf.Vec3d(0.57, -0.8, 0.5)
    UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Camera")).AddTranslateOp().Set(camera_offset)
    positions = torch.tensor([[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0]])
    quaternions = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.70710678, 0.70710678]])
    root_before_child_copy = []
    copy_spec = Sdf.CopySpec

    def capture_copy(source_layer, source_path, destination_layer, destination_path):
        if str(destination_path) == "/World/envs/env_1/Camera":
            root_path = "/World/envs/env_1"
            root = destination_layer.GetPrimAtPath(root_path)
            translation = root.GetAttributeAtPath(root_path + ".xformOp:translate").default
            orientation = root.GetAttributeAtPath(root_path + ".xformOp:orient").default
            root_before_child_copy.append((root.specifier, root.typeName, translation, orientation))
        return copy_spec(source_layer, source_path, destination_layer, destination_path)

    monkeypatch.setattr(Sdf, "CopySpec", capture_copy)
    usd_replicate(
        stage,
        sources=["/World/envs/env_0/Camera"],
        destinations=["/World/envs/env_{}/Camera"],
        env_ids=torch.arange(2),
        positions=positions,
        quaternions=quaternions,
    )

    assert len(root_before_child_copy) == 1
    specifier, type_name, translation, orientation = root_before_child_copy[0]
    assert specifier == Sdf.SpecifierDef and type_name == "Xform"
    assert tuple(translation) == pytest.approx(positions[1].tolist())
    assert orientation.GetReal() == pytest.approx(quaternions[1, 3].item())
    assert tuple(orientation.GetImaginary()) == pytest.approx(quaternions[1, :3].tolist())

    root = stage.GetPrimAtPath("/World/envs/env_1")
    assert root.IsDefined() and root.GetTypeName() == "Xform"
    camera = stage.GetPrimAtPath("/World/envs/env_1/Camera")
    assert camera.IsDefined()
    assert tuple(camera.GetAttribute("xformOp:translate").Get()) == pytest.approx(tuple(camera_offset))


def test_sparse_environment_ids_use_column_aligned_poses():
    """Sparse environment ids use their clone-plan columns to select root poses."""
    stage = _make_stage_with_source("/World/scenes/scene_2/Camera")
    positions = torch.tensor([[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0], [7.0, -8.0, 9.0]])
    quaternions = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.70710678, 0.70710678], [0.0, 0.70710678, 0.0, 0.70710678]]
    )

    usd_replicate(
        stage,
        sources=["/World/scenes/scene_2/Camera"],
        destinations=["/World/scenes/scene_{}/Camera"],
        env_ids=torch.tensor([2, 7, 11]),
        mask=torch.tensor([[True, False, True]]),
        positions=positions,
        quaternions=quaternions,
    )

    assert not stage.GetPrimAtPath("/World/scenes/scene_7").IsValid()
    root = stage.GetPrimAtPath("/World/scenes/scene_11")
    assert tuple(root.GetAttribute("xformOp:translate").Get()) == pytest.approx(positions[2].tolist())
    orientation = root.GetAttribute("xformOp:orient").Get()
    assert orientation.GetReal() == pytest.approx(quaternions[2, 3].item())
    assert tuple(orientation.GetImaginary()) == pytest.approx(quaternions[2, :3].tolist())
