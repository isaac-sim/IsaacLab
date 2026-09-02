# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for OvPhysX cloning."""

import math
from types import SimpleNamespace

import pytest
import torch
from isaaclab_ov.cloner import OvPhysxReplicateContext
from isaaclab_ov.physics.ovphysx_manager import OvPhysxManager

from pxr import Gf, Usd, UsdGeom

from isaaclab.cloner import ClonePlan


def _context(stage) -> OvPhysxReplicateContext:
    """Build a context with the manager surface used by cloning."""
    return OvPhysxReplicateContext(SimpleNamespace(stage=stage, physics_manager=OvPhysxManager))


def _plan(sources, destinations, env_ids, mapping, positions=None) -> ClonePlan:
    """Build one direct context input for backend-focused tests."""
    if positions is None:
        positions = torch.zeros((len(env_ids), 3))
    return ClonePlan(
        sources=tuple(sources),
        destinations=tuple(destinations),
        clone_mask=mapping,
        env_ids=env_ids,
        positions=positions,
        context_rows={OvPhysxReplicateContext: tuple(range(len(sources)))},
    )


def test_context_is_simulation_owned():
    stage = Usd.Stage.CreateInMemory()
    context = _context(stage)

    assert context._sim.stage is stage


def _pose_matrix(position: tuple[float, float, float], quaternion: tuple[float, float, float, float]) -> Gf.Matrix4d:
    """Build a USD pose matrix from an xyzw quaternion."""
    matrix = Gf.Matrix4d(1.0)
    matrix.SetTranslateOnly(Gf.Vec3d(*position))
    matrix.SetRotateOnly(Gf.Quatd(quaternion[3], Gf.Vec3d(*quaternion[:3])))
    return matrix


def test_nested_clone_uses_final_target_pose(monkeypatch):
    """Nested clone rows keep their source-local pose under the target environment."""
    monkeypatch.setattr(OvPhysxManager, "_active_clone_recipes", [])
    monkeypatch.setattr(OvPhysxManager, "_pending_clones", [])
    half_sqrt_two = math.sqrt(0.5)
    source_half_angle_sin = 0.5
    source_half_angle_cos = math.sqrt(0.75)
    stage = Usd.Stage.CreateInMemory()

    source_env = UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    source_env.AddTransformOp().Set(_pose_matrix((4.0, 5.0, 6.0), (0.0, half_sqrt_two, 0.0, half_sqrt_two)))
    source_row = UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot")
    source_row.AddTransformOp().Set(
        _pose_matrix((0.0, 1.0, 2.0), (source_half_angle_sin, 0.0, 0.0, source_half_angle_cos))
    )

    context = _context(stage)
    context.replicate(
        _plan(
            ["/World/envs/env_0/Robot", "/World/envs/env_9/Inactive"],
            ["/World/envs/env_{}/Robot", "/World/envs/env_{}/Inactive"],
            torch.tensor([0, 1]),
            torch.tensor([[True, True], [False, False]]),
            positions=torch.tensor([[4.0, 5.0, 6.0], [10.0, 20.0, 30.0]]),
        )
    )

    expected_orientation = torch.tensor([source_half_angle_sin, 0.0, 0.0, source_half_angle_cos])
    expected_transform = (10.0, 21.0, 32.0, *expected_orientation.tolist())

    assert len(OvPhysxManager._pending_clones) == 1
    pending_source, pending_targets, pending_transforms = OvPhysxManager._pending_clones[0]
    assert pending_source == "/World/envs/env_0/Robot"
    assert pending_targets == ["/World/envs/env_1/Robot"]
    assert len(pending_transforms) == 1
    assert pending_transforms[0][:3] == pytest.approx(expected_transform[:3])
    orientation = torch.tensor(pending_transforms[0][3:])
    if torch.dot(orientation, expected_orientation) < 0.0:
        orientation = -orientation
    assert orientation.tolist() == pytest.approx(expected_orientation.tolist())


def test_register_clone_preserves_translation_only_compatibility(monkeypatch):
    """World positions become target-root poses with identity rotations."""
    monkeypatch.setattr(OvPhysxManager, "_active_clone_recipes", [])
    monkeypatch.setattr(OvPhysxManager, "_pending_clones", [])

    OvPhysxManager.register_clone("/World/env_0", ["/World/env_1"], [(1.0, 2.0, 3.0)])

    expected_recipes = [("/World/env_0", ["/World/env_1"], [(1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0)])]
    assert OvPhysxManager._active_clone_recipes == expected_recipes
    assert OvPhysxManager._pending_clones == expected_recipes


def test_replicate_rejects_invalid_source_prim():
    """Active clone rows require a valid source prim."""
    stage = Usd.Stage.CreateInMemory()
    context = _context(stage)

    with pytest.raises(ValueError, match="/World/envs/env_0"):
        context.replicate(
            _plan(
                ["/World/envs/env_0/Robot"],
                ["/World/envs/env_{}/Robot"],
                torch.tensor([0, 1]),
                torch.tensor([[True, True]]),
            )
        )


def test_replicate_rejects_invalid_source_anchor():
    """Active nested clone rows require a valid source-environment anchor."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot")

    class StageWithoutAnchor:
        def GetPrimAtPath(self, path):
            if str(path) == "/World/envs/env_0":
                return Usd.Prim()
            return stage.GetPrimAtPath(path)

    context = _context(StageWithoutAnchor())
    with pytest.raises(ValueError, match="/World/envs/env_0"):
        context.replicate(
            _plan(
                ["/World/envs/env_0/Robot"],
                ["/World/envs/env_{}/Robot"],
                torch.tensor([0, 1]),
                torch.tensor([[True, True]]),
            )
        )


def test_replicate_rejects_pose_tensor_missing_selected_environment():
    """Provided pose tensors include every selected environment."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot")
    context = _context(stage)

    with pytest.raises(ValueError, match="positions"):
        context.replicate(
            _plan(
                ["/World/envs/env_0/Robot"],
                ["/World/envs/env_{}/Robot"],
                torch.tensor([0, 1]),
                torch.tensor([[True, True]]),
                positions=torch.zeros((1, 3)),
            )
        )


def test_replicate_rejects_malformed_pose_tensor():
    """Provided pose tensors use the documented component counts."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot")
    context = _context(stage)

    with pytest.raises(ValueError, match="positions must have shape"):
        context.replicate(
            _plan(
                ["/World/envs/env_0/Robot"],
                ["/World/envs/env_{}/Robot"],
                torch.tensor([0, 1]),
                torch.tensor([[True, True]]),
                positions=torch.zeros((2, 2)),
            )
        )
