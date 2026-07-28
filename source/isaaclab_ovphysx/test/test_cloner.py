# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for OvPhysX cloning."""

import math

import pytest
import torch
from isaaclab_ovphysx.cloner import OvPhysxReplicateContext

from pxr import Gf, Usd, UsdGeom


def _pose_matrix(position: tuple[float, float, float], quaternion: tuple[float, float, float, float]) -> Gf.Matrix4d:
    """Build a USD pose matrix from an xyzw quaternion."""
    matrix = Gf.Matrix4d(1.0)
    matrix.SetTranslateOnly(Gf.Vec3d(*position))
    matrix.SetRotateOnly(Gf.Quatd(quaternion[3], Gf.Vec3d(*quaternion[:3])))
    return matrix


def test_nested_clone_uses_final_target_pose():
    """Nested clone rows keep their source-local pose under the target environment."""
    half_sqrt_two = math.sqrt(0.5)
    stage = Usd.Stage.CreateInMemory()

    source_env = UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    source_env.AddTransformOp().Set(_pose_matrix((4.0, 5.0, 6.0), (0.0, half_sqrt_two, 0.0, half_sqrt_two)))
    source_row = UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot")
    source_row.AddTransformOp().Set(_pose_matrix((0.0, 1.0, 2.0), (half_sqrt_two, 0.0, 0.0, half_sqrt_two)))

    context = OvPhysxReplicateContext(stage)
    context.queue_mapping(
        sources=["/World/envs/env_0/Robot", "/World/envs/env_9/Inactive"],
        destinations=["/World/envs/env_{}/Robot", "/World/envs/env_{}/Inactive"],
        env_ids=torch.tensor([0, 1]),
        mapping=torch.tensor([[True, True], [False, False]]),
        positions=torch.tensor([[4.0, 5.0, 6.0], [10.0, 20.0, 30.0]]),
        quaternions=torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, half_sqrt_two, half_sqrt_two]]),
    )

    assert len(context._queue) == 1
    source, targets, target_transforms = context._queue[0]
    assert source == "/World/envs/env_0/Robot"
    assert targets == ["/World/envs/env_1/Robot"]
    assert len(target_transforms) == 1
    assert target_transforms[0][:3] == pytest.approx((9.0, 20.0, 32.0))
    orientation = torch.tensor(target_transforms[0][3:])
    expected_orientation = torch.full((4,), 0.5)
    assert torch.abs(torch.dot(orientation, expected_orientation)).item() == pytest.approx(1.0)
