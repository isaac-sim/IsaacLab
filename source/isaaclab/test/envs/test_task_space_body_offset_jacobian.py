# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU-only tests for the body-offset Jacobian of the task-space action terms."""

import math
from types import SimpleNamespace

import pytest
import torch

from isaaclab.envs.mdp.actions.task_space_actions import (
    DifferentialInverseKinematicsAction,
    OperationalSpaceControllerAction,
)

pytestmark = pytest.mark.unit


def _stub(offset_pos, offset_rot):
    """One revolute joint about the root z axis, with the body rotated 90 degrees about z.

    The body-origin Jacobian is ``v = 0`` and ``w = z``. A point ``r`` fixed on the body moves with
    ``w x (R_body_b @ r)`` and turns with the body's angular velocity.
    """
    body_quat_w = torch.tensor([[[0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)]]])
    jacobian_b = torch.tensor([[[0.0], [0.0], [0.0], [0.0], [0.0], [1.0]]])
    return SimpleNamespace(
        cfg=SimpleNamespace(body_offset=SimpleNamespace(pos=offset_pos, rot=offset_rot)),
        _offset_pos=torch.tensor([offset_pos]),
        _offset_rot=torch.tensor([offset_rot]),
        _jacobian_b=torch.zeros(1, 6, 1),
        jacobian_b=jacobian_b,
        _body_idx=0,
        _ee_body_idx=0,
        _asset=SimpleNamespace(
            data=SimpleNamespace(
                root_quat_w=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, 0.0, 1.0]])),
                body_quat_w=SimpleNamespace(torch=body_quat_w),
            )
        ),
    )


@pytest.mark.parametrize(
    "compute",
    [
        DifferentialInverseKinematicsAction._compute_frame_jacobian,
        OperationalSpaceControllerAction._compute_ee_jacobian,
    ],
    ids=["diff_ik", "osc"],
)
def test_body_offset_jacobian_uses_offset_in_root_frame(compute):
    """The offset is rotated by the body orientation, and the offset rotation leaves angular rows unchanged."""
    # A 90 degree rotation about x for the offset frame must not change the angular rows.
    stub = _stub(offset_pos=[1.0, 0.0, 0.0], offset_rot=[math.sin(math.pi / 4.0), 0.0, 0.0, math.cos(math.pi / 4.0)])
    compute(stub)

    # R_body_b @ (1, 0, 0) = (0, 1, 0), and z x (0, 1, 0) = (-1, 0, 0).
    expected = torch.tensor([[[-1.0], [0.0], [0.0], [0.0], [0.0], [1.0]]])
    torch.testing.assert_close(stub._jacobian_b, expected)
