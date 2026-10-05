# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kinematics of the target frame shared by the Newton task-space action terms."""

import ast
import inspect
import math
import textwrap
from types import SimpleNamespace

import pytest
import torch
from isaaclab_newton.envs.mdp.actions.newton_task_space_actions import _NewtonTaskSpaceAction

from isaaclab.envs.mdp.actions.task_space_actions import (
    DifferentialInverseKinematicsAction,
    OperationalSpaceControllerAction,
)
from isaaclab.utils import math as math_utils

pytestmark = pytest.mark.unit

_S = math.sin(math.pi / 4.0)


def test_body_offset_jacobian_uses_offset_in_root_frame():
    """The offset uses root axes, and a rigid offset rotation leaves angular rows unchanged."""
    # a tilted root, so that a body-frame or world-frame lever arm gives a different result
    root_quat_w = torch.tensor([[_S, 0.0, 0.0, _S]])
    # One revolute joint about the root z axis through the body origin; the body is yawed 90 degrees from the root.
    body_quat_w = math_utils.quat_mul(root_quat_w, torch.tensor([[0.0, 0.0, _S, _S]]))
    jacobian_w = torch.zeros(1, 1, 6, 1)
    jacobian_w[0, 0, 3:, 0] = math_utils.quat_apply(root_quat_w, torch.tensor([[0.0, 0.0, 1.0]]))[0]
    data = SimpleNamespace(
        body_link_jacobian_w=SimpleNamespace(torch=jacobian_w),
        root_quat_w=SimpleNamespace(torch=root_quat_w),
        body_quat_w=SimpleNamespace(torch=torch.stack([root_quat_w, body_quat_w], dim=1)),  # [root, hand]
    )
    term = SimpleNamespace(
        _asset=SimpleNamespace(data=data),
        _body_idx=1,  # a fixed base has no Jacobian row for its root, so the hand stays Jacobian row 0
        _jacobi_body_idx=0,
        _jacobi_joint_ids=[0],
        _jacobian_b=torch.zeros(1, 6, 1),
        _offset_pos=torch.tensor([[1.0, 0.0, 0.0]]),
        # a 90 degree rotation about x for the target frame
        _offset_rot=torch.tensor([[_S, 0.0, 0.0, _S]]),
    )

    jacobian_b = _NewtonTaskSpaceAction._compute_ee_jacobian(term)

    # The lever arm in root axes is (0, 1, 0), and z x (0, 1, 0) = (-1, 0, 0).
    expected = torch.tensor([[[-1.0], [0.0], [0.0], [0.0], [0.0], [1.0]]])
    torch.testing.assert_close(jacobian_b, expected, atol=1e-6, rtol=0.0)


def test_task_space_actions_do_not_own_point_shift_math():
    """All task-space actions must use the shared point shift instead of recopying its formula."""
    methods = (
        DifferentialInverseKinematicsAction._compute_frame_jacobian,
        OperationalSpaceControllerAction._compute_ee_jacobian,
        _NewtonTaskSpaceAction._compute_ee_jacobian,
    )
    for method in methods:
        tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
        calls = [node.func for node in ast.walk(tree) if isinstance(node, ast.Call)]
        names = [call.attr for call in calls if isinstance(call, ast.Attribute)]
        assert names.count("velocity_at_point") == 1, method.__qualname__
        assert not {"skew_symmetric_matrix", "cross"}.intersection(names), method.__qualname__
