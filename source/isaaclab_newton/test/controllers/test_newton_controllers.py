# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton model-free controllers and the task-space Jacobian supplied to them."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from isaaclab_newton.controllers import (
    NewtonDifferentialIKController,
    NewtonDifferentialIKControllerCfg,
    NewtonJointImpedanceController,
    NewtonJointImpedanceControllerCfg,
    NewtonOperationalSpaceController,
    NewtonOperationalSpaceControllerCfg,
)
from isaaclab_newton.envs.mdp.actions.newton_task_space_actions import _NewtonTaskSpaceAction

from isaaclab.utils import math as math_utils

_NUM_ENVS = 3
_NUM_JOINTS = 7
_DEVICE = "cpu"


def _rand(*shape: int) -> torch.Tensor:
    return torch.rand(*shape) * 2.0 - 1.0


def _pose(pos: torch.Tensor) -> torch.Tensor:
    """Poses at ``pos`` with identity orientation, so the orientation error is zero."""
    quat = torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(pos.shape[0], 4)
    return torch.cat((pos, quat), dim=-1)


def _spd(num_envs: int, n: int) -> torch.Tensor:
    a = _rand(num_envs, n, n)
    return a @ a.mT + n * torch.eye(n)


@pytest.mark.parametrize("inertia_decoupling", [False, True])
@pytest.mark.parametrize("live_gains", [False, True])
def test_joint_impedance_matches_impedance_law(inertia_decoupling: bool, live_gains: bool):
    """Efforts follow ``M (Kp (q_des - q) + Kd (qd_des - qd)) + g + c``, with baked or live gains."""
    torch.manual_seed(0)
    kp, kd = torch.rand(_NUM_ENVS, _NUM_JOINTS) * 50.0, torch.rand(_NUM_ENVS, _NUM_JOINTS) * 5.0
    cfg = NewtonJointImpedanceControllerCfg(
        stiffness=None if live_gains else 40.0,
        damping=None if live_gains else 3.0,
        use_inertia_decoupling=inertia_decoupling,
    )
    if not live_gains:
        kp, kd = torch.full_like(kp, 40.0), torch.full_like(kd, 3.0)
    controller = NewtonJointImpedanceController(cfg, _NUM_ENVS, _NUM_JOINTS, _DEVICE)
    q_des, q, qd, qd_des = (_rand(_NUM_ENVS, _NUM_JOINTS) for _ in range(4))
    mass, gravity, coriolis = _spd(_NUM_ENVS, _NUM_JOINTS), _rand(_NUM_ENVS, _NUM_JOINTS), _rand(_NUM_ENVS, _NUM_JOINTS)

    efforts = controller.compute(
        q_des,
        q,
        qd,
        joint_vel_des=qd_des,
        mass_matrix=mass if inertia_decoupling else None,
        gravity=gravity,
        coriolis=coriolis,
        stiffness=kp if live_gains else None,
        damping=kd if live_gains else None,
    )

    acc = kp * (q_des - q) + kd * (qd_des - qd)
    if inertia_decoupling:
        acc = (mass @ acc.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(efforts, acc + gravity + coriolis, rtol=1e-4, atol=1e-4)


def test_disabled_input_is_rejected():
    """A tensor for an input the configuration disables raises instead of being ignored."""
    cfg = NewtonJointImpedanceControllerCfg(stiffness=1.0, damping=1.0, use_inertia_decoupling=False)
    controller = NewtonJointImpedanceController(cfg, _NUM_ENVS, _NUM_JOINTS, _DEVICE)
    q = torch.zeros(_NUM_ENVS, _NUM_JOINTS)
    with pytest.raises(ValueError, match="mass_matrix"):
        controller.compute(q, q, q, mass_matrix=torch.eye(_NUM_JOINTS).repeat(_NUM_ENVS, 1, 1))


def test_differential_ik_dls_matches_reference():
    """An offset-frame Jacobian reaches position-only DLS with the correct frame and control law."""
    torch.manual_seed(0)
    bandwidth, damping, dt = 20.0, 0.1, 0.05
    cfg = NewtonDifferentialIKControllerCfg(bandwidth=bandwidth, damping=damping, axis_weight=(1, 1, 1, 0, 0, 0))
    controller = NewtonDifferentialIKController(cfg, _NUM_ENVS, _NUM_JOINTS, _DEVICE)
    s = math.sqrt(0.5)
    root_quat_w = torch.tensor([[s, 0.0, 0.0, s]]).repeat(_NUM_ENVS, 1)
    body_quat_w = math_utils.quat_mul(root_quat_w, torch.tensor([[0.0, 0.0, s, s]]).repeat(_NUM_ENVS, 1))
    jacobian_w = _rand(_NUM_ENVS, 1, 6, _NUM_JOINTS)
    jacobian_w[:, 0, :3, 0] = 0.0
    jacobian_w[:, 0, 3:, 0] = math_utils.quat_apply(root_quat_w, torch.tensor([[0.0, 0.0, 1.0]]))
    data = SimpleNamespace(
        body_link_jacobian_w=SimpleNamespace(torch=jacobian_w),
        root_quat_w=SimpleNamespace(torch=root_quat_w),
        body_quat_w=SimpleNamespace(torch=torch.stack([root_quat_w, body_quat_w], dim=1)),
    )
    action = SimpleNamespace(
        _asset=SimpleNamespace(data=data),
        _body_idx=1,  # fixed-root body 1 uses Jacobian row 0
        _jacobi_body_idx=0,
        _jacobi_joint_ids=list(range(_NUM_JOINTS)),
        _jacobian_b=torch.zeros(_NUM_ENVS, 6, _NUM_JOINTS),
        _offset_pos=torch.tensor([[1.0, 0.0, 0.0]]).repeat(_NUM_ENVS, 1),
        _offset_rot=torch.tensor([[s, 0.0, 0.0, s]]).repeat(_NUM_ENVS, 1),
    )
    jacobian = _NewtonTaskSpaceAction._compute_ee_jacobian(action)
    # The body-local x offset is root-local y; a root-z rotation gives linear velocity -x.
    expected = torch.tensor([[-1.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).repeat(_NUM_ENVS, 1)
    torch.testing.assert_close(jacobian[:, :, 0], expected, atol=1e-6, rtol=0.0)
    assert not hasattr(math_utils, "velocity_at_point")

    q = _rand(_NUM_ENVS, _NUM_JOINTS)
    pos, pos_des = _rand(_NUM_ENVS, 3), _rand(_NUM_ENVS, 3)

    q_target = controller.compute(_pose(pos), _pose(pos_des), jacobian, q, dt)

    j = jacobian[:, :3]
    jjt = j @ j.mT + damping**2 * torch.eye(3)
    dq = bandwidth * (j.mT @ torch.linalg.solve(jjt, (pos_des - pos).unsqueeze(-1))).squeeze(-1)
    torch.testing.assert_close(q_target, q + dt * dq, rtol=1e-4, atol=1e-4)


def test_differential_ik_null_space_posture_preserves_task():
    """Posture control moves the joints toward the target without changing the task-space motion."""
    torch.manual_seed(0)
    common = dict(bandwidth=1.0, damping=0.0, ik_method="damped_least_squares", axis_weight=(1, 1, 1, 0, 0, 0))
    base = NewtonDifferentialIKController(NewtonDifferentialIKControllerCfg(**common), _NUM_ENVS, _NUM_JOINTS, _DEVICE)
    posture = NewtonDifferentialIKController(
        NewtonDifferentialIKControllerCfg(
            **common, use_null_space_posture_control=True, null_space_stiffness=1.0, null_space_damping=0.0
        ),
        _NUM_ENVS,
        _NUM_JOINTS,
        _DEVICE,
    )
    jacobian, q, q_null = _rand(_NUM_ENVS, 6, _NUM_JOINTS), _rand(_NUM_ENVS, _NUM_JOINTS), _rand(_NUM_ENVS, _NUM_JOINTS)
    pose, pose_des = _pose(_rand(_NUM_ENVS, 3)), _pose(_rand(_NUM_ENVS, 3))

    dq_base = base.compute(pose, pose_des, jacobian, q, 1.0).clone() - q
    dq_posture = posture.compute(pose, pose_des, jacobian, q, 1.0, null_space_joint_pos_target=q_null) - q

    dq_null = dq_posture - dq_base
    # the posture term lies in the task null space and reduces the posture error
    torch.testing.assert_close(jacobian[:, :3] @ dq_null.unsqueeze(-1), torch.zeros(_NUM_ENVS, 3, 1), atol=1e-4, rtol=0)
    assert torch.all((dq_null * (q_null - q)).sum(dim=-1) > 0.0)


def test_operational_space_matches_pd_law():
    """Without inertia decoupling, efforts follow ``J^T (Kp e - Kd v) + g`` for a position error."""
    torch.manual_seed(0)
    kp, kd = (100.0, 100.0, 100.0, 50.0, 50.0, 50.0), (10.0, 10.0, 10.0, 5.0, 5.0, 5.0)
    cfg = NewtonOperationalSpaceControllerCfg(motion_stiffness=kp, motion_damping=kd, use_inertia_decoupling=False)
    controller = NewtonOperationalSpaceController(cfg, _NUM_ENVS, _NUM_JOINTS, _DEVICE)
    jacobian, vel, gravity = _rand(_NUM_ENVS, 6, _NUM_JOINTS), _rand(_NUM_ENVS, 6), _rand(_NUM_ENVS, _NUM_JOINTS)
    pos, pos_des = _rand(_NUM_ENVS, 3), _rand(_NUM_ENVS, 3)

    efforts = controller.compute(jacobian, _pose(pos), vel, _pose(pos_des), gravity=gravity)

    error = torch.cat((pos_des - pos, torch.zeros(_NUM_ENVS, 3)), dim=-1)
    wrench = torch.tensor(kp) * error - torch.tensor(kd) * vel
    expected = (jacobian.mT @ wrench.unsqueeze(-1)).squeeze(-1) + gravity
    torch.testing.assert_close(efforts, expected, rtol=1e-4, atol=1e-3)
