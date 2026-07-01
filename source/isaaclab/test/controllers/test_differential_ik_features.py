# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sim-free unit tests for the optional :class:`DifferentialIKController` features.

Covers the ``adaptive_dls`` ik-method, per-axis orientation weighting, null-space joint-limit
avoidance, and quaternion renormalization -- all exercised with hand-built tensors (no gym.make,
USD, or GPU). The simulated convergence tests live in ``test_differential_ik.py``.
"""

import math

import pytest
import torch

from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg

_NUM_JOINTS = 5
_ID_QUAT = [0.0, 0.0, 0.0, 1.0]  # xyzw identity


def _quat_xyzw(axis: list[float], angle: float) -> list[float]:
    """Build a unit xyzw quaternion from an axis (need not be unit) and angle [rad]."""
    norm = math.sqrt(sum(a * a for a in axis)) or 1.0
    s = math.sin(angle / 2.0)
    return [axis[0] / norm * s, axis[1] / norm * s, axis[2] / norm * s, math.cos(angle / 2.0)]


def _make_controller(
    command_type: str = "pose",
    ik_method: str = "adaptive_dls",
    ik_params: dict | None = None,
    orientation_weight=None,
    joint_limit_avoidance_gain: float = 0.0,
    joint_limit_avoidance_margin: float = 0.3,
    num_envs: int = 1,
) -> DifferentialIKController:
    cfg = DifferentialIKControllerCfg(
        command_type=command_type,
        use_relative_mode=False,
        ik_method=ik_method,
        ik_params=ik_params,
        orientation_weight=orientation_weight,
        joint_limit_avoidance_gain=joint_limit_avoidance_gain,
        joint_limit_avoidance_margin=joint_limit_avoidance_margin,
    )
    return DifferentialIKController(cfg, num_envs=num_envs, device="cpu")


def test_adaptive_dls_default_params():
    """The cfg fills the adaptive_dls defaults when ``ik_params`` is not provided."""
    cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="adaptive_dls")
    assert set(cfg.ik_params) == {"lambda_min", "lambda_max", "sigma_thresh"}


def test_cfg_rejects_bad_orientation_weight():
    with pytest.raises(ValueError):
        DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls", orientation_weight=(0.3, 0.3)
        )


def test_cfg_rejects_bad_adaptive_params():
    with pytest.raises(ValueError):
        DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="adaptive_dls",
            ik_params={"lambda_min": 0.5, "lambda_max": 0.1, "sigma_thresh": 0.02},
        )


def test_set_command_renormalizes_quat():
    """A non-unit commanded quaternion is stored renormalized in pose (absolute) mode."""
    c = _make_controller()
    raw = torch.tensor([0.2588, 0.0, 0.0, 0.9659])  # xyzw
    cmd = torch.cat([torch.tensor([0.3, -0.1, 0.2]), raw * 3.0]).unsqueeze(0)  # non-unit
    c.set_command(cmd)
    stored = c.ee_quat_des[0]
    assert torch.linalg.norm(stored).item() == pytest.approx(1.0, abs=1e-6)
    torch.testing.assert_close(stored, raw / torch.linalg.norm(raw), atol=1e-6, rtol=0.0)


def test_orientation_weight_none_is_unweighted():
    """With no orientation weight, the pose task Jacobian equals the raw Jacobian."""
    c = _make_controller(orientation_weight=None)
    ee_pos = torch.tensor([[0.3, 0.0, 0.2]])
    ee_quat = torch.tensor([_ID_QUAT])
    c.set_command(torch.tensor([[0.31, 0.0, 0.2] + _quat_xyzw([1.0, 0.0, 0.0], 0.5)]))
    jac = torch.arange(6 * _NUM_JOINTS, dtype=torch.float32).reshape(1, 6, _NUM_JOINTS)
    task_jac, _ = c._compute_pose_task(ee_pos, ee_quat, jac)
    torch.testing.assert_close(task_jac, jac)


def test_orientation_weight_per_axis_scales_rows_and_error():
    """A per-axis (wx, wy, wz) weight scales each base-frame orientation row and error; wz=0 drops
    the yaw row. Position rows/error are untouched."""
    ee_pos = torch.tensor([[0.3, 0.0, 0.2]])
    ee_quat = torch.tensor([_ID_QUAT])
    jac = torch.arange(6 * _NUM_JOINTS, dtype=torch.float32).reshape(1, 6, _NUM_JOINTS)
    cmd = torch.tensor([[0.31, 0.0, 0.2] + _quat_xyzw([0.3, 0.5, 0.8], 0.7)])

    base = _make_controller(orientation_weight=None)
    base.set_command(cmd)
    jb, eb = base._compute_pose_task(ee_pos, ee_quat, jac)

    c = _make_controller(orientation_weight=(0.4, 0.2, 0.0))
    c.set_command(cmd)
    jp, ep = c._compute_pose_task(ee_pos, ee_quat, jac)

    torch.testing.assert_close(jp[:, :3, :], jb[:, :3, :])  # position rows unchanged
    torch.testing.assert_close(jp[:, 3, :], 0.4 * jb[:, 3, :])
    torch.testing.assert_close(jp[:, 4, :], 0.2 * jb[:, 4, :])
    torch.testing.assert_close(jp[:, 5, :], torch.zeros_like(jb[:, 5, :]))  # yaw dropped
    torch.testing.assert_close(ep[:, :3], eb[:, :3])
    torch.testing.assert_close(ep[:, 3], 0.4 * eb[:, 3])
    torch.testing.assert_close(ep[:, 5], torch.zeros_like(eb[:, 5]))


def test_compute_pose_task_quat_convention_xyzw():
    """Discriminating regression for the xyzw quaternion convention: commanding the EE's current
    orientation yields zero orientation error. A wxyz mis-read would corrupt this."""
    rot = pytest.importorskip("scipy.spatial.transform").Rotation.from_euler("x", 30.0, degrees=True)
    q_xyzw = rot.as_quat()  # [x, y, z, w]
    ee_quat = torch.tensor(q_xyzw, dtype=torch.float32).unsqueeze(0)
    ee_pos = torch.tensor([[0.3, 0.0, 0.2]])
    c = _make_controller(orientation_weight=1.0)
    cmd = torch.cat([torch.tensor([0.3, 0.0, 0.2]), torch.tensor(q_xyzw, dtype=torch.float32)]).unsqueeze(0)
    c.set_command(cmd)
    _, err = c._compute_pose_task(ee_pos, ee_quat, torch.zeros(1, 6, _NUM_JOINTS))
    assert torch.linalg.norm(err[0, 3:6]).item() == pytest.approx(0.0, abs=1e-6)


def test_adaptive_dls_damps_singularity():
    """Near a task-Jacobian singularity, the adaptive ramp produces a smaller (more damped) and
    finite step than a fixed ``lambda_min`` solve would."""
    c = _make_controller(ik_params={"lambda_min": 0.01, "lambda_max": 0.5, "sigma_thresh": 0.1})
    j_task = torch.zeros(1, 6, _NUM_JOINTS)
    j_task[0, 0, 0] = j_task[0, 1, 1] = j_task[0, 2, 2] = 1.0  # well-conditioned position block
    eps = 1e-3
    j_task[0, 3, 3] = j_task[0, 4, 4] = eps  # near-singular orientation block
    err = torch.zeros(1, 6)
    err[0, 3] = err[0, 4] = 1.0

    dq = c._compute_delta_joint_pos(delta_pose=err, jacobian=j_task)
    # reference: fixed lambda_min damped least squares
    jt = j_task.transpose(1, 2)
    a_min = torch.bmm(j_task, jt) + (0.01**2) * torch.eye(6)
    dq_min = torch.bmm(jt, torch.linalg.solve(a_min, err.unsqueeze(-1))).squeeze(-1)
    assert torch.isfinite(dq).all()
    assert dq.norm().item() < dq_min.norm().item()


def test_joint_limit_avoidance_zero_when_disabled():
    """JLA returns zeros when joint_limit_avoidance_gain == 0 (default) or before limits are provided."""
    c = _make_controller(joint_limit_avoidance_gain=0.0)
    out = c._joint_limit_avoidance(torch.zeros(1, _NUM_JOINTS), torch.ones(1, 6, _NUM_JOINTS))
    torch.testing.assert_close(out, torch.zeros(1, _NUM_JOINTS))
    # enabled but limits not set yet -> still zeros
    c2 = _make_controller(joint_limit_avoidance_gain=1.0)
    out2 = c2._joint_limit_avoidance(torch.zeros(1, _NUM_JOINTS), torch.ones(1, 6, _NUM_JOINTS))
    torch.testing.assert_close(out2, torch.zeros(1, _NUM_JOINTS))


def test_joint_limit_avoidance_stays_in_position_nullspace():
    """When enabled, the JLA correction lies in the null space of the position rows, so it does not
    perturb the commanded end-effector position (``J_pos @ correction ~= 0``)."""
    c = _make_controller(joint_limit_avoidance_gain=2.0, joint_limit_avoidance_margin=0.3)
    c.set_joint_pos_limits(torch.full((_NUM_JOINTS,), -1.0), torch.full((_NUM_JOINTS,), 1.0))
    # a generic well-conditioned task Jacobian
    torch.manual_seed(0)
    j_task = torch.randn(1, 6, _NUM_JOINTS)
    # joints near their limits -> non-zero center-seeking bias
    joint_pos = torch.tensor([[0.95, -0.9, 0.0, 0.8, -0.85]])
    correction = c._joint_limit_avoidance(joint_pos, j_task)
    assert correction.norm().item() > 0.0  # bias is active
    residual = torch.bmm(j_task[:, :3, :], correction.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-5, rtol=0.0)


def test_compute_returns_joint_targets_shape():
    """compute returns one target per joint (joint_pos + delta)."""
    c = _make_controller(orientation_weight=(0.5, 0.5, 0.0), joint_limit_avoidance_gain=0.5)
    c.set_joint_pos_limits(torch.full((_NUM_JOINTS,), -1.0), torch.full((_NUM_JOINTS,), 1.0))
    ee_pos = torch.tensor([[0.3, 0.0, 0.2]])
    ee_quat = torch.tensor([_ID_QUAT])
    c.set_command(torch.tensor([[0.31, 0.0, 0.2] + _ID_QUAT]))
    jac = torch.zeros(1, 6, _NUM_JOINTS)
    for i in range(5):
        jac[0, i, i] = 1.0
    out = c.compute(ee_pos, ee_quat, jac, torch.zeros(1, _NUM_JOINTS))
    assert out.shape == (1, _NUM_JOINTS)
