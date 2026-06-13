# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sim-free unit tests for the SO-101 full-pose SE3 IK controller.

These exercise the controller's task assembly (6-row pose Jacobian: 3 linear rows + 3
soft-weighted orientation rows over 5 arm joints) and command storage with hand-built tensors --
no gym.make, USD, or GPU.
"""

import dataclasses
import math

import pytest
import torch

from isaaclab_tasks.contrib.stack.config.so101.pose_ik_controller import (
    SO101PoseIKController,
    SO101PoseIKControllerCfg,
)

# The SO-101 arm IK acts over 5 joints; the geometric Jacobian is (N, 6, 5).
_NUM_JOINTS = 5
_ID_QUAT = [0.0, 0.0, 0.0, 1.0]  # xyzw identity


def _make_controller(
    num_envs=1,
    orientation_task_weight=1.0,
    lambda_min=0.05,
    lambda_max=0.2,
    sigma_thresh=0.02,
    jla_gain=0.0,
    jla_margin=0.3,
):
    cfg = SO101PoseIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
        orientation_task_weight=orientation_task_weight,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        sigma_thresh=sigma_thresh,
        jla_gain=jla_gain,
        jla_margin=jla_margin,
    )
    return SO101PoseIKController(cfg=cfg, num_envs=num_envs, device="cpu")


def test_action_dim_is_seven():
    """The controller advertises a 7D pose command: [pos_xyz, quat_xyzw]."""
    assert _make_controller().action_dim == 7


def test_set_command_stores_position_and_quat():
    """set_command splits the 7D command into a 3D position target and a unit xyzw quat target."""
    c = _make_controller()
    cmd = torch.tensor([[0.3, -0.1, 0.2, 0.0, 0.0, 0.0, 1.0]])
    c.set_command(cmd, ee_pos=torch.zeros(1, 3), ee_quat=torch.tensor([_ID_QUAT]))
    torch.testing.assert_close(c.ee_pos_des, cmd[:, :3])
    torch.testing.assert_close(c.ee_quat_des, cmd[:, 3:7])


def test_set_command_renormalizes_quat():
    """A non-unit commanded quaternion is stored renormalized, preserving direction."""
    c = _make_controller()
    raw = torch.tensor([0.2588, 0.0, 0.0, 0.9659])  # xyzw, slightly off unit after scaling
    scaled = raw * 3.0  # non-unit
    cmd = torch.cat([torch.tensor([0.3, -0.1, 0.2]), scaled]).unsqueeze(0)
    c.set_command(cmd, ee_pos=torch.zeros(1, 3), ee_quat=torch.tensor([_ID_QUAT]))
    stored = c.ee_quat_des[0]
    assert torch.linalg.norm(stored).item() == pytest.approx(1.0, abs=1e-6)
    # Direction preserved: stored == raw / |raw| (same as scaled / |scaled|).
    expected = raw / torch.linalg.norm(raw)
    torch.testing.assert_close(stored, expected, atol=1e-6, rtol=0.0)


def test_assemble_task_is_six_rows_and_position_rows_unweighted():
    """The task is 6 rows; the 3 linear rows equal the Jacobian linear block unscaled."""
    c = _make_controller(orientation_task_weight=0.5)
    ee_pos = torch.tensor([[0.3, 0.0, 0.2]])
    ee_quat = torch.tensor([_ID_QUAT])
    c.set_command(torch.tensor([[0.31, 0.0, 0.2] + _ID_QUAT]), ee_pos=ee_pos, ee_quat=ee_quat)
    jac = torch.arange(6 * _NUM_JOINTS, dtype=torch.float32).reshape(1, 6, _NUM_JOINTS)
    J_task, err = c._assemble_task(ee_pos, ee_quat, jac)
    assert J_task.shape == (1, 6, _NUM_JOINTS)
    assert err.shape == (1, 6)
    torch.testing.assert_close(J_task[:, :3, :], jac[:, 0:3, :])  # linear rows unscaled
    # Orientation rows are the angular block scaled by w.
    torch.testing.assert_close(J_task[:, 3:6, :], 0.5 * jac[:, 3:6, :])


def test_orientation_weight_scales_orientation_rows_and_error():
    """orientation_task_weight scales the orientation rows of J_task and the orientation error
    vs the w=1 task; the position rows/error are unchanged."""
    ee_pos = torch.tensor([[0.3, 0.0, 0.2]])
    ee_quat = torch.tensor([_ID_QUAT])
    jac = torch.arange(6 * _NUM_JOINTS, dtype=torch.float32).reshape(1, 6, _NUM_JOINTS)
    # Non-identity orientation command so there is a real orientation error to scale.
    cmd_quat = _quat_xyzw([1.0, 0.0, 0.0], 0.6)
    cmd = torch.tensor([[0.31, 0.0, 0.2] + cmd_quat])

    full = _make_controller(orientation_task_weight=1.0)
    full.set_command(cmd, ee_pos=ee_pos, ee_quat=ee_quat)
    Jf, ef = full._assemble_task(ee_pos, ee_quat, jac)

    half = _make_controller(orientation_task_weight=0.5)
    half.set_command(cmd, ee_pos=ee_pos, ee_quat=ee_quat)
    Jh, eh = half._assemble_task(ee_pos, ee_quat, jac)

    torch.testing.assert_close(Jh[:, :3, :], Jf[:, :3, :])  # position rows unchanged
    torch.testing.assert_close(Jh[:, 3:6, :], 0.5 * Jf[:, 3:6, :])  # orientation rows scaled
    torch.testing.assert_close(eh[:, :3], ef[:, :3])  # position error unchanged
    torch.testing.assert_close(eh[:, 3:6], 0.5 * ef[:, 3:6])  # orientation error scaled


def test_orientation_weight_per_axis_scales_each_row():
    """A per-axis (wx, wy, wz) weight scales each base-frame orientation row independently; wz=0
    zeros the yaw (base-Z) row, leaving a 5-row [pos + 2 tilt] task."""
    ee_pos = torch.tensor([[0.3, 0.0, 0.2]])
    ee_quat = torch.tensor([_ID_QUAT])
    jac = torch.arange(6 * _NUM_JOINTS, dtype=torch.float32).reshape(1, 6, _NUM_JOINTS)
    cmd = torch.tensor([[0.31, 0.0, 0.2] + _quat_xyzw([0.3, 0.5, 0.8], 0.7)])

    base = _make_controller(orientation_task_weight=1.0)
    base.set_command(cmd, ee_pos=ee_pos, ee_quat=ee_quat)
    Jb, eb = base._assemble_task(ee_pos, ee_quat, jac)

    c = _make_controller(orientation_task_weight=(0.4, 0.2, 0.0))
    c.set_command(cmd, ee_pos=ee_pos, ee_quat=ee_quat)
    Jp, ep = c._assemble_task(ee_pos, ee_quat, jac)

    torch.testing.assert_close(Jp[:, :3, :], Jb[:, :3, :])  # position rows unchanged
    torch.testing.assert_close(Jp[:, 3, :], 0.4 * Jb[:, 3, :])  # base-X (tilt) row
    torch.testing.assert_close(Jp[:, 4, :], 0.2 * Jb[:, 4, :])  # base-Y (tilt) row
    torch.testing.assert_close(Jp[:, 5, :], torch.zeros_like(Jb[:, 5, :]))  # base-Z (yaw) dropped
    torch.testing.assert_close(ep[:, 3], 0.4 * eb[:, 3])
    torch.testing.assert_close(ep[:, 4], 0.2 * eb[:, 4])
    torch.testing.assert_close(ep[:, 5], torch.zeros_like(eb[:, 5]))


def test_cfg_rejects_bad_orientation_weight():
    with pytest.raises(ValueError):
        _make_controller(orientation_task_weight=(0.3, 0.3))  # not length 3


def test_orientation_weight_zero_drops_orientation_task():
    """At w=0 the orientation rows and error are zero, so the solve reduces to a position-only
    3-row DLS. The ramp now keys off the full task Jacobian, whose smallest singular value vanishes
    when the orientation rows are zeroed, so lambda saturates to lambda_max (the documented weight~0
    over-damping); compute reproduces the position-only solve at that saturated damping."""
    ee_pos = torch.tensor([[0.3, 0.0, 0.2]])
    ee_quat = torch.tensor([_ID_QUAT])
    jac = torch.arange(6 * _NUM_JOINTS, dtype=torch.float32).reshape(1, 6, _NUM_JOINTS) * 0.01
    # Non-identity orientation command: at w=0 it must have no effect.
    cmd = torch.tensor([[0.5, 0.3, 0.2] + _quat_xyzw([0.0, 1.0, 0.0], 0.9)])
    joint_pos = torch.zeros(1, _NUM_JOINTS)

    c = _make_controller(orientation_task_weight=0.0)
    c.set_command(cmd, ee_pos=ee_pos, ee_quat=ee_quat)
    J_task, err = c._assemble_task(ee_pos, ee_quat, jac)
    torch.testing.assert_close(J_task[:, 3:6, :], torch.zeros(1, 3, _NUM_JOINTS))
    torch.testing.assert_close(err[:, 3:6], torch.zeros(1, 3))

    out = c.compute(ee_pos, ee_quat, jac, joint_pos)
    delta = out - joint_pos

    # Reference: the zeroed orientation rows make the solve block-diagonal, so it reduces to a
    # position-only 3-row DLS at the same (saturated) adaptive damping the controller computes.
    J_pos = J_task[:, :3, :]
    sigma_min = torch.linalg.svdvals(J_task)[:, -1]
    lam2 = c._adaptive_lambda_sq(sigma_min)
    jt = J_pos.transpose(1, 2)
    a = torch.bmm(J_pos, jt) + lam2.view(-1, 1, 1) * torch.eye(3)
    pos_err = err[:, :3]
    dq_pos = torch.bmm(jt, torch.linalg.solve(a, pos_err.unsqueeze(-1))).squeeze(-1)
    torch.testing.assert_close(delta, dq_pos, atol=1e-6, rtol=1e-5)


def test_damping_ramp_catches_orientation_singularity():
    """The ramp keys off the FULL task Jacobian, so an orientation-induced near-singularity (tiny
    smallest singular value of J_task while the position block stays well-conditioned) triggers
    damping. Fails with a position-block-only ramp, which would leave lambda at lambda_min and let
    the solve run away."""
    c = _make_controller(lambda_min=0.01, lambda_max=0.5, sigma_thresh=0.1)
    M = _NUM_JOINTS
    J_task = torch.zeros(1, 6, M)
    # Position block: unit gain on joints 0,1,2 -> position is well-conditioned (sigma_min ~ 1).
    J_task[0, 0, 0] = J_task[0, 1, 1] = J_task[0, 2, 2] = 1.0
    # Orientation block: only a tiny coupling to joints 3,4 -> the full 6x5 task is near-singular.
    eps = 1e-3
    J_task[0, 3, 3] = eps
    J_task[0, 4, 4] = eps

    sig_full = torch.linalg.svdvals(J_task)[:, -1]
    sig_pos = torch.linalg.svdvals(J_task[:, :3, :])[:, -1]
    # The full task sees a singularity the position block alone does not.
    assert sig_full.item() < 0.5 * sig_pos.item()
    # The ramp (keyed off the full task) saturates toward lambda_max, not lambda_min.
    lam2 = c._adaptive_lambda_sq(sig_full)
    assert lam2.item() > (0.5 * c.cfg.lambda_max) ** 2

    # The adaptive solve is more damped (smaller delta) than a fixed-lambda_min solve would be --
    # this is the runaway the position-block ramp (lambda_min here) would NOT have prevented.
    err = torch.zeros(1, 6)
    err[0, 3] = err[0, 4] = 1.0
    dq = c._damped_least_squares(J_task, err)
    jt = J_task.transpose(1, 2)
    a_min = torch.bmm(J_task, jt) + (c.cfg.lambda_min**2) * torch.eye(6)
    dq_min = torch.bmm(jt, torch.linalg.solve(a_min, err.unsqueeze(-1))).squeeze(-1)
    assert torch.isfinite(dq).all()
    assert dq.norm().item() < dq_min.norm().item()


def test_compute_returns_joint_targets_shape():
    """compute returns one target per IK joint (joint_pos + delta)."""
    c = _make_controller()
    ee_pos = torch.tensor([[0.3, 0.0, 0.2]])
    ee_quat = torch.tensor([_ID_QUAT])
    c.set_command(torch.tensor([[0.31, 0.0, 0.2] + _ID_QUAT]), ee_pos=ee_pos, ee_quat=ee_quat)
    # Position rows on the first 3 joints, angular rows on the last 2: well-conditioned (1,6,5).
    jac = torch.zeros(1, 6, _NUM_JOINTS)
    jac[0, 0, 0] = 1.0
    jac[0, 1, 1] = 1.0
    jac[0, 2, 2] = 1.0
    jac[0, 3, 3] = 1.0
    jac[0, 4, 4] = 1.0
    joint_pos = torch.zeros(1, _NUM_JOINTS)
    out = c.compute(ee_pos, ee_quat, jac, joint_pos)
    assert out.shape == (1, _NUM_JOINTS)


def test_orientation_error_zero_for_non_identity_quat_xyzw():
    """Discriminating regression for the xyzw quaternion convention.

    Builds a known rotation (30 deg about +X) via scipy in xyzw (scalar-last), which is the
    IsaacLab convention. Commanding that exact orientation as the target while the current EE is
    at the same orientation must yield zero orientation error (axis-angle of q_des * q_cur^-1).

    A version that mis-converts the quaternion (e.g. treats an xyzw quat as wxyz) corrupts the
    rotation and produces a non-zero orientation error. This test catches that class of
    convention bug.
    """
    R = pytest.importorskip("scipy.spatial.transform").Rotation
    rot = R.from_euler("x", 30.0, degrees=True)
    q_xyzw = rot.as_quat()  # shape (4,) = [x, y, z, w]
    ee_quat = torch.tensor(q_xyzw, dtype=torch.float32).unsqueeze(0)  # (1, 4) xyzw

    c = _make_controller()
    ee_pos = torch.tensor([[0.3, 0.0, 0.2]])
    # Command the EE at its current pose -> zero position AND orientation error.
    cmd = torch.cat([torch.tensor([0.3, 0.0, 0.2]), torch.tensor(q_xyzw, dtype=torch.float32)]).unsqueeze(0)
    c.set_command(cmd, ee_pos=ee_pos, ee_quat=ee_quat)

    jac = torch.zeros(1, 6, _NUM_JOINTS)  # only task error is tested here
    _, err = c._assemble_task(ee_pos, ee_quat, jac)

    assert torch.linalg.norm(err[0, 3:6]).item() == pytest.approx(0.0, abs=1e-6), (
        f"orientation error should be 0 but got {err[0, 3:6].tolist()} -- "
        "this likely indicates a quaternion convention mismatch (xyzw vs wxyz)"
    )


def test_orientation_error_matches_axis_angle_xyzw():
    """A known relative rotation between current and commanded orientation yields the expected
    axis-angle orientation error (base-frame q_des * q_cur^-1, xyzw)."""
    R = pytest.importorskip("scipy.spatial.transform").Rotation
    cur = R.from_euler("z", 10.0, degrees=True)
    phi = 0.4
    des = R.from_rotvec([0.0, 0.0, phi]) * cur  # left-multiply a +phi roll about base Z
    ee_quat = torch.tensor(cur.as_quat(), dtype=torch.float32).unsqueeze(0)
    cmd = torch.cat([torch.tensor([0.3, 0.0, 0.2]), torch.tensor(des.as_quat(), dtype=torch.float32)]).unsqueeze(0)

    c = _make_controller(orientation_task_weight=1.0)
    ee_pos = torch.tensor([[0.3, 0.0, 0.2]])
    c.set_command(cmd, ee_pos=ee_pos, ee_quat=ee_quat)
    jac = torch.zeros(1, 6, _NUM_JOINTS)
    _, err = c._assemble_task(ee_pos, ee_quat, jac)
    # The orientation error is the axis-angle [0, 0, phi].
    torch.testing.assert_close(err[0, 3:6], torch.tensor([0.0, 0.0, phi]), atol=1e-5, rtol=0.0)


def test_action_cfg_points_at_custom_action_and_controller():
    """The action cfg wires the custom action class and the full-pose controller cfg."""
    pytest.importorskip("pxr")  # the action term imports UsdPhysics at module load
    from isaaclab.utils.string import string_to_callable

    from isaaclab_tasks.contrib.stack.config.so101.pose_ik_action import SO101PoseIKActionCfg
    from isaaclab_tasks.contrib.stack.config.so101.pose_ik_action_term import SO101PoseIKAction

    cfg = SO101PoseIKActionCfg(
        asset_name="robot",
        joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
        body_name="gripper",
        controller=SO101PoseIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
    )
    # ``class_type`` is a lazy ``{DIR}.pose_ik_action_term:SO101PoseIKAction`` string (so the cfg
    # stays importable without Kit); resolve it to confirm it points at the custom term.
    assert string_to_callable(str(cfg.class_type)) is SO101PoseIKAction
    assert isinstance(cfg.controller, SO101PoseIKControllerCfg)


def test_action_cfg_accepts_clip_field():
    """The cfg dataclass accepts a clip value as a plain field (no sim required).

    Verifying that NotImplementedError is raised when clip is set requires constructing
    the action term, which needs a live articulation (sim).  That path is sim-gated and
    is NOT tested here -- consistent with the other sim-dependent behavior in this suite.
    """
    pytest.importorskip("pxr")  # action term imports UsdPhysics at module load
    from isaaclab_tasks.contrib.stack.config.so101.pose_ik_action import (
        SO101PoseIKActionCfg,
    )

    clip_value = {"shoulder_pan": (-1.0, 1.0)}
    cfg = SO101PoseIKActionCfg(
        asset_name="robot",
        joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
        body_name="gripper",
        controller=SO101PoseIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        clip=clip_value,
    )
    # The cfg stores the value -- the refusal happens at action-term construction (sim-gated).
    assert cfg.clip == clip_value


def test_env_cfg_arm_action_is_pose_and_ordering_matches_pipeline():
    """The IK-Abs env wires the full-pose arm action and the 8D action ordering."""
    pytest.importorskip("pxr")
    from isaaclab_tasks.contrib.stack.config.so101.pose_ik_action import (
        SO101PoseIKActionCfg,
    )
    from isaaclab_tasks.contrib.stack.config.so101.stack_ik_abs_env_cfg import (
        SO101CubeStackEnvCfg,
        SO101IkActionsCfg,
    )

    cfg = SO101CubeStackEnvCfg()
    assert isinstance(cfg.actions.arm_action, SO101PoseIKActionCfg)
    assert cfg.actions.arm_action.controller.command_type == "pose"
    assert cfg.actions.arm_action.joint_names == [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    ]
    # Field order is the positional contract with the pipeline output_order:
    # [arm(pos+quat), gripper] -> [pos_xyz, quat_xyzw, gripper].
    assert [f.name for f in dataclasses.fields(SO101IkActionsCfg)] == [
        "arm_action",
        "gripper_action",
    ]


def test_adaptive_lambda_schedule():
    """lambda^2 equals lambda_min^2 when well-conditioned, lambda_max^2 at a singularity, and is
    monotonic non-increasing in sigma_min between."""
    c = _make_controller(lambda_min=0.05, lambda_max=0.2, sigma_thresh=0.02)
    big = c._adaptive_lambda_sq(torch.tensor([0.1]))  # sigma_min >> thresh
    zero = c._adaptive_lambda_sq(torch.tensor([0.0]))  # fully singular
    mid = c._adaptive_lambda_sq(torch.tensor([0.01]))  # between
    assert big.item() == pytest.approx(0.05**2, rel=1e-5)
    assert zero.item() == pytest.approx(0.2**2, rel=1e-5)
    assert 0.05**2 < mid.item() < 0.2**2


def test_damped_least_squares_tracks_on_well_conditioned_jacobian():
    """On a full-rank, well-conditioned task the damped solve steps toward the target."""
    c = _make_controller()
    # 6-row identity-like task over 6 columns so J_task is full row rank.
    J_task = torch.eye(6).unsqueeze(0)
    err = torch.tensor([[0.1, -0.2, 0.05, 0.0, 0.0, 0.0]])
    dq = c._damped_least_squares(J_task, err)
    # With J = I and small lambda, dq ~ err / (1 + lambda^2): same sign, slightly shrunk.
    assert torch.all(torch.sign(dq) == torch.sign(err))
    assert torch.all(dq.abs() <= err.abs() + 1e-6)
    assert torch.linalg.norm(dq - err).item() < 0.05  # close to err for small damping


def test_cfg_rejects_bad_damping_params():
    with pytest.raises(ValueError):
        _make_controller(sigma_thresh=0.0)
    with pytest.raises(ValueError):
        _make_controller(lambda_min=0.5, lambda_max=0.1)


def test_adaptive_damping_bounds_delta_near_singularity():
    """Near a singular position task, adaptive damping keeps the joint delta bounded, whereas
    lambda_min-only damping blows it up. Regression for the runaway (delta=4.3) seen in sim."""
    c = _make_controller(lambda_min=0.05, lambda_max=0.2, sigma_thresh=0.02)
    # Position block near-singular (smallest sv 1e-3); orientation rows zero.
    J_task = torch.zeros(1, 6, 3)
    J_task[0, 0, 0] = 0.2
    J_task[0, 1, 1] = 0.2
    J_task[0, 2, 2] = 1e-3
    err = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])  # unit error in the degenerate position dir
    dq_adaptive = c._damped_least_squares(J_task, err)
    # Reference: same DLS pinned at lambda_min (the pre-fix behavior).
    lam2 = torch.tensor(0.05**2)
    jt = J_task.transpose(1, 2)
    a = torch.bmm(J_task, jt) + lam2 * torch.eye(6).unsqueeze(0)
    dq_fixed = torch.bmm(jt, torch.linalg.solve(a, err.unsqueeze(-1))).squeeze(-1)
    assert torch.linalg.norm(dq_adaptive).item() < 0.1
    assert torch.linalg.norm(dq_fixed).item() > 0.3


def test_jla_disabled_by_default_returns_zero():
    """With jla_gain=0 (default) the joint-limit-avoidance term is zero."""
    c = _make_controller(jla_gain=0.0)
    J_task = torch.eye(6).unsqueeze(0)[:, :, :_NUM_JOINTS]
    dq = c._joint_limit_avoidance(torch.zeros(1, _NUM_JOINTS), J_task)
    torch.testing.assert_close(dq, torch.zeros(1, _NUM_JOINTS))


def test_jla_pushes_joint_off_its_limit_within_position_nullspace():
    """A joint within jla_margin of its limit is pushed toward center, and the correction lies in
    the position task's null space (does not perturb commanded EE position)."""
    c = _make_controller(jla_gain=2.0, jla_margin=0.3)
    c.set_joint_pos_limits(torch.full((_NUM_JOINTS,), -1.5), torch.full((_NUM_JOINTS,), 1.5))
    # Joint 4 appears only in the orientation rows, not the position rows -> it sits in the
    # position null space and the avoidance correction is non-zero for it.
    J_task = torch.zeros(1, 6, _NUM_JOINTS)
    J_task[0, 0, 0] = 0.2
    J_task[0, 1, 1] = 0.2
    J_task[0, 2, 2] = 0.2
    J_task[0, 3, 4] = 1.0  # orientation row uses joint 4
    q = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.45]])  # joint 4 near +1.5 upper limit
    dq = c._joint_limit_avoidance(q, J_task)
    assert dq[0, 4].item() < -1e-4  # joint 4 pushed toward center (negative)
    residual = torch.bmm(J_task[:, :3, :], dq.unsqueeze(-1)).squeeze(-1)
    assert torch.linalg.norm(residual).item() < 1e-5  # in position null space


def test_jla_inactive_when_all_joints_mid_range():
    """All joints mid-range -> activation ~0 -> no correction."""
    c = _make_controller(jla_gain=2.0, jla_margin=0.3)
    c.set_joint_pos_limits(torch.full((_NUM_JOINTS,), -1.5), torch.full((_NUM_JOINTS,), 1.5))
    J_task = torch.eye(6).unsqueeze(0)[:, :, :_NUM_JOINTS]
    dq = c._joint_limit_avoidance(torch.zeros(1, _NUM_JOINTS), J_task)
    assert torch.linalg.norm(dq).item() < 1e-9


def test_jla_multi_env_shapes():
    """The null-space JLA term has the right shape and broadcasts across multiple envs."""
    c = _make_controller(num_envs=3, jla_gain=2.0, jla_margin=0.3)
    c.set_joint_pos_limits(torch.full((_NUM_JOINTS,), -1.5), torch.full((_NUM_JOINTS,), 1.5))
    J_single = torch.zeros(6, _NUM_JOINTS)
    J_single[0, 0] = 0.2
    J_single[1, 1] = 0.2
    J_single[2, 2] = 0.2
    J_single[3, 4] = 1.0  # joint 4 only in an orientation row -> position null space
    J_task = J_single.unsqueeze(0).expand(3, -1, -1).contiguous()
    q = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.45]] * 3)
    dq = c._joint_limit_avoidance(q, J_task)
    assert dq.shape == (3, _NUM_JOINTS)
    assert torch.all(dq[:, 4] < -1e-4)  # joint 4 near its limit in every env


def test_cfg_rejects_bad_jla_params():
    with pytest.raises(ValueError):
        _make_controller(jla_margin=0.0)
    with pytest.raises(ValueError):
        _make_controller(jla_gain=-1.0)


def test_compute_stays_bounded_near_singular_and_limit_with_all_levers():
    """End-to-end compute with soft orientation weight + adaptive damping + JLA enabled produces a
    finite, bounded joint step even at a near-singular, near-limit state (no runaway)."""
    c = _make_controller(orientation_task_weight=0.5, jla_gain=2.0, jla_margin=0.3)
    c.set_joint_pos_limits(torch.full((_NUM_JOINTS,), -1.5), torch.full((_NUM_JOINTS,), 1.5))
    # Near-singular geometric Jacobian: the y position row is tiny -> smallest position sv ~0.
    jac = torch.zeros(1, 6, _NUM_JOINTS)
    jac[0, 0, :] = torch.tensor([0.2, 0.2, 0.2, 0.0, 0.0])  # x linear row
    jac[0, 1, :] = torch.tensor([0.0, 1e-3, 1e-3, 0.0, 0.0])  # y linear row (near-degenerate)
    jac[0, 2, :] = torch.tensor([0.0, 0.0, 0.0, 0.2, 0.0])  # z linear row
    jac[0, 4, :] = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])  # angular-y row
    ee_pos = torch.tensor([[0.3, 0.0, 0.2]])
    ee_quat = torch.tensor([_ID_QUAT])
    joint_pos = torch.tensor([[0.0, 1.45, 0.0, 0.0, 0.0]])  # joint 1 near its +1.5 limit
    c.set_command(torch.tensor([[0.5, 0.3, 0.2] + _quat_xyzw([0.0, 1.0, 0.0], 0.5)]), ee_pos=ee_pos, ee_quat=ee_quat)
    out = c.compute(ee_pos, ee_quat, jac, joint_pos)
    delta = out - joint_pos
    assert torch.isfinite(delta).all()
    assert torch.linalg.norm(delta).item() < 5.0  # adaptive damping prevents the runaway


def _quat_xyzw(axis, angle_rad: float) -> list[float]:
    """Build an [x, y, z, w] quaternion (Python list) for a rotation about a unit axis."""
    ax = torch.tensor(axis, dtype=torch.float64)
    ax = ax / torch.linalg.norm(ax)
    half = 0.5 * angle_rad
    xyz = ax * math.sin(half)
    return [float(xyz[0]), float(xyz[1]), float(xyz[2]), float(math.cos(half))]
