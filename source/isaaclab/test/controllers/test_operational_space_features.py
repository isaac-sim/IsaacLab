# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")),
    ],
)
@pytest.mark.parametrize("partial_inertial_decoupling", [False, True])
def test_inertial_decoupling_damps_near_singular_directions(device, partial_inertial_decoupling):
    """Finite near-singular modes must not amplify commands or couple retained tasks to posture control."""
    cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs"],
        inertial_dynamics_decoupling=True,
        partial_inertial_dynamics_decoupling=partial_inertial_decoupling,
        nullspace_control="position",
        nullspace_stiffness=1.0,
    )
    controller = OperationalSpaceController(cfg, num_envs=4, device=device)
    # Solve for forces each step; do not reintroduce cached inverse matrices or duplicate inertia state.
    assert not hasattr(controller, "_mass_matrix_inv")
    assert not hasattr(controller, "_os_mass_matrix_b")
    pose = torch.zeros(4, 7, device=device)
    pose[:, -1] = 1.0
    controller.set_command(pose)

    # Known task directions, rotated within each translation/rotation block.
    basis = torch.eye(6, device=device)
    basis[:2, :2] = basis[3:5, 3:5] = torch.tensor([[0.6, -0.8], [0.8, 0.6]], device=device)
    scales = torch.ones(4, 6, device=device)
    scales[0] = torch.tensor([0.5, 0.7, 1.0, 0.4, 0.6, 0.9], device=device)
    scales[1, 0] = scales[2, 3] = 0.002
    scales[3] = 0.0
    masses = torch.arange(2.0, 9.0, device=device)
    joint_basis = torch.eye(7, device=device)
    joint_basis[0, 0] = joint_basis[6, 6] = 0.6
    joint_basis[0, 6], joint_basis[6, 0] = -0.8, 0.8
    mass = (joint_basis @ torch.diag(masses) @ joint_basis.mT).repeat(4, 1, 1)
    jacobian = torch.zeros(4, 6, 7, device=device)
    jacobian[:, :, :6] = basis @ torch.diag_embed(scales * masses[:6].sqrt())
    jacobian = jacobian @ joint_basis.mT
    acceleration = torch.arange(1.0, 7.0, device=device).repeat(4, 1)
    inputs = dict(
        jacobian_b=jacobian,
        mass_matrix=mass,
        current_ee_pose_b=pose,
        current_ee_vel_b=-acceleration / 20.0,
        current_joint_pos=torch.zeros(4, 7, device=device),
        current_joint_vel=torch.zeros(4, 7, device=device),
    )
    efforts = controller.compute(**inputs)
    retained = scales > 0.1
    # All nonzero weak modes are below the lower threshold: the scalar damped response is s / (s² + d).
    damping = cfg.inertia_conditioning_thresholds[0]
    gains = torch.where(retained, scales.clamp_min(0.1).reciprocal(), scales / (scales.square() + damping))
    expected = torch.zeros_like(efforts)
    expected[:, :6] = masses[:6].sqrt() * (acceleration @ basis) * gains
    expected = expected @ joint_basis.mT
    torch.testing.assert_close(efforts, expected, atol=2e-4, rtol=2e-4)

    # Batch composition must not change the result when only some environments need damping.
    for env_id in range(4):
        single_controller = OperationalSpaceController(cfg, num_envs=1, device=device)
        single_controller.set_command(pose[env_id : env_id + 1])
        single_efforts = single_controller.compute(**{key: value[env_id : env_id + 1] for key, value in inputs.items()})
        torch.testing.assert_close(single_efforts[0], efforts[env_id], atol=2e-4, rtol=2e-4)

    # Full inertia decoupling must isolate every retained task direction from posture torques.
    if not partial_inertial_decoupling:
        with_posture = controller.compute(
            **inputs, nullspace_joint_pos_target=torch.ones_like(efforts) @ joint_basis.mT
        )
        null_acceleration = (jacobian @ torch.linalg.solve(mass, (with_posture - efforts).unsqueeze(-1))).squeeze(-1)
        torch.testing.assert_close(
            (null_acceleration @ basis) * retained, torch.zeros(4, 6, device=device), atol=2e-4, rtol=0.0
        )
        expected_null = masses.expand_as(efforts).clone()
        expected_null[:, :6] *= 1.0 - scales * gains
        expected_null = expected_null @ joint_basis.mT
        torch.testing.assert_close(with_posture - efforts, expected_null, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")),
    ],
)
def test_inertial_decoupling_smoothly_releases_weak_directions(device):
    """Task and posture efforts are continuous at both conditioning thresholds."""
    cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs"],
        inertial_dynamics_decoupling=True,
        nullspace_control="position",
        nullspace_stiffness=1.0,
        inertia_conditioning_thresholds=(1e-4, 1e-3),
    )
    lower, upper = cfg.inertia_conditioning_thresholds
    offsets = torch.tensor([1 - 1e-5, 1.0, 1 + 1e-5], device=device)
    ratios = torch.cat((lower * offsets, torch.tensor([(lower + upper) / 2], device=device), upper * offsets))
    num_envs = len(ratios)
    controller = OperationalSpaceController(cfg, num_envs=num_envs, device=device)
    pose = torch.zeros(num_envs, 7, device=device)
    pose[:, -1] = 1.0
    controller.set_command(pose)
    jacobian = torch.eye(6, 7, device=device).repeat(num_envs, 1, 1)
    jacobian[:, 0, 0] = ratios.sqrt()
    inputs = dict(
        jacobian_b=jacobian,
        mass_matrix=torch.eye(7, device=device).repeat(num_envs, 1, 1),
        current_ee_pose_b=pose,
        current_ee_vel_b=torch.full((num_envs, 6), -0.05, device=device),
        current_joint_pos=torch.zeros(num_envs, 7, device=device),
        current_joint_vel=torch.zeros(num_envs, 7, device=device),
    )
    task_efforts = controller.compute(**inputs)
    posture_efforts = controller.compute(**inputs, nullspace_joint_pos_target=torch.ones(num_envs, 7, device=device))
    posture_efforts -= task_efforts

    # Damping equals the lower threshold below the band, halves at its midpoint, and vanishes above it.
    reference_damping = torch.tensor([lower, lower / 2, 0.0], device=device)
    response = ratios[[1, 3, 5]] / (ratios[[1, 3, 5]] + reference_damping)
    torch.testing.assert_close(task_efforts[[1, 3, 5], 0], response / ratios[[1, 3, 5]].sqrt())
    torch.testing.assert_close(posture_efforts[[1, 3, 5], 0], 1.0 - response)
    for efforts in (task_efforts, posture_efforts):
        torch.testing.assert_close(efforts[:3, 0], efforts[1, 0].expand(3), atol=2e-4, rtol=1e-5)
        torch.testing.assert_close(efforts[-3:, 0], efforts[5, 0].expand(3), atol=2e-4, rtol=1e-5)
    torch.testing.assert_close(task_efforts[:, 1:6], torch.ones(num_envs, 5, device=device))
    torch.testing.assert_close(posture_efforts[:, 1:6], torch.zeros(num_envs, 5, device=device))
    torch.testing.assert_close(posture_efforts[:, 6], torch.ones(num_envs, device=device))


@pytest.mark.parametrize(
    "thresholds",
    [(0.0, 1e-4), (-1.0, 1e-4), (1e-4, 1e-4), (1e-3, 1e-4), (1e-4, 1.1), (1e-4, float("nan")), (1e-4, float("inf"))],
)
def test_inertial_decoupling_rejects_invalid_conditioning_thresholds(thresholds):
    cfg = OperationalSpaceControllerCfg(target_types=["pose_abs"], inertia_conditioning_thresholds=thresholds)
    with pytest.raises(ValueError, match="conditioning thresholds"):
        OperationalSpaceController(cfg, num_envs=1, device="cpu")


@pytest.mark.parametrize("partial_inertial_decoupling", [False, True])
def test_inertial_decoupling_handles_singular_task_inertia(partial_inertial_decoupling: bool):
    """Inertial decoupling produces finite efforts for rank-deficient Jacobians in a mixed batch."""
    num_envs = 3
    num_joints = 7
    cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs"],
        inertial_dynamics_decoupling=True,
        partial_inertial_dynamics_decoupling=partial_inertial_decoupling,
    )
    controller = OperationalSpaceController(cfg, num_envs=num_envs, device="cpu")

    target_pose = torch.tensor([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).repeat(num_envs, 1)
    controller.set_command(target_pose)

    jacobian = torch.zeros(num_envs, 6, num_joints)
    jacobian[:, :6, :6] = torch.eye(6)
    jacobian[1, 1] = 0.0  # singular translational task-space inertia
    jacobian[2, 5] = 0.0  # singular rotational task-space inertia

    current_pose = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).repeat(num_envs, 1)

    joint_efforts = controller.compute(
        jacobian_b=jacobian,
        current_ee_pose_b=current_pose,
        current_ee_vel_b=torch.zeros(num_envs, 6),
        mass_matrix=torch.eye(num_joints).repeat(num_envs, 1, 1),
    )

    assert torch.isfinite(joint_efforts).all()
    torch.testing.assert_close(
        joint_efforts[:, 0],
        torch.full((num_envs,), 10.0),
    )
    torch.testing.assert_close(
        joint_efforts[:, 1:],
        torch.zeros(num_envs, num_joints - 1),
    )


def _pose_abs_controller(num_envs: int) -> OperationalSpaceController:
    cfg = OperationalSpaceControllerCfg(target_types=["pose_abs"], inertial_dynamics_decoupling=False)
    return OperationalSpaceController(cfg, num_envs=num_envs, device="cpu")


def test_pose_abs_target_quaternion_is_normalized():
    """Scaling an absolute pose quaternion must not change the target orientation or the commanded efforts."""
    num_envs = 2
    unit_quat = torch.tensor([[0.0, 0.0, 0.3826834, 0.9238795]]).repeat(num_envs, 1)  # 45 deg about z
    unit_target = torch.cat([torch.tensor([[0.1, 0.0, 0.0]]).repeat(num_envs, 1), unit_quat], dim=-1)
    scaled_target = unit_target.clone()
    scaled_target[0, 3:7] *= 3.0
    scaled_target[1, 3:7] *= -0.25  # sign flip encodes the same rotation

    current_pose = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).repeat(num_envs, 1)
    jacobian = torch.zeros(num_envs, 6, 7)
    jacobian[:, :6, :6] = torch.eye(6)

    efforts = []
    for target in (unit_target, scaled_target):
        controller = _pose_abs_controller(num_envs)
        controller.set_command(target, current_ee_pose_b=current_pose)
        torch.testing.assert_close(
            torch.linalg.norm(controller.desired_ee_pose_task[:, 3:7], dim=-1), torch.ones(num_envs)
        )
        efforts.append(
            controller.compute(
                jacobian_b=jacobian, current_ee_pose_b=current_pose, current_ee_vel_b=torch.zeros(num_envs, 6)
            )
        )

    torch.testing.assert_close(efforts[0], efforts[1])
    assert torch.isfinite(efforts[1]).all()


def test_pose_abs_degenerate_quaternion_falls_back_to_current_orientation():
    """Zero and non-finite quaternions keep the current orientation, or identity without a current pose."""
    num_envs = 3
    current_quat = torch.tensor([[0.0, 0.7071068, 0.0, 0.7071068]]).repeat(num_envs, 1)  # 90 deg about y
    current_pose = torch.cat([torch.zeros(num_envs, 3), current_quat], dim=-1)
    target = torch.zeros(num_envs, 7)
    target[0, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    target[1, 3:7] = 0.0
    target[2, 3:7] = torch.tensor([float("nan"), 0.0, 0.0, 1.0])

    controller = _pose_abs_controller(num_envs)
    controller.set_command(target, current_ee_pose_b=current_pose)
    torch.testing.assert_close(controller.desired_ee_pose_task[0, 3:7], torch.tensor([0.0, 0.0, 0.0, 1.0]))
    torch.testing.assert_close(controller.desired_ee_pose_task[1:, 3:7], current_quat[1:])

    controller = _pose_abs_controller(num_envs)
    controller.set_command(target)
    torch.testing.assert_close(
        controller.desired_ee_pose_task[1:, 3:7], torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(num_envs - 1, 1)
    )
