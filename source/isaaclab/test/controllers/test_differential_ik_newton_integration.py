# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parity tests for the Newton-backed differential inverse-kinematics controller."""

import pytest
import torch
import warp as wp

from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.utils.math import compute_pose_error

pytestmark = pytest.mark.integration

_NUM_ENVS = 4
_NUM_JOINTS = 7
_IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)


def _make_controller(
    ik_method: str,
    *,
    command_type: str = "pose",
    orientation_weight: float | tuple[float, float, float] | None = None,
    use_relative_mode: bool = False,
    joint_limit_avoidance_gain: float = 0.0,
    device: str = "cpu",
) -> DifferentialIKController:
    cfg = DifferentialIKControllerCfg(
        use_newton=True,
        command_type=command_type,
        use_relative_mode=use_relative_mode,
        ik_method=ik_method,
        orientation_weight=orientation_weight,
        joint_limit_avoidance_gain=joint_limit_avoidance_gain,
    )
    return DifferentialIKController(cfg, num_envs=_NUM_ENVS, device=device)


def _well_conditioned_jacobian(device: str) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(7)
    jacobian = 0.05 * torch.randn(_NUM_ENVS, 6, _NUM_JOINTS, generator=generator, device=device)
    jacobian[:, :, :6] += torch.eye(6, device=device)
    return jacobian


def _pose_inputs(device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ee_pos = torch.tensor([0.2, -0.1, 0.4], device=device).repeat(_NUM_ENVS, 1)
    ee_quat = torch.tensor(_IDENTITY_QUAT, device=device).repeat(_NUM_ENVS, 1)
    command = torch.tensor([0.21, -0.12, 0.43, 0.04997917, 0.0, 0.0, 0.99875027], device=device).repeat(_NUM_ENVS, 1)
    joint_pos = torch.linspace(-0.25, 0.35, _NUM_JOINTS, device=device).repeat(_NUM_ENVS, 1)
    return ee_pos, ee_quat, command, joint_pos


def _reference_pose_task(controller, ee_pos, ee_quat, jacobian):
    """Assemble the previous weighted task independently of the Newton adapter."""
    position_error, rotation_error = compute_pose_error(
        ee_pos, ee_quat, controller.ee_pos_des, controller.ee_quat_des, rot_error_type="axis_angle"
    )
    weight = controller.cfg.orientation_weight
    weight = torch.as_tensor(1.0 if weight is None else weight, device=jacobian.device)
    return (
        torch.cat((jacobian[:, :3], jacobian[:, 3:] * weight.reshape(1, -1, 1)), dim=1),
        torch.cat((position_error, rotation_error * weight), dim=1),
    )


def _previous_delta_joint_pos(
    controller: DifferentialIKController, task_error: torch.Tensor, task_jacobian: torch.Tensor
) -> torch.Tensor:
    """Evaluate the previous Torch solver implementation as a parity oracle."""
    params = controller.cfg.ik_params
    assert params is not None
    if controller.cfg.ik_method == "pinv":
        return params["k_val"] * torch.bmm(torch.linalg.pinv(task_jacobian), task_error.unsqueeze(-1)).squeeze(-1)
    if controller.cfg.ik_method == "svd":
        u, singular_values, vh = torch.linalg.svd(task_jacobian, full_matrices=False)
        singular_values_inv = torch.where(
            singular_values > params["min_singular_value"],
            singular_values.reciprocal(),
            torch.zeros_like(singular_values),
        )
        jacobian_pinv = vh.mT @ torch.diag_embed(singular_values_inv) @ u.mT
        return params["k_val"] * torch.bmm(jacobian_pinv, task_error.unsqueeze(-1)).squeeze(-1)
    if controller.cfg.ik_method == "trans":
        return params["k_val"] * torch.bmm(task_jacobian.mT, task_error.unsqueeze(-1)).squeeze(-1)
    if controller.cfg.ik_method == "dls":
        jacobian_t = task_jacobian.mT
        regularization = params["lambda_val"] ** 2 * torch.eye(task_jacobian.shape[1])
        return torch.bmm(
            jacobian_t,
            torch.linalg.solve(torch.bmm(task_jacobian, jacobian_t) + regularization, task_error.unsqueeze(-1)),
        ).squeeze(-1)
    if controller.cfg.ik_method == "adaptive_dls":
        sigma_min = torch.linalg.svdvals(task_jacobian)[:, -1]
        ratio = (sigma_min / params["sigma_thresh"]).clamp(max=1.0)
        lambda_sq = params["lambda_min"] ** 2 + (1.0 - ratio**2) * (
            params["lambda_max"] ** 2 - params["lambda_min"] ** 2
        )
        jacobian_t = task_jacobian.mT
        regularization = lambda_sq.view(-1, 1, 1) * torch.eye(task_jacobian.shape[1])
        return torch.bmm(
            jacobian_t,
            torch.linalg.solve(torch.bmm(task_jacobian, jacobian_t) + regularization, task_error.unsqueeze(-1)),
        ).squeeze(-1)
    raise AssertionError(f"Unexpected IK method: {controller.cfg.ik_method}")


def _previous_joint_limit_correction(
    controller: DifferentialIKController,
    joint_pos: torch.Tensor,
    task_jacobian: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the previous Torch null-space correction as a parity oracle."""
    joint_mid = 0.5 * (lower + upper)
    distance = torch.minimum(joint_pos - lower, upper - joint_pos)
    activation = 1.0 - (distance / controller.cfg.joint_limit_avoidance_margin).clamp(0.0, 1.0)
    center_delta = -controller.cfg.joint_limit_avoidance_gain * activation * (joint_pos - joint_mid)
    position_jacobian = task_jacobian[:, :3]
    null_projector = torch.eye(task_jacobian.shape[2]) - torch.bmm(
        torch.linalg.pinv(position_jacobian), position_jacobian
    )
    return torch.bmm(null_projector, center_delta.unsqueeze(-1)).squeeze(-1)


@pytest.mark.parametrize("ik_method", ["pinv", "svd", "trans", "dls", "adaptive_dls"])
@pytest.mark.parametrize("orientation_weight", [None, (0.4, 0.2, 0.0), (2.0, 1.0, 1.0)])
@pytest.mark.parametrize("use_newton", [False, True], ids=["lab", "newton"])
def test_backend_matches_previous_pose_solver(ik_method: str, orientation_weight, use_newton: bool):
    """Every configured solver produces the previous Isaac Lab joint-position target."""
    controller = _make_controller(ik_method, orientation_weight=orientation_weight)
    controller.cfg.use_newton = use_newton
    ee_pos, ee_quat, command, joint_pos = _pose_inputs("cpu")
    jacobian = _well_conditioned_jacobian("cpu")
    controller.set_command(command)

    task_jacobian, task_error = _reference_pose_task(controller, ee_pos, ee_quat, jacobian)
    expected = joint_pos + _previous_delta_joint_pos(controller, task_error, task_jacobian)
    actual = controller.compute(ee_pos, ee_quat, jacobian, joint_pos)

    torch.testing.assert_close(actual, expected, atol=2.0e-4, rtol=2.0e-4)


@pytest.mark.parametrize("ik_method", ["pinv", "svd", "trans", "dls", "adaptive_dls"])
@pytest.mark.parametrize("use_newton", [False, True], ids=["lab", "newton"])
def test_backend_matches_previous_position_solver(ik_method: str, use_newton: bool):
    """Position-only control passes the matching three-row site Jacobian to every solver."""
    controller = _make_controller(ik_method, command_type="position")
    controller.cfg.use_newton = use_newton
    ee_pos, ee_quat, _, joint_pos = _pose_inputs("cpu")
    command = ee_pos + torch.tensor([0.01, -0.02, 0.03])
    jacobian = _well_conditioned_jacobian("cpu")
    controller.set_command(command, ee_quat=ee_quat)

    task_jacobian = jacobian[:, :3]
    task_error = command - ee_pos
    expected = joint_pos + _previous_delta_joint_pos(controller, task_error, task_jacobian)
    actual = controller.compute(ee_pos, ee_quat, jacobian, joint_pos)

    torch.testing.assert_close(actual, expected, atol=2.0e-4, rtol=2.0e-4)


def test_pinv_handles_rank_deficient_jacobian():
    """The Newton Moore-Penrose solver stays finite and matches Torch for a rank-deficient task."""
    controller = _make_controller("pinv")
    ee_pos, ee_quat, command, joint_pos = _pose_inputs("cpu")
    controller.set_command(command)
    jacobian = torch.zeros(_NUM_ENVS, 6, _NUM_JOINTS)
    jacobian[:, 0, 0] = 1.0
    jacobian[:, 1, 0] = 2.0
    jacobian[:, 2, 1] = 1.0

    task_jacobian, task_error = _reference_pose_task(controller, ee_pos, ee_quat, jacobian)
    expected = joint_pos + torch.bmm(torch.linalg.pinv(task_jacobian), task_error.unsqueeze(-1)).squeeze(-1)
    actual = controller.compute(ee_pos, ee_quat, jacobian, joint_pos)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=2.0e-4, rtol=2.0e-4)


@pytest.mark.parametrize("late_limits", [False, True])
def test_orientation_weight_and_joint_limit_avoidance_match_previous_behavior(late_limits):
    """Task shaping and the null-space correction remain numerically compatible."""
    controller = _make_controller("adaptive_dls", orientation_weight=(0.5, 0.25, 0.0), joint_limit_avoidance_gain=0.4)
    ee_pos, ee_quat, command, joint_pos = _pose_inputs("cpu")
    jacobian = _well_conditioned_jacobian("cpu")
    joint_pos[:, 0] = 0.95
    lower = torch.full((_NUM_JOINTS,), -1.0)
    upper = torch.full((_NUM_JOINTS,), 1.0)
    controller.set_command(command)
    if late_limits:
        controller.compute(ee_pos, ee_quat, jacobian, joint_pos)
    controller.set_joint_pos_limits(lower, upper)

    task_jacobian, task_error = _reference_pose_task(controller, ee_pos, ee_quat, jacobian)
    expected = joint_pos + _previous_delta_joint_pos(controller, task_error, task_jacobian)
    expected += _previous_joint_limit_correction(controller, joint_pos, task_jacobian, lower, upper)
    actual = controller.compute(ee_pos, ee_quat, jacobian, joint_pos)

    torch.testing.assert_close(actual, expected, atol=5.0e-4, rtol=5.0e-4)


def test_command_and_bridge_buffers_keep_stable_addresses():
    """Commands and per-step inputs copy into stable storage instead of rebinding Warp views."""
    controller = _make_controller("dls")
    ee_pos, ee_quat, command, joint_pos = _pose_inputs("cpu")
    jacobian = _well_conditioned_jacobian("cpu")

    controller.set_command(command)
    controller.compute(ee_pos, ee_quat, jacobian, joint_pos)
    pointers = (
        controller._controller_input.tool_pose_world.ptr,
        controller._controller_input.jacobian_tool_world.ptr,
        controller._controller_input.joint_q.ptr,
    )

    controller.set_command(command.clone())
    controller.compute(ee_pos.clone(), ee_quat.clone(), jacobian.clone(), joint_pos.clone())
    assert pointers == (
        controller._controller_input.tool_pose_world.ptr,
        controller._controller_input.jacobian_tool_world.ptr,
        controller._controller_input.joint_q.ptr,
    )


def test_output_is_an_independent_snapshot():
    """Returned results remain independent of later calls and caller mutations."""
    controller = _make_controller("trans")
    ee_pos, ee_quat, command, joint_pos = _pose_inputs("cpu")
    jacobian = _well_conditioned_jacobian("cpu")
    controller.set_command(command)

    first = controller.compute(ee_pos, ee_quat, jacobian, joint_pos)
    first_snapshot = first.clone()
    controller.set_command(command + torch.tensor([0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    second = controller.compute(ee_pos, ee_quat, jacobian, joint_pos)
    assert first.data_ptr() != second.data_ptr()
    torch.testing.assert_close(first, first_snapshot)

    assert first.data_ptr() != joint_pos.data_ptr()
    second.zero_()
    torch.testing.assert_close(first, first_snapshot)


@pytest.mark.parametrize("use_relative_mode", [False, True])
def test_floating_input_and_output_dtypes_are_preserved_at_public_boundary(use_relative_mode):
    """The float32 Warp bridge accepts floating inputs and returns the requested public dtype."""
    controller = _make_controller("trans", use_relative_mode=use_relative_mode)
    ee_pos, ee_quat, command, joint_pos = (value.to(torch.float64) for value in _pose_inputs("cpu"))
    jacobian = _well_conditioned_jacobian("cpu").to(torch.float64)
    controller.set_command(command[:, :6] if use_relative_mode else command, ee_pos, ee_quat)

    result = controller.compute(ee_pos, ee_quat, jacobian, joint_pos)
    assert result.dtype == torch.float64


@pytest.mark.parametrize("device", ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else []))
def test_joint_limits_accept_float64_cpu_tensors_before_and_after_initialization(device: str):
    """Joint limits retain the legacy conversion behavior at the float32 Warp boundary."""
    controller = _make_controller("trans", joint_limit_avoidance_gain=0.2, device=device)
    lower = torch.full((_NUM_JOINTS,), -1.0, dtype=torch.float64, device="cpu")
    upper = torch.full((_NUM_JOINTS,), 1.0, dtype=torch.float64, device="cpu")
    controller.set_joint_pos_limits(lower, upper)

    ee_pos, ee_quat, command, joint_pos = _pose_inputs(device)
    jacobian = _well_conditioned_jacobian(device)
    controller.set_command(command)
    controller.compute(ee_pos, ee_quat, jacobian, joint_pos)
    assert controller._joint_pos_lower.dtype == torch.float32
    assert controller._joint_pos_lower.device == torch.device(device)

    backend = controller._controller
    controller.set_joint_pos_limits(lower - 0.5, upper + 0.5)
    controller.compute(ee_pos, ee_quat, jacobian, joint_pos)
    assert controller._controller is backend
    torch.testing.assert_close(controller._joint_pos_lower.cpu(), torch.full((_NUM_JOINTS,), -1.5))
    torch.testing.assert_close(controller._joint_pos_upper.cpu(), torch.full((_NUM_JOINTS,), 1.5))


def test_joint_limit_count_matches_initialized_controller():
    """The setter rejects limits that do not match the fixed joint count."""
    controller = _make_controller("trans", joint_limit_avoidance_gain=0.2)
    ee_pos, ee_quat, command, joint_pos = _pose_inputs("cpu")
    controller.set_command(command)
    controller.compute(ee_pos, ee_quat, _well_conditioned_jacobian("cpu"), joint_pos)
    with pytest.raises(ValueError, match="limits for 7 joints"):
        controller.set_joint_pos_limits(torch.full((_NUM_JOINTS - 1,), -1.0), torch.full((_NUM_JOINTS - 1,), 1.0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Warp graph capture")
@pytest.mark.parametrize("use_relative_mode", [False, True])
def test_dls_backend_captures_with_stable_bridge_buffers(use_relative_mode):
    """The graphable DLS backend captures and replays through the wrapper's stable buffers."""
    device = "cuda:0"
    controller = _make_controller("dls", device=device, use_relative_mode=use_relative_mode)
    ee_pos, ee_quat, command, joint_pos = _pose_inputs(device)
    jacobian = _well_conditioned_jacobian(device)
    controller.set_command(command[:, :6] if use_relative_mode else command, ee_pos, ee_quat)
    controller.compute(ee_pos, ee_quat, jacobian, joint_pos)
    wp.synchronize_device(device)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), wp.ScopedStream(wp.stream_from_torch(stream)):
        with wp.ScopedCapture(device=device) as capture:
            result = controller.compute(ee_pos, ee_quat, jacobian, joint_pos)
        controller.set_command(
            torch.zeros(_NUM_ENVS, 6, device=device) if use_relative_mode else torch.cat((ee_pos, ee_quat), dim=-1),
            ee_pos,
            ee_quat,
        )
        wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    torch.testing.assert_close(result, joint_pos)


@pytest.mark.parametrize("use_newton", [False, True])
def test_joint_count_is_inferred_from_compute(use_newton):
    """Both solvers retain standalone construction and accept changing joint counts."""
    cfg = DifferentialIKControllerCfg(command_type="position", ik_method="dls", use_newton=use_newton)
    controller = DifferentialIKController(cfg, 1, "cpu")
    quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    controller.set_command(torch.ones(1, 3) * 0.1, ee_quat=quat)
    for count in (7, 6, 8):
        actual = controller.compute(torch.zeros(1, 3), quat, torch.eye(6, count).unsqueeze(0), torch.zeros(1, count))
        expected = torch.zeros(1, count)
        expected[:, :3] = 0.1 / (1.0 + cfg.ik_params["lambda_val"] ** 2)
        torch.testing.assert_close(actual, expected)
