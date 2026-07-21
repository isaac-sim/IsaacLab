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

pytestmark = pytest.mark.integration

_NUM_ENVS = 4
_NUM_JOINTS = 7
_IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)


def _make_controller(
    ik_method: str,
    *,
    command_type: str = "pose",
    orientation_weight: float | tuple[float, float, float] | None = None,
    joint_limit_avoidance_gain: float = 0.0,
    device: str = "cpu",
) -> DifferentialIKController:
    cfg = DifferentialIKControllerCfg(
        command_type=command_type,
        use_relative_mode=False,
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
def test_newton_backend_matches_previous_pose_solver(ik_method: str):
    """Every configured solver produces the previous Isaac Lab joint-position target."""
    controller = _make_controller(ik_method)
    ee_pos, ee_quat, command, joint_pos = _pose_inputs("cpu")
    jacobian = _well_conditioned_jacobian("cpu")
    controller.set_command(command)

    task_jacobian, task_error = controller._compute_pose_task(ee_pos, ee_quat, jacobian)
    expected = joint_pos + _previous_delta_joint_pos(controller, task_error, task_jacobian)
    actual = controller.compute(ee_pos, ee_quat, jacobian, joint_pos)

    torch.testing.assert_close(actual, expected, atol=2.0e-4, rtol=2.0e-4)


@pytest.mark.parametrize("ik_method", ["pinv", "svd", "trans", "dls", "adaptive_dls"])
def test_newton_backend_matches_previous_position_solver(ik_method: str):
    """Position-only control passes the matching three-row site Jacobian to every solver."""
    controller = _make_controller(ik_method, command_type="position")
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

    task_jacobian, task_error = controller._compute_pose_task(ee_pos, ee_quat, jacobian)
    expected = joint_pos + torch.bmm(torch.linalg.pinv(task_jacobian), task_error.unsqueeze(-1)).squeeze(-1)
    actual = controller.compute(ee_pos, ee_quat, jacobian, joint_pos)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=2.0e-4, rtol=2.0e-4)


def test_orientation_weight_and_joint_limit_avoidance_match_previous_behavior():
    """Task shaping and the null-space correction remain numerically compatible."""
    controller = _make_controller("adaptive_dls", orientation_weight=(0.5, 0.25, 0.0), joint_limit_avoidance_gain=0.4)
    ee_pos, ee_quat, command, joint_pos = _pose_inputs("cpu")
    jacobian = _well_conditioned_jacobian("cpu")
    joint_pos[:, 0] = 0.95
    lower = torch.full((_NUM_JOINTS,), -1.0)
    upper = torch.full((_NUM_JOINTS,), 1.0)
    controller.set_joint_pos_limits(lower, upper)
    controller.set_command(command)

    task_jacobian, task_error = controller._compute_pose_task(ee_pos, ee_quat, jacobian)
    expected = joint_pos + _previous_delta_joint_pos(controller, task_error, task_jacobian)
    expected += _previous_joint_limit_correction(controller, joint_pos, task_jacobian, lower, upper)
    actual = controller.compute(ee_pos, ee_quat, jacobian, joint_pos)

    torch.testing.assert_close(actual, expected, atol=5.0e-4, rtol=5.0e-4)


def test_command_and_bridge_buffers_keep_stable_addresses():
    """Commands and per-step inputs copy into stable storage instead of rebinding Warp views."""
    controller = _make_controller("dls")
    ee_pos, ee_quat, command, joint_pos = _pose_inputs("cpu")
    jacobian = _well_conditioned_jacobian("cpu")
    target_pos_ptr = controller.ee_pos_des.data_ptr()
    target_quat_ptr = controller.ee_quat_des.data_ptr()
    out = torch.empty_like(joint_pos)

    controller.set_command(command)
    first_output = controller.compute(ee_pos, ee_quat, jacobian, joint_pos, out=out)
    pointers = (
        controller._controller_input.task_error.ptr,
        controller._controller_input.jacobian.ptr,
        controller._controller_input.joint_q.ptr,
        first_output.data_ptr(),
    )

    controller.set_command(command.clone())
    second_output = controller.compute(ee_pos.clone(), ee_quat.clone(), jacobian.clone(), joint_pos.clone(), out=out)
    assert controller.ee_pos_des.data_ptr() == target_pos_ptr
    assert controller.ee_quat_des.data_ptr() == target_quat_ptr
    assert pointers == (
        controller._controller_input.task_error.ptr,
        controller._controller_input.jacobian.ptr,
        controller._controller_input.joint_q.ptr,
        second_output.data_ptr(),
    )


def test_default_output_is_a_snapshot_and_out_is_caller_owned():
    """Default results remain independent while the additive out path returns the caller's buffer."""
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

    out = torch.empty_like(joint_pos)
    result = controller.compute(ee_pos, ee_quat, jacobian, joint_pos, out=out)
    assert result is out
    assert out.data_ptr() != joint_pos.data_ptr()


def test_floating_input_and_output_dtypes_are_preserved_at_public_boundary():
    """The float32 Warp bridge accepts floating inputs and returns the requested public dtype."""
    controller = _make_controller("trans")
    ee_pos, ee_quat, command, joint_pos = (value.to(torch.float64) for value in _pose_inputs("cpu"))
    jacobian = _well_conditioned_jacobian("cpu").to(torch.float64)
    controller.set_command(command)

    result = controller.compute(ee_pos, ee_quat, jacobian, joint_pos)
    assert result.dtype == torch.float64

    out = torch.empty_like(joint_pos, dtype=torch.float16)
    returned = controller.compute(ee_pos, ee_quat, jacobian, joint_pos, out=out)
    assert returned is out
    assert returned.dtype == torch.float16
    torch.testing.assert_close(returned.float(), result.float(), atol=5.0e-4, rtol=5.0e-4)


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

    controller.set_joint_pos_limits(lower - 0.5, upper + 0.5)
    torch.testing.assert_close(controller._joint_pos_lower.cpu(), torch.full((_NUM_JOINTS,), -1.5))
    torch.testing.assert_close(controller._joint_pos_upper.cpu(), torch.full((_NUM_JOINTS,), 1.5))


def test_joint_limit_count_is_checked_when_controller_initializes():
    """Limits supplied before the joint count is known must match the first compute call."""
    controller = _make_controller("trans", joint_limit_avoidance_gain=0.2)
    controller.set_joint_pos_limits(torch.full((_NUM_JOINTS - 1,), -1.0), torch.full((_NUM_JOINTS - 1,), 1.0))
    ee_pos, ee_quat, command, joint_pos = _pose_inputs("cpu")
    controller.set_command(command)
    with pytest.raises(ValueError, match="limits for 7 joints"):
        controller.compute(ee_pos, ee_quat, _well_conditioned_jacobian("cpu"), joint_pos)


@pytest.mark.parametrize(
    ("argument", "value", "error"),
    [
        ("ee_pos", torch.zeros(_NUM_ENVS, 2), ValueError),
        ("jacobian", torch.zeros(_NUM_ENVS, 5, _NUM_JOINTS), ValueError),
        ("joint_pos", torch.zeros(_NUM_ENVS - 1, _NUM_JOINTS), ValueError),
    ],
)
def test_compute_validates_bridge_inputs(argument: str, value: torch.Tensor, error: type[Exception]):
    """Invalid input shapes fail before reaching Warp."""
    controller = _make_controller("dls")
    ee_pos, ee_quat, command, joint_pos = _pose_inputs("cpu")
    jacobian = _well_conditioned_jacobian("cpu")
    controller.set_command(command)
    inputs = {"ee_pos": ee_pos, "ee_quat": ee_quat, "jacobian": jacobian, "joint_pos": joint_pos}
    inputs[argument] = value
    with pytest.raises(error):
        controller.compute(**inputs)


def test_compute_rejects_integral_out_buffer():
    """The additive output-buffer API requires a floating-point destination."""
    controller = _make_controller("dls")
    ee_pos, ee_quat, command, joint_pos = _pose_inputs("cpu")
    controller.set_command(command)
    out = torch.empty_like(joint_pos, dtype=torch.int64)
    with pytest.raises(TypeError, match="out to be a floating-point tensor"):
        controller.compute(ee_pos, ee_quat, _well_conditioned_jacobian("cpu"), joint_pos, out=out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for a device mismatch")
def test_compute_rejects_tensors_on_a_different_device():
    """Per-step tensors must reside on the controller device."""
    controller = _make_controller("trans", device="cuda:0")
    ee_pos, ee_quat, command, joint_pos = _pose_inputs("cuda:0")
    controller.set_command(command)
    with pytest.raises(ValueError, match="Expected ee_pos on cuda:0"):
        controller.compute(ee_pos.cpu(), ee_quat, _well_conditioned_jacobian("cuda:0"), joint_pos)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Warp graph capture")
def test_dls_backend_captures_with_stable_bridge_buffers():
    """The graphable DLS backend captures and replays through the wrapper's stable buffers."""
    device = "cuda:0"
    controller = _make_controller("dls", device=device)
    ee_pos, ee_quat, command, joint_pos = _pose_inputs(device)
    jacobian = _well_conditioned_jacobian(device)
    controller.set_command(command)
    controller.compute(ee_pos, ee_quat, jacobian, joint_pos)
    wp.synchronize_device(device)

    with wp.ScopedCapture(device=device) as capture:
        controller._controller.compute(
            controller._controller_input, controller._controller_output, None, None, controller._time_step
        )

    controller._task_error.zero_()
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    torch.testing.assert_close(controller._joint_pos_des, controller._joint_pos)
