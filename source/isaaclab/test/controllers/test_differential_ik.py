# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import math

import numpy as np
import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import Articulation
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.test.utils import test_devices

from isaaclab.utils.math import (  # isort:skip
    compute_pose_error,
    matrix_from_quat,
    quat_inv,
    random_yaw_orientation,
    subtract_frame_transforms,
)

##
# Pre-defined configs
##
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG, UR10_CFG  # isort:skip

pytestmark = pytest.mark.integration

_IK_METHODS = ("pinv", "svd", "trans", "dls", "adaptive_dls")
_NUM_ENVS = 4
_NUM_JOINTS = 7
_ID_QUAT = [0.0, 0.0, 0.0, 1.0]  # xyzw identity


##
# Helpers
##


def _make_cfg(ik_method: str = "dls", **kwargs) -> DifferentialIKControllerCfg:
    kwargs.setdefault("command_type", "pose")
    return DifferentialIKControllerCfg(ik_method=ik_method, **kwargs)


def _compute(
    cfg: DifferentialIKControllerCfg,
    ee_pos: torch.Tensor,
    ee_quat: torch.Tensor,
    jacobian: torch.Tensor,
    joint_pos: torch.Tensor,
    command: torch.Tensor,
    joint_limits: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Set the command on a fresh controller and return its joint-position targets."""
    controller = DifferentialIKController(cfg, num_envs=joint_pos.shape[0], device="cpu")
    if joint_limits is not None:
        controller.set_joint_pos_limits(*joint_limits)
    controller.set_command(command, ee_pos, ee_quat)
    return controller.compute(ee_pos, ee_quat, jacobian, joint_pos)


def _quat_xyzw(axis: list[float], angle: float) -> list[float]:
    """Build a unit xyzw quaternion from an axis (need not be unit) and angle [rad]."""
    norm = math.sqrt(sum(a * a for a in axis)) or 1.0
    s = math.sin(angle / 2.0)
    return [axis[0] / norm * s, axis[1] / norm * s, axis[2] / norm * s, math.cos(angle / 2.0)]


def _well_conditioned_jacobian(device: str = "cpu") -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(7)
    jacobian = 0.05 * torch.randn(_NUM_ENVS, 6, _NUM_JOINTS, generator=generator, device=device)
    jacobian[:, :, :6] += torch.eye(6, device=device)
    return jacobian


def _pose_inputs(device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ee_pos = torch.tensor([0.2, -0.1, 0.4], device=device).repeat(_NUM_ENVS, 1)
    ee_quat = torch.tensor(_ID_QUAT, device=device).repeat(_NUM_ENVS, 1)
    command = torch.tensor([0.21, -0.12, 0.43, 0.04997917, 0.0, 0.0, 0.99875027], device=device).repeat(_NUM_ENVS, 1)
    joint_pos = torch.linspace(-0.25, 0.35, _NUM_JOINTS, device=device).repeat(_NUM_ENVS, 1)
    return ee_pos, ee_quat, command, joint_pos


def _reference_pose_task(
    cfg: DifferentialIKControllerCfg, ee_pos: torch.Tensor, ee_quat: torch.Tensor, command: torch.Tensor, jacobian
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble the weighted pose task Jacobian and error for a unit-quaternion absolute command."""
    position_error, rotation_error = compute_pose_error(
        ee_pos, ee_quat, command[:, :3], command[:, 3:7], rot_error_type="axis_angle"
    )
    weight = torch.as_tensor(1.0 if cfg.orientation_weight is None else cfg.orientation_weight)
    return (
        torch.cat((jacobian[:, :3], jacobian[:, 3:] * weight.reshape(1, -1, 1)), dim=1),
        torch.cat((position_error, rotation_error * weight), dim=1),
    )


def _reference_delta_joint_pos(
    cfg: DifferentialIKControllerCfg, task_error: torch.Tensor, task_jacobian: torch.Tensor
) -> torch.Tensor:
    """Reference Torch solve for each IK method."""
    params = cfg.ik_params
    if cfg.ik_method == "pinv":
        return params["k_val"] * torch.bmm(torch.linalg.pinv(task_jacobian), task_error.unsqueeze(-1)).squeeze(-1)
    if cfg.ik_method == "svd":
        u, singular_values, vh = torch.linalg.svd(task_jacobian, full_matrices=False)
        singular_values_inv = torch.where(
            singular_values > params["min_singular_value"],
            singular_values.reciprocal(),
            torch.zeros_like(singular_values),
        )
        jacobian_pinv = vh.mT @ torch.diag_embed(singular_values_inv) @ u.mT
        return params["k_val"] * torch.bmm(jacobian_pinv, task_error.unsqueeze(-1)).squeeze(-1)
    if cfg.ik_method == "trans":
        return params["k_val"] * torch.bmm(task_jacobian.mT, task_error.unsqueeze(-1)).squeeze(-1)
    if cfg.ik_method == "dls":
        lambda_sq = torch.full((task_jacobian.shape[0],), params["lambda_val"] ** 2)
    elif cfg.ik_method == "adaptive_dls":
        sigma_min = torch.linalg.svdvals(task_jacobian)[:, -1]
        ratio = (sigma_min / params["sigma_thresh"]).clamp(max=1.0)
        lambda_sq = params["lambda_min"] ** 2 + (1.0 - ratio**2) * (
            params["lambda_max"] ** 2 - params["lambda_min"] ** 2
        )
    else:
        raise AssertionError(f"Unexpected IK method: {cfg.ik_method}")
    jacobian_t = task_jacobian.mT
    regularization = lambda_sq.view(-1, 1, 1) * torch.eye(task_jacobian.shape[1])
    return torch.bmm(
        jacobian_t,
        torch.linalg.solve(torch.bmm(task_jacobian, jacobian_t) + regularization, task_error.unsqueeze(-1)),
    ).squeeze(-1)


def _reference_joint_limit_correction(
    cfg: DifferentialIKControllerCfg,
    joint_pos: torch.Tensor,
    task_jacobian: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Reference Torch null-space joint-limit-avoidance correction."""
    joint_mid = 0.5 * (lower + upper)
    distance = torch.minimum(joint_pos - lower, upper - joint_pos)
    activation = 1.0 - (distance / cfg.joint_limit_avoidance_margin).clamp(0.0, 1.0)
    center_delta = -cfg.joint_limit_avoidance_gain * activation * (joint_pos - joint_mid)
    position_jacobian = task_jacobian[:, :3]
    null_projector = torch.eye(task_jacobian.shape[2]) - torch.bmm(
        torch.linalg.pinv(position_jacobian), position_jacobian
    )
    return torch.bmm(null_projector, center_delta.unsqueeze(-1)).squeeze(-1)


##
# Configuration and command handling
##


def test_adaptive_dls_default_params():
    """The cfg fills the adaptive_dls defaults when ``ik_params`` is not provided."""
    cfg = _make_cfg("adaptive_dls")
    assert set(cfg.ik_params) == {"lambda_min", "lambda_max", "sigma_thresh"}


def test_cfg_rejects_bad_orientation_weight():
    with pytest.raises(ValueError):
        _make_cfg(orientation_weight=(0.3, 0.3))


def test_cfg_rejects_bad_adaptive_params():
    with pytest.raises(ValueError):
        _make_cfg("adaptive_dls", ik_params={"lambda_min": 0.5, "lambda_max": 0.1, "sigma_thresh": 0.02})


def test_set_command_renormalizes_quat():
    """A non-unit commanded quaternion is stored renormalized in pose (absolute) mode."""
    c = DifferentialIKController(_make_cfg(), num_envs=1, device="cpu")
    raw = torch.tensor([0.2588, 0.0, 0.0, 0.9659])  # xyzw
    cmd = torch.cat([torch.tensor([0.3, -0.1, 0.2]), raw * 3.0]).unsqueeze(0)  # non-unit
    c.set_command(cmd)
    stored = c.ee_quat_des[0]
    assert torch.linalg.norm(stored).item() == pytest.approx(1.0, abs=1e-6)
    torch.testing.assert_close(stored, raw / torch.linalg.norm(raw), atol=1e-6, rtol=0.0)


@pytest.mark.parametrize("bad_quat", [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1e-38]])
def test_set_command_unnormalizable_quat_holds_current_orientation(bad_quat):
    """An unnormalizable commanded quaternion holds that env's current orientation instead of NaN."""
    c = DifferentialIKController(_make_cfg(), num_envs=2, device="cpu")
    good_quat = _quat_xyzw([0.0, 1.0, 0.0], 0.4)
    held_quat = _quat_xyzw([1.0, 0.0, 0.0], 0.5)
    ee_pos = torch.tensor([[0.3, -0.1, 0.2], [0.3, -0.1, 0.2]])
    ee_quat = torch.tensor([_ID_QUAT, held_quat])
    cmd = torch.tensor([[0.3, -0.1, 0.2] + good_quat, [0.3, -0.1, 0.2] + bad_quat])
    c.set_command(cmd, ee_pos, ee_quat)
    torch.testing.assert_close(c.ee_quat_des[0], torch.tensor(good_quat), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(c.ee_quat_des[1], torch.tensor(held_quat), atol=1e-6, rtol=0.0)


def test_set_command_unnormalizable_quat_without_current_orientation_is_identity():
    """Without a current orientation to hold, an unnormalizable command falls back to identity."""
    c = DifferentialIKController(_make_cfg(), num_envs=1, device="cpu")
    cmd = torch.cat([torch.tensor([[0.3, -0.1, 0.2]]), torch.zeros(1, 4)], dim=-1)
    c.set_command(cmd)
    torch.testing.assert_close(c.ee_quat_des, torch.tensor([_ID_QUAT]), atol=1e-6, rtol=0.0)


@pytest.mark.parametrize("scale", [1e-7, 1e-20])
def test_set_command_tiny_normalizable_quat_is_still_normalized(scale):
    """A tiny but normalizable quaternion keeps its meaning instead of taking the fallback."""
    c = DifferentialIKController(_make_cfg(), num_envs=1, device="cpu")
    ee_pos = torch.tensor([[0.3, -0.1, 0.2]])
    ee_quat = torch.tensor([_quat_xyzw([1.0, 0.0, 0.0], 0.5)])  # a fallback that is NOT identity
    cmd = torch.cat([ee_pos, torch.tensor([[0.0, 0.0, 0.0, scale]])], dim=-1)  # scaled identity
    c.set_command(cmd, ee_pos, ee_quat)
    torch.testing.assert_close(c.ee_quat_des, torch.tensor([_ID_QUAT]), atol=1e-5, rtol=0.0)


##
# Solver results: controller == reference
##


@pytest.mark.parametrize("ik_method", _IK_METHODS)
@pytest.mark.parametrize("orientation_weight", [None, (0.4, 0.2, 0.0), (2.0, 1.0, 1.0)])
def test_compute_pose_matches_reference(ik_method: str, orientation_weight):
    """Every solver produces the reference joint-position target for a (weighted) pose task."""
    cfg = _make_cfg(ik_method, orientation_weight=orientation_weight)
    ee_pos, ee_quat, command, joint_pos = _pose_inputs()
    jacobian = _well_conditioned_jacobian()

    actual = _compute(cfg, ee_pos, ee_quat, jacobian, joint_pos, command)

    task_jacobian, task_error = _reference_pose_task(cfg, ee_pos, ee_quat, command, jacobian)
    expected = joint_pos + _reference_delta_joint_pos(cfg, task_error, task_jacobian)
    torch.testing.assert_close(actual, expected, atol=2.0e-4, rtol=2.0e-4)


@pytest.mark.parametrize("ik_method", _IK_METHODS)
def test_compute_position_matches_reference(ik_method: str):
    """Position-only control solves against the three position rows of the Jacobian."""
    cfg = _make_cfg(ik_method, command_type="position")
    ee_pos, ee_quat, _, joint_pos = _pose_inputs()
    command = ee_pos + torch.tensor([0.01, -0.02, 0.03])
    jacobian = _well_conditioned_jacobian()

    actual = _compute(cfg, ee_pos, ee_quat, jacobian, joint_pos, command)

    expected = joint_pos + _reference_delta_joint_pos(cfg, command - ee_pos, jacobian[:, :3])
    torch.testing.assert_close(actual, expected, atol=2.0e-4, rtol=2.0e-4)


def test_pinv_handles_rank_deficient_jacobian():
    """The pseudo-inverse stays finite and matches Torch for a rank-deficient task."""
    cfg = _make_cfg("pinv")
    ee_pos, ee_quat, command, joint_pos = _pose_inputs()
    jacobian = torch.zeros(_NUM_ENVS, 6, _NUM_JOINTS)
    jacobian[:, 0, 0] = 1.0
    jacobian[:, 1, 0] = 2.0
    jacobian[:, 2, 1] = 1.0

    actual = _compute(cfg, ee_pos, ee_quat, jacobian, joint_pos, command)

    task_jacobian, task_error = _reference_pose_task(cfg, ee_pos, ee_quat, command, jacobian)
    expected = joint_pos + torch.bmm(torch.linalg.pinv(task_jacobian), task_error.unsqueeze(-1)).squeeze(-1)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=2.0e-4, rtol=2.0e-4)


def test_compute_quat_convention_xyzw():
    """Commanding the current xyzw pose yields zero motion; a wxyz mis-read would not."""
    q_xyzw = _quat_xyzw([1.0, 0.0, 0.0], math.radians(30.0))
    ee_pos = torch.tensor([[0.3, 0.0, 0.2]])
    ee_quat = torch.tensor([q_xyzw])
    command = torch.cat([ee_pos, ee_quat], dim=-1)
    joint_pos = torch.zeros(1, _NUM_JOINTS)
    cfg = _make_cfg("adaptive_dls", orientation_weight=1.0)

    actual = _compute(cfg, ee_pos, ee_quat, torch.ones(1, 6, _NUM_JOINTS), joint_pos, command)

    torch.testing.assert_close(actual, joint_pos, atol=1e-6, rtol=0.0)


def test_adaptive_dls_damps_singularity():
    """Near a task-Jacobian singularity, the adaptive ramp produces a smaller (more damped) and
    finite step than a fixed ``lambda_min`` solve would."""
    cfg = _make_cfg("adaptive_dls", ik_params={"lambda_min": 0.01, "lambda_max": 0.5, "sigma_thresh": 0.1})
    j_task = torch.zeros(1, 6, _NUM_JOINTS)
    j_task[0, 0, 0] = j_task[0, 1, 1] = j_task[0, 2, 2] = 1.0  # well-conditioned position block
    eps = 1e-3
    j_task[0, 3, 3] = j_task[0, 4, 4] = eps  # near-singular orientation block
    ee_pos = torch.zeros(1, 3)
    ee_quat = torch.tensor([_ID_QUAT])
    command = torch.tensor([[0.0, 0.0, 0.0] + _quat_xyzw([1.0, 1.0, 0.0], math.sqrt(2.0))])

    dq = _compute(cfg, ee_pos, ee_quat, j_task, torch.zeros(1, _NUM_JOINTS), command)

    # reference: fixed lambda_min damped least squares
    err = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 0.0]])
    jt = j_task.transpose(1, 2)
    a_min = torch.bmm(j_task, jt) + (0.01**2) * torch.eye(6)
    dq_min = torch.bmm(jt, torch.linalg.solve(a_min, err.unsqueeze(-1))).squeeze(-1)
    assert torch.isfinite(dq).all()
    assert dq.norm().item() < dq_min.norm().item()


@pytest.mark.parametrize("gain, with_limits", [(0.0, True), (1.0, False)])
def test_joint_limit_avoidance_inactive_when_disabled_or_without_limits(gain: float, with_limits: bool):
    """Avoidance is a no-op when disabled (gain 0) or before joint limits are provided."""
    ee_pos = torch.zeros(1, 3)
    ee_quat = torch.tensor([_ID_QUAT])
    command = torch.tensor([[0.0, 0.0, 0.0] + _ID_QUAT])
    # joints near their limits, so an active avoidance term would move them
    joint_pos = torch.linspace(-0.95, 0.95, _NUM_JOINTS).unsqueeze(0)
    limits = (torch.full((_NUM_JOINTS,), -1.0), torch.full((_NUM_JOINTS,), 1.0)) if with_limits else None
    cfg = _make_cfg("adaptive_dls", joint_limit_avoidance_gain=gain)

    actual = _compute(cfg, ee_pos, ee_quat, torch.ones(1, 6, _NUM_JOINTS), joint_pos, command, limits)

    torch.testing.assert_close(actual, joint_pos)


def test_joint_limit_avoidance_stays_in_position_nullspace():
    """The avoidance correction lies in the null space of the position rows, so it does not
    perturb the commanded end-effector position (``J_pos @ correction ~= 0``)."""
    cfg = _make_cfg("adaptive_dls", joint_limit_avoidance_gain=2.0, joint_limit_avoidance_margin=0.3)
    limits = (torch.full((_NUM_JOINTS,), -1.0), torch.full((_NUM_JOINTS,), 1.0))
    j_task = torch.randn(1, 6, _NUM_JOINTS, generator=torch.Generator().manual_seed(0))
    # joints near their limits -> non-zero center-seeking bias
    joint_pos = torch.tensor([[0.95, -0.9, 0.0, 0.8, -0.85, 0.0, 0.0]])
    ee_pos = torch.zeros(1, 3)
    ee_quat = torch.tensor([_ID_QUAT])
    command = torch.tensor([[0.0, 0.0, 0.0] + _ID_QUAT])

    correction = _compute(cfg, ee_pos, ee_quat, j_task, joint_pos, command, limits) - joint_pos

    assert correction.norm().item() > 0.0  # bias is active
    residual = torch.bmm(j_task[:, :3, :], correction.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-5, rtol=0.0)


def test_orientation_weight_and_joint_limit_avoidance_match_reference():
    """Orientation weighting combined with the null-space correction matches the reference."""
    cfg = _make_cfg("adaptive_dls", orientation_weight=(0.5, 0.25, 0.0), joint_limit_avoidance_gain=0.4)
    ee_pos, ee_quat, command, joint_pos = _pose_inputs()
    jacobian = _well_conditioned_jacobian()
    joint_pos[:, 0] = 0.95
    lower = torch.full((_NUM_JOINTS,), -1.0)
    upper = torch.full((_NUM_JOINTS,), 1.0)

    actual = _compute(cfg, ee_pos, ee_quat, jacobian, joint_pos, command, (lower, upper))

    task_jacobian, task_error = _reference_pose_task(cfg, ee_pos, ee_quat, command, jacobian)
    expected = joint_pos + _reference_delta_joint_pos(cfg, task_error, task_jacobian)
    expected += _reference_joint_limit_correction(cfg, joint_pos, task_jacobian, lower, upper)
    torch.testing.assert_close(actual, expected, atol=5.0e-4, rtol=5.0e-4)


##
# Public boundary contracts
##


@pytest.mark.parametrize("device", test_devices())
def test_joint_limits_accept_float64_cpu_tensors_and_later_updates(device: str):
    """Float64 CPU limits are accepted before the first compute and can be updated afterwards."""
    cfg = _make_cfg("trans", joint_limit_avoidance_gain=0.2)
    controller = DifferentialIKController(cfg, num_envs=_NUM_ENVS, device=device)
    ee_pos, ee_quat, command, joint_pos = _pose_inputs(device)
    joint_pos[:, 0] = 0.95
    jacobian = _well_conditioned_jacobian(device)
    lower = torch.full((_NUM_JOINTS,), -1.0, dtype=torch.float64)
    upper = torch.full((_NUM_JOINTS,), 1.0, dtype=torch.float64)
    controller.set_command(command)

    controller.set_joint_pos_limits(lower, upper)
    near_limit = controller.compute(ee_pos, ee_quat, jacobian, joint_pos)
    # Widening the limits moves joint 0 outside the avoidance margin.
    controller.set_joint_pos_limits(lower - 0.5, upper + 0.5)
    away_from_limit = controller.compute(ee_pos, ee_quat, jacobian, joint_pos)
    assert not torch.allclose(near_limit, away_from_limit)


##
# Simulated convergence
##


@pytest.fixture
def sim():
    """Create a simulation context for testing."""
    # Wait for spawning
    stage = sim_utils.create_new_stage()
    # Constants
    num_envs = 1
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    # TODO: Remove this once we have a better way to handle this.
    sim._app_control_on_stop_handle = None

    # Create a ground plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/GroundPlane", cfg)

    # Create environment clones using Isaac Lab's cloner utilities
    env_prim_paths = [f"/World/envs/env_{i}" for i in range(num_envs)]
    env_fmt = "/World/envs/env_{}"
    env_ids = np.arange(num_envs, dtype=np.int64)
    env_origins, _ = cloner.grid_transforms(num_envs, spacing=2.0)
    # create source prim
    stage.DefinePrim(env_prim_paths[0], "Xform")
    # clone the env xform
    cloner.usd_replicate(stage, [env_fmt.format(0)], [env_fmt], env_ids, positions=env_origins)

    # Define goals for the arm (x, y, z, qx, qy, qz, qw)
    ee_goals_set = [
        [0.5, 0.5, 0.7, 0, 0.707, 0, 0.707],
        [0.5, -0.4, 0.6, 0.707, 0, 0, 0.707],
        [0.5, 0, 0.5, 1.0, 0.0, 0.0, 0.0],
    ]
    ee_pose_b_des_set = torch.tensor(ee_goals_set, device=sim.device)

    yield sim, num_envs, ee_pose_b_des_set

    # Cleanup
    sim.stop()
    sim.clear_instance()


def test_franka_ik_pose_abs(sim):
    """Test IK controller for Franka arm with Franka hand."""
    sim_context, num_envs, ee_pose_b_des_set = sim

    # Create robot instance
    robot_cfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot = Articulation(cfg=robot_cfg)

    # Create IK controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=num_envs, device=sim_context.device)

    # Run the controller and check that it converges to the goal
    _run_ik_controller(
        robot, diff_ik_controller, "panda_hand", ["panda_joint.*"], sim_context, num_envs, ee_pose_b_des_set
    )


def test_ur10_ik_pose_abs(sim):
    """Test IK controller for UR10 arm."""
    sim_context, num_envs, ee_pose_b_des_set = sim

    # Create robot instance
    robot_cfg = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot_cfg.spawn.rigid_props.disable_gravity = True
    robot = Articulation(cfg=robot_cfg)

    # Create IK controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=num_envs, device=sim_context.device)

    # Run the controller and check that it converges to the goal
    _run_ik_controller(robot, diff_ik_controller, "ee_link", [".*"], sim_context, num_envs, ee_pose_b_des_set)


def _run_ik_controller(
    robot: Articulation,
    diff_ik_controller: DifferentialIKController,
    ee_frame_name: str,
    arm_joint_names: list[str],
    sim: sim_utils.SimulationContext,
    num_envs: int,
    ee_pose_b_des_set: torch.Tensor,
):
    """Run the IK controller with the given parameters.

    Args:
        robot (Articulation): The robot to control.
        diff_ik_controller (DifferentialIKController): The differential IK controller.
        ee_frame_name (str): The name of the end-effector frame.
        arm_joint_names (list[str]): The names of the arm joints.
        sim (sim_utils.SimulationContext): The simulation context.
        num_envs (int): The number of environments.
        ee_pose_b_des_set (torch.Tensor): The set of desired end-effector poses.
    """
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    # Play the simulator
    sim.reset()

    # Obtain the frame index of the end-effector
    ee_frame_idx = robot.find_bodies(ee_frame_name)[0][0]
    ee_jacobi_idx = ee_frame_idx - 1
    # Obtain joint indices
    arm_joint_ids = robot.find_joints(arm_joint_names)[0]
    # Update existing buffers
    # Note: We need to update buffers before the first step for the controller.
    robot.update(dt=sim_dt)

    # Track the given command
    current_goal_idx = 0
    # Current goal for the arm
    ee_pose_b_des = torch.zeros(num_envs, diff_ik_controller.action_dim, device=sim.device)
    ee_pose_b_des[:] = ee_pose_b_des_set[current_goal_idx]
    # Compute current pose of the end-effector
    ee_pose_w = robot.data.body_pose_w.torch[:, ee_frame_idx]
    root_pose_w = robot.data.root_pose_w.torch
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )

    # Now we are ready!
    for count in range(1500):
        # reset every 150 steps
        if count % 250 == 0:
            # check that we converged to the goal
            if count > 0:
                pos_error, rot_error = compute_pose_error(
                    ee_pos_b, ee_quat_b, ee_pose_b_des[:, 0:3], ee_pose_b_des[:, 3:7]
                )
                pos_error_norm = torch.linalg.norm(pos_error, dim=-1)
                rot_error_norm = torch.linalg.norm(rot_error, dim=-1)
                # desired error (zer)
                des_error = torch.zeros_like(pos_error_norm)
                # check convergence
                torch.testing.assert_close(pos_error_norm, des_error, rtol=0.0, atol=1e-3)
                torch.testing.assert_close(rot_error_norm, des_error, rtol=0.0, atol=1e-3)
            # reset joint state
            joint_pos = robot.data.default_joint_pos.torch.clone()
            joint_vel = robot.data.default_joint_vel.torch.clone()
            # joint_pos *= sample_uniform(0.9, 1.1, joint_pos.shape, joint_pos.device)
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.set_joint_position_target(joint_pos)
            robot.write_data_to_sim()
            # randomize root state yaw, ik should work regardless base rotation
            root_state = robot.data.root_state_w.torch.clone()
            root_state[:, 3:7] = random_yaw_orientation(num_envs, sim.device)
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            robot.reset()
            # reset actions
            ee_pose_b_des[:] = ee_pose_b_des_set[current_goal_idx]
            joint_pos_des = joint_pos[:, arm_joint_ids].clone()
            # update goal for next iteration
            current_goal_idx = (current_goal_idx + 1) % len(ee_pose_b_des_set)
            # set the controller commands
            diff_ik_controller.reset()
            diff_ik_controller.set_command(ee_pose_b_des)
        else:
            # at reset, the jacobians are not updated to the latest state
            # so we MUST skip the first step
            # obtain quantities from simulation
            jacobian = robot.data.body_link_jacobian_w.torch[:, ee_jacobi_idx, :, arm_joint_ids]
            ee_pose_w = robot.data.body_pose_w.torch[:, ee_frame_idx]
            root_pose_w = robot.data.root_pose_w.torch
            base_rot = root_pose_w[:, 3:7]
            base_rot_matrix = matrix_from_quat(quat_inv(base_rot))
            jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
            jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
            joint_pos = robot.data.joint_pos.torch[:, arm_joint_ids]
            # compute frame in root frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            # compute the joint commands
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        # apply actions
        robot.set_joint_position_target(joint_pos_des, arm_joint_ids)
        robot.write_data_to_sim()
        # perform step
        sim.step(render=False)
        # update buffers
        robot.update(sim_dt)
