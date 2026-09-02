# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for singularity handling in the operational-space controller.

The task-space inertia ``J M^-1 J^T`` loses rank at kinematic singularities. Without
regularization its inverse produces unbounded command forces, which shows up as violent motion and
eventually NaN. Non-redundant (6-DoF) arms are most exposed because they cannot reconfigure through
a singularity in the null space.

These tests drive the controller with a synthetic near-singular Jacobian so the failure is
deterministic and does not require the simulator.
"""

import pytest
import torch

from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg

pytestmark = pytest.mark.unit

NUM_ENVS = 2
NUM_DOF = 6
"""A 6-DoF arm: the task Jacobian is square, so rank loss cannot be absorbed by redundancy."""

SMALLEST_SINGULAR_VALUE = 1e-3
"""Smallest singular value of the synthetic Jacobian, i.e. how close it sits to a singularity.

Chosen so the resulting task inertia stays well clear of ``float32`` precision limits: below roughly
1e-4 the ``sigma_min**2`` term underflows against the unit-scale terms and the undamped inverse
returns numerical noise whose magnitude is neither stable nor comparable across devices.
"""


def _orthonormal(dim: int, seed: int, device: str) -> torch.Tensor:
    """Build a deterministic orthonormal matrix of shape (``dim``, ``dim``)."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randn(dim, dim, generator=generator)
    q, _ = torch.linalg.qr(matrix)
    return q.to(device)


def _near_singular_jacobian(device: str, sigma_min: float = SMALLEST_SINGULAR_VALUE) -> torch.Tensor:
    """Build a batched 6x6 Jacobian whose smallest singular value is ``sigma_min``.

    Constructed through its SVD so the conditioning is exact rather than incidental.
    """
    u = _orthonormal(6, seed=0, device=device)
    v = _orthonormal(NUM_DOF, seed=1, device=device)
    singular_values = torch.ones(6, device=device)
    singular_values[-1] = sigma_min
    jacobian = u @ torch.diag(singular_values) @ v.mT
    return jacobian.unsqueeze(0).repeat(NUM_ENVS, 1, 1)


def _make_controller(device: str, **cfg_kwargs) -> OperationalSpaceController:
    """Create a pose-tracking controller with inertial decoupling enabled."""
    cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs"],
        impedance_mode="fixed",
        inertial_dynamics_decoupling=True,
        partial_inertial_dynamics_decoupling=False,
        gravity_compensation=False,
        motion_stiffness_task=100.0,
        motion_damping_ratio_task=1.0,
        **cfg_kwargs,
    )
    return OperationalSpaceController(cfg, num_envs=NUM_ENVS, device=device)


def _compute_efforts(
    osc: OperationalSpaceController, jacobian_b: torch.Tensor, device: str, mass_scale: float = 1.0
) -> torch.Tensor:
    """Command a fixed pose offset and return the resulting joint efforts.

    ``mass_scale`` stands in for a heavier or lighter robot: it scales the joint-space inertia
    without changing the kinematics, so only the dynamic scale of ``J M^-1 J^T`` moves.
    """
    # identity orientation in (x, y, z, w) order
    current_ee_pose_b = torch.tensor([[0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]], device=device).repeat(NUM_ENVS, 1)
    current_ee_vel_b = torch.zeros(NUM_ENVS, 6, device=device)
    # a well-conditioned joint-space inertia, so the only ill-conditioning comes from the Jacobian
    mass_matrix = mass_scale * torch.eye(NUM_DOF, device=device).unsqueeze(0).repeat(NUM_ENVS, 1, 1)

    # target is offset by 0.1 m in x, giving a non-zero task-space error to act on
    command = current_ee_pose_b.clone()
    command[:, 0] += 0.1
    osc.set_command(command=command, current_ee_pose_b=current_ee_pose_b)

    return osc.compute(
        jacobian_b=jacobian_b,
        current_ee_pose_b=current_ee_pose_b,
        current_ee_vel_b=current_ee_vel_b,
        mass_matrix=mass_matrix,
    )


def _peak_effort(
    device: str,
    method: str,
    sigma_min: float = SMALLEST_SINGULAR_VALUE,
    mass_scale: float = 1.0,
    **params,
) -> float:
    """Return the largest joint effort the controller commands for the given method."""
    kwargs = {"inertial_decoupling_method": method}
    if params:
        kwargs["inertial_decoupling_params"] = params
    osc = _make_controller(device, **kwargs)
    efforts = _compute_efforts(osc, _near_singular_jacobian(device, sigma_min=sigma_min), device, mass_scale)
    assert torch.isfinite(efforts).all(), f"'{method}' produced non-finite efforts"
    return efforts.abs().max().item()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_clamp_bounds_efforts_near_singularity(device):
    """The clamp keeps efforts finite and far below the unregularized inverse."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    undamped = _peak_effort(device, "inv")
    clamped = _peak_effort(device, "cond_clamp", max_condition_number=1e3)

    assert clamped * 20.0 < undamped, (
        f"expected the clamp to bound efforts near a singularity, got {clamped:.3e} against"
        f" {undamped:.3e} for the unregularized inverse"
    )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_undamped_efforts_scale_with_singularity_proximity(device):
    """The defect itself: without the clamp, effort grows as the arm nears a singularity.

    The clamp holds its output essentially flat over the same range, which is the contrast the fix
    provides.
    """
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    far, near = 1e-2, 1e-3

    undamped_far = _peak_effort(device, "inv", sigma_min=far)
    undamped_near = _peak_effort(device, "inv", sigma_min=near)
    assert undamped_near > 5.0 * undamped_far, (
        "expected the unregularized inverse to amplify as the singularity is approached, got"
        f" {undamped_far:.3e} -> {undamped_near:.3e}"
    )

    clamped_far = _peak_effort(device, "cond_clamp", sigma_min=far, max_condition_number=1e3)
    clamped_near = _peak_effort(device, "cond_clamp", sigma_min=near, max_condition_number=1e3)
    assert clamped_near < 1.5 * clamped_far, (
        "expected the clamp to stay bounded as the singularity is approached, got"
        f" {clamped_far:.3e} -> {clamped_near:.3e}"
    )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_clamp_strength_is_independent_of_robot_inertia(device):
    """A given bound must act identically on a heavy arm and a light one.

    ``J M^-1 J^T`` scales inversely with the robot's inertia, so any criterion expressed as an
    absolute magnitude acts far more aggressively on a heavy arm. Bounding a ratio does not.
    """
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    light, heavy = 1.0, 100.0

    light_ratio = _peak_effort(device, "cond_clamp", mass_scale=light, max_condition_number=1e3) / _peak_effort(
        device, "inv", mass_scale=light
    )
    heavy_ratio = _peak_effort(device, "cond_clamp", mass_scale=heavy, max_condition_number=1e3) / _peak_effort(
        device, "inv", mass_scale=heavy
    )

    torch.testing.assert_close(
        torch.tensor(heavy_ratio),
        torch.tensor(light_ratio),
        rtol=0.1,
        atol=1e-3,
        msg=(
            f"clamped a {heavy}x heavier arm to {heavy_ratio:.4f} of its undamped effort but a light arm to"
            f" {light_ratio:.4f}; the bound must not depend on robot inertia"
        ),
    )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_clamp_is_inert_on_well_conditioned_configurations(device):
    """Away from singularities the clamp must reproduce the plain inverse exactly.

    This is the property that makes it safe to enable by default on a working setup: it intervenes
    only once the configured condition bound is actually exceeded.
    """
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    # all singular values equal to 1, so the task inertia is perfectly conditioned
    jacobian_b = _near_singular_jacobian(device, sigma_min=1.0)

    undamped = _compute_efforts(_make_controller(device, inertial_decoupling_method="inv"), jacobian_b, device)
    clamped = _compute_efforts(
        _make_controller(
            device,
            inertial_decoupling_method="cond_clamp",
            inertial_decoupling_params={"max_condition_number": 1e6},
        ),
        jacobian_b,
        device,
    )

    torch.testing.assert_close(clamped, undamped, rtol=1e-4, atol=1e-5)


def test_default_method_is_the_unregularized_inverse():
    """The default must preserve pre-existing behavior so upgrades are not silently altered."""
    cfg = OperationalSpaceControllerCfg(target_types=["pose_abs"])
    assert cfg.inertial_decoupling_method == "inv"
    assert cfg.inertial_decoupling_params == {}


def test_unknown_method_is_rejected():
    """An unrecognized method must fail at config time rather than mid-rollout."""
    with pytest.raises(ValueError, match="Unsupported inertial decoupling method"):
        OperationalSpaceControllerCfg(target_types=["pose_abs"], inertial_decoupling_method="dls")


def test_method_defaults_are_filled_and_overridable():
    """Unspecified parameters fall back to documented defaults; provided ones win."""
    cfg = OperationalSpaceControllerCfg(target_types=["pose_abs"], inertial_decoupling_method="cond_clamp")
    assert cfg.inertial_decoupling_params == {"max_condition_number": 1e6}

    cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs"],
        inertial_decoupling_method="cond_clamp",
        inertial_decoupling_params={"max_condition_number": 1e4},
    )
    assert cfg.inertial_decoupling_params["max_condition_number"] == 1e4


@pytest.mark.parametrize("bound", [1.0, 0.0, -5.0])
def test_invalid_condition_bound_is_rejected(bound):
    """A bound of 1 or below cannot be satisfied by any non-degenerate matrix."""
    with pytest.raises(ValueError, match="max_condition_number must be > 1"):
        OperationalSpaceControllerCfg(
            target_types=["pose_abs"],
            inertial_decoupling_method="cond_clamp",
            inertial_decoupling_params={"max_condition_number": bound},
        )
