# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the joint impedance controller against an independent impedance-law reference."""

import math

import pytest
import torch

from isaaclab.controllers.joint_impedance import JointImpedanceController
from isaaclab.controllers.joint_impedance_cfg import JointImpedanceControllerCfg

pytestmark = pytest.mark.integration

_NUM_ROBOTS = 4
_NUM_DOF = 7


# The branches are independent and additive, so each mode rotates across every value of the other axes.
@pytest.mark.parametrize(
    "mode, command_type, inertial, gravity, dtype",
    [
        ("fixed", "p_abs", False, False, torch.float32),
        ("fixed", "p_rel", True, True, torch.float64),
        ("variable_kp", "p_rel", False, True, torch.float32),
        ("variable_kp", "p_abs", True, False, torch.float64),
        ("variable", "p_abs", True, True, torch.float32),
        ("variable", "p_rel", False, False, torch.float64),
    ],
)
def test_compute_matches_impedance_law(
    mode: str, command_type: str, inertial: bool, gravity: bool, dtype: torch.dtype
) -> None:
    """The controller matches an independent impedance-law reference."""
    device = "cpu"
    generator = torch.Generator(device=device).manual_seed(0)
    stiffness, damping_ratio, offset = 50.0, 1.0, 0.1
    cfg = JointImpedanceControllerCfg(
        impedance_mode=mode,
        command_type=command_type,
        inertial_compensation=inertial,
        gravity_compensation=gravity,
        stiffness=stiffness,
        damping_ratio=damping_ratio,
        dof_pos_offset=[offset] * _NUM_DOF,
    )
    # tight enough that some desired positions are clipped to the joint limits
    limits = torch.stack(
        [
            -0.3 * torch.ones(_NUM_ROBOTS, _NUM_DOF, device=device),
            0.3 * torch.ones(_NUM_ROBOTS, _NUM_DOF, device=device),
        ],
        dim=-1,
    )

    dof_pos = 0.3 * torch.randn(_NUM_ROBOTS, _NUM_DOF, generator=generator, device=device)
    dof_vel = 0.2 * torch.randn(_NUM_ROBOTS, _NUM_DOF, generator=generator, device=device)
    jacobian = torch.randn(_NUM_ROBOTS, _NUM_DOF, _NUM_DOF, generator=generator, device=device)
    mass_matrix = torch.einsum("nij,nkj->nik", jacobian, jacobian) + torch.eye(_NUM_DOF, device=device)  # SPD
    gravity_vec = 0.5 * torch.randn(_NUM_ROBOTS, _NUM_DOF, generator=generator, device=device)

    target_pos = 0.4 * torch.randn(_NUM_ROBOTS, _NUM_DOF, generator=generator, device=device)
    p_gains = torch.full_like(target_pos, stiffness)
    d_gains = torch.full_like(target_pos, 2.0 * math.sqrt(stiffness) * damping_ratio)
    if mode == "fixed":
        command = target_pos
    elif mode == "variable_kp":
        p_gains = torch.rand(_NUM_ROBOTS, _NUM_DOF, generator=generator, device=device) * 100 + 10
        d_gains = 2.0 * p_gains.sqrt()
        command = torch.cat([target_pos, p_gains], dim=-1)
    else:  # variable
        p_gains = torch.rand(_NUM_ROBOTS, _NUM_DOF, generator=generator, device=device) * 100 + 10
        damping_cmd = torch.rand(_NUM_ROBOTS, _NUM_DOF, generator=generator, device=device) * 2
        d_gains = 2.0 * p_gains.sqrt() * damping_cmd
        command = torch.cat([target_pos, p_gains, damping_cmd], dim=-1)

    dof_pos, dof_vel, mass_matrix, gravity_vec = (
        value.to(dtype=dtype) for value in (dof_pos, dof_vel, mass_matrix, gravity_vec)
    )

    # reference impedance law
    desired = target_pos + (offset if command_type == "p_abs" else dof_pos)
    desired = desired.clip(min=limits[..., 0], max=limits[..., 1])
    expected = p_gains * (desired - dof_pos) + d_gains * (-dof_vel)
    if inertial:
        expected = torch.einsum("nij,nj->ni", mass_matrix, expected)
    if gravity:
        expected = expected + gravity_vec

    controller = JointImpedanceController(cfg, _NUM_ROBOTS, limits, device)
    controller.set_command(command)
    torques = controller.compute(dof_pos, dof_vel, mass_matrix if inertial else None, gravity_vec if gravity else None)
    assert torques.dtype == dtype
    torch.testing.assert_close(torques, expected, atol=1e-4, rtol=1e-4)
