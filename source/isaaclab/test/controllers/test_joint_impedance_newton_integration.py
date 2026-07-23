# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parity tests for the Newton-backed joint impedance controller."""

import pytest
import torch

# The controller delegates to ``newton.controllers``; skip if that in-core module is unavailable.
pytest.importorskip("newton.controllers")

from isaaclab.controllers.joint_impedance import JointImpedanceController
from isaaclab.controllers.joint_impedance_cfg import JointImpedanceControllerCfg

pytestmark = pytest.mark.integration

_NUM_ROBOTS = 4
_NUM_DOF = 7


def _make_cfg(mode: str, command_type: str, inertial: bool, gravity: bool) -> JointImpedanceControllerCfg:
    cfg = JointImpedanceControllerCfg()
    cfg.impedance_mode = mode
    cfg.command_type = command_type
    cfg.inertial_compensation = inertial
    cfg.gravity_compensation = gravity
    cfg.stiffness = 50.0
    cfg.damping_ratio = 1.0
    cfg.dof_pos_offset = [0.1] * _NUM_DOF
    return cfg


def _reference_torques(
    cfg: JointImpedanceControllerCfg,
    p_gains: torch.Tensor,
    d_gains: torch.Tensor,
    limits: torch.Tensor,
    target: torch.Tensor,
    offset: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_vel: torch.Tensor,
    mass_matrix: torch.Tensor,
    gravity: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the previous Torch impedance law as a parity oracle."""
    if cfg.command_type == "p_abs":
        desired = target + offset
    else:
        desired = target + dof_pos
    desired = desired.clip(min=limits[..., 0], max=limits[..., 1])
    acc = p_gains * (desired - dof_pos) + d_gains * (-dof_vel)
    if cfg.inertial_compensation:
        torques = torch.einsum("nij,nj->ni", mass_matrix, acc)
    else:
        torques = acc
    if cfg.gravity_compensation:
        torques = torques + gravity
    return torques


@pytest.mark.parametrize("mode", ["fixed", "variable_kp", "variable"])
@pytest.mark.parametrize("command_type", ["p_abs", "p_rel"])
@pytest.mark.parametrize("inertial", [False, True])
@pytest.mark.parametrize("gravity", [False, True])
def test_newton_backend_matches_previous_impedance_law(
    mode: str, command_type: str, inertial: bool, gravity: bool
) -> None:
    """The Newton-backed controller reproduces the previous Torch impedance law."""
    device = "cpu"
    generator = torch.Generator(device=device).manual_seed(0)
    cfg = _make_cfg(mode, command_type, inertial, gravity)
    limits = torch.stack(
        [
            -3.0 * torch.ones(_NUM_ROBOTS, _NUM_DOF, device=device),
            3.0 * torch.ones(_NUM_ROBOTS, _NUM_DOF, device=device),
        ],
        dim=-1,
    )
    controller = JointImpedanceController(cfg, _NUM_ROBOTS, limits, device)

    dof_pos = 0.3 * torch.randn(_NUM_ROBOTS, _NUM_DOF, generator=generator, device=device)
    dof_vel = 0.2 * torch.randn(_NUM_ROBOTS, _NUM_DOF, generator=generator, device=device)
    jacobian = torch.randn(_NUM_ROBOTS, _NUM_DOF, _NUM_DOF, generator=generator, device=device)
    mass_matrix = torch.einsum("nij,nkj->nik", jacobian, jacobian) + torch.eye(_NUM_DOF, device=device)  # SPD
    gravity_vec = 0.5 * torch.randn(_NUM_ROBOTS, _NUM_DOF, generator=generator, device=device)

    target_pos = 0.4 * torch.randn(_NUM_ROBOTS, _NUM_DOF, generator=generator, device=device)
    if mode == "fixed":
        command = target_pos
    elif mode == "variable_kp":
        stiffness_cmd = torch.rand(_NUM_ROBOTS, _NUM_DOF, generator=generator, device=device) * 100 + 10
        command = torch.cat([target_pos, stiffness_cmd], dim=-1)
    else:  # variable
        stiffness_cmd = torch.rand(_NUM_ROBOTS, _NUM_DOF, generator=generator, device=device) * 100 + 10
        damping_cmd = torch.rand(_NUM_ROBOTS, _NUM_DOF, generator=generator, device=device) * 2
        command = torch.cat([target_pos, stiffness_cmd, damping_cmd], dim=-1)
    controller.set_command(command)

    torques = controller.compute(
        dof_pos, dof_vel, mass_matrix if inertial else None, gravity_vec if gravity else None
    ).clone()
    reference = _reference_torques(
        cfg,
        controller._p_gains,
        controller._d_gains,
        limits,
        controller._dof_pos_target,
        controller._dof_pos_offset,
        dof_pos,
        dof_vel,
        mass_matrix,
        gravity_vec,
    )
    torch.testing.assert_close(torques, reference, atol=1e-4, rtol=1e-4)
