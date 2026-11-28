# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the joint impedance controller command handling.

These exercise the controller in isolation, without launching the simulator.
"""

import pytest
import torch

from isaaclab.controllers.joint_impedance import JointImpedanceController
from isaaclab.controllers.joint_impedance_cfg import JointImpedanceControllerCfg

pytestmark = [pytest.mark.unit, pytest.mark.kitless]

_NUM_ROBOTS = 4
_NUM_DOF = 7
_TARGET = 0.1
# a non-zero lower limit, so that a parameter below it is distinguishable from an unset one
_STIFFNESS_LIMITS = (10.0, 300.0)
_DAMPING_RATIO_LIMITS = (0.2, 1.0)


@pytest.mark.parametrize("impedance_mode", ["variable_kp", "variable"])
def test_set_command_clamps_impedance_parameters_to_their_limits(impedance_mode: str) -> None:
    """A parameter above, below or on a limit maps onto that limit, and an in-range one is kept.

    The clamped gains are observed through the commanded torques, and the caller's command is asserted
    unchanged after the same call.
    """
    controller = _controller(impedance_mode)
    command = _command(controller).repeat(_NUM_ROBOTS, 1)
    command_before = command.clone()

    controller.set_command(command)

    # the caller keeps its buffer: set_command reads the command, it does not write into it
    torch.testing.assert_close(command, command_before)
    # at zero velocity the torque is the position term alone, so it carries the clamped gains
    torque = controller.compute(torch.zeros(_NUM_ROBOTS, _NUM_DOF), torch.zeros(_NUM_ROBOTS, _NUM_DOF))
    torch.testing.assert_close(torque, _expected_torque(impedance_mode).repeat(_NUM_ROBOTS, 1))


@pytest.mark.parametrize("impedance_mode", ["variable_kp", "variable"])
def test_set_command_accepts_overlapping_and_gradient_commands(impedance_mode: str) -> None:
    """set_command accepts commands whose storage it cannot write through.

    ``torch.tensor_split`` returns views that alias the command, so an in-place clamp fails for a command
    whose elements share one storage slot (``expand``) or that carries ``requires_grad``.
    """
    controller = _controller(impedance_mode)
    command = _command(controller)

    # one command broadcast across robots: every row aliases the same storage
    controller.set_command(command.expand(_NUM_ROBOTS, -1))
    # a command that carries a gradient: a leaf view cannot be written through
    controller.set_command(command.repeat(_NUM_ROBOTS, 1).requires_grad_(True))


def _controller(impedance_mode: str) -> JointImpedanceController:
    cfg = JointImpedanceControllerCfg(
        impedance_mode=impedance_mode,
        command_type="p_abs",
        stiffness=50.0,
        dof_pos_offset=[0.1] * _NUM_DOF,
        stiffness_limits=_STIFFNESS_LIMITS,
        damping_ratio_limits=_DAMPING_RATIO_LIMITS,
    )
    # symmetric limits wide enough that the clamp on the joint positions is not exercised
    dof_pos_limits = torch.stack(
        [torch.full((_NUM_ROBOTS, _NUM_DOF), -0.3), torch.full((_NUM_ROBOTS, _NUM_DOF), 0.3)], dim=-1
    )
    return JointImpedanceController(cfg, _NUM_ROBOTS, dof_pos_limits, "cpu")


def _command(controller: JointImpedanceController) -> torch.Tensor:
    """Return one command whose impedance parameters span above, below and on their configured limits."""
    stiffness_lower, stiffness_upper = controller.cfg.stiffness_limits
    command = torch.zeros(controller.num_actions)
    command[: _NUM_DOF] = _TARGET
    mid = 0.5 * (stiffness_lower + stiffness_upper)
    command[_NUM_DOF : _NUM_DOF * 2] = torch.tensor(
        [2.0 * stiffness_upper, 0.0, stiffness_upper, stiffness_lower, mid, mid, mid]
    )
    if controller.cfg.impedance_mode == "variable":
        ratio_lower, ratio_upper = controller.cfg.damping_ratio_limits
        ratio_mid = 0.5 * (ratio_lower + ratio_upper)
        command[_NUM_DOF * 2 :] = torch.tensor(
            [2.0 * ratio_upper, 0.0, ratio_upper, ratio_lower, ratio_mid, ratio_mid, ratio_mid]
        )
    return command


def _expected_torque(impedance_mode: str) -> torch.Tensor:
    """Return the torque the clamped command produces at zero joint position and velocity."""
    # the clamped values, written out rather than read back from the controller
    stiffness = torch.tensor([300.0, 10.0, 300.0, 10.0, 155.0, 155.0, 155.0])
    if impedance_mode == "variable":
        damping_ratio = torch.tensor([1.0, 0.2, 1.0, 0.2, 0.6, 0.6, 0.6])
    else:
        damping_ratio = torch.ones(_NUM_DOF)
    damping = 2.0 * stiffness.sqrt() * damping_ratio
    # p_abs: the target is the command plus the offset; at zero velocity only the position term acts
    pos_error = _TARGET + 0.1
    return stiffness * pos_error
