# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the operational-space controller's command handling.

These exercise the controller in isolation, without launching the simulator.
"""

import pytest
import torch

from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg

pytestmark = pytest.mark.unit

_NUM_ENVS = 4
_IDENTITY_POSE_B = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
# a non-zero lower limit, so that a parameter below it is distinguishable from an unset one
_STIFFNESS_LIMITS = (10.0, 100.0)
_DAMPING_RATIO_LIMITS = (0.2, 1.0)


def _variable_impedance_cfg(impedance_mode: str) -> OperationalSpaceControllerCfg:
    return OperationalSpaceControllerCfg(
        target_types=["pose_abs"],
        impedance_mode=impedance_mode,
        motion_stiffness_limits_task=_STIFFNESS_LIMITS,
        motion_damping_ratio_limits_task=_DAMPING_RATIO_LIMITS,
    )


def _impedance_command(controller: OperationalSpaceController) -> torch.Tensor:
    """Return one command whose impedance parameters span above, below and on their configured limits."""
    stiffness_lower, stiffness_upper = controller.cfg.motion_stiffness_limits_task
    command = torch.zeros(controller.action_dim)
    command[3:7] = _IDENTITY_POSE_B[3:]
    command[7:13] = torch.tensor(
        [
            2.0 * stiffness_upper,
            0.0,
            stiffness_upper,
            stiffness_lower,
            0.5 * (stiffness_lower + stiffness_upper),
            0.5 * (stiffness_lower + stiffness_upper),
        ]
    )
    if controller.cfg.impedance_mode == "variable":
        ratio_lower, ratio_upper = controller.cfg.motion_damping_ratio_limits_task
        command[13:19] = torch.tensor(
            [
                2.0 * ratio_upper,
                0.0,
                ratio_upper,
                ratio_lower,
                0.5 * (ratio_lower + ratio_upper),
                0.5 * (ratio_lower + ratio_upper),
            ]
        )
    return command


def _expected_gains(controller: OperationalSpaceController) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the stiffness and damping gains the clamped command is expected to produce."""
    stiffness = torch.tensor([100.0, 10.0, 100.0, 10.0, 55.0, 55.0])
    if controller.cfg.impedance_mode == "variable":
        damping_ratio = torch.tensor([1.0, 0.2, 1.0, 0.2, 0.6, 0.6])
    else:
        damping_ratio = torch.tensor(controller.cfg.motion_damping_ratio_task, dtype=torch.float)
    return stiffness, 2.0 * stiffness.sqrt() * damping_ratio


@pytest.mark.parametrize("impedance_mode", ["variable_kp", "variable"])
def test_set_command_clamps_impedance_parameters_to_their_limits(impedance_mode: str) -> None:
    """A parameter above, below or on a limit maps onto that limit, and an in-range one is kept."""
    controller = OperationalSpaceController(_variable_impedance_cfg(impedance_mode), num_envs=_NUM_ENVS, device="cpu")
    command = _impedance_command(controller).repeat(_NUM_ENVS, 1)

    controller.set_command(command, current_ee_pose_b=_IDENTITY_POSE_B.repeat(_NUM_ENVS, 1))

    expected_stiffness, expected_damping = _expected_gains(controller)
    gains = torch.diagonal(controller._motion_p_gains_task, dim1=-2, dim2=-1)
    torch.testing.assert_close(gains, expected_stiffness.repeat(_NUM_ENVS, 1))
    gains = torch.diagonal(controller._motion_d_gains_task, dim1=-2, dim2=-1)
    torch.testing.assert_close(gains, expected_damping.repeat(_NUM_ENVS, 1))


@pytest.mark.parametrize("impedance_mode", ["variable_kp", "variable"])
def test_set_command_accepts_read_only_command_tensors(impedance_mode: str) -> None:
    """The impedance parameters are clamped without writing through the caller's ``command``.

    ``torch.split`` returns views that alias ``command``, so clamping in place raises for a command that is
    broadcast across environments, or that carries ``requires_grad``.
    """
    controller = OperationalSpaceController(_variable_impedance_cfg(impedance_mode), num_envs=_NUM_ENVS, device="cpu")
    command = _impedance_command(controller)
    current_ee_pose_b = _IDENTITY_POSE_B.repeat(_NUM_ENVS, 1)

    # one command shared by every environment: the caller's tensor is not writable
    controller.set_command(command.expand(_NUM_ENVS, -1), current_ee_pose_b=current_ee_pose_b)
    # a command that carries a gradient: its views are not writable either
    controller.set_command(command.repeat(_NUM_ENVS, 1).requires_grad_(True), current_ee_pose_b=current_ee_pose_b)


@pytest.mark.parametrize("impedance_mode", ["variable_kp", "variable"])
def test_set_command_leaves_a_writable_command_untouched(impedance_mode: str) -> None:
    """``set_command`` reads its ``command``, so a caller can keep using the buffer afterwards."""
    controller = OperationalSpaceController(_variable_impedance_cfg(impedance_mode), num_envs=_NUM_ENVS, device="cpu")
    command = _impedance_command(controller).repeat(_NUM_ENVS, 1)
    command_before = command.clone()

    controller.set_command(command, current_ee_pose_b=_IDENTITY_POSE_B.repeat(_NUM_ENVS, 1))

    torch.testing.assert_close(command, command_before)
