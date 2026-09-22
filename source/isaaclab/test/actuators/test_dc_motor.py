# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from isaaclab.actuators import DCMotorCfg

pytestmark = pytest.mark.unit

_DEVICES = [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")),
]
_EFFORT_LIMIT = 60.0
_SATURATION_EFFORT = 100.0
_VELOCITY_LIMIT = 50.0


def _make_dc_motor(
    joint_names: list[str], device: str, saturation_effort=_SATURATION_EFFORT, effort_limit: float = _EFFORT_LIMIT
):
    cfg = DCMotorCfg(
        joint_names_expr=joint_names,
        stiffness=200.0,
        damping=10.0,
        actuator_effort_limit=effort_limit,
        actuator_velocity_limit=_VELOCITY_LIMIT,
        saturation_effort=saturation_effort,
    )
    return cfg.class_type(
        cfg, joint_names=joint_names, joint_ids=list(range(len(joint_names))), num_envs=2, device=device
    )


@pytest.mark.parametrize("device", _DEVICES)
def test_dc_motor_init_minimum(device):
    actuator = _make_dc_motor(["joint_0", "joint_1"], device)

    expected = torch.zeros(2, 2, device=device)
    torch.testing.assert_close(actuator.computed_effort, expected)
    torch.testing.assert_close(actuator.applied_effort, expected)
    torch.testing.assert_close(actuator.actuator_effort_limit, torch.full_like(expected, _EFFORT_LIMIT))
    torch.testing.assert_close(actuator.actuator_velocity_limit, torch.full_like(expected, _VELOCITY_LIMIT))


@pytest.mark.parametrize("device", _DEVICES)
def test_dc_motor_clip(device):
    r"""Test the computation of the dc motor actuator 4 quadrant torque speed curve.

    Each (torque, speed) pair below probes one region of the curve, and is mirrored into the opposite
    quadrant (negated pair) so both halves of the curve are covered:

    0 - fully inside torque speed curve and effort limit (quadrant 1)
    1 - greater than effort limit but under torque-speed curve (quadrant 1)
    2 - greater than effort limit and outside torque-speed curve (quadrant 1)
    3 - less than effort limit but outside torque speed curve (quadrant 1)
    4 - less than effort limit but outside torque speed curve and outside corner velocity (quadrant 4)
    5 - fully inside torque speed curve and effort limit (quadrant 4)
    6 - fully outside torque speed curve and -effort limit (quadrant 4)
    7 - fully inside torque speed curve, outside -effort limit, and inside corner velocity (quadrant 4)
    8 - fully inside torque speed curves, outside -effort limit, and outside corner velocity (quadrant 4)
    9 - less than effort limit but outside torque speed curve and inside corner velocity (quadrant 4)
    e - effort_limit
    s - saturation_effort
    v - actuator_velocity_limit
    c - corner velocity
    \ - torque-speed linear boundary between v and s
    ===========================================================
                            Torque
                             \  (+)
                               \ |
                Q2               s                   Q1
                                 | \        2
        \                        | 1 \
          c ---------------------e-----\
            \                    |       \
              \                  |  0      \ 3
                \                |           \
    (-)-----------v -------------o-------------v --------------(+) Speed
                    \            |               \   9    4
                      \          |    5            \
                        \        |                   \
                          \ -----e---------------------c
                            \    |                      \  6
                Q3            \  |              7    Q4   \
                                \s                          \
                                 |\                       8   \
                                (-) \
    ============================================================
    """
    torque_speed_pairs = torch.tensor(
        [
            [30.0, 10.0],  # 0
            [70.0, 10.0],  # 1
            [80.0, 40.0],  # 2
            [30.0, 40.0],  # 3
            [-20.0, 90.0],  # 4
            [-30.0, 10.0],  # 5
            [-80.0, 110.0],  # 6
            [-80.0, 50.0],  # 7
            [-120.0, 90.0],  # 8
            [-10.0, 70.0],  # 9
        ],
        device=device,
    )
    expected_clipped_effort = torch.tensor(
        [30.0, 60.0, 20.0, 20.0, -60.0, -30.0, -60.0, -60.0, -60.0, -40.0], device=device
    )
    # mirror every point into the opposite quadrant
    torque_speed_pairs = torch.cat([torque_speed_pairs, -torque_speed_pairs])
    expected_clipped_effort = torch.cat([expected_clipped_effort, -expected_clipped_effort])

    # one joint per test point, evaluated for both environments at once
    actuator = _make_dc_motor([f"joint_{i}" for i in range(len(torque_speed_pairs))], device)
    actuator._joint_vel[:] = torque_speed_pairs[:, 1]
    clipped_effort = actuator._clip_effort(torque_speed_pairs[:, 0].expand(2, -1))

    torch.testing.assert_close(clipped_effort, expected_clipped_effort.expand(2, -1))


@pytest.mark.parametrize("device", _DEVICES)
def test_dc_motor_clip_with_per_joint_saturation_effort(device):
    """A per-joint ``saturation_effort`` gives each joint its own torque-speed curve.

    Joints behind different gear reductions belong to one actuator group but do not share a stall
    torque, e.g. the Unitree Go2 calf, which sits behind an extra knee reduction.
    """
    actuator = _make_dc_motor(
        ["hip", "calf"], device, saturation_effort={"hip": 100.0, "calf": 190.0}, effort_limit=100.0
    )

    # at half the no-load speed each joint delivers half of its own stall torque, and the shared
    # effort limit is high enough to clip neither
    actuator._joint_vel[:] = 25.0
    clipped_effort = actuator._clip_effort(torch.full((2, 2), 500.0, device=device))
    torch.testing.assert_close(clipped_effort, torch.tensor([[50.0, 95.0]], device=device).expand(2, -1))
