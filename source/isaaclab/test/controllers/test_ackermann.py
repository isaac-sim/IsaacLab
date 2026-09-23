# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import pytest
import torch
import warp as wp

from isaaclab.controllers import AckermannController, AckermannControllerCfg


def _controller_cfg(**kwargs) -> AckermannControllerCfg:
    values = {
        "wheel_radius": 0.25,
        "wheel_base": 1.6,
        "track_width": 1.0,
        "non_steerable_wheel_offsets": (0.5, -0.5),
        "max_linear_speed": 3.0,
        "max_steering_angle": 0.6,
    }
    values.update(kwargs)
    return AckermannControllerCfg(**values)


def test_straight_command_produces_equal_wheel_speeds() -> None:
    controller = AckermannController(_controller_cfg(), num_envs=2, device="cpu")

    steering_targets, wheel_targets = controller.compute(torch.tensor([[1.0, 0.0], [-2.0, 0.0]]))

    torch.testing.assert_close(steering_targets, torch.zeros((2, 2)))
    torch.testing.assert_close(wheel_targets[0], torch.full((4,), 4.0))
    torch.testing.assert_close(wheel_targets[1], torch.full((4,), -8.0))


def test_turn_command_uses_body_forward_speed_convention() -> None:
    cfg = _controller_cfg()
    controller = AckermannController(cfg, num_envs=1, device="cpu")
    speed = 2.0
    steering_angle = 0.3

    steering_targets, wheel_targets = controller.compute(torch.tensor([[speed, steering_angle]]))

    yaw_rate = speed * math.tan(steering_angle) / cfg.wheel_base
    left_longitudinal_speed = speed - 0.5 * yaw_rate * cfg.track_width
    right_longitudinal_speed = speed + 0.5 * yaw_rate * cfg.track_width
    lateral_speed = yaw_rate * cfg.wheel_base
    expected_steering = torch.tensor(
        [
            math.atan2(lateral_speed, left_longitudinal_speed),
            math.atan2(lateral_speed, right_longitudinal_speed),
        ]
    )
    expected_wheel_speeds = torch.tensor(
        [
            math.hypot(left_longitudinal_speed, lateral_speed) / cfg.wheel_radius,
            math.hypot(right_longitudinal_speed, lateral_speed) / cfg.wheel_radius,
            (speed - yaw_rate * cfg.non_steerable_wheel_offsets[0]) / cfg.wheel_radius,
            (speed - yaw_rate * cfg.non_steerable_wheel_offsets[1]) / cfg.wheel_radius,
        ]
    )
    torch.testing.assert_close(steering_targets[0], expected_steering, atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(wheel_targets[0], expected_wheel_speeds, atol=1.0e-6, rtol=1.0e-6)


def test_reverse_command_preserves_steering_and_reverses_wheels() -> None:
    controller = AckermannController(_controller_cfg(), num_envs=1, device="cpu")

    forward_steering, forward_wheels = controller.compute(torch.tensor([[1.0, 0.25]]))
    forward_steering = forward_steering.clone()
    forward_wheels = forward_wheels.clone()
    reverse_steering, reverse_wheels = controller.compute(torch.tensor([[-1.0, 0.25]]))

    torch.testing.assert_close(reverse_steering, forward_steering)
    torch.testing.assert_close(reverse_wheels, -forward_wheels)


def test_rear_steering_reverses_steering_joint_signs() -> None:
    front_controller = AckermannController(_controller_cfg(), num_envs=1, device="cpu")
    rear_controller = AckermannController(_controller_cfg(steerable_wheels_at_rear=True), num_envs=1, device="cpu")
    command = torch.tensor([[1.0, 0.25]])

    front_steering, front_wheels = front_controller.compute(command)
    rear_steering, rear_wheels = rear_controller.compute(command)

    torch.testing.assert_close(rear_steering, -front_steering)
    torch.testing.assert_close(rear_wheels, front_wheels)


def test_controller_clamps_speed_and_steering_angle() -> None:
    cfg = _controller_cfg(max_linear_speed=1.5, max_steering_angle=0.2)
    controller = AckermannController(cfg, num_envs=1, device="cpu")

    saturated_outputs = controller.compute(torch.tensor([[10.0, 1.0]]))
    saturated_outputs = tuple(output.clone() for output in saturated_outputs)
    limit_outputs = controller.compute(torch.tensor([[1.5, 0.2]]))

    torch.testing.assert_close(saturated_outputs[0], limit_outputs[0])
    torch.testing.assert_close(saturated_outputs[1], limit_outputs[1])


def test_float64_steering_limit_just_below_half_pi_is_safe_at_float32_bridge() -> None:
    """A valid float64 limit that rounds to float32 pi/2 remains constructible and finite."""
    max_steering_angle = math.nextafter(math.pi / 2.0, 0.0)
    controller = AckermannController(_controller_cfg(max_steering_angle=max_steering_angle), num_envs=1, device="cpu")

    steering_targets, wheel_targets = controller.compute(torch.tensor([[1.0, max_steering_angle]]))

    assert torch.isfinite(steering_targets).all()
    assert torch.isfinite(wheel_targets).all()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("wheel_radius", 0.0),
        ("wheel_base", -1.0),
        ("track_width", math.inf),
        ("max_linear_speed", -1.0),
        ("max_steering_angle", -0.1),
        ("max_steering_angle", math.pi / 2.0),
        ("non_steerable_wheel_offsets", (0.5, math.nan)),
    ],
)
def test_config_rejects_invalid_geometry(field: str, value: float | tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        _controller_cfg(**{field: value})


def test_compute_rejects_wrong_command_shape() -> None:
    controller = AckermannController(_controller_cfg(), num_envs=2, device="cpu")

    with pytest.raises(ValueError, match="command must have shape"):
        controller.compute(torch.zeros((2, 3)))


def test_zero_command_limits_safely_disable_motion_and_steering() -> None:
    """Zero clamps are valid safety settings and produce stationary straight targets."""
    controller = AckermannController(
        _controller_cfg(max_linear_speed=0.0, max_steering_angle=0.0), num_envs=1, device="cpu"
    )

    steering_targets, wheel_targets = controller.compute(torch.tensor([[2.0, 0.4]]))

    torch.testing.assert_close(steering_targets, torch.zeros_like(steering_targets))
    torch.testing.assert_close(wheel_targets, torch.zeros_like(wheel_targets))


def test_non_finite_commands_produce_finite_targets() -> None:
    controller = AckermannController(_controller_cfg(), num_envs=3, device="cpu")

    steering_targets, wheel_targets = controller.compute(
        torch.tensor([[math.nan, 0.2], [1.0, math.inf], [-math.inf, -math.inf]])
    )

    assert torch.isfinite(steering_targets).all()
    assert torch.isfinite(wheel_targets).all()
    torch.testing.assert_close(steering_targets, torch.zeros_like(steering_targets))
    torch.testing.assert_close(wheel_targets, torch.zeros_like(wheel_targets))


def test_reset_clears_only_selected_command_buffers() -> None:
    controller = AckermannController(_controller_cfg(), num_envs=3, device="cpu")
    controller.compute(torch.tensor([[1.0, 0.1], [2.0, 0.2], [3.0, 0.3]]))

    controller.reset(torch.tensor([0, 2]))

    torch.testing.assert_close(controller._command, torch.tensor([[0.0, 0.0], [2.0, 0.2], [0.0, 0.0]]))
    torch.testing.assert_close(controller._linear_speed_input, torch.tensor([0.0, 2.0, 0.0]))
    torch.testing.assert_close(controller._steering_angle_input, torch.tensor([0.0, 0.2, 0.0]))


def test_compute_reuses_persistent_zero_copy_output_buffers() -> None:
    controller = AckermannController(_controller_cfg(), num_envs=1, device="cpu")
    first_steering, first_wheels = controller.compute(torch.tensor([[1.0, 0.1]]))
    steering_pointer = first_steering.data_ptr()
    wheel_pointer = first_wheels.data_ptr()

    second_steering, second_wheels = controller.compute(torch.tensor([[2.0, -0.2]]))

    assert first_steering is second_steering
    assert first_wheels is second_wheels
    assert second_steering.data_ptr() == steering_pointer
    assert second_wheels.data_ptr() == wheel_pointer
    torch.testing.assert_close(first_steering, second_steering)
    torch.testing.assert_close(first_wheels, second_wheels)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_bridge_uses_zero_copy_buffers_and_replays_warp_graph() -> None:
    controller = AckermannController(_controller_cfg(), num_envs=2, device="cuda:0")
    command = torch.tensor([[1.0, 0.1], [2.0, -0.2]], device="cuda:0")
    steering_targets, wheel_targets = controller.compute(command)
    wp.synchronize_device("cuda:0")
    steering_pointer = steering_targets.data_ptr()
    wheel_pointer = wheel_targets.data_ptr()

    with wp.ScopedCapture(device="cuda:0") as capture:
        controller._controller.compute(controller._inputs, controller._outputs, None, None, time_step=0.0)

    first_wheels = wheel_targets.clone()
    controller._linear_speed_input.copy_(torch.tensor([0.5, -1.0], device="cuda:0"))
    controller._steering_angle_input.copy_(torch.tensor([-0.3, 0.25], device="cuda:0"))
    wp.capture_launch(capture.graph)
    wp.synchronize_device("cuda:0")

    assert steering_targets.data_ptr() == steering_pointer
    assert wheel_targets.data_ptr() == wheel_pointer
    assert not torch.equal(wheel_targets, first_wheels)
    assert torch.isfinite(steering_targets).all()
    assert torch.isfinite(wheel_targets).all()
