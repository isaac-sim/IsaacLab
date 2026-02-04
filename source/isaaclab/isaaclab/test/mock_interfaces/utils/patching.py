# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for patching real classes with mock implementations in tests."""

from __future__ import annotations

import functools
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any, TypeVar
from unittest.mock import patch

F = TypeVar("F", bound=Callable[..., Any])


@contextmanager
def patch_articulation(
    target: str,
    num_instances: int = 1,
    num_joints: int = 12,
    num_bodies: int = 13,
    joint_names: list[str] | None = None,
    body_names: list[str] | None = None,
    is_fixed_base: bool = False,
    device: str = "cpu",
    **kwargs: Any,
) -> Generator[Any, None, None]:
    """Context manager for patching an articulation class with MockArticulation.

    Args:
        target: The target to patch (e.g., "my_module.Articulation").
        num_instances: Number of articulation instances.
        num_joints: Number of joints.
        num_bodies: Number of bodies.
        joint_names: Names of joints.
        body_names: Names of bodies.
        is_fixed_base: Whether the articulation has a fixed base.
        device: Device for tensor allocation.
        **kwargs: Additional keyword arguments for MockArticulation.

    Yields:
        The mock articulation class.

    Example:
        >>> with patch_articulation("my_module.robot", num_joints=12) as MockRobot:
        ...     robot = MockRobot()
        ...     robot.data.set_joint_pos(torch.zeros(1, 12))
        ...     result = my_function_using_robot(robot)
    """
    from ..assets import MockArticulation

    def create_mock(*args: Any, **create_kwargs: Any) -> MockArticulation:
        # Merge configuration with any runtime kwargs
        return MockArticulation(
            num_instances=create_kwargs.get("num_instances", num_instances),
            num_joints=create_kwargs.get("num_joints", num_joints),
            num_bodies=create_kwargs.get("num_bodies", num_bodies),
            joint_names=create_kwargs.get("joint_names", joint_names),
            body_names=create_kwargs.get("body_names", body_names),
            is_fixed_base=create_kwargs.get("is_fixed_base", is_fixed_base),
            device=create_kwargs.get("device", device),
            **{k: v for k, v in kwargs.items() if k not in create_kwargs},
        )

    with patch(target, side_effect=create_mock) as mock_class:
        yield mock_class


@contextmanager
def patch_sensor(
    target: str,
    sensor_type: str,
    num_instances: int = 1,
    device: str = "cpu",
    **kwargs: Any,
) -> Generator[Any, None, None]:
    """Context manager for patching a sensor class with a mock sensor.

    Args:
        target: The target to patch (e.g., "my_module.ContactSensor").
        sensor_type: Type of sensor ("contact", "imu", or "frame_transformer").
        num_instances: Number of sensor instances.
        device: Device for tensor allocation.
        **kwargs: Additional keyword arguments for the mock sensor.

    Yields:
        The mock sensor class.

    Example:
        >>> with patch_sensor("my_env.ContactSensor", "contact", num_bodies=4) as MockSensor:
        ...     sensor = MockSensor()
        ...     sensor.data.set_net_forces_w(torch.randn(1, 4, 3))
        ...     result = my_function(sensor)
    """
    if sensor_type == "contact":
        from ..sensors import MockContactSensor

        def create_mock(*args: Any, **create_kwargs: Any) -> MockContactSensor:
            return MockContactSensor(
                num_instances=create_kwargs.get("num_instances", num_instances),
                num_bodies=create_kwargs.get("num_bodies", kwargs.get("num_bodies", 1)),
                body_names=create_kwargs.get("body_names", kwargs.get("body_names")),
                device=create_kwargs.get("device", device),
                history_length=create_kwargs.get("history_length", kwargs.get("history_length", 0)),
                num_filter_bodies=create_kwargs.get("num_filter_bodies", kwargs.get("num_filter_bodies", 0)),
            )

    elif sensor_type == "imu":
        from ..sensors import MockImu

        def create_mock(*args: Any, **create_kwargs: Any) -> MockImu:
            return MockImu(
                num_instances=create_kwargs.get("num_instances", num_instances),
                device=create_kwargs.get("device", device),
            )

    elif sensor_type == "frame_transformer":
        from ..sensors import MockFrameTransformer

        def create_mock(*args: Any, **create_kwargs: Any) -> MockFrameTransformer:
            return MockFrameTransformer(
                num_instances=create_kwargs.get("num_instances", num_instances),
                num_target_frames=create_kwargs.get("num_target_frames", kwargs.get("num_target_frames", 1)),
                target_frame_names=create_kwargs.get("target_frame_names", kwargs.get("target_frame_names")),
                device=create_kwargs.get("device", device),
            )

    else:
        raise ValueError(f"Unknown sensor type: {sensor_type}")

    with patch(target, side_effect=create_mock) as mock_class:
        yield mock_class


def mock_articulation(
    num_instances: int = 1,
    num_joints: int = 12,
    num_bodies: int = 13,
    joint_names: list[str] | None = None,
    body_names: list[str] | None = None,
    is_fixed_base: bool = False,
    device: str = "cpu",
    **kwargs: Any,
) -> Callable[[F], F]:
    """Decorator for injecting a MockArticulation into a test function.

    The mock articulation is passed as the first argument to the decorated function.

    Args:
        num_instances: Number of articulation instances.
        num_joints: Number of joints.
        num_bodies: Number of bodies.
        joint_names: Names of joints.
        body_names: Names of bodies.
        is_fixed_base: Whether the articulation has a fixed base.
        device: Device for tensor allocation.
        **kwargs: Additional keyword arguments for MockArticulation.

    Returns:
        Decorator function.

    Example:
        >>> @mock_articulation(num_joints=12, num_bodies=13)
        ... def test_my_function(mock_robot):
        ...     mock_robot.data.set_joint_pos(torch.zeros(1, 12))
        ...     result = compute_something(mock_robot)
        ...     assert result.shape == (1, 12)
    """
    from ..assets import MockArticulation

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **wrap_kwargs: Any) -> Any:
            mock = MockArticulation(
                num_instances=num_instances,
                num_joints=num_joints,
                num_bodies=num_bodies,
                joint_names=joint_names,
                body_names=body_names,
                is_fixed_base=is_fixed_base,
                device=device,
                **kwargs,
            )
            return func(mock, *args, **wrap_kwargs)

        return wrapper  # type: ignore

    return decorator


def mock_sensor(
    sensor_type: str,
    num_instances: int = 1,
    device: str = "cpu",
    **kwargs: Any,
) -> Callable[[F], F]:
    """Decorator for injecting a mock sensor into a test function.

    The mock sensor is passed as the first argument to the decorated function.

    Args:
        sensor_type: Type of sensor ("contact", "imu", or "frame_transformer").
        num_instances: Number of sensor instances.
        device: Device for tensor allocation.
        **kwargs: Additional keyword arguments for the mock sensor.

    Returns:
        Decorator function.

    Example:
        >>> @mock_sensor("contact", num_instances=4, num_bodies=4)
        ... def test_contact_reward(mock_contact):
        ...     mock_contact.data.set_net_forces_w(torch.randn(4, 4, 3))
        ...     reward = compute_contact_reward(mock_contact)
        ...     assert reward.shape == (4,)
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **wrap_kwargs: Any) -> Any:
            if sensor_type == "contact":
                from ..sensors import MockContactSensor

                mock = MockContactSensor(
                    num_instances=num_instances,
                    num_bodies=kwargs.get("num_bodies", 1),
                    body_names=kwargs.get("body_names"),
                    device=device,
                    history_length=kwargs.get("history_length", 0),
                    num_filter_bodies=kwargs.get("num_filter_bodies", 0),
                )
            elif sensor_type == "imu":
                from ..sensors import MockImu

                mock = MockImu(
                    num_instances=num_instances,
                    device=device,
                )
            elif sensor_type == "frame_transformer":
                from ..sensors import MockFrameTransformer

                mock = MockFrameTransformer(
                    num_instances=num_instances,
                    num_target_frames=kwargs.get("num_target_frames", 1),
                    target_frame_names=kwargs.get("target_frame_names"),
                    device=device,
                )
            else:
                raise ValueError(f"Unknown sensor type: {sensor_type}")

            return func(mock, *args, **wrap_kwargs)

        return wrapper  # type: ignore

    return decorator
