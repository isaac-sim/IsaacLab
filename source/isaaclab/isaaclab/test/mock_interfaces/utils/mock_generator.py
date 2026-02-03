# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for generating custom mock objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..assets import MockArticulation
    from ..sensors import MockContactSensor, MockFrameTransformer, MockImu


class MockArticulationBuilder:
    """Builder class for creating custom MockArticulation instances.

    Example:
        >>> robot = (
        ...     MockArticulationBuilder()
        ...     .with_joints(["hip", "knee", "ankle"], default_pos=[0.0, 0.5, -0.5])
        ...     .with_bodies(["base", "thigh", "shin", "foot"])
        ...     .with_fixed_base(True)
        ...     .with_num_instances(8)
        ...     .build()
        ... )
    """

    def __init__(self):
        """Initialize the builder with default values."""
        self._num_instances = 1
        self._joint_names: list[str] = []
        self._body_names: list[str] = []
        self._is_fixed_base = False
        self._device = "cpu"
        self._default_joint_pos: list[float] | None = None
        self._default_joint_vel: list[float] | None = None
        self._joint_pos_limits: tuple[float, float] | None = None
        self._num_fixed_tendons = 0
        self._num_spatial_tendons = 0
        self._fixed_tendon_names: list[str] | None = None
        self._spatial_tendon_names: list[str] | None = None

    def with_num_instances(self, num_instances: int) -> MockArticulationBuilder:
        """Set the number of articulation instances.

        Args:
            num_instances: Number of instances.

        Returns:
            Self for method chaining.
        """
        self._num_instances = num_instances
        return self

    def with_joints(
        self,
        joint_names: list[str],
        default_pos: list[float] | None = None,
        default_vel: list[float] | None = None,
    ) -> MockArticulationBuilder:
        """Set joint configuration.

        Args:
            joint_names: Names of the joints.
            default_pos: Default joint positions.
            default_vel: Default joint velocities.

        Returns:
            Self for method chaining.
        """
        self._joint_names = joint_names
        self._default_joint_pos = default_pos
        self._default_joint_vel = default_vel
        return self

    def with_bodies(self, body_names: list[str]) -> MockArticulationBuilder:
        """Set body configuration.

        Args:
            body_names: Names of the bodies.

        Returns:
            Self for method chaining.
        """
        self._body_names = body_names
        return self

    def with_fixed_base(self, is_fixed: bool) -> MockArticulationBuilder:
        """Set whether the articulation has a fixed base.

        Args:
            is_fixed: True for fixed base, False for floating base.

        Returns:
            Self for method chaining.
        """
        self._is_fixed_base = is_fixed
        return self

    def with_device(self, device: str) -> MockArticulationBuilder:
        """Set the device for tensor allocation.

        Args:
            device: Device string (e.g., "cpu", "cuda:0").

        Returns:
            Self for method chaining.
        """
        self._device = device
        return self

    def with_joint_limits(self, lower: float, upper: float) -> MockArticulationBuilder:
        """Set uniform joint position limits for all joints.

        Args:
            lower: Lower joint limit.
            upper: Upper joint limit.

        Returns:
            Self for method chaining.
        """
        self._joint_pos_limits = (lower, upper)
        return self

    def with_fixed_tendons(self, tendon_names: list[str]) -> MockArticulationBuilder:
        """Set fixed tendon configuration.

        Args:
            tendon_names: Names of fixed tendons.

        Returns:
            Self for method chaining.
        """
        self._fixed_tendon_names = tendon_names
        self._num_fixed_tendons = len(tendon_names)
        return self

    def with_spatial_tendons(self, tendon_names: list[str]) -> MockArticulationBuilder:
        """Set spatial tendon configuration.

        Args:
            tendon_names: Names of spatial tendons.

        Returns:
            Self for method chaining.
        """
        self._spatial_tendon_names = tendon_names
        self._num_spatial_tendons = len(tendon_names)
        return self

    def build(self) -> MockArticulation:
        """Build the MockArticulation instance.

        Returns:
            Configured MockArticulation instance.
        """
        from ..assets import MockArticulation

        num_joints = len(self._joint_names) if self._joint_names else 1
        num_bodies = len(self._body_names) if self._body_names else num_joints + 1

        robot = MockArticulation(
            num_instances=self._num_instances,
            num_joints=num_joints,
            num_bodies=num_bodies,
            joint_names=self._joint_names or None,
            body_names=self._body_names or None,
            is_fixed_base=self._is_fixed_base,
            num_fixed_tendons=self._num_fixed_tendons,
            num_spatial_tendons=self._num_spatial_tendons,
            fixed_tendon_names=self._fixed_tendon_names,
            spatial_tendon_names=self._spatial_tendon_names,
            device=self._device,
        )

        # Set default joint positions
        if self._default_joint_pos is not None:
            default_pos = torch.tensor(
                [self._default_joint_pos] * self._num_instances,
                device=self._device,
            )
            robot.data.set_default_joint_pos(default_pos)
            robot.data.set_joint_pos(default_pos)

        # Set default joint velocities
        if self._default_joint_vel is not None:
            default_vel = torch.tensor(
                [self._default_joint_vel] * self._num_instances,
                device=self._device,
            )
            robot.data.set_default_joint_vel(default_vel)
            robot.data.set_joint_vel(default_vel)

        # Set joint limits
        if self._joint_pos_limits is not None:
            limits = torch.zeros(self._num_instances, num_joints, 2, device=self._device)
            limits[..., 0] = self._joint_pos_limits[0]
            limits[..., 1] = self._joint_pos_limits[1]
            robot.data.set_joint_pos_limits(limits)

        return robot


class MockSensorBuilder:
    """Builder class for creating custom mock sensor instances.

    Example:
        >>> sensor = (
        ...     MockSensorBuilder("contact")
        ...     .with_num_instances(4)
        ...     .with_bodies(["FL_foot", "FR_foot", "RL_foot", "RR_foot"])
        ...     .with_device("cuda")
        ...     .build()
        ... )
    """

    def __init__(self, sensor_type: str):
        """Initialize the builder.

        Args:
            sensor_type: Type of sensor ("contact", "imu", or "frame_transformer").
        """
        if sensor_type not in ("contact", "imu", "frame_transformer"):
            raise ValueError(f"Unknown sensor type: {sensor_type}")
        self._sensor_type = sensor_type
        self._num_instances = 1
        self._device = "cpu"

        # Contact sensor specific
        self._body_names: list[str] = []
        self._history_length = 0
        self._num_filter_bodies = 0

        # Frame transformer specific
        self._target_frame_names: list[str] = []

    def with_num_instances(self, num_instances: int) -> MockSensorBuilder:
        """Set the number of sensor instances.

        Args:
            num_instances: Number of instances.

        Returns:
            Self for method chaining.
        """
        self._num_instances = num_instances
        return self

    def with_device(self, device: str) -> MockSensorBuilder:
        """Set the device for tensor allocation.

        Args:
            device: Device string.

        Returns:
            Self for method chaining.
        """
        self._device = device
        return self

    def with_bodies(self, body_names: list[str]) -> MockSensorBuilder:
        """Set body names (for contact sensor).

        Args:
            body_names: Names of bodies with contact sensors.

        Returns:
            Self for method chaining.
        """
        self._body_names = body_names
        return self

    def with_history_length(self, length: int) -> MockSensorBuilder:
        """Set history buffer length (for contact sensor).

        Args:
            length: Length of history buffer.

        Returns:
            Self for method chaining.
        """
        self._history_length = length
        return self

    def with_filter_bodies(self, num_filter_bodies: int) -> MockSensorBuilder:
        """Set number of filter bodies (for contact sensor).

        Args:
            num_filter_bodies: Number of filter bodies for force matrix.

        Returns:
            Self for method chaining.
        """
        self._num_filter_bodies = num_filter_bodies
        return self

    def with_target_frames(self, frame_names: list[str]) -> MockSensorBuilder:
        """Set target frame names (for frame transformer).

        Args:
            frame_names: Names of target frames.

        Returns:
            Self for method chaining.
        """
        self._target_frame_names = frame_names
        return self

    def build(self) -> MockContactSensor | MockImu | MockFrameTransformer:
        """Build the mock sensor instance.

        Returns:
            Configured mock sensor instance.
        """
        if self._sensor_type == "contact":
            from ..sensors import MockContactSensor

            num_bodies = len(self._body_names) if self._body_names else 1
            return MockContactSensor(
                num_instances=self._num_instances,
                num_bodies=num_bodies,
                body_names=self._body_names or None,
                device=self._device,
                history_length=self._history_length,
                num_filter_bodies=self._num_filter_bodies,
            )
        elif self._sensor_type == "imu":
            from ..sensors import MockImu

            return MockImu(
                num_instances=self._num_instances,
                device=self._device,
            )
        elif self._sensor_type == "frame_transformer":
            from ..sensors import MockFrameTransformer

            num_frames = len(self._target_frame_names) if self._target_frame_names else 1
            return MockFrameTransformer(
                num_instances=self._num_instances,
                num_target_frames=num_frames,
                target_frame_names=self._target_frame_names or None,
                device=self._device,
            )
        else:
            raise ValueError(f"Unknown sensor type: {self._sensor_type}")
