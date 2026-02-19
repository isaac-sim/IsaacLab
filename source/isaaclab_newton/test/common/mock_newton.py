# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared mock interfaces for testing and benchmarking Newton-based asset classes.

This module provides mock implementations of Newton simulation components that can
be used to test ArticulationData, RigidObjectData, and related classes without
requiring an actual simulation environment.
"""

from __future__ import annotations

import torch
from unittest.mock import MagicMock, patch

import warp as wp


class MockNewtonModel:
    """Mock Newton model that provides gravity."""

    def __init__(self, gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)):
        self._gravity = wp.array([gravity], dtype=wp.vec3f, device="cuda:0")

    @property
    def gravity(self):
        return self._gravity


class MockWrenchComposer:
    """Mock WrenchComposer for testing without full simulation infrastructure.

    This class mimics the interface of :class:`isaaclab.utils.wrench_composer.WrenchComposer`
    and can be used to test Articulation and RigidObject classes.
    """

    def __init__(self, asset):
        """Initialize the mock wrench composer.

        Args:
            asset: The asset (Articulation or RigidObject) to compose wrenches for.
        """
        self.num_envs = asset.num_instances
        self.num_bodies = asset.num_bodies
        self.device = asset.device
        self._active = False

        # Create buffers matching the real WrenchComposer
        self._composed_force_b = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)
        self._composed_torque_b = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)
        self._ALL_ENV_MASK_WP = wp.ones((self.num_envs,), dtype=wp.bool, device=self.device)
        self._ALL_BODY_MASK_WP = wp.ones((self.num_bodies,), dtype=wp.bool, device=self.device)

    @property
    def active(self) -> bool:
        """Whether the wrench composer is active."""
        return self._active

    @property
    def composed_force(self) -> wp.array:
        """Composed force at the body's com frame."""
        return self._composed_force_b

    @property
    def composed_torque(self) -> wp.array:
        """Composed torque at the body's com frame."""
        return self._composed_torque_b

    def set_forces_and_torques(
        self,
        forces=None,
        torques=None,
        positions=None,
        body_ids=None,
        env_ids=None,
        body_mask=None,
        env_mask=None,
        is_global: bool = False,
    ):
        """Mock set_forces_and_torques - just marks as active."""
        self._active = True

    def add_forces_and_torques(
        self,
        forces=None,
        torques=None,
        positions=None,
        body_ids=None,
        env_ids=None,
        body_mask=None,
        env_mask=None,
        is_global: bool = False,
    ):
        """Mock add_forces_and_torques - just marks as active."""
        self._active = True

    def reset(self, env_ids=None, env_mask=None):
        """Reset the composed force and torque."""
        self._composed_force_b.zero_()
        self._composed_torque_b.zero_()
        self._active = False


class MockNewtonArticulationView:
    """Mock NewtonArticulationView that provides simulation bindings.

    This class mimics the interface that ArticulationData and RigidObjectData
    expect from Newton. It can be used for both articulation and rigid object testing.
    """

    def __init__(
        self,
        num_instances: int,
        num_bodies: int,
        num_joints: int,
        device: str = "cuda:0",
        is_fixed_base: bool = False,
        joint_names: list[str] | None = None,
        body_names: list[str] | None = None,
    ):
        """Initialize the mock NewtonArticulationView.

        Args:
            num_instances: Number of instances.
            num_bodies: Number of bodies.
            num_joints: Number of joints.
            device: Device to use.
            is_fixed_base: Whether the articulation is fixed-base.
            joint_names: Names of joints. Defaults to ["joint_0", ...].
            body_names: Names of bodies. Defaults to ["body_0", ...].
        """
        # Set the parameters
        self._count = num_instances
        self._link_count = num_bodies
        self._joint_dof_count = num_joints
        self._device = device
        self._is_fixed_base = is_fixed_base

        # Set joint and body names
        if joint_names is None:
            self._joint_dof_names = [f"joint_{i}" for i in range(num_joints)]
        else:
            self._joint_dof_names = joint_names

        if body_names is None:
            self._body_names = [f"body_{i}" for i in range(num_bodies)]
        else:
            self._body_names = body_names

        # Storage for mock data
        # Note: These are set via set_mock_data() before any property access in tests
        print("num_instances:", num_instances)
        print("num_bodies:", num_bodies)
        print("num_joints:", num_joints)
        print("is_fixed_base:", is_fixed_base)
        if is_fixed_base:
            self._root_transforms = wp.zeros((num_instances, 1, 1), dtype=wp.transformf, device=device)
            self._root_velocities = None
            self._link_velocities = None
        else:
            self._root_transforms = wp.zeros((num_instances, 1), dtype=wp.transformf, device=device)
            self._root_velocities = wp.zeros((num_instances, 1), dtype=wp.spatial_vectorf, device=device)
            self._link_velocities = wp.zeros((num_instances, 1, num_bodies), dtype=wp.spatial_vectorf, device=device)

        self._link_transforms = wp.zeros((num_instances, 1, num_bodies), dtype=wp.transformf, device=device)

        self._dof_positions = wp.zeros((num_instances, 1, num_joints), dtype=wp.float32, device=device)
        self._dof_velocities = wp.zeros((num_instances, 1, num_joints), dtype=wp.float32, device=device)

        # Initialize default attributes
        self._attributes: dict = {}
        self._attributes["body_com"] = wp.zeros((self._count, 1, self._link_count), dtype=wp.vec3f, device=self._device)
        self._attributes["body_mass"] = wp.zeros(
            (self._count, 1, self._link_count), dtype=wp.float32, device=self._device
        )
        self._attributes["body_inertia"] = wp.zeros(
            (self._count, 1, self._link_count), dtype=wp.mat33f, device=self._device
        )
        self._attributes["joint_limit_lower"] = wp.zeros(
            (self._count, 1, self._joint_dof_count), dtype=wp.float32, device=self._device
        )
        self._attributes["joint_limit_upper"] = wp.zeros(
            (self._count, 1, self._joint_dof_count), dtype=wp.float32, device=self._device
        )
        self._attributes["joint_target_ke"] = wp.zeros(
            (self._count, 1, self._joint_dof_count), dtype=wp.float32, device=self._device
        )
        self._attributes["joint_target_kd"] = wp.zeros(
            (self._count, 1, self._joint_dof_count), dtype=wp.float32, device=self._device
        )
        self._attributes["joint_armature"] = wp.zeros(
            (self._count, 1, self._joint_dof_count), dtype=wp.float32, device=self._device
        )
        self._attributes["joint_friction"] = wp.zeros(
            (self._count, 1, self._joint_dof_count), dtype=wp.float32, device=self._device
        )
        self._attributes["joint_velocity_limit"] = wp.zeros(
            (self._count, 1, self._joint_dof_count), dtype=wp.float32, device=self._device
        )
        self._attributes["joint_effort_limit"] = wp.zeros(
            (self._count, 1, self._joint_dof_count), dtype=wp.float32, device=self._device
        )
        self._attributes["body_f"] = wp.zeros(
            (self._count, 1, self._link_count), dtype=wp.spatial_vectorf, device=self._device
        )
        self._attributes["joint_f"] = wp.zeros(
            (self._count, 1, self._joint_dof_count), dtype=wp.float32, device=self._device
        )
        self._attributes["joint_target_pos"] = wp.zeros(
            (self._count, 1, self._joint_dof_count), dtype=wp.float32, device=self._device
        )
        self._attributes["joint_target_vel"] = wp.zeros(
            (self._count, 1, self._joint_dof_count), dtype=wp.float32, device=self._device
        )
        self._attributes["joint_limit_ke"] = wp.zeros(
            (self._count, 1, self._joint_dof_count), dtype=wp.float32, device=self._device
        )
        self._attributes["joint_limit_kd"] = wp.zeros(
            (self._count, 1, self._joint_dof_count), dtype=wp.float32, device=self._device
        )

    @property
    def count(self) -> int:
        return self._count

    @property
    def link_count(self) -> int:
        return self._link_count

    @property
    def joint_dof_count(self) -> int:
        return self._joint_dof_count

    @property
    def is_fixed_base(self) -> bool:
        return self._is_fixed_base

    @property
    def joint_dof_names(self) -> list[str]:
        return self._joint_dof_names

    @property
    def body_names(self) -> list[str]:
        return self._body_names

    def get_root_transforms(self, state) -> wp.array:
        return self._root_transforms

    def get_root_velocities(self, state) -> wp.array:
        return self._root_velocities

    def get_link_transforms(self, state) -> wp.array:
        return self._link_transforms

    def get_link_velocities(self, state) -> wp.array:
        return self._link_velocities

    def get_dof_positions(self, state) -> wp.array:
        return self._dof_positions

    def get_dof_velocities(self, state) -> wp.array:
        return self._dof_velocities

    def get_attribute(self, name: str, model_or_state) -> wp.array:
        return self._attributes[name]

    def set_root_transforms(self, state, transforms: wp.array):
        self._root_transforms.assign(transforms)

    def set_root_velocities(self, state, velocities: wp.array):
        self._root_velocities.assign(velocities)

    def set_mock_data(
        self,
        root_transforms: wp.array | None = None,
        root_velocities: wp.array | None = None,
        link_transforms: wp.array | None = None,
        link_velocities: wp.array | None = None,
        body_com_pos: wp.array | None = None,
        dof_positions: wp.array | None = None,
        dof_velocities: wp.array | None = None,
        body_mass: wp.array | None = None,
        body_inertia: wp.array | None = None,
        joint_limit_lower: wp.array | None = None,
        joint_limit_upper: wp.array | None = None,
    ):
        """Set mock simulation data."""
        if root_transforms is None:
            self._root_transforms.assign(wp.zeros((self._count,), dtype=wp.transformf, device=self._device))
        else:
            self._root_transforms.assign(root_transforms)
        if root_velocities is None:
            if self._root_velocities is not None:
                self._root_velocities.assign(wp.zeros((self._count,), dtype=wp.spatial_vectorf, device=self._device))
            else:
                self._root_velocities = root_velocities
        else:
            if self._root_velocities is not None:
                self._root_velocities.assign(root_velocities)
            else:
                self._root_velocities = root_velocities
        if link_transforms is None:
            self._link_transforms.assign(
                wp.zeros((self._count, self._link_count), dtype=wp.transformf, device=self._device)
            )
        else:
            self._link_transforms.assign(link_transforms)
        if link_velocities is None:
            if self._link_velocities is not None:
                self._link_velocities.assign(
                    wp.zeros((self._count, self._link_count), dtype=wp.spatial_vectorf, device=self._device)
                )
            else:
                self._link_velocities = link_velocities
        else:
            if self._link_velocities is not None:
                self._link_velocities.assign(link_velocities)
            else:
                self._link_velocities = link_velocities

        # Set attributes that ArticulationData expects
        if body_com_pos is None:
            self._attributes["body_com"].assign(
                wp.zeros((self._count, self._link_count), dtype=wp.vec3f, device=self._device)
            )
        else:
            self._attributes["body_com"].assign(body_com_pos)

        if dof_positions is None:
            self._dof_positions.assign(
                wp.zeros((self._count, self._joint_dof_count), dtype=wp.float32, device=self._device)
            )
        else:
            self._dof_positions.assign(dof_positions)

        if dof_velocities is None:
            self._dof_velocities.assign(
                wp.zeros((self._count, self._joint_dof_count), dtype=wp.float32, device=self._device)
            )
        else:
            self._dof_velocities.assign(dof_velocities)

        if body_mass is None:
            self._attributes["body_mass"].assign(
                wp.zeros((self._count, self._link_count), dtype=wp.float32, device=self._device)
            )
        else:
            self._attributes["body_mass"].assign(body_mass)

        if body_inertia is None:
            # Initialize as identity inertia tensors
            self._attributes["body_inertia"].assign(
                wp.zeros((self._count, self._link_count), dtype=wp.mat33f, device=self._device)
            )
        else:
            self._attributes["body_inertia"].assign(body_inertia)

        if joint_limit_lower is not None:
            self._attributes["joint_limit_lower"].assign(joint_limit_lower)

        if joint_limit_upper is not None:
            self._attributes["joint_limit_upper"].assign(joint_limit_upper)

    def set_random_mock_data(self):
        """Set randomized mock simulation data for benchmarking."""
        # Generate random root transforms (position + normalized quaternion)
        root_pose = torch.zeros((self._count, 7), device=self._device)
        root_pose[:, :3] = torch.rand((self._count, 3), device=self._device) * 10.0 - 5.0  # Random positions
        root_pose[:, 3:] = torch.randn((self._count, 4), device=self._device)
        root_pose[:, 3:] = torch.nn.functional.normalize(root_pose[:, 3:], p=2.0, dim=-1, eps=1e-12)

        # Generate random velocities
        root_vel = torch.rand((self._count, 6), device=self._device) * 2.0 - 1.0

        # Generate random link transforms
        link_pose = torch.zeros((self._count, self._link_count, 7), device=self._device)
        link_pose[:, :, :3] = torch.rand((self._count, self._link_count, 3), device=self._device) * 10.0 - 5.0
        link_pose[:, :, 3:] = torch.randn((self._count, self._link_count, 4), device=self._device)
        link_pose[:, :, 3:] = torch.nn.functional.normalize(link_pose[:, :, 3:], p=2.0, dim=-1, eps=1e-12)

        # Generate random link velocities
        link_vel = torch.rand((self._count, self._link_count, 6), device=self._device) * 2.0 - 1.0

        # Generate random body COM positions
        body_com_pos = torch.rand((self._count, self._link_count, 3), device=self._device) * 0.2 - 0.1

        # Generate random joint positions and velocities
        dof_pos = torch.rand((self._count, self._joint_dof_count), device=self._device) * 6.28 - 3.14
        dof_vel = torch.rand((self._count, self._joint_dof_count), device=self._device) * 2.0 - 1.0

        # Generate random body masses (positive values)
        body_mass = torch.rand((self._count, self._link_count), device=self._device) * 10.0 + 0.1

        # Set the mock data
        self.set_mock_data(
            root_transforms=wp.from_torch(root_pose, dtype=wp.transformf),
            root_velocities=wp.from_torch(root_vel, dtype=wp.spatial_vectorf),
            link_transforms=wp.from_torch(link_pose, dtype=wp.transformf),
            link_velocities=wp.from_torch(link_vel, dtype=wp.spatial_vectorf),
            body_com_pos=wp.from_torch(body_com_pos, dtype=wp.vec3f),
            dof_positions=wp.from_torch(dof_pos, dtype=wp.float32),
            dof_velocities=wp.from_torch(dof_vel, dtype=wp.float32),
            body_mass=wp.from_torch(body_mass, dtype=wp.float32),
        )


def create_mock_newton_manager(
    patch_path: str,
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
):
    """Create a mock NewtonManager for testing.

    Args:
        patch_path: The module path to patch (e.g., "isaaclab_newton.assets.articulation.articulation_data.NewtonManager").
        gravity: Gravity vector to use for the mock model.

    Returns:
        A context manager that patches the NewtonManager.
    """
    mock_model = MockNewtonModel(gravity)
    mock_state = MagicMock()
    mock_control = MagicMock()

    return patch(
        patch_path,
        **{
            "get_model.return_value": mock_model,
            "get_state_0.return_value": mock_state,
            "get_control.return_value": mock_control,
            "get_dt.return_value": 0.01,
        },
    )


class MockNewtonContactSensor:
    """Mock Newton contact sensor for testing without full simulation infrastructure.

    This class mimics the interface of Newton's SensorContact and can be used to test
    ContactSensor classes without requiring an actual simulation environment.
    """

    def __init__(
        self,
        num_sensing_objs: int,
        num_counterparts: int = 1,
        device: str = "cuda:0",
    ):
        """Initialize the mock contact sensor.

        Args:
            num_sensing_objs: Number of sensing objects (e.g., bodies or shapes).
            num_counterparts: Number of counterparts per sensing object.
            device: Device to use.
        """
        self.device = device
        self.shape: tuple[int, int] = (num_sensing_objs, num_counterparts)
        self.sensing_objs: list[tuple[int, int]] = [(i, 1) for i in range(num_sensing_objs)]
        self.counterparts: list[tuple[int, int]] = [(i, 1) for i in range(num_counterparts)]
        self.reading_indices: list[list[int]] = [list(range(num_counterparts)) for _ in range(num_sensing_objs)]

        # Net force array (n_sensing_objs, n_counterparts) of vec3
        self._net_force = wp.zeros(num_sensing_objs * num_counterparts, dtype=wp.vec3, device=device)
        self.net_force = self._net_force.reshape(self.shape)

    def eval(self, contacts):
        """Mock eval - does nothing since forces are set directly via set_mock_data."""
        pass

    def get_total_force(self) -> wp.array:
        """Get the total net force measured by the contact sensor."""
        return self.net_force

    def set_mock_data(self, net_force: wp.array | None = None):
        """Set mock contact force data.

        Args:
            net_force: Force data shaped (num_sensing_objs, num_counterparts) of vec3.
                       If None, zeros the force data.
        """
        if net_force is None:
            self._net_force.zero_()
        else:
            self._net_force.assign(net_force)
