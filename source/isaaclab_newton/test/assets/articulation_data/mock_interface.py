# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mock interfaces for testing and benchmarking ArticulationData class."""

from __future__ import annotations

import torch
import warp as wp
from unittest.mock import MagicMock, patch

##
# Mock classes for Newton
##


class MockNewtonModel:
    """Mock Newton model that provides gravity."""

    def __init__(self, gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)):
        self._gravity = wp.array([gravity], dtype=wp.vec3f, device="cuda:0")

    @property
    def gravity(self):
        return self._gravity


class MockNewtonArticulationView:
    """Mock NewtonArticulationView that provides simulation bindings.

    This class mimics the interface that ArticulationData expects from Newton.
    """

    def __init__(
        self,
        num_instances: int,
        num_bodies: int,
        num_joints: int,
        device: str = "cuda:0",
    ):
        """Initialize the mock NewtonArticulationView.

        Args:
            num_instances: Number of instances.
            num_bodies: Number of bodies.
            num_joints: Number of joints.
            device: Device to use.
        """
        # Set the parameters
        self._count = num_instances
        self._link_count = num_bodies
        self._joint_dof_count = num_joints
        self._device = device

        # Storage for mock data
        # Note: These are set via set_mock_data() before any property access in tests
        self._root_transforms = wp.zeros((num_instances,), dtype=wp.transformf, device=device)
        self._root_velocities = wp.zeros((num_instances,), dtype=wp.spatial_vectorf, device=device)
        self._link_transforms = wp.zeros((num_instances, num_bodies), dtype=wp.transformf, device=device)
        self._link_velocities = wp.zeros((num_instances, num_bodies), dtype=wp.spatial_vectorf, device=device)
        self._dof_positions = wp.zeros((num_instances, num_joints), dtype=wp.float32, device=device)
        self._dof_velocities = wp.zeros((num_instances, num_joints), dtype=wp.float32, device=device)

        # Initialize default attributes
        self._attributes: dict = {}
        self._attributes["body_com"] = wp.zeros(
            (self._count, self._link_count), dtype=wp.vec3f, device=self._device
        )
        self._attributes["body_mass"] = wp.ones(
            (self._count, self._link_count), dtype=wp.float32, device=self._device
        )
        self._attributes["body_inertia"] = wp.zeros(
            (self._count, self._link_count), dtype=wp.mat33f, device=self._device
        )
        self._attributes["joint_limit_lower"] = wp.zeros(
            (self._count, self._joint_dof_count), dtype=wp.float32, device=self._device
        )
        self._attributes["joint_limit_upper"] = wp.zeros(
            (self._count, self._joint_dof_count), dtype=wp.float32, device=self._device
        )
        self._attributes["joint_target_ke"] = wp.zeros(
            (self._count, self._joint_dof_count), dtype=wp.float32, device=self._device
        )
        self._attributes["joint_target_kd"] = wp.zeros(
            (self._count, self._joint_dof_count), dtype=wp.float32, device=self._device
        )
        self._attributes["joint_armature"] = wp.zeros(
            (self._count, self._joint_dof_count), dtype=wp.float32, device=self._device
        )
        self._attributes["joint_friction"] = wp.zeros(
            (self._count, self._joint_dof_count), dtype=wp.float32, device=self._device
        )
        self._attributes["joint_velocity_limit"] = wp.ones(
            (self._count, self._joint_dof_count), dtype=wp.float32, device=self._device
        )
        self._attributes["joint_effort_limit"] = wp.ones(
            (self._count, self._joint_dof_count), dtype=wp.float32, device=self._device
        )
        self._attributes["body_f"] = wp.zeros(
            (self._count, self._link_count), dtype=wp.spatial_vectorf, device=self._device
        )
        self._attributes["joint_f"] = wp.zeros(
            (self._count, self._joint_dof_count), dtype=wp.float32, device=self._device
        )
        self._attributes["joint_target_pos"] = wp.zeros(
            (self._count, self._joint_dof_count), dtype=wp.float32, device=self._device
        )
        self._attributes["joint_target_vel"] = wp.zeros(
            (self._count, self._joint_dof_count), dtype=wp.float32, device=self._device
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
    ):
        """Set mock simulation data."""
        if root_transforms is None:
            self._root_transforms.assign(wp.zeros((self._count,), dtype=wp.transformf, device=self._device))
        else:
            self._root_transforms.assign(root_transforms)
        if root_velocities is None:
            self._root_velocities.assign(wp.zeros((self._count,), dtype=wp.spatial_vectorf, device=self._device))
        else:
            self._root_velocities.assign(root_velocities)
        if link_transforms is None:
            self._link_transforms.assign(
                wp.zeros((self._count, self._link_count), dtype=wp.transformf, device=self._device)
            )
        else:
            self._link_transforms.assign(link_transforms)
        if link_velocities is None:
            self._link_velocities.assign(
                wp.zeros((self._count, self._link_count), dtype=wp.spatial_vectorf, device=self._device)
            )
        else:
            self._link_velocities.assign(link_velocities)

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
                wp.ones((self._count, self._link_count), dtype=wp.float32, device=self._device)
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


class MockSharedMetaDataType:
    """Mock shared meta data types."""

    def __init__(
        self, fixed_base: bool, dof_count: int, link_count: int, dof_names: list[str], link_names: list[str]
    ):
        self._fixed_base: bool = fixed_base
        self._dof_count: int = dof_count
        self._link_count: int = link_count
        self._dof_names: list[str] = dof_names
        self._link_names: list[str] = link_names

    @property
    def fixed_base(self) -> bool:
        return self._fixed_base

    @property
    def dof_count(self) -> int:
        return self._dof_count

    @property
    def link_count(self) -> int:
        return self._link_count

    @property
    def dof_names(self) -> list[str]:
        return self._dof_names

    @property
    def link_names(self) -> list[str]:
        return self._link_names


class MockArticulationTensorAPI:
    """Mock ArticulationView that provides tensor API like interface.

    This is for testing against the PhysX implementation which uses torch.Tensor.
    """

    def __init__(
        self,
        num_instances: int,
        num_bodies: int,
        num_joints: int,
        device: str,
        fixed_base: bool = False,
        dof_names: list[str] = [],
        link_names: list[str] = [],
    ):
        """Initialize the mock ArticulationTensorAPI.

        Args:
            num_instances: Number of instances.
            num_bodies: Number of bodies.
            num_joints: Number of joints.
            device: Device to use.
            fixed_base: Whether the articulation is a fixed-base or floating-base system. (default: False)
            dof_names: Names of the joints. (default: [])
            link_names: Names of the bodies. (default: [])
        """
        # Set the parameters
        self._count = num_instances
        self._link_count = num_bodies
        self._joint_dof_count = num_joints
        self._device = device

        # Mock shared meta data type
        if not dof_names:
            dof_names = [f"dof_{i}" for i in range(num_joints)]
        else:
            assert len(dof_names) == num_joints, "The number of dof names must be equal to the number of joints."
        if not link_names:
            link_names = [f"link_{i}" for i in range(num_bodies)]
        else:
            assert len(link_names) == num_bodies, "The number of link names must be equal to the number of bodies."
        self._shared_metatype = MockSharedMetaDataType(fixed_base, num_joints, num_bodies, dof_names, link_names)

        # Storage for mock data
        # Note: These are set via set_mock_data() before any property access in tests
        self._root_transforms: torch.Tensor
        self._root_velocities: torch.Tensor
        self._link_transforms: torch.Tensor
        self._link_velocities: torch.Tensor
        self._link_acceleration: torch.Tensor
        self._body_com: torch.Tensor
        self._dof_positions: torch.Tensor
        self._dof_velocities: torch.Tensor
        self._body_mass: torch.Tensor
        self._body_inertia: torch.Tensor

        # Initialize default attributes
        self._attributes: dict = {}

    @property
    def count(self) -> int:
        return self._count

    @property
    def shared_metatype(self) -> MockSharedMetaDataType:
        return self._shared_metatype

    def get_dof_positions(self) -> torch.Tensor:
        return self._dof_positions

    def get_dof_velocities(self) -> torch.Tensor:
        return self._dof_velocities

    def get_root_transforms(self) -> torch.Tensor:
        return self._root_transforms

    def get_root_velocities(self) -> torch.Tensor:
        return self._root_velocities

    def get_link_transforms(self) -> torch.Tensor:
        return self._link_transforms

    def get_link_velocities(self) -> torch.Tensor:
        return self._link_velocities

    def get_link_acceleration(self) -> torch.Tensor:
        return self._link_acceleration

    def get_coms(self) -> torch.Tensor:
        return self._body_com

    def get_masses(self) -> torch.Tensor:
        return self._body_mass

    def get_inertias(self) -> torch.Tensor:
        return self._body_inertia

    def set_mock_data(
        self,
        root_transforms: torch.Tensor,
        root_velocities: torch.Tensor,
        link_transforms: torch.Tensor,
        link_velocities: torch.Tensor,
        body_com: torch.Tensor,
        link_acceleration: torch.Tensor | None = None,
        dof_positions: torch.Tensor | None = None,
        dof_velocities: torch.Tensor | None = None,
        body_mass: torch.Tensor | None = None,
        body_inertia: torch.Tensor | None = None,
    ):
        """Set mock simulation data."""
        self._root_transforms = root_transforms
        self._root_velocities = root_velocities
        self._link_transforms = link_transforms
        self._link_velocities = link_velocities
        if link_acceleration is None:
            self._link_acceleration = torch.zeros_like(link_velocities)
        else:
            self._link_acceleration = link_acceleration
        self._body_com = body_com

        # Set attributes that ArticulationData expects
        self._attributes["body_com"] = body_com

        if dof_positions is None:
            self._dof_positions = torch.zeros(
                (self._count, self._joint_dof_count), dtype=torch.float32, device=self._device
            )
        else:
            self._dof_positions = dof_positions

        if dof_velocities is None:
            self._dof_velocities = torch.zeros(
                (self._count, self._joint_dof_count), dtype=torch.float32, device=self._device
            )
        else:
            self._dof_velocities = dof_velocities

        if body_mass is None:
            self._body_mass = torch.ones((self._count, self._link_count), dtype=torch.float32, device=self._device)
        else:
            self._body_mass = body_mass
        self._attributes["body_mass"] = self._body_mass

        if body_inertia is None:
            # Initialize as identity inertia tensors
            self._body_inertia = torch.zeros(
                (self._count, self._link_count, 9), dtype=torch.float32, device=self._device
            )
        else:
            self._body_inertia = body_inertia
        self._attributes["body_inertia"] = self._body_inertia

        # Initialize other required attributes with defaults
        self._attributes["joint_limit_lower"] = wp.zeros(
            (self._count, self._joint_dof_count), dtype=wp.float32, device=self._device
        )
        self._attributes["joint_limit_upper"] = wp.zeros(
            (self._count, self._joint_dof_count), dtype=wp.float32, device=self._device
        )
        self._attributes["joint_target_ke"] = wp.zeros(
            (self._count, self._joint_dof_count), dtype=wp.float32, device=self._device
        )
        self._attributes["joint_target_kd"] = wp.zeros(
            (self._count, self._joint_dof_count), dtype=wp.float32, device=self._device
        )
        self._attributes["joint_armature"] = wp.zeros(
            (self._count, self._joint_dof_count), dtype=wp.float32, device=self._device
        )
        self._attributes["joint_friction"] = wp.zeros(
            (self._count, self._joint_dof_count), dtype=wp.float32, device=self._device
        )
        self._attributes["joint_velocity_limit"] = wp.ones(
            (self._count, self._joint_dof_count), dtype=wp.float32, device=self._device
        )
        self._attributes["joint_effort_limit"] = wp.ones(
            (self._count, self._joint_dof_count), dtype=wp.float32, device=self._device
        )
        self._attributes["body_f"] = wp.zeros(
            (self._count, self._link_count), dtype=wp.spatial_vectorf, device=self._device
        )
        self._attributes["joint_f"] = wp.zeros(
            (self._count, self._joint_dof_count), dtype=wp.float32, device=self._device
        )
        self._attributes["joint_target_pos"] = wp.zeros(
            (self._count, self._joint_dof_count), dtype=wp.float32, device=self._device
        )
        self._attributes["joint_target_vel"] = wp.zeros(
            (self._count, self._joint_dof_count), dtype=wp.float32, device=self._device
        )


def create_mock_newton_manager(gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)):
    """Create a mock NewtonManager for testing.

    Returns a context manager that patches the NewtonManager.
    """
    mock_model = MockNewtonModel(gravity)
    mock_state = MagicMock()
    mock_control = MagicMock()

    return patch(
        "isaaclab_newton.assets.articulation.articulation_data.NewtonManager",
        **{
            "get_model.return_value": mock_model,
            "get_state_0.return_value": mock_state,
            "get_control.return_value": mock_control,
            "get_dt.return_value": 0.01,
        },
    )

