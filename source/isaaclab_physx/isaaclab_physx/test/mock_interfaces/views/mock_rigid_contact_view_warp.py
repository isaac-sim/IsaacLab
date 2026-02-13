# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mock implementation of PhysX RigidContactView using Warp arrays."""

from __future__ import annotations

import warp as wp


class MockRigidContactViewWarp:
    """Mock implementation of physx.RigidContactView using Warp arrays for unit testing.

    This class mimics the interface of the PhysX TensorAPI RigidContactView using
    flat float32 arrays (matching real PhysX behavior), allowing tests to run
    without Isaac Sim or GPU simulation.

    Data Shapes (flat float32, matching real PhysX views):
        - net_contact_forces: (N*B, 3) dtype=wp.float32 - flattened net forces
        - contact_force_matrix: (N*B, F, 3) dtype=wp.float32 - per-filter forces
        - contact_data: tuple of arrays with float32 for positions/normals/impulses
        - friction_data: tuple of arrays with float32 for forces/impulses/points

    Where:
        - N = count (number of instances)
        - B = num_bodies (bodies per instance)
        - F = filter_count (number of filter bodies)
    """

    def __init__(
        self,
        count: int = 1,
        num_bodies: int = 1,
        filter_count: int = 0,
        max_contact_data_count: int = 16,
        device: str = "cpu",
    ):
        """Initialize the mock rigid contact view.

        Args:
            count: Number of instances.
            num_bodies: Number of bodies per instance.
            filter_count: Number of filter bodies for contact filtering.
            max_contact_data_count: Maximum number of contact data points.
            device: Device for array allocation ("cpu" or "cuda:N").
        """
        self._count = count
        self._num_bodies = num_bodies
        self._filter_count = filter_count
        self._max_contact_data_count = max_contact_data_count
        self._device = device
        self._backend = "warp"

        # Total number of bodies (flattened)
        self._total_bodies = count * num_bodies

        # Internal state (lazily initialized)
        self._net_contact_forces: wp.array | None = None
        self._contact_force_matrix: wp.array | None = None
        self._contact_positions: wp.array | None = None
        self._contact_normals: wp.array | None = None
        self._contact_impulses: wp.array | None = None
        self._contact_separations: wp.array | None = None
        self._contact_num_found: wp.array | None = None
        self._contact_patch_id: wp.array | None = None
        self._friction_forces: wp.array | None = None
        self._friction_impulses: wp.array | None = None
        self._friction_points: wp.array | None = None
        self._friction_patch_id: wp.array | None = None

    # -- Properties --

    @property
    def filter_count(self) -> int:
        """Number of filter bodies for contact filtering."""
        return self._filter_count

    @property
    def count(self) -> int:
        """Number of instances."""
        return self._count

    @property
    def num_bodies(self) -> int:
        """Number of bodies per instance."""
        return self._num_bodies

    # -- Getters --

    def get_net_contact_forces(self, dt: float) -> wp.array:
        """Get net contact forces on all bodies.

        Args:
            dt: Physics timestep (unused in mock, but required for API compatibility).

        Returns:
            Warp array of shape (N*B, 3) with dtype=wp.float32.
        """
        if self._net_contact_forces is None:
            self._net_contact_forces = wp.zeros((self._total_bodies, 3), dtype=wp.float32, device=self._device)
        return wp.clone(self._net_contact_forces)

    def get_contact_force_matrix(self, dt: float) -> wp.array:
        """Get contact force matrix (per-filter forces).

        Args:
            dt: Physics timestep (unused in mock, but required for API compatibility).

        Returns:
            Warp array of shape (N*B, F, 3) with dtype=wp.float32.
        """
        if self._contact_force_matrix is None:
            self._contact_force_matrix = wp.zeros(
                (self._total_bodies, self._filter_count, 3), dtype=wp.float32, device=self._device
            )
        return wp.clone(self._contact_force_matrix)

    def get_contact_data(self, dt: float) -> tuple[wp.array, wp.array, wp.array, wp.array, wp.array, wp.array]:
        """Get detailed contact data.

        Args:
            dt: Physics timestep (unused in mock, but required for API compatibility).

        Returns:
            Tuple of 6 arrays:
                - contact_positions: (N*B, max_contacts, 3) dtype=wp.float32
                - contact_normals: (N*B, max_contacts, 3) dtype=wp.float32
                - contact_impulses: (N*B, max_contacts, 3) dtype=wp.float32
                - contact_separations: (N*B, max_contacts) dtype=wp.float32
                - contact_num_found: (N*B,) dtype=wp.int32
                - contact_patch_id: (N*B, max_contacts) dtype=wp.int32
        """
        max_contacts = self._max_contact_data_count

        if self._contact_positions is None:
            self._contact_positions = wp.zeros(
                (self._total_bodies, max_contacts, 3), dtype=wp.float32, device=self._device
            )
        if self._contact_normals is None:
            self._contact_normals = wp.zeros(
                (self._total_bodies, max_contacts, 3), dtype=wp.float32, device=self._device
            )
        if self._contact_impulses is None:
            self._contact_impulses = wp.zeros(
                (self._total_bodies, max_contacts, 3), dtype=wp.float32, device=self._device
            )
        if self._contact_separations is None:
            self._contact_separations = wp.zeros(
                (self._total_bodies, max_contacts), dtype=wp.float32, device=self._device
            )
        if self._contact_num_found is None:
            self._contact_num_found = wp.zeros(self._total_bodies, dtype=wp.int32, device=self._device)
        if self._contact_patch_id is None:
            self._contact_patch_id = wp.zeros((self._total_bodies, max_contacts), dtype=wp.int32, device=self._device)

        return (
            wp.clone(self._contact_positions),
            wp.clone(self._contact_normals),
            wp.clone(self._contact_impulses),
            wp.clone(self._contact_separations),
            wp.clone(self._contact_num_found),
            wp.clone(self._contact_patch_id),
        )

    def get_friction_data(self, dt: float) -> tuple[wp.array, wp.array, wp.array, wp.array]:
        """Get friction data.

        Args:
            dt: Physics timestep (unused in mock, but required for API compatibility).

        Returns:
            Tuple of 4 arrays:
                - friction_forces: (N*B, max_contacts, 3) dtype=wp.float32
                - friction_impulses: (N*B, max_contacts, 3) dtype=wp.float32
                - friction_points: (N*B, max_contacts, 3) dtype=wp.float32
                - friction_patch_id: (N*B, max_contacts) dtype=wp.int32
        """
        max_contacts = self._max_contact_data_count

        if self._friction_forces is None:
            self._friction_forces = wp.zeros(
                (self._total_bodies, max_contacts, 3), dtype=wp.float32, device=self._device
            )
        if self._friction_impulses is None:
            self._friction_impulses = wp.zeros(
                (self._total_bodies, max_contacts, 3), dtype=wp.float32, device=self._device
            )
        if self._friction_points is None:
            self._friction_points = wp.zeros(
                (self._total_bodies, max_contacts, 3), dtype=wp.float32, device=self._device
            )
        if self._friction_patch_id is None:
            self._friction_patch_id = wp.zeros((self._total_bodies, max_contacts), dtype=wp.int32, device=self._device)

        return (
            wp.clone(self._friction_forces),
            wp.clone(self._friction_impulses),
            wp.clone(self._friction_points),
            wp.clone(self._friction_patch_id),
        )

    # -- Mock setters (direct test data injection) --

    def set_mock_net_contact_forces(self, forces: wp.array) -> None:
        """Set mock net contact force data directly for testing.

        Args:
            forces: Warp array of shape (N*B, 3) with dtype=wp.float32.
        """
        self._net_contact_forces = wp.clone(forces)
        if self._net_contact_forces.device.alias != self._device:
            self._net_contact_forces = self._net_contact_forces.to(self._device)

    def set_mock_contact_force_matrix(self, matrix: wp.array) -> None:
        """Set mock contact force matrix data directly for testing.

        Args:
            matrix: Warp array of shape (N*B, F, 3) with dtype=wp.float32.
        """
        self._contact_force_matrix = wp.clone(matrix)
        if self._contact_force_matrix.device.alias != self._device:
            self._contact_force_matrix = self._contact_force_matrix.to(self._device)

    def set_mock_contact_data(
        self,
        positions: wp.array | None = None,
        normals: wp.array | None = None,
        impulses: wp.array | None = None,
        separations: wp.array | None = None,
        num_found: wp.array | None = None,
        patch_id: wp.array | None = None,
    ) -> None:
        """Set mock contact data directly for testing.

        Args:
            positions: Contact positions, shape (N*B, max_contacts, 3) dtype=wp.float32.
            normals: Contact normals, shape (N*B, max_contacts, 3) dtype=wp.float32.
            impulses: Contact impulses, shape (N*B, max_contacts, 3) dtype=wp.float32.
            separations: Contact separations, shape (N*B, max_contacts) dtype=wp.float32.
            num_found: Number of contacts found, shape (N*B,) dtype=wp.int32.
            patch_id: Contact patch IDs, shape (N*B, max_contacts) dtype=wp.int32.
        """
        if positions is not None:
            self._contact_positions = wp.clone(positions)
            if self._contact_positions.device.alias != self._device:
                self._contact_positions = self._contact_positions.to(self._device)
        if normals is not None:
            self._contact_normals = wp.clone(normals)
            if self._contact_normals.device.alias != self._device:
                self._contact_normals = self._contact_normals.to(self._device)
        if impulses is not None:
            self._contact_impulses = wp.clone(impulses)
            if self._contact_impulses.device.alias != self._device:
                self._contact_impulses = self._contact_impulses.to(self._device)
        if separations is not None:
            self._contact_separations = wp.clone(separations)
            if self._contact_separations.device.alias != self._device:
                self._contact_separations = self._contact_separations.to(self._device)
        if num_found is not None:
            self._contact_num_found = wp.clone(num_found)
            if self._contact_num_found.device.alias != self._device:
                self._contact_num_found = self._contact_num_found.to(self._device)
        if patch_id is not None:
            self._contact_patch_id = wp.clone(patch_id)
            if self._contact_patch_id.device.alias != self._device:
                self._contact_patch_id = self._contact_patch_id.to(self._device)

    def set_mock_friction_data(
        self,
        forces: wp.array | None = None,
        impulses: wp.array | None = None,
        points: wp.array | None = None,
        patch_id: wp.array | None = None,
    ) -> None:
        """Set mock friction data directly for testing.

        Args:
            forces: Friction forces, shape (N*B, max_contacts, 3) dtype=wp.float32.
            impulses: Friction impulses, shape (N*B, max_contacts, 3) dtype=wp.float32.
            points: Friction application points, shape (N*B, max_contacts, 3) dtype=wp.float32.
            patch_id: Friction patch IDs, shape (N*B, max_contacts) dtype=wp.int32.
        """
        if forces is not None:
            self._friction_forces = wp.clone(forces)
            if self._friction_forces.device.alias != self._device:
                self._friction_forces = self._friction_forces.to(self._device)
        if impulses is not None:
            self._friction_impulses = wp.clone(impulses)
            if self._friction_impulses.device.alias != self._device:
                self._friction_impulses = self._friction_impulses.to(self._device)
        if points is not None:
            self._friction_points = wp.clone(points)
            if self._friction_points.device.alias != self._device:
                self._friction_points = self._friction_points.to(self._device)
        if patch_id is not None:
            self._friction_patch_id = wp.clone(patch_id)
            if self._friction_patch_id.device.alias != self._device:
                self._friction_patch_id = self._friction_patch_id.to(self._device)
