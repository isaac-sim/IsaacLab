# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mock implementation of PhysX RigidContactView."""

from __future__ import annotations

import torch


class MockRigidContactView:
    """Mock implementation of physx.RigidContactView for unit testing.

    This class mimics the interface of the PhysX TensorAPI RigidContactView,
    allowing tests to run without Isaac Sim or GPU simulation.

    Data Shapes:
        - net_contact_forces: (N*B, 3) - flattened net forces
        - contact_force_matrix: (N*B, F, 3) - per-filter forces
        - contact_data: tuple of 6 tensors (positions, normals, impulses, separations, num_found, patch_id)
        - friction_data: tuple of 4 tensors (friction_forces, friction_impulses, friction_points, patch_id)

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
            device: Device for tensor allocation ("cpu" or "cuda").
        """
        self._count = count
        self._num_bodies = num_bodies
        self._filter_count = filter_count
        self._max_contact_data_count = max_contact_data_count
        self._device = device

        # Total number of bodies (flattened)
        self._total_bodies = count * num_bodies

        # Internal state (lazily initialized)
        self._net_contact_forces: torch.Tensor | None = None
        self._contact_force_matrix: torch.Tensor | None = None
        self._contact_positions: torch.Tensor | None = None
        self._contact_normals: torch.Tensor | None = None
        self._contact_impulses: torch.Tensor | None = None
        self._contact_separations: torch.Tensor | None = None
        self._contact_num_found: torch.Tensor | None = None
        self._contact_patch_id: torch.Tensor | None = None
        self._friction_forces: torch.Tensor | None = None
        self._friction_impulses: torch.Tensor | None = None
        self._friction_points: torch.Tensor | None = None
        self._friction_patch_id: torch.Tensor | None = None

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

    def get_net_contact_forces(self, dt: float) -> torch.Tensor:
        """Get net contact forces on all bodies.

        Args:
            dt: Physics timestep (unused in mock, but required for API compatibility).

        Returns:
            Tensor of shape (N*B, 3) with net forces per body.
        """
        if self._net_contact_forces is None:
            self._net_contact_forces = torch.zeros(self._total_bodies, 3, device=self._device)
        return self._net_contact_forces.clone()

    def get_contact_force_matrix(self, dt: float) -> torch.Tensor:
        """Get contact force matrix (per-filter forces).

        Args:
            dt: Physics timestep (unused in mock, but required for API compatibility).

        Returns:
            Tensor of shape (N*B, F, 3) with forces per body per filter.
        """
        if self._contact_force_matrix is None:
            self._contact_force_matrix = torch.zeros(self._total_bodies, self._filter_count, 3, device=self._device)
        return self._contact_force_matrix.clone()

    def get_contact_data(
        self, dt: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get detailed contact data.

        Args:
            dt: Physics timestep (unused in mock, but required for API compatibility).

        Returns:
            Tuple of 6 tensors:
                - contact_positions: (N*B, max_contacts, 3) - contact point positions
                - contact_normals: (N*B, max_contacts, 3) - contact normals
                - contact_impulses: (N*B, max_contacts, 3) - contact impulses
                - contact_separations: (N*B, max_contacts) - contact separations
                - contact_num_found: (N*B,) - number of contacts found
                - contact_patch_id: (N*B, max_contacts) - contact patch IDs
        """
        max_contacts = self._max_contact_data_count

        if self._contact_positions is None:
            self._contact_positions = torch.zeros(self._total_bodies, max_contacts, 3, device=self._device)
        if self._contact_normals is None:
            self._contact_normals = torch.zeros(self._total_bodies, max_contacts, 3, device=self._device)
        if self._contact_impulses is None:
            self._contact_impulses = torch.zeros(self._total_bodies, max_contacts, 3, device=self._device)
        if self._contact_separations is None:
            self._contact_separations = torch.zeros(self._total_bodies, max_contacts, device=self._device)
        if self._contact_num_found is None:
            self._contact_num_found = torch.zeros(self._total_bodies, dtype=torch.int32, device=self._device)
        if self._contact_patch_id is None:
            self._contact_patch_id = torch.zeros(
                self._total_bodies, max_contacts, dtype=torch.int32, device=self._device
            )

        return (
            self._contact_positions.clone(),
            self._contact_normals.clone(),
            self._contact_impulses.clone(),
            self._contact_separations.clone(),
            self._contact_num_found.clone(),
            self._contact_patch_id.clone(),
        )

    def get_friction_data(self, dt: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get friction data.

        Args:
            dt: Physics timestep (unused in mock, but required for API compatibility).

        Returns:
            Tuple of 4 tensors:
                - friction_forces: (N*B, max_contacts, 3) - friction forces
                - friction_impulses: (N*B, max_contacts, 3) - friction impulses
                - friction_points: (N*B, max_contacts, 3) - friction application points
                - friction_patch_id: (N*B, max_contacts) - friction patch IDs
        """
        max_contacts = self._max_contact_data_count

        if self._friction_forces is None:
            self._friction_forces = torch.zeros(self._total_bodies, max_contacts, 3, device=self._device)
        if self._friction_impulses is None:
            self._friction_impulses = torch.zeros(self._total_bodies, max_contacts, 3, device=self._device)
        if self._friction_points is None:
            self._friction_points = torch.zeros(self._total_bodies, max_contacts, 3, device=self._device)
        if self._friction_patch_id is None:
            self._friction_patch_id = torch.zeros(
                self._total_bodies, max_contacts, dtype=torch.int32, device=self._device
            )

        return (
            self._friction_forces.clone(),
            self._friction_impulses.clone(),
            self._friction_points.clone(),
            self._friction_patch_id.clone(),
        )

    # -- Mock setters (direct test data injection) --

    def set_mock_net_contact_forces(self, forces: torch.Tensor) -> None:
        """Set mock net contact force data directly for testing.

        Args:
            forces: Tensor of shape (N*B, 3).
        """
        self._net_contact_forces = forces.to(self._device)

    def set_mock_contact_force_matrix(self, matrix: torch.Tensor) -> None:
        """Set mock contact force matrix data directly for testing.

        Args:
            matrix: Tensor of shape (N*B, F, 3).
        """
        self._contact_force_matrix = matrix.to(self._device)

    def set_mock_contact_data(
        self,
        positions: torch.Tensor | None = None,
        normals: torch.Tensor | None = None,
        impulses: torch.Tensor | None = None,
        separations: torch.Tensor | None = None,
        num_found: torch.Tensor | None = None,
        patch_id: torch.Tensor | None = None,
    ) -> None:
        """Set mock contact data directly for testing.

        Args:
            positions: Contact positions, shape (N*B, max_contacts, 3).
            normals: Contact normals, shape (N*B, max_contacts, 3).
            impulses: Contact impulses, shape (N*B, max_contacts, 3).
            separations: Contact separations, shape (N*B, max_contacts).
            num_found: Number of contacts found, shape (N*B,).
            patch_id: Contact patch IDs, shape (N*B, max_contacts).
        """
        if positions is not None:
            self._contact_positions = positions.to(self._device)
        if normals is not None:
            self._contact_normals = normals.to(self._device)
        if impulses is not None:
            self._contact_impulses = impulses.to(self._device)
        if separations is not None:
            self._contact_separations = separations.to(self._device)
        if num_found is not None:
            self._contact_num_found = num_found.to(self._device)
        if patch_id is not None:
            self._contact_patch_id = patch_id.to(self._device)

    def set_mock_friction_data(
        self,
        forces: torch.Tensor | None = None,
        impulses: torch.Tensor | None = None,
        points: torch.Tensor | None = None,
        patch_id: torch.Tensor | None = None,
    ) -> None:
        """Set mock friction data directly for testing.

        Args:
            forces: Friction forces, shape (N*B, max_contacts, 3).
            impulses: Friction impulses, shape (N*B, max_contacts, 3).
            points: Friction application points, shape (N*B, max_contacts, 3).
            patch_id: Friction patch IDs, shape (N*B, max_contacts).
        """
        if forces is not None:
            self._friction_forces = forces.to(self._device)
        if impulses is not None:
            self._friction_impulses = impulses.to(self._device)
        if points is not None:
            self._friction_points = points.to(self._device)
        if patch_id is not None:
            self._friction_patch_id = patch_id.to(self._device)
