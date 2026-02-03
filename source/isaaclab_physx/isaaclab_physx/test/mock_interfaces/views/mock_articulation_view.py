# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mock implementation of PhysX ArticulationView."""

from __future__ import annotations

import torch

from ..utils.mock_shared_metatype import MockSharedMetatype


class MockArticulationView:
    """Mock implementation of physx.ArticulationView for unit testing.

    This class mimics the interface of the PhysX TensorAPI ArticulationView,
    allowing tests to run without Isaac Sim or GPU simulation.

    Data Shapes:
        - root_transforms: (N, 7) - [pos(3), quat_xyzw(4)]
        - root_velocities: (N, 6) - [lin_vel(3), ang_vel(3)]
        - link_transforms: (N, L, 7) - per-link poses
        - link_velocities: (N, L, 6) - per-link velocities
        - dof_positions: (N, J) - joint positions
        - dof_velocities: (N, J) - joint velocities
        - dof_limits: (N, J, 2) - [lower, upper] limits
        - dof_stiffnesses: (N, J) - joint stiffnesses
        - dof_dampings: (N, J) - joint dampings
        - dof_max_forces: (N, J) - maximum joint forces
        - dof_max_velocities: (N, J) - maximum joint velocities
        - masses: (N, L) - per-link masses
        - coms: (N, L, 7) - per-link centers of mass
        - inertias: (N, L, 3, 3) - per-link inertia tensors

    Where N = count, L = num_links, J = num_dofs
    """

    def __init__(
        self,
        count: int = 1,
        num_dofs: int = 1,
        num_links: int = 2,
        dof_names: list[str] | None = None,
        link_names: list[str] | None = None,
        fixed_base: bool = False,
        prim_paths: list[str] | None = None,
        device: str = "cpu",
    ):
        """Initialize the mock articulation view.

        Args:
            count: Number of articulation instances.
            num_dofs: Number of degrees of freedom (joints).
            num_links: Number of links (bodies).
            dof_names: Names of the DOFs. Defaults to auto-generated names.
            link_names: Names of the links. Defaults to auto-generated names.
            fixed_base: Whether the articulation has a fixed base.
            prim_paths: USD prim paths for each instance.
            device: Device for tensor allocation ("cpu" or "cuda").
        """
        self._count = count
        self._num_dofs = num_dofs
        self._num_links = num_links
        self._device = device
        self._prim_paths = prim_paths or [f"/World/Articulation_{i}" for i in range(count)]

        # Create shared metatype
        self._shared_metatype = MockSharedMetatype(
            dof_count=num_dofs,
            link_count=num_links,
            dof_names=dof_names,
            link_names=link_names,
            fixed_base=fixed_base,
        )

        # Tendon properties (fixed values for mock)
        self._max_fixed_tendons = 0
        self._max_spatial_tendons = 0

        # Internal state (lazily initialized)
        self._root_transforms: torch.Tensor | None = None
        self._root_velocities: torch.Tensor | None = None
        self._link_transforms: torch.Tensor | None = None
        self._link_velocities: torch.Tensor | None = None
        self._link_accelerations: torch.Tensor | None = None
        self._link_incoming_joint_force: torch.Tensor | None = None
        self._dof_positions: torch.Tensor | None = None
        self._dof_velocities: torch.Tensor | None = None
        self._dof_projected_joint_forces: torch.Tensor | None = None
        self._dof_limits: torch.Tensor | None = None
        self._dof_stiffnesses: torch.Tensor | None = None
        self._dof_dampings: torch.Tensor | None = None
        self._dof_max_forces: torch.Tensor | None = None
        self._dof_max_velocities: torch.Tensor | None = None
        self._dof_armatures: torch.Tensor | None = None
        self._dof_friction_coefficients: torch.Tensor | None = None
        self._dof_friction_properties: torch.Tensor | None = None
        self._masses: torch.Tensor | None = None
        self._coms: torch.Tensor | None = None
        self._inertias: torch.Tensor | None = None

    # -- Helper Methods --

    def _check_cpu_tensor(self, tensor: torch.Tensor, name: str) -> None:
        """Check that tensor is on CPU, raise RuntimeError if on GPU.

        This mimics PhysX behavior where joint/body properties must be on CPU.
        """
        if tensor.is_cuda:
            raise RuntimeError(
                f"Expected CPU tensor for {name}, but got tensor on {tensor.device}. "
                "Joint and body properties must be set with CPU tensors."
            )

    # -- Properties --

    @property
    def count(self) -> int:
        """Number of articulation instances."""
        return self._count

    @property
    def shared_metatype(self) -> MockSharedMetatype:
        """Shared metatype containing articulation structure metadata."""
        return self._shared_metatype

    @property
    def max_fixed_tendons(self) -> int:
        """Maximum number of fixed tendons."""
        return self._max_fixed_tendons

    @property
    def max_spatial_tendons(self) -> int:
        """Maximum number of spatial tendons."""
        return self._max_spatial_tendons

    @property
    def prim_paths(self) -> list[str]:
        """USD prim paths for each instance."""
        return self._prim_paths

    # -- Root Getters --

    def get_root_transforms(self) -> torch.Tensor:
        """Get world transforms of root links.

        Returns:
            Tensor of shape (N, 7) with [pos(3), quat_xyzw(4)].
        """
        if self._root_transforms is None:
            self._root_transforms = torch.zeros(self._count, 7, device=self._device)
            self._root_transforms[:, 6] = 1.0  # w=1 for identity quaternion
        return self._root_transforms.clone()

    def get_root_velocities(self) -> torch.Tensor:
        """Get velocities of root links.

        Returns:
            Tensor of shape (N, 6) with [lin_vel(3), ang_vel(3)].
        """
        if self._root_velocities is None:
            self._root_velocities = torch.zeros(self._count, 6, device=self._device)
        return self._root_velocities.clone()

    # -- Link Getters --

    def get_link_transforms(self) -> torch.Tensor:
        """Get world transforms of all links.

        Returns:
            Tensor of shape (N, L, 7) with [pos(3), quat_xyzw(4)] per link.
        """
        if self._link_transforms is None:
            self._link_transforms = torch.zeros(self._count, self._num_links, 7, device=self._device)
            self._link_transforms[:, :, 6] = 1.0  # w=1 for identity quaternion
        return self._link_transforms.clone()

    def get_link_velocities(self) -> torch.Tensor:
        """Get velocities of all links.

        Returns:
            Tensor of shape (N, L, 6) with [lin_vel(3), ang_vel(3)] per link.
        """
        if self._link_velocities is None:
            self._link_velocities = torch.zeros(self._count, self._num_links, 6, device=self._device)
        return self._link_velocities.clone()

    # -- DOF Getters --

    def get_dof_positions(self) -> torch.Tensor:
        """Get positions of all DOFs.

        Returns:
            Tensor of shape (N, J) with joint positions.
        """
        if self._dof_positions is None:
            self._dof_positions = torch.zeros(self._count, self._num_dofs, device=self._device)
        return self._dof_positions.clone()

    def get_dof_velocities(self) -> torch.Tensor:
        """Get velocities of all DOFs.

        Returns:
            Tensor of shape (N, J) with joint velocities.
        """
        if self._dof_velocities is None:
            self._dof_velocities = torch.zeros(self._count, self._num_dofs, device=self._device)
        return self._dof_velocities.clone()

    def get_dof_projected_joint_forces(self) -> torch.Tensor:
        """Get projected joint forces of all DOFs.

        Returns:
            Tensor of shape (N, J) with projected joint forces.
        """
        if self._dof_projected_joint_forces is None:
            self._dof_projected_joint_forces = torch.zeros(self._count, self._num_dofs, device=self._device)
        return self._dof_projected_joint_forces.clone()

    def get_dof_limits(self) -> torch.Tensor:
        """Get position limits of all DOFs.

        Returns:
            Tensor of shape (N, J, 2) with [lower, upper] limits. Always on CPU.
        """
        if self._dof_limits is None:
            # Default: no limits (infinite) - stored on CPU
            self._dof_limits = torch.zeros(self._count, self._num_dofs, 2, device="cpu")
            self._dof_limits[:, :, 0] = float("-inf")  # lower limit
            self._dof_limits[:, :, 1] = float("inf")  # upper limit
        return self._dof_limits.clone()

    def get_dof_stiffnesses(self) -> torch.Tensor:
        """Get stiffnesses of all DOFs.

        Returns:
            Tensor of shape (N, J) with joint stiffnesses. Always on CPU.
        """
        if self._dof_stiffnesses is None:
            self._dof_stiffnesses = torch.zeros(self._count, self._num_dofs, device="cpu")
        return self._dof_stiffnesses.clone()

    def get_dof_dampings(self) -> torch.Tensor:
        """Get dampings of all DOFs.

        Returns:
            Tensor of shape (N, J) with joint dampings. Always on CPU.
        """
        if self._dof_dampings is None:
            self._dof_dampings = torch.zeros(self._count, self._num_dofs, device="cpu")
        return self._dof_dampings.clone()

    def get_dof_max_forces(self) -> torch.Tensor:
        """Get maximum forces of all DOFs.

        Returns:
            Tensor of shape (N, J) with maximum joint forces. Always on CPU.
        """
        if self._dof_max_forces is None:
            # Default: infinite max force - stored on CPU
            self._dof_max_forces = torch.full((self._count, self._num_dofs), float("inf"), device="cpu")
        return self._dof_max_forces.clone()

    def get_dof_max_velocities(self) -> torch.Tensor:
        """Get maximum velocities of all DOFs.

        Returns:
            Tensor of shape (N, J) with maximum joint velocities. Always on CPU.
        """
        if self._dof_max_velocities is None:
            # Default: infinite max velocity - stored on CPU
            self._dof_max_velocities = torch.full((self._count, self._num_dofs), float("inf"), device="cpu")
        return self._dof_max_velocities.clone()

    def get_dof_armatures(self) -> torch.Tensor:
        """Get armatures of all DOFs.

        Returns:
            Tensor of shape (N, J) with joint armatures. Always on CPU.
        """
        if self._dof_armatures is None:
            self._dof_armatures = torch.zeros(self._count, self._num_dofs, device="cpu")
        return self._dof_armatures.clone()

    def get_dof_friction_coefficients(self) -> torch.Tensor:
        """Get friction coefficients of all DOFs.

        Returns:
            Tensor of shape (N, J) with joint friction coefficients. Always on CPU.
        """
        if self._dof_friction_coefficients is None:
            self._dof_friction_coefficients = torch.zeros(self._count, self._num_dofs, device="cpu")
        return self._dof_friction_coefficients.clone()

    # -- Mass Property Getters --

    def get_masses(self) -> torch.Tensor:
        """Get masses of all links.

        Returns:
            Tensor of shape (N, L) with link masses. Always on CPU.
        """
        if self._masses is None:
            self._masses = torch.ones(self._count, self._num_links, device="cpu")
        return self._masses.clone()

    def get_coms(self) -> torch.Tensor:
        """Get centers of mass of all links.

        Returns:
            Tensor of shape (N, L, 7) with [pos(3), quat_xyzw(4)] per link. Always on CPU.
        """
        if self._coms is None:
            self._coms = torch.zeros(self._count, self._num_links, 7, device="cpu")
            self._coms[:, :, 6] = 1.0  # w=1 for identity quaternion
        return self._coms.clone()

    def get_inertias(self) -> torch.Tensor:
        """Get inertia tensors of all links.

        Returns:
            Tensor of shape (N, L, 9) with flattened 3x3 inertia matrices per link (row-major). Always on CPU.
        """
        if self._inertias is None:
            # Default: identity inertia - flattened [1,0,0,0,1,0,0,0,1] - stored on CPU
            self._inertias = torch.zeros(self._count, self._num_links, 9, device="cpu")
            self._inertias[:, :, 0] = 1.0  # [0,0]
            self._inertias[:, :, 4] = 1.0  # [1,1]
            self._inertias[:, :, 8] = 1.0  # [2,2]
        return self._inertias.clone()

    # -- Root Setters --

    def set_root_transforms(
        self,
        transforms: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> None:
        """Set world transforms of root links.

        Args:
            transforms: Tensor of shape (N, 7) or (len(indices), 7).
            indices: Optional indices of articulations to update.
        """
        transforms = transforms.to(self._device)
        if self._root_transforms is None:
            self._root_transforms = torch.zeros(self._count, 7, device=self._device)
            self._root_transforms[:, 6] = 1.0
        if indices is not None:
            self._root_transforms[indices] = transforms
        else:
            self._root_transforms = transforms

    def set_root_velocities(
        self,
        velocities: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> None:
        """Set velocities of root links.

        Args:
            velocities: Tensor of shape (N, 6) or (len(indices), 6).
            indices: Optional indices of articulations to update.
        """
        velocities = velocities.to(self._device)
        if self._root_velocities is None:
            self._root_velocities = torch.zeros(self._count, 6, device=self._device)
        if indices is not None:
            self._root_velocities[indices] = velocities
        else:
            self._root_velocities = velocities

    # -- DOF Setters --

    def set_dof_positions(
        self,
        positions: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> None:
        """Set positions of all DOFs.

        Args:
            positions: Tensor of shape (N, J) or (len(indices), J).
            indices: Optional indices of articulations to update.
        """
        positions = positions.to(self._device)
        if self._dof_positions is None:
            self._dof_positions = torch.zeros(self._count, self._num_dofs, device=self._device)
        if indices is not None:
            self._dof_positions[indices] = positions
        else:
            self._dof_positions = positions

    def set_dof_velocities(
        self,
        velocities: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> None:
        """Set velocities of all DOFs.

        Args:
            velocities: Tensor of shape (N, J) or (len(indices), J).
            indices: Optional indices of articulations to update.
        """
        velocities = velocities.to(self._device)
        if self._dof_velocities is None:
            self._dof_velocities = torch.zeros(self._count, self._num_dofs, device=self._device)
        if indices is not None:
            self._dof_velocities[indices] = velocities
        else:
            self._dof_velocities = velocities

    def set_dof_position_targets(
        self,
        targets: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> None:
        """Set position targets for all DOFs (no-op in mock).

        Args:
            targets: Tensor of shape (N, J) or (len(indices), J).
            indices: Optional indices of articulations to update.
        """
        pass  # No-op for mock

    def set_dof_velocity_targets(
        self,
        targets: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> None:
        """Set velocity targets for all DOFs (no-op in mock).

        Args:
            targets: Tensor of shape (N, J) or (len(indices), J).
            indices: Optional indices of articulations to update.
        """
        pass  # No-op for mock

    def set_dof_actuation_forces(
        self,
        forces: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> None:
        """Set actuation forces for all DOFs (no-op in mock).

        Args:
            forces: Tensor of shape (N, J) or (len(indices), J).
            indices: Optional indices of articulations to update.
        """
        pass  # No-op for mock

    def set_dof_limits(
        self,
        limits: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> None:
        """Set position limits of all DOFs.

        Args:
            limits: Tensor of shape (N, J, 2) with [lower, upper] limits. Must be on CPU.
            indices: Optional indices of articulations to update.

        Raises:
            RuntimeError: If limits tensor is on GPU.
        """
        self._check_cpu_tensor(limits, "dof_limits")
        if self._dof_limits is None:
            self._dof_limits = torch.zeros(self._count, self._num_dofs, 2, device="cpu")
            self._dof_limits[:, :, 0] = float("-inf")
            self._dof_limits[:, :, 1] = float("inf")
        if indices is not None:
            self._dof_limits[indices] = limits
        else:
            self._dof_limits = limits

    def set_dof_stiffnesses(
        self,
        stiffnesses: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> None:
        """Set stiffnesses of all DOFs.

        Args:
            stiffnesses: Tensor of shape (N, J). Must be on CPU.
            indices: Optional indices of articulations to update.

        Raises:
            RuntimeError: If stiffnesses tensor is on GPU.
        """
        self._check_cpu_tensor(stiffnesses, "dof_stiffnesses")
        if self._dof_stiffnesses is None:
            self._dof_stiffnesses = torch.zeros(self._count, self._num_dofs, device="cpu")
        if indices is not None:
            self._dof_stiffnesses[indices] = stiffnesses
        else:
            self._dof_stiffnesses = stiffnesses

    def set_dof_dampings(
        self,
        dampings: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> None:
        """Set dampings of all DOFs.

        Args:
            dampings: Tensor of shape (N, J). Must be on CPU.
            indices: Optional indices of articulations to update.

        Raises:
            RuntimeError: If dampings tensor is on GPU.
        """
        self._check_cpu_tensor(dampings, "dof_dampings")
        if self._dof_dampings is None:
            self._dof_dampings = torch.zeros(self._count, self._num_dofs, device="cpu")
        if indices is not None:
            self._dof_dampings[indices] = dampings
        else:
            self._dof_dampings = dampings

    def set_dof_max_forces(
        self,
        max_forces: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> None:
        """Set maximum forces of all DOFs.

        Args:
            max_forces: Tensor of shape (N, J). Must be on CPU.
            indices: Optional indices of articulations to update.

        Raises:
            RuntimeError: If max_forces tensor is on GPU.
        """
        self._check_cpu_tensor(max_forces, "dof_max_forces")
        if self._dof_max_forces is None:
            self._dof_max_forces = torch.full((self._count, self._num_dofs), float("inf"), device="cpu")
        if indices is not None:
            self._dof_max_forces[indices] = max_forces
        else:
            self._dof_max_forces = max_forces

    def set_dof_max_velocities(
        self,
        max_velocities: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> None:
        """Set maximum velocities of all DOFs.

        Args:
            max_velocities: Tensor of shape (N, J). Must be on CPU.
            indices: Optional indices of articulations to update.

        Raises:
            RuntimeError: If max_velocities tensor is on GPU.
        """
        self._check_cpu_tensor(max_velocities, "dof_max_velocities")
        if self._dof_max_velocities is None:
            self._dof_max_velocities = torch.full((self._count, self._num_dofs), float("inf"), device="cpu")
        if indices is not None:
            self._dof_max_velocities[indices] = max_velocities
        else:
            self._dof_max_velocities = max_velocities

    def set_dof_armatures(
        self,
        armatures: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> None:
        """Set armatures of all DOFs.

        Args:
            armatures: Tensor of shape (N, J). Must be on CPU.
            indices: Optional indices of articulations to update.

        Raises:
            RuntimeError: If armatures tensor is on GPU.
        """
        self._check_cpu_tensor(armatures, "dof_armatures")
        if self._dof_armatures is None:
            self._dof_armatures = torch.zeros(self._count, self._num_dofs, device="cpu")
        if indices is not None:
            self._dof_armatures[indices] = armatures
        else:
            self._dof_armatures = armatures

    def set_dof_friction_coefficients(
        self,
        friction_coefficients: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> None:
        """Set friction coefficients of all DOFs.

        Args:
            friction_coefficients: Tensor of shape (N, J). Must be on CPU.
            indices: Optional indices of articulations to update.

        Raises:
            RuntimeError: If friction_coefficients tensor is on GPU.
        """
        self._check_cpu_tensor(friction_coefficients, "dof_friction_coefficients")
        if self._dof_friction_coefficients is None:
            self._dof_friction_coefficients = torch.zeros(self._count, self._num_dofs, device="cpu")
        if indices is not None:
            self._dof_friction_coefficients[indices] = friction_coefficients
        else:
            self._dof_friction_coefficients = friction_coefficients

    # -- Mass Property Setters --

    def set_masses(
        self,
        masses: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> None:
        """Set masses of all links.

        Args:
            masses: Tensor of shape (N, L). Must be on CPU.
            indices: Optional indices of articulations to update.

        Raises:
            RuntimeError: If masses tensor is on GPU.
        """
        self._check_cpu_tensor(masses, "masses")
        if self._masses is None:
            self._masses = torch.ones(self._count, self._num_links, device="cpu")
        if indices is not None:
            self._masses[indices] = masses
        else:
            self._masses = masses

    def set_coms(
        self,
        coms: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> None:
        """Set centers of mass of all links.

        Args:
            coms: Tensor of shape (N, L, 7). Must be on CPU.
            indices: Optional indices of articulations to update.

        Raises:
            RuntimeError: If coms tensor is on GPU.
        """
        self._check_cpu_tensor(coms, "coms")
        if self._coms is None:
            self._coms = torch.zeros(self._count, self._num_links, 7, device="cpu")
            self._coms[:, :, 6] = 1.0
        if indices is not None:
            self._coms[indices] = coms
        else:
            self._coms = coms

    def set_inertias(
        self,
        inertias: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> None:
        """Set inertia tensors of all links.

        Args:
            inertias: Tensor of shape (N, L, 9) - flattened 3x3 matrices. Must be on CPU.
            indices: Optional indices of articulations to update.

        Raises:
            RuntimeError: If inertias tensor is on GPU.
        """
        self._check_cpu_tensor(inertias, "inertias")
        if self._inertias is None:
            self._inertias = torch.zeros(self._count, self._num_links, 9, device="cpu")
            self._inertias[:, :, 0] = 1.0
            self._inertias[:, :, 4] = 1.0
            self._inertias[:, :, 8] = 1.0
        if indices is not None:
            self._inertias[indices] = inertias
        else:
            self._inertias = inertias

    # -- Mock setters (direct test data injection) --

    def set_mock_root_transforms(self, transforms: torch.Tensor) -> None:
        """Set mock root transform data directly for testing.

        Args:
            transforms: Tensor of shape (N, 7).
        """
        self._root_transforms = transforms.to(self._device)

    def set_mock_root_velocities(self, velocities: torch.Tensor) -> None:
        """Set mock root velocity data directly for testing.

        Args:
            velocities: Tensor of shape (N, 6).
        """
        self._root_velocities = velocities.to(self._device)

    def set_mock_link_transforms(self, transforms: torch.Tensor) -> None:
        """Set mock link transform data directly for testing.

        Args:
            transforms: Tensor of shape (N, L, 7).
        """
        self._link_transforms = transforms.to(self._device)

    def set_mock_link_velocities(self, velocities: torch.Tensor) -> None:
        """Set mock link velocity data directly for testing.

        Args:
            velocities: Tensor of shape (N, L, 6).
        """
        self._link_velocities = velocities.to(self._device)

    def set_mock_dof_positions(self, positions: torch.Tensor) -> None:
        """Set mock DOF position data directly for testing.

        Args:
            positions: Tensor of shape (N, J).
        """
        self._dof_positions = positions.to(self._device)

    def set_mock_dof_velocities(self, velocities: torch.Tensor) -> None:
        """Set mock DOF velocity data directly for testing.

        Args:
            velocities: Tensor of shape (N, J).
        """
        self._dof_velocities = velocities.to(self._device)

    def set_mock_dof_projected_joint_forces(self, forces: torch.Tensor) -> None:
        """Set mock projected joint force data directly for testing.

        Args:
            forces: Tensor of shape (N, J).
        """
        self._dof_projected_joint_forces = forces.to(self._device)

    def set_mock_dof_limits(self, limits: torch.Tensor) -> None:
        """Set mock DOF limit data directly for testing.

        Args:
            limits: Tensor of shape (N, J, 2).
        """
        self._dof_limits = limits.to(self._device)

    def set_mock_dof_stiffnesses(self, stiffnesses: torch.Tensor) -> None:
        """Set mock DOF stiffness data directly for testing.

        Args:
            stiffnesses: Tensor of shape (N, J).
        """
        self._dof_stiffnesses = stiffnesses.to(self._device)

    def set_mock_dof_dampings(self, dampings: torch.Tensor) -> None:
        """Set mock DOF damping data directly for testing.

        Args:
            dampings: Tensor of shape (N, J).
        """
        self._dof_dampings = dampings.to(self._device)

    def set_mock_dof_max_forces(self, max_forces: torch.Tensor) -> None:
        """Set mock DOF max force data directly for testing.

        Args:
            max_forces: Tensor of shape (N, J).
        """
        self._dof_max_forces = max_forces.to(self._device)

    def set_mock_dof_max_velocities(self, max_velocities: torch.Tensor) -> None:
        """Set mock DOF max velocity data directly for testing.

        Args:
            max_velocities: Tensor of shape (N, J).
        """
        self._dof_max_velocities = max_velocities.to(self._device)

    def set_mock_dof_armatures(self, armatures: torch.Tensor) -> None:
        """Set mock DOF armature data directly for testing.

        Args:
            armatures: Tensor of shape (N, J).
        """
        self._dof_armatures = armatures.to(self._device)

    def set_mock_dof_friction_coefficients(self, friction_coefficients: torch.Tensor) -> None:
        """Set mock DOF friction coefficient data directly for testing.

        Args:
            friction_coefficients: Tensor of shape (N, J).
        """
        self._dof_friction_coefficients = friction_coefficients.to(self._device)

    def set_mock_masses(self, masses: torch.Tensor) -> None:
        """Set mock mass data directly for testing.

        Args:
            masses: Tensor of shape (N, L).
        """
        self._masses = masses.to(self._device)

    def set_mock_coms(self, coms: torch.Tensor) -> None:
        """Set mock center of mass data directly for testing.

        Args:
            coms: Tensor of shape (N, L, 7).
        """
        self._coms = coms.to(self._device)

    def set_mock_inertias(self, inertias: torch.Tensor) -> None:
        """Set mock inertia data directly for testing.

        Args:
            inertias: Tensor of shape (N, L, 3, 3).
        """
        self._inertias = inertias.to(self._device)

    # -- Additional mock state for extended properties --

    def get_dof_friction_properties(self) -> torch.Tensor:
        """Get friction properties of all DOFs.

        Returns:
            Tensor of shape (N, J, 3) with [static_friction, dynamic_friction, viscous_friction]. Always on CPU.
        """
        if self._dof_friction_properties is None:
            self._dof_friction_properties = torch.zeros(self._count, self._num_dofs, 3, device="cpu")
        return self._dof_friction_properties.clone()

    def get_link_accelerations(self) -> torch.Tensor:
        """Get accelerations of all links.

        Returns:
            Tensor of shape (N, L, 6) with [lin_acc(3), ang_acc(3)] per link.
        """
        if self._link_accelerations is None:
            self._link_accelerations = torch.zeros(self._count, self._num_links, 6, device=self._device)
        return self._link_accelerations.clone()

    def get_link_incoming_joint_force(self) -> torch.Tensor:
        """Get incoming joint forces for all links.

        Returns:
            Tensor of shape (N, L, 6) with [force(3), torque(3)] per link.
        """
        if self._link_incoming_joint_force is None:
            self._link_incoming_joint_force = torch.zeros(self._count, self._num_links, 6, device=self._device)
        return self._link_incoming_joint_force.clone()

    def set_mock_dof_friction_properties(self, friction_properties: torch.Tensor) -> None:
        """Set mock DOF friction properties data directly for testing.

        Args:
            friction_properties: Tensor of shape (N, J, 3).
        """
        self._dof_friction_properties = friction_properties.to(self._device)

    def set_mock_link_accelerations(self, accelerations: torch.Tensor) -> None:
        """Set mock link acceleration data directly for testing.

        Args:
            accelerations: Tensor of shape (N, L, 6).
        """
        self._link_accelerations = accelerations.to(self._device)

    def set_mock_link_incoming_joint_force(self, forces: torch.Tensor) -> None:
        """Set mock link incoming joint force data directly for testing.

        Args:
            forces: Tensor of shape (N, L, 6).
        """
        self._link_incoming_joint_force = forces.to(self._device)

    def set_random_mock_data(self) -> None:
        """Set all internal state to random values for benchmarking.

        This method initializes all mock data with random values,
        useful for benchmarking where the actual values don't matter.
        """
        # Root state
        self._root_transforms = torch.randn(self._count, 7, device=self._device)
        self._root_transforms[:, 3:7] = torch.nn.functional.normalize(self._root_transforms[:, 3:7], dim=-1)
        self._root_velocities = torch.randn(self._count, 6, device=self._device)

        # Link state
        self._link_transforms = torch.randn(self._count, self._num_links, 7, device=self._device)
        self._link_transforms[:, :, 3:7] = torch.nn.functional.normalize(self._link_transforms[:, :, 3:7], dim=-1)
        self._link_velocities = torch.randn(self._count, self._num_links, 6, device=self._device)
        self._link_accelerations = torch.randn(self._count, self._num_links, 6, device=self._device)
        self._link_incoming_joint_force = torch.randn(self._count, self._num_links, 6, device=self._device)

        # DOF state
        self._dof_positions = torch.randn(self._count, self._num_dofs, device=self._device)
        self._dof_velocities = torch.randn(self._count, self._num_dofs, device=self._device)
        self._dof_projected_joint_forces = torch.randn(self._count, self._num_dofs, device=self._device)

        # DOF properties - stored on CPU (PhysX requirement)
        self._dof_limits = torch.randn(self._count, self._num_dofs, 2, device="cpu")
        self._dof_stiffnesses = torch.rand(self._count, self._num_dofs, device="cpu") * 100
        self._dof_dampings = torch.rand(self._count, self._num_dofs, device="cpu") * 10
        self._dof_max_forces = torch.rand(self._count, self._num_dofs, device="cpu") * 100
        self._dof_max_velocities = torch.rand(self._count, self._num_dofs, device="cpu") * 10
        self._dof_armatures = torch.rand(self._count, self._num_dofs, device="cpu") * 0.1
        self._dof_friction_coefficients = torch.rand(self._count, self._num_dofs, device="cpu")
        self._dof_friction_properties = torch.rand(self._count, self._num_dofs, 3, device="cpu")

        # Mass properties - stored on CPU (PhysX requirement)
        self._masses = torch.rand(self._count, self._num_links, device="cpu") * 10
        self._coms = torch.randn(self._count, self._num_links, 7, device="cpu")
        self._coms[:, :, 3:7] = torch.nn.functional.normalize(self._coms[:, :, 3:7], dim=-1)
        # Inertias: (N, L, 9) flattened format
        self._inertias = torch.zeros(self._count, self._num_links, 9, device="cpu")
        self._inertias[:, :, 0] = torch.rand(self._count, self._num_links)  # [0,0]
        self._inertias[:, :, 4] = torch.rand(self._count, self._num_links)  # [1,1]
        self._inertias[:, :, 8] = torch.rand(self._count, self._num_links)  # [2,2]
