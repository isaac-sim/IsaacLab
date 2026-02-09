# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mock implementation of PhysX ArticulationView using Warp arrays."""

from __future__ import annotations

import warp as wp

from ..utils.kernels_mock import (
    copy_spatial_vectors_1d,
    copy_spatial_vectors_2d,
    copy_transforms_1d,
    copy_transforms_2d,
    init_identity_inertias_2d,
    init_identity_transforms_1d,
    init_identity_transforms_2d,
    init_zero_spatial_vectors_1d,
    init_zero_spatial_vectors_2d,
    scatter_spatial_vectors_1d,
    scatter_transforms_1d,
)
from ..utils.mock_shared_metatype import MockSharedMetatype


class MockArticulationViewWarp:
    """Mock implementation of physx.ArticulationView using Warp arrays for unit testing.

    This class mimics the interface of the PhysX TensorAPI ArticulationView using
    Warp structured types, allowing tests to run without Isaac Sim or GPU simulation.

    Data Shapes (using Warp types):
        - root_transforms: (N,) dtype=wp.transformf - [pos(3), quat_xyzw(4)]
        - root_velocities: (N,) dtype=wp.spatial_vectorf - [ang_vel(3), lin_vel(3)]
        - link_transforms: (N, L) dtype=wp.transformf - per-link poses
        - link_velocities: (N, L) dtype=wp.spatial_vectorf - per-link velocities
        - dof_positions: (N, J) dtype=wp.float32 - joint positions
        - dof_velocities: (N, J) dtype=wp.float32 - joint velocities
        - dof_limits: (N, J, 2) dtype=wp.float32 - [lower, upper] limits
        - dof_stiffnesses: (N, J) dtype=wp.float32 - joint stiffnesses
        - dof_dampings: (N, J) dtype=wp.float32 - joint dampings
        - masses: (N, L) dtype=wp.float32 - per-link masses
        - coms: (N, L) dtype=wp.transformf - per-link centers of mass
        - inertias: (N, L, 9) dtype=wp.float32 - per-link inertia tensors (flattened)

    Where N = count, L = num_links, J = num_dofs

    Note:
        wp.spatial_vectorf stores [angular(3), linear(3)] which differs from
        torch's [linear(3), angular(3)] convention.
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
            device: Device for array allocation ("cpu" or "cuda:N").
        """
        self._count = count
        self._num_dofs = num_dofs
        self._num_links = num_links
        self._device = device
        self._prim_paths = prim_paths or [f"/World/Articulation_{i}" for i in range(count)]
        self._backend = "warp"

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
        self._root_transforms: wp.array | None = None
        self._root_velocities: wp.array | None = None
        self._link_transforms: wp.array | None = None
        self._link_velocities: wp.array | None = None
        self._link_accelerations: wp.array | None = None
        self._link_incoming_joint_force: wp.array | None = None
        self._dof_positions: wp.array | None = None
        self._dof_velocities: wp.array | None = None
        self._dof_projected_joint_forces: wp.array | None = None
        self._dof_limits: wp.array | None = None
        self._dof_stiffnesses: wp.array | None = None
        self._dof_dampings: wp.array | None = None
        self._dof_max_forces: wp.array | None = None
        self._dof_max_velocities: wp.array | None = None
        self._dof_armatures: wp.array | None = None
        self._dof_friction_coefficients: wp.array | None = None
        self._dof_friction_properties: wp.array | None = None
        self._masses: wp.array | None = None
        self._coms: wp.array | None = None
        self._inertias: wp.array | None = None

    # -- Helper Methods --

    def _check_cpu_array(self, arr: wp.array, name: str) -> None:
        """Check that array is on CPU, raise RuntimeError if on GPU.

        This mimics PhysX behavior where joint/body properties must be on CPU.
        """
        if arr.device.is_cuda:
            raise RuntimeError(
                f"Expected CPU array for {name}, but got array on {arr.device}. "
                "Joint and body properties must be set with CPU arrays."
            )

    def _create_identity_transforms_1d(self, count: int) -> wp.array:
        """Create 1D array of identity transforms."""
        arr = wp.zeros(count, dtype=wp.transformf, device=self._device)
        wp.launch(init_identity_transforms_1d, dim=count, inputs=[arr], device=self._device)
        return arr

    def _create_identity_transforms_2d(self, count: int, num_links: int) -> wp.array:
        """Create 2D array of identity transforms."""
        arr = wp.zeros((count, num_links), dtype=wp.transformf, device=self._device)
        wp.launch(init_identity_transforms_2d, dim=(count, num_links), inputs=[arr], device=self._device)
        return arr

    def _create_zero_spatial_vectors_1d(self, count: int) -> wp.array:
        """Create 1D array of zero spatial vectors."""
        arr = wp.zeros(count, dtype=wp.spatial_vectorf, device=self._device)
        wp.launch(init_zero_spatial_vectors_1d, dim=count, inputs=[arr], device=self._device)
        return arr

    def _create_zero_spatial_vectors_2d(self, count: int, num_links: int) -> wp.array:
        """Create 2D array of zero spatial vectors."""
        arr = wp.zeros((count, num_links), dtype=wp.spatial_vectorf, device=self._device)
        wp.launch(init_zero_spatial_vectors_2d, dim=(count, num_links), inputs=[arr], device=self._device)
        return arr

    def _clone_transforms_1d(self, src: wp.array) -> wp.array:
        """Clone a 1D transform array."""
        dst = wp.zeros_like(src)
        wp.launch(copy_transforms_1d, dim=src.shape[0], inputs=[src, dst], device=self._device)
        return dst

    def _clone_transforms_2d(self, src: wp.array) -> wp.array:
        """Clone a 2D transform array."""
        dst = wp.zeros_like(src)
        wp.launch(copy_transforms_2d, dim=src.shape, inputs=[src, dst], device=self._device)
        return dst

    def _clone_spatial_vectors_1d(self, src: wp.array) -> wp.array:
        """Clone a 1D spatial vector array."""
        dst = wp.zeros_like(src)
        wp.launch(copy_spatial_vectors_1d, dim=src.shape[0], inputs=[src, dst], device=self._device)
        return dst

    def _clone_spatial_vectors_2d(self, src: wp.array) -> wp.array:
        """Clone a 2D spatial vector array."""
        dst = wp.zeros_like(src)
        wp.launch(copy_spatial_vectors_2d, dim=src.shape, inputs=[src, dst], device=self._device)
        return dst

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

    def get_root_transforms(self) -> wp.array:
        """Get world transforms of root links.

        Returns:
            Warp array of shape (N,) with dtype=wp.transformf.
        """
        if self._root_transforms is None:
            self._root_transforms = self._create_identity_transforms_1d(self._count)
        return self._clone_transforms_1d(self._root_transforms)

    def get_root_velocities(self) -> wp.array:
        """Get velocities of root links.

        Returns:
            Warp array of shape (N,) with dtype=wp.spatial_vectorf.
        """
        if self._root_velocities is None:
            self._root_velocities = self._create_zero_spatial_vectors_1d(self._count)
        return self._clone_spatial_vectors_1d(self._root_velocities)

    # -- Link Getters --

    def get_link_transforms(self) -> wp.array:
        """Get world transforms of all links.

        Returns:
            Warp array of shape (N, L) with dtype=wp.transformf.
        """
        if self._link_transforms is None:
            self._link_transforms = self._create_identity_transforms_2d(self._count, self._num_links)
        return self._clone_transforms_2d(self._link_transforms)

    def get_link_velocities(self) -> wp.array:
        """Get velocities of all links.

        Returns:
            Warp array of shape (N, L) with dtype=wp.spatial_vectorf.
        """
        if self._link_velocities is None:
            self._link_velocities = self._create_zero_spatial_vectors_2d(self._count, self._num_links)
        return self._clone_spatial_vectors_2d(self._link_velocities)

    def get_link_accelerations(self) -> wp.array:
        """Get accelerations of all links.

        Returns:
            Warp array of shape (N, L) with dtype=wp.spatial_vectorf.
        """
        if self._link_accelerations is None:
            self._link_accelerations = self._create_zero_spatial_vectors_2d(self._count, self._num_links)
        return self._clone_spatial_vectors_2d(self._link_accelerations)

    def get_link_incoming_joint_force(self) -> wp.array:
        """Get incoming joint forces for all links.

        Returns:
            Warp array of shape (N, L) with dtype=wp.spatial_vectorf.
        """
        if self._link_incoming_joint_force is None:
            self._link_incoming_joint_force = self._create_zero_spatial_vectors_2d(self._count, self._num_links)
        return self._clone_spatial_vectors_2d(self._link_incoming_joint_force)

    # -- DOF Getters --

    def get_dof_positions(self) -> wp.array:
        """Get positions of all DOFs.

        Returns:
            Warp array of shape (N, J) with dtype=wp.float32.
        """
        if self._dof_positions is None:
            self._dof_positions = wp.zeros((self._count, self._num_dofs), dtype=wp.float32, device=self._device)
        return wp.clone(self._dof_positions)

    def get_dof_velocities(self) -> wp.array:
        """Get velocities of all DOFs.

        Returns:
            Warp array of shape (N, J) with dtype=wp.float32.
        """
        if self._dof_velocities is None:
            self._dof_velocities = wp.zeros((self._count, self._num_dofs), dtype=wp.float32, device=self._device)
        return wp.clone(self._dof_velocities)

    def get_dof_projected_joint_forces(self) -> wp.array:
        """Get projected joint forces of all DOFs.

        Returns:
            Warp array of shape (N, J) with dtype=wp.float32.
        """
        if self._dof_projected_joint_forces is None:
            self._dof_projected_joint_forces = wp.zeros(
                (self._count, self._num_dofs), dtype=wp.float32, device=self._device
            )
        return wp.clone(self._dof_projected_joint_forces)

    def get_dof_limits(self) -> wp.array:
        """Get position limits of all DOFs.

        Returns:
            Warp array of shape (N, J, 2) with dtype=wp.float32. Always on CPU.
        """
        if self._dof_limits is None:
            import numpy as np

            limits = np.zeros((self._count, self._num_dofs, 2), dtype=np.float32)
            limits[:, :, 0] = float("-inf")  # lower limit
            limits[:, :, 1] = float("inf")  # upper limit
            self._dof_limits = wp.array(limits, dtype=wp.float32, device="cpu")
        return wp.clone(self._dof_limits)

    def get_dof_stiffnesses(self) -> wp.array:
        """Get stiffnesses of all DOFs.

        Returns:
            Warp array of shape (N, J) with dtype=wp.float32. Always on CPU.
        """
        if self._dof_stiffnesses is None:
            self._dof_stiffnesses = wp.zeros((self._count, self._num_dofs), dtype=wp.float32, device="cpu")
        return wp.clone(self._dof_stiffnesses)

    def get_dof_dampings(self) -> wp.array:
        """Get dampings of all DOFs.

        Returns:
            Warp array of shape (N, J) with dtype=wp.float32. Always on CPU.
        """
        if self._dof_dampings is None:
            self._dof_dampings = wp.zeros((self._count, self._num_dofs), dtype=wp.float32, device="cpu")
        return wp.clone(self._dof_dampings)

    def get_dof_max_forces(self) -> wp.array:
        """Get maximum forces of all DOFs.

        Returns:
            Warp array of shape (N, J) with dtype=wp.float32. Always on CPU.
        """
        if self._dof_max_forces is None:
            import numpy as np

            arr = np.full((self._count, self._num_dofs), float("inf"), dtype=np.float32)
            self._dof_max_forces = wp.array(arr, dtype=wp.float32, device="cpu")
        return wp.clone(self._dof_max_forces)

    def get_dof_max_velocities(self) -> wp.array:
        """Get maximum velocities of all DOFs.

        Returns:
            Warp array of shape (N, J) with dtype=wp.float32. Always on CPU.
        """
        if self._dof_max_velocities is None:
            import numpy as np

            arr = np.full((self._count, self._num_dofs), float("inf"), dtype=np.float32)
            self._dof_max_velocities = wp.array(arr, dtype=wp.float32, device="cpu")
        return wp.clone(self._dof_max_velocities)

    def get_dof_armatures(self) -> wp.array:
        """Get armatures of all DOFs.

        Returns:
            Warp array of shape (N, J) with dtype=wp.float32. Always on CPU.
        """
        if self._dof_armatures is None:
            self._dof_armatures = wp.zeros((self._count, self._num_dofs), dtype=wp.float32, device="cpu")
        return wp.clone(self._dof_armatures)

    def get_dof_friction_coefficients(self) -> wp.array:
        """Get friction coefficients of all DOFs.

        Returns:
            Warp array of shape (N, J) with dtype=wp.float32. Always on CPU.
        """
        if self._dof_friction_coefficients is None:
            self._dof_friction_coefficients = wp.zeros((self._count, self._num_dofs), dtype=wp.float32, device="cpu")
        return wp.clone(self._dof_friction_coefficients)

    def get_dof_friction_properties(self) -> wp.array:
        """Get friction properties of all DOFs.

        Returns:
            Warp array of shape (N, J, 3) with dtype=wp.float32. Always on CPU.
        """
        if self._dof_friction_properties is None:
            self._dof_friction_properties = wp.zeros(
                (self._count, self._num_dofs, 3), dtype=wp.float32, device="cpu"
            )
        return wp.clone(self._dof_friction_properties)

    # -- Mass Property Getters --

    def get_masses(self) -> wp.array:
        """Get masses of all links.

        Returns:
            Warp array of shape (N, L) with dtype=wp.float32. Always on CPU.
        """
        if self._masses is None:
            self._masses = wp.ones((self._count, self._num_links), dtype=wp.float32, device="cpu")
        return wp.clone(self._masses)

    def get_coms(self) -> wp.array:
        """Get centers of mass of all links.

        Returns:
            Warp array of shape (N, L) with dtype=wp.transformf. Always on CPU.
        """
        if self._coms is None:
            self._coms = wp.zeros((self._count, self._num_links), dtype=wp.transformf, device="cpu")
            wp.launch(
                init_identity_transforms_2d,
                dim=(self._count, self._num_links),
                inputs=[self._coms],
                device="cpu",
            )
        dst = wp.zeros_like(self._coms)
        wp.launch(copy_transforms_2d, dim=self._coms.shape, inputs=[self._coms, dst], device="cpu")
        return dst

    def get_inertias(self) -> wp.array:
        """Get inertia tensors of all links.

        Returns:
            Warp array of shape (N, L, 9) with dtype=wp.float32 - flattened 3x3 matrices. Always on CPU.
        """
        if self._inertias is None:
            self._inertias = wp.zeros((self._count, self._num_links, 9), dtype=wp.float32, device="cpu")
            wp.launch(
                init_identity_inertias_2d,
                dim=(self._count, self._num_links),
                inputs=[self._inertias],
                device="cpu",
            )
        return wp.clone(self._inertias)

    # -- Root Setters --

    def set_root_transforms(
        self,
        transforms: wp.array,
        indices: wp.array | None = None,
    ) -> None:
        """Set world transforms of root links.

        Args:
            transforms: Warp array of shape (N,) or (len(indices),) with dtype=wp.transformf.
            indices: Optional indices of articulations to update.
        """
        if self._root_transforms is None:
            self._root_transforms = self._create_identity_transforms_1d(self._count)
        if indices is not None:
            wp.launch(
                scatter_transforms_1d,
                dim=indices.shape[0],
                inputs=[transforms, indices, self._root_transforms],
                device=self._device,
            )
        else:
            wp.copy(self._root_transforms, transforms)

    def set_root_velocities(
        self,
        velocities: wp.array,
        indices: wp.array | None = None,
    ) -> None:
        """Set velocities of root links.

        Args:
            velocities: Warp array of shape (N,) or (len(indices),) with dtype=wp.spatial_vectorf.
            indices: Optional indices of articulations to update.
        """
        if self._root_velocities is None:
            self._root_velocities = self._create_zero_spatial_vectors_1d(self._count)
        if indices is not None:
            wp.launch(
                scatter_spatial_vectors_1d,
                dim=indices.shape[0],
                inputs=[velocities, indices, self._root_velocities],
                device=self._device,
            )
        else:
            wp.copy(self._root_velocities, velocities)

    # -- DOF Setters --

    def set_dof_positions(
        self,
        positions: wp.array,
        indices: wp.array | None = None,
    ) -> None:
        """Set positions of all DOFs.

        Args:
            positions: Warp array of shape (N, J) or (len(indices), J) with dtype=wp.float32.
            indices: Optional indices of articulations to update.
        """
        if self._dof_positions is None:
            self._dof_positions = wp.zeros((self._count, self._num_dofs), dtype=wp.float32, device=self._device)
        if indices is not None:
            positions_np = positions.numpy()
            indices_np = indices.numpy()
            self_positions_np = self._dof_positions.numpy()
            self_positions_np[indices_np] = positions_np
            self._dof_positions = wp.array(self_positions_np, dtype=wp.float32, device=self._device)
        else:
            wp.copy(self._dof_positions, positions)

    def set_dof_velocities(
        self,
        velocities: wp.array,
        indices: wp.array | None = None,
    ) -> None:
        """Set velocities of all DOFs.

        Args:
            velocities: Warp array of shape (N, J) or (len(indices), J) with dtype=wp.float32.
            indices: Optional indices of articulations to update.
        """
        if self._dof_velocities is None:
            self._dof_velocities = wp.zeros((self._count, self._num_dofs), dtype=wp.float32, device=self._device)
        if indices is not None:
            velocities_np = velocities.numpy()
            indices_np = indices.numpy()
            self_velocities_np = self._dof_velocities.numpy()
            self_velocities_np[indices_np] = velocities_np
            self._dof_velocities = wp.array(self_velocities_np, dtype=wp.float32, device=self._device)
        else:
            wp.copy(self._dof_velocities, velocities)

    def set_dof_position_targets(
        self,
        targets: wp.array,
        indices: wp.array | None = None,
    ) -> None:
        """Set position targets for all DOFs (no-op in mock).

        Args:
            targets: Warp array of shape (N, J) or (len(indices), J) with dtype=wp.float32.
            indices: Optional indices of articulations to update.
        """
        pass  # No-op for mock

    def set_dof_velocity_targets(
        self,
        targets: wp.array,
        indices: wp.array | None = None,
    ) -> None:
        """Set velocity targets for all DOFs (no-op in mock).

        Args:
            targets: Warp array of shape (N, J) or (len(indices), J) with dtype=wp.float32.
            indices: Optional indices of articulations to update.
        """
        pass  # No-op for mock

    def set_dof_actuation_forces(
        self,
        forces: wp.array,
        indices: wp.array | None = None,
    ) -> None:
        """Set actuation forces for all DOFs (no-op in mock).

        Args:
            forces: Warp array of shape (N, J) or (len(indices), J) with dtype=wp.float32.
            indices: Optional indices of articulations to update.
        """
        pass  # No-op for mock

    def set_dof_limits(
        self,
        limits: wp.array,
        indices: wp.array | None = None,
    ) -> None:
        """Set position limits of all DOFs.

        Args:
            limits: Warp array of shape (N, J, 2) with dtype=wp.float32. Must be on CPU.
            indices: Optional indices of articulations to update.

        Raises:
            RuntimeError: If limits array is on GPU.
        """
        self._check_cpu_array(limits, "dof_limits")
        if self._dof_limits is None:
            import numpy as np

            arr = np.zeros((self._count, self._num_dofs, 2), dtype=np.float32)
            arr[:, :, 0] = float("-inf")
            arr[:, :, 1] = float("inf")
            self._dof_limits = wp.array(arr, dtype=wp.float32, device="cpu")
        if indices is not None:
            limits_np = limits.numpy()
            indices_np = indices.numpy()
            self_limits_np = self._dof_limits.numpy()
            self_limits_np[indices_np] = limits_np
            self._dof_limits = wp.array(self_limits_np, dtype=wp.float32, device="cpu")
        else:
            wp.copy(self._dof_limits, limits)

    def set_dof_stiffnesses(
        self,
        stiffnesses: wp.array,
        indices: wp.array | None = None,
    ) -> None:
        """Set stiffnesses of all DOFs.

        Args:
            stiffnesses: Warp array of shape (N, J) with dtype=wp.float32. Must be on CPU.
            indices: Optional indices of articulations to update.

        Raises:
            RuntimeError: If stiffnesses array is on GPU.
        """
        self._check_cpu_array(stiffnesses, "dof_stiffnesses")
        if self._dof_stiffnesses is None:
            self._dof_stiffnesses = wp.zeros((self._count, self._num_dofs), dtype=wp.float32, device="cpu")
        if indices is not None:
            stiffnesses_np = stiffnesses.numpy()
            indices_np = indices.numpy()
            self_stiffnesses_np = self._dof_stiffnesses.numpy()
            self_stiffnesses_np[indices_np] = stiffnesses_np
            self._dof_stiffnesses = wp.array(self_stiffnesses_np, dtype=wp.float32, device="cpu")
        else:
            wp.copy(self._dof_stiffnesses, stiffnesses)

    def set_dof_dampings(
        self,
        dampings: wp.array,
        indices: wp.array | None = None,
    ) -> None:
        """Set dampings of all DOFs.

        Args:
            dampings: Warp array of shape (N, J) with dtype=wp.float32. Must be on CPU.
            indices: Optional indices of articulations to update.

        Raises:
            RuntimeError: If dampings array is on GPU.
        """
        self._check_cpu_array(dampings, "dof_dampings")
        if self._dof_dampings is None:
            self._dof_dampings = wp.zeros((self._count, self._num_dofs), dtype=wp.float32, device="cpu")
        if indices is not None:
            dampings_np = dampings.numpy()
            indices_np = indices.numpy()
            self_dampings_np = self._dof_dampings.numpy()
            self_dampings_np[indices_np] = dampings_np
            self._dof_dampings = wp.array(self_dampings_np, dtype=wp.float32, device="cpu")
        else:
            wp.copy(self._dof_dampings, dampings)

    def set_dof_max_forces(
        self,
        max_forces: wp.array,
        indices: wp.array | None = None,
    ) -> None:
        """Set maximum forces of all DOFs.

        Args:
            max_forces: Warp array of shape (N, J) with dtype=wp.float32. Must be on CPU.
            indices: Optional indices of articulations to update.

        Raises:
            RuntimeError: If max_forces array is on GPU.
        """
        self._check_cpu_array(max_forces, "dof_max_forces")
        if self._dof_max_forces is None:
            import numpy as np

            arr = np.full((self._count, self._num_dofs), float("inf"), dtype=np.float32)
            self._dof_max_forces = wp.array(arr, dtype=wp.float32, device="cpu")
        if indices is not None:
            max_forces_np = max_forces.numpy()
            indices_np = indices.numpy()
            self_max_forces_np = self._dof_max_forces.numpy()
            self_max_forces_np[indices_np] = max_forces_np
            self._dof_max_forces = wp.array(self_max_forces_np, dtype=wp.float32, device="cpu")
        else:
            wp.copy(self._dof_max_forces, max_forces)

    def set_dof_max_velocities(
        self,
        max_velocities: wp.array,
        indices: wp.array | None = None,
    ) -> None:
        """Set maximum velocities of all DOFs.

        Args:
            max_velocities: Warp array of shape (N, J) with dtype=wp.float32. Must be on CPU.
            indices: Optional indices of articulations to update.

        Raises:
            RuntimeError: If max_velocities array is on GPU.
        """
        self._check_cpu_array(max_velocities, "dof_max_velocities")
        if self._dof_max_velocities is None:
            import numpy as np

            arr = np.full((self._count, self._num_dofs), float("inf"), dtype=np.float32)
            self._dof_max_velocities = wp.array(arr, dtype=wp.float32, device="cpu")
        if indices is not None:
            max_velocities_np = max_velocities.numpy()
            indices_np = indices.numpy()
            self_max_velocities_np = self._dof_max_velocities.numpy()
            self_max_velocities_np[indices_np] = max_velocities_np
            self._dof_max_velocities = wp.array(self_max_velocities_np, dtype=wp.float32, device="cpu")
        else:
            wp.copy(self._dof_max_velocities, max_velocities)

    def set_dof_armatures(
        self,
        armatures: wp.array,
        indices: wp.array | None = None,
    ) -> None:
        """Set armatures of all DOFs.

        Args:
            armatures: Warp array of shape (N, J) with dtype=wp.float32. Must be on CPU.
            indices: Optional indices of articulations to update.

        Raises:
            RuntimeError: If armatures array is on GPU.
        """
        self._check_cpu_array(armatures, "dof_armatures")
        if self._dof_armatures is None:
            self._dof_armatures = wp.zeros((self._count, self._num_dofs), dtype=wp.float32, device="cpu")
        if indices is not None:
            armatures_np = armatures.numpy()
            indices_np = indices.numpy()
            self_armatures_np = self._dof_armatures.numpy()
            self_armatures_np[indices_np] = armatures_np
            self._dof_armatures = wp.array(self_armatures_np, dtype=wp.float32, device="cpu")
        else:
            wp.copy(self._dof_armatures, armatures)

    def set_dof_friction_coefficients(
        self,
        friction_coefficients: wp.array,
        indices: wp.array | None = None,
    ) -> None:
        """Set friction coefficients of all DOFs.

        Args:
            friction_coefficients: Warp array of shape (N, J) with dtype=wp.float32. Must be on CPU.
            indices: Optional indices of articulations to update.

        Raises:
            RuntimeError: If friction_coefficients array is on GPU.
        """
        self._check_cpu_array(friction_coefficients, "dof_friction_coefficients")
        if self._dof_friction_coefficients is None:
            self._dof_friction_coefficients = wp.zeros((self._count, self._num_dofs), dtype=wp.float32, device="cpu")
        if indices is not None:
            friction_coefficients_np = friction_coefficients.numpy()
            indices_np = indices.numpy()
            self_friction_coefficients_np = self._dof_friction_coefficients.numpy()
            self_friction_coefficients_np[indices_np] = friction_coefficients_np
            self._dof_friction_coefficients = wp.array(self_friction_coefficients_np, dtype=wp.float32, device="cpu")
        else:
            wp.copy(self._dof_friction_coefficients, friction_coefficients)

    # -- Mass Property Setters --

    def set_masses(
        self,
        masses: wp.array,
        indices: wp.array | None = None,
    ) -> None:
        """Set masses of all links.

        Args:
            masses: Warp array of shape (N, L) with dtype=wp.float32. Must be on CPU.
            indices: Optional indices of articulations to update.

        Raises:
            RuntimeError: If masses array is on GPU.
        """
        self._check_cpu_array(masses, "masses")
        if self._masses is None:
            self._masses = wp.ones((self._count, self._num_links), dtype=wp.float32, device="cpu")
        if indices is not None:
            masses_np = masses.numpy()
            indices_np = indices.numpy()
            self_masses_np = self._masses.numpy()
            self_masses_np[indices_np] = masses_np
            self._masses = wp.array(self_masses_np, dtype=wp.float32, device="cpu")
        else:
            wp.copy(self._masses, masses)

    def set_coms(
        self,
        coms: wp.array,
        indices: wp.array | None = None,
    ) -> None:
        """Set centers of mass of all links.

        Args:
            coms: Warp array of shape (N, L) with dtype=wp.transformf. Must be on CPU.
            indices: Optional indices of articulations to update.

        Raises:
            RuntimeError: If coms array is on GPU.
        """
        self._check_cpu_array(coms, "coms")
        if self._coms is None:
            self._coms = wp.zeros((self._count, self._num_links), dtype=wp.transformf, device="cpu")
            wp.launch(
                init_identity_transforms_2d,
                dim=(self._count, self._num_links),
                inputs=[self._coms],
                device="cpu",
            )
        if indices is not None:
            # For 2D transform arrays with indices, use numpy for scatter
            coms_np = coms.numpy()
            indices_np = indices.numpy()
            self_coms_np = self._coms.numpy()
            self_coms_np[indices_np] = coms_np
            self._coms = wp.array(self_coms_np, dtype=wp.transformf, device="cpu")
        else:
            wp.copy(self._coms, coms)

    def set_inertias(
        self,
        inertias: wp.array,
        indices: wp.array | None = None,
    ) -> None:
        """Set inertia tensors of all links.

        Args:
            inertias: Warp array of shape (N, L, 9) with dtype=wp.float32 - flattened 3x3 matrices. Must be on CPU.
            indices: Optional indices of articulations to update.

        Raises:
            RuntimeError: If inertias array is on GPU.
        """
        self._check_cpu_array(inertias, "inertias")
        if self._inertias is None:
            self._inertias = wp.zeros((self._count, self._num_links, 9), dtype=wp.float32, device="cpu")
            wp.launch(
                init_identity_inertias_2d,
                dim=(self._count, self._num_links),
                inputs=[self._inertias],
                device="cpu",
            )
        if indices is not None:
            inertias_np = inertias.numpy()
            indices_np = indices.numpy()
            self_inertias_np = self._inertias.numpy()
            self_inertias_np[indices_np] = inertias_np
            self._inertias = wp.array(self_inertias_np, dtype=wp.float32, device="cpu")
        else:
            wp.copy(self._inertias, inertias)

    # -- Mock setters (direct test data injection) --

    def set_mock_root_transforms(self, transforms: wp.array) -> None:
        """Set mock root transform data directly for testing.

        Args:
            transforms: Warp array of shape (N,) with dtype=wp.transformf.
        """
        self._root_transforms = wp.clone(transforms)
        if self._root_transforms.device.alias != self._device:
            self._root_transforms = self._root_transforms.to(self._device)

    def set_mock_root_velocities(self, velocities: wp.array) -> None:
        """Set mock root velocity data directly for testing.

        Args:
            velocities: Warp array of shape (N,) with dtype=wp.spatial_vectorf.
        """
        self._root_velocities = wp.clone(velocities)
        if self._root_velocities.device.alias != self._device:
            self._root_velocities = self._root_velocities.to(self._device)

    def set_mock_link_transforms(self, transforms: wp.array) -> None:
        """Set mock link transform data directly for testing.

        Args:
            transforms: Warp array of shape (N, L) with dtype=wp.transformf.
        """
        self._link_transforms = wp.clone(transforms)
        if self._link_transforms.device.alias != self._device:
            self._link_transforms = self._link_transforms.to(self._device)

    def set_mock_link_velocities(self, velocities: wp.array) -> None:
        """Set mock link velocity data directly for testing.

        Args:
            velocities: Warp array of shape (N, L) with dtype=wp.spatial_vectorf.
        """
        self._link_velocities = wp.clone(velocities)
        if self._link_velocities.device.alias != self._device:
            self._link_velocities = self._link_velocities.to(self._device)

    def set_mock_link_accelerations(self, accelerations: wp.array) -> None:
        """Set mock link acceleration data directly for testing.

        Args:
            accelerations: Warp array of shape (N, L) with dtype=wp.spatial_vectorf.
        """
        self._link_accelerations = wp.clone(accelerations)
        if self._link_accelerations.device.alias != self._device:
            self._link_accelerations = self._link_accelerations.to(self._device)

    def set_mock_link_incoming_joint_force(self, forces: wp.array) -> None:
        """Set mock link incoming joint force data directly for testing.

        Args:
            forces: Warp array of shape (N, L) with dtype=wp.spatial_vectorf.
        """
        self._link_incoming_joint_force = wp.clone(forces)
        if self._link_incoming_joint_force.device.alias != self._device:
            self._link_incoming_joint_force = self._link_incoming_joint_force.to(self._device)

    def set_mock_dof_positions(self, positions: wp.array) -> None:
        """Set mock DOF position data directly for testing.

        Args:
            positions: Warp array of shape (N, J) with dtype=wp.float32.
        """
        self._dof_positions = wp.clone(positions)
        if self._dof_positions.device.alias != self._device:
            self._dof_positions = self._dof_positions.to(self._device)

    def set_mock_dof_velocities(self, velocities: wp.array) -> None:
        """Set mock DOF velocity data directly for testing.

        Args:
            velocities: Warp array of shape (N, J) with dtype=wp.float32.
        """
        self._dof_velocities = wp.clone(velocities)
        if self._dof_velocities.device.alias != self._device:
            self._dof_velocities = self._dof_velocities.to(self._device)

    def set_mock_dof_projected_joint_forces(self, forces: wp.array) -> None:
        """Set mock projected joint force data directly for testing.

        Args:
            forces: Warp array of shape (N, J) with dtype=wp.float32.
        """
        self._dof_projected_joint_forces = wp.clone(forces)
        if self._dof_projected_joint_forces.device.alias != self._device:
            self._dof_projected_joint_forces = self._dof_projected_joint_forces.to(self._device)

    def set_mock_dof_limits(self, limits: wp.array) -> None:
        """Set mock DOF limit data directly for testing.

        Args:
            limits: Warp array of shape (N, J, 2) with dtype=wp.float32.
        """
        self._dof_limits = wp.clone(limits)

    def set_mock_dof_stiffnesses(self, stiffnesses: wp.array) -> None:
        """Set mock DOF stiffness data directly for testing.

        Args:
            stiffnesses: Warp array of shape (N, J) with dtype=wp.float32.
        """
        self._dof_stiffnesses = wp.clone(stiffnesses)

    def set_mock_dof_dampings(self, dampings: wp.array) -> None:
        """Set mock DOF damping data directly for testing.

        Args:
            dampings: Warp array of shape (N, J) with dtype=wp.float32.
        """
        self._dof_dampings = wp.clone(dampings)

    def set_mock_dof_max_forces(self, max_forces: wp.array) -> None:
        """Set mock DOF max force data directly for testing.

        Args:
            max_forces: Warp array of shape (N, J) with dtype=wp.float32.
        """
        self._dof_max_forces = wp.clone(max_forces)

    def set_mock_dof_max_velocities(self, max_velocities: wp.array) -> None:
        """Set mock DOF max velocity data directly for testing.

        Args:
            max_velocities: Warp array of shape (N, J) with dtype=wp.float32.
        """
        self._dof_max_velocities = wp.clone(max_velocities)

    def set_mock_dof_armatures(self, armatures: wp.array) -> None:
        """Set mock DOF armature data directly for testing.

        Args:
            armatures: Warp array of shape (N, J) with dtype=wp.float32.
        """
        self._dof_armatures = wp.clone(armatures)

    def set_mock_dof_friction_coefficients(self, friction_coefficients: wp.array) -> None:
        """Set mock DOF friction coefficient data directly for testing.

        Args:
            friction_coefficients: Warp array of shape (N, J) with dtype=wp.float32.
        """
        self._dof_friction_coefficients = wp.clone(friction_coefficients)

    def set_mock_dof_friction_properties(self, friction_properties: wp.array) -> None:
        """Set mock DOF friction properties data directly for testing.

        Args:
            friction_properties: Warp array of shape (N, J, 3) with dtype=wp.float32.
        """
        self._dof_friction_properties = wp.clone(friction_properties)

    def set_mock_masses(self, masses: wp.array) -> None:
        """Set mock mass data directly for testing.

        Args:
            masses: Warp array of shape (N, L) with dtype=wp.float32.
        """
        self._masses = wp.clone(masses)

    def set_mock_coms(self, coms: wp.array) -> None:
        """Set mock center of mass data directly for testing.

        Args:
            coms: Warp array of shape (N, L) with dtype=wp.transformf.
        """
        self._coms = wp.clone(coms)

    def set_mock_inertias(self, inertias: wp.array) -> None:
        """Set mock inertia data directly for testing.

        Args:
            inertias: Warp array of shape (N, L, 9) with dtype=wp.float32.
        """
        self._inertias = wp.clone(inertias)
