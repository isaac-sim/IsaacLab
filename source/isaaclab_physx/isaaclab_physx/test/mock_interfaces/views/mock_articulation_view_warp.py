# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mock implementation of PhysX ArticulationView using Warp arrays."""

from __future__ import annotations

import warp as wp

from ..utils.kernels_mock import (
    init_identity_inertias_2d,
    init_identity_transforms_1d_flat,
    init_identity_transforms_2d_flat,
    scatter_floats_2d,
)
from ..utils.mock_shared_metatype import MockSharedMetatype


class MockArticulationViewWarp:
    """Mock implementation of physx.ArticulationView using Warp arrays for unit testing.

    This class mimics the interface of the PhysX TensorAPI ArticulationView using
    flat float32 arrays (matching real PhysX behavior), allowing tests to run
    without Isaac Sim or GPU simulation.

    Data Shapes (flat float32, matching real PhysX views):
        - root_transforms: (N, 7) dtype=wp.float32 - [pos(3), quat_xyzw(4)]
        - root_velocities: (N, 6) dtype=wp.float32 - [ang_vel(3), lin_vel(3)]
        - link_transforms: (N, L, 7) dtype=wp.float32 - per-link poses
        - link_velocities: (N, L, 6) dtype=wp.float32 - per-link velocities
        - link_accelerations: (N, L, 6) dtype=wp.float32 - per-link accelerations
        - link_incoming_joint_force: (N, L, 6) dtype=wp.float32 - per-link forces
        - dof_positions: (N, J) dtype=wp.float32 - joint positions
        - dof_velocities: (N, J) dtype=wp.float32 - joint velocities
        - dof_limits: (N, J, 2) dtype=wp.float32 - [lower, upper] limits
        - dof_stiffnesses: (N, J) dtype=wp.float32 - joint stiffnesses
        - dof_dampings: (N, J) dtype=wp.float32 - joint dampings
        - masses: (N, L) dtype=wp.float32 - per-link masses
        - coms: (N, L, 7) dtype=wp.float32 - per-link centers of mass
        - inertias: (N, L, 9) dtype=wp.float32 - per-link inertia tensors (flattened)

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
            device: Device for array allocation ("cpu" or "cuda:N").
        """
        self._count = count
        self._num_dofs = num_dofs
        self._num_links = num_links
        self._device = device
        self._prim_paths = prim_paths or [f"/World/Articulation_{i}" for i in range(count)]
        self._backend = "warp"
        self._noop_setters = False

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

    def _create_identity_transforms_1d(self, count: int, device: str | None = None) -> wp.array:
        """Create 1D array of identity transforms as (count, 7) float32."""
        dev = device or self._device
        arr = wp.zeros((count, 7), dtype=wp.float32, device=dev)
        wp.launch(init_identity_transforms_1d_flat, dim=count, inputs=[arr], device=dev)
        return arr

    def _create_identity_transforms_2d(self, count: int, num_links: int, device: str | None = None) -> wp.array:
        """Create 2D array of identity transforms as (count, num_links, 7) float32."""
        dev = device or self._device
        arr = wp.zeros((count, num_links, 7), dtype=wp.float32, device=dev)
        wp.launch(init_identity_transforms_2d_flat, dim=(count, num_links), inputs=[arr], device=dev)
        return arr

    @staticmethod
    def _as_structured(flat: wp.array, dtype, shape: tuple) -> wp.array:
        """Zero-copy reinterpretation of flat float32 array as structured type array.

        This is needed because real PhysX views return structured types (e.g. transformf)
        and the data classes call .view() on the results (which requires same byte-size dtype).
        """
        return wp.array(
            ptr=flat.ptr,
            dtype=dtype,
            shape=shape,
            device=flat.device,
            copy=False,
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

    def get_root_transforms(self) -> wp.array:
        """Get world transforms of root links.

        Returns:
            Warp array of shape (N,) with dtype=wp.transformf (matching real PhysX).
        """
        if self._root_transforms is None:
            self._root_transforms = self._create_identity_transforms_1d(self._count)
        return wp.clone(self._as_structured(self._root_transforms, wp.transformf, (self._count,)))

    def get_root_velocities(self) -> wp.array:
        """Get velocities of root links.

        Returns:
            Warp array of shape (N,) with dtype=wp.spatial_vectorf (matching real PhysX).
        """
        if self._root_velocities is None:
            self._root_velocities = wp.zeros((self._count, 6), dtype=wp.float32, device=self._device)
        return wp.clone(self._as_structured(self._root_velocities, wp.spatial_vectorf, (self._count,)))

    # -- Link Getters --

    def get_link_transforms(self) -> wp.array:
        """Get world transforms of all links.

        Returns:
            Warp array of shape (N, L) with dtype=wp.transformf (matching real PhysX).
        """
        if self._link_transforms is None:
            self._link_transforms = self._create_identity_transforms_2d(self._count, self._num_links)
        return wp.clone(self._as_structured(self._link_transforms, wp.transformf, (self._count, self._num_links)))

    def get_link_velocities(self) -> wp.array:
        """Get velocities of all links.

        Returns:
            Warp array of shape (N, L) with dtype=wp.spatial_vectorf (matching real PhysX).
        """
        if self._link_velocities is None:
            self._link_velocities = wp.zeros((self._count, self._num_links, 6), dtype=wp.float32, device=self._device)
        return wp.clone(self._as_structured(self._link_velocities, wp.spatial_vectorf, (self._count, self._num_links)))

    def get_link_accelerations(self) -> wp.array:
        """Get accelerations of all links.

        Returns:
            Warp array of shape (N, L) with dtype=wp.spatial_vectorf (matching real PhysX).
        """
        if self._link_accelerations is None:
            self._link_accelerations = wp.zeros(
                (self._count, self._num_links, 6), dtype=wp.float32, device=self._device
            )
        return wp.clone(
            self._as_structured(self._link_accelerations, wp.spatial_vectorf, (self._count, self._num_links))
        )

    def get_link_incoming_joint_force(self) -> wp.array:
        """Get incoming joint forces for all links.

        Returns:
            Warp array of shape (N, L) with dtype=wp.spatial_vectorf (matching real PhysX).
        """
        if self._link_incoming_joint_force is None:
            self._link_incoming_joint_force = wp.zeros(
                (self._count, self._num_links, 6), dtype=wp.float32, device=self._device
            )
        return wp.clone(
            self._as_structured(self._link_incoming_joint_force, wp.spatial_vectorf, (self._count, self._num_links))
        )

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
            Warp array of shape (N, J) with dtype=wp.vec2f (matching real PhysX). Always on CPU.
        """
        if self._dof_limits is None:
            import numpy as np

            limits = np.zeros((self._count, self._num_dofs, 2), dtype=np.float32)
            limits[:, :, 0] = float("-inf")  # lower limit
            limits[:, :, 1] = float("inf")  # upper limit
            self._dof_limits = wp.array(limits, dtype=wp.float32, device="cpu")
        return wp.clone(self._as_structured(self._dof_limits, wp.vec2f, (self._count, self._num_dofs)))

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
            self._dof_friction_properties = wp.zeros((self._count, self._num_dofs, 3), dtype=wp.float32, device="cpu")
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
            Warp array of shape (N, L) with dtype=wp.transformf (matching real PhysX). Always on CPU.
        """
        if self._coms is None:
            self._coms = self._create_identity_transforms_2d(self._count, self._num_links, device="cpu")
        return wp.clone(self._as_structured(self._coms, wp.transformf, (self._count, self._num_links)))

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
            transforms: Warp array of shape (N, 7) or (len(indices), 7) with dtype=wp.float32.
            indices: Optional indices of articulations to update.
        """
        if self._noop_setters:
            return
        if self._root_transforms is None:
            self._root_transforms = self._create_identity_transforms_1d(self._count)
        if indices is not None:
            wp.launch(
                scatter_floats_2d,
                dim=(indices.shape[0], 7),
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
            velocities: Warp array of shape (N, 6) or (len(indices), 6) with dtype=wp.float32.
            indices: Optional indices of articulations to update.
        """
        if self._noop_setters:
            return
        if self._root_velocities is None:
            self._root_velocities = wp.zeros((self._count, 6), dtype=wp.float32, device=self._device)
        if indices is not None:
            wp.launch(
                scatter_floats_2d,
                dim=(indices.shape[0], 6),
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
        if self._noop_setters:
            return
        if self._dof_positions is None:
            self._dof_positions = wp.zeros((self._count, self._num_dofs), dtype=wp.float32, device=self._device)
        if indices is not None:
            wp.launch(
                scatter_floats_2d,
                dim=(indices.shape[0], self._num_dofs),
                inputs=[positions, indices, self._dof_positions],
                device=self._device,
            )
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
        if self._noop_setters:
            return
        if self._dof_velocities is None:
            self._dof_velocities = wp.zeros((self._count, self._num_dofs), dtype=wp.float32, device=self._device)
        if indices is not None:
            wp.launch(
                scatter_floats_2d,
                dim=(indices.shape[0], self._num_dofs),
                inputs=[velocities, indices, self._dof_velocities],
                device=self._device,
            )
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
        if self._noop_setters:
            return
        self._check_cpu_array(limits, "dof_limits")
        if self._dof_limits is None:
            import numpy as np

            arr = np.zeros((self._count, self._num_dofs, 2), dtype=np.float32)
            arr[:, :, 0] = float("-inf")
            arr[:, :, 1] = float("inf")
            self._dof_limits = wp.array(arr, dtype=wp.float32, device="cpu")
        if indices is not None:
            # numpy() on CPU warp arrays is zero-copy; in-place write modifies the warp array directly
            self._dof_limits.numpy()[indices.numpy()] = limits.numpy()
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
        if self._noop_setters:
            return
        self._check_cpu_array(stiffnesses, "dof_stiffnesses")
        if self._dof_stiffnesses is None:
            self._dof_stiffnesses = wp.zeros((self._count, self._num_dofs), dtype=wp.float32, device="cpu")
        if indices is not None:
            self._dof_stiffnesses.numpy()[indices.numpy()] = stiffnesses.numpy()
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
        if self._noop_setters:
            return
        self._check_cpu_array(dampings, "dof_dampings")
        if self._dof_dampings is None:
            self._dof_dampings = wp.zeros((self._count, self._num_dofs), dtype=wp.float32, device="cpu")
        if indices is not None:
            self._dof_dampings.numpy()[indices.numpy()] = dampings.numpy()
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
        if self._noop_setters:
            return
        self._check_cpu_array(max_forces, "dof_max_forces")
        if self._dof_max_forces is None:
            import numpy as np

            arr = np.full((self._count, self._num_dofs), float("inf"), dtype=np.float32)
            self._dof_max_forces = wp.array(arr, dtype=wp.float32, device="cpu")
        if indices is not None:
            self._dof_max_forces.numpy()[indices.numpy()] = max_forces.numpy()
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
        if self._noop_setters:
            return
        self._check_cpu_array(max_velocities, "dof_max_velocities")
        if self._dof_max_velocities is None:
            import numpy as np

            arr = np.full((self._count, self._num_dofs), float("inf"), dtype=np.float32)
            self._dof_max_velocities = wp.array(arr, dtype=wp.float32, device="cpu")
        if indices is not None:
            self._dof_max_velocities.numpy()[indices.numpy()] = max_velocities.numpy()
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
        if self._noop_setters:
            return
        self._check_cpu_array(armatures, "dof_armatures")
        if self._dof_armatures is None:
            self._dof_armatures = wp.zeros((self._count, self._num_dofs), dtype=wp.float32, device="cpu")
        if indices is not None:
            self._dof_armatures.numpy()[indices.numpy()] = armatures.numpy()
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
        if self._noop_setters:
            return
        self._check_cpu_array(friction_coefficients, "dof_friction_coefficients")
        if self._dof_friction_coefficients is None:
            self._dof_friction_coefficients = wp.zeros((self._count, self._num_dofs), dtype=wp.float32, device="cpu")
        if indices is not None:
            self._dof_friction_coefficients.numpy()[indices.numpy()] = friction_coefficients.numpy()
        else:
            wp.copy(self._dof_friction_coefficients, friction_coefficients)

    def set_dof_friction_properties(
        self,
        friction_properties: wp.array,
        indices: wp.array | None = None,
    ) -> None:
        """Set friction properties of all DOFs.

        Args:
            friction_properties: Warp array of shape (N, J, 3) with [static, dynamic, viscous]. Must be on CPU.
            indices: Optional indices of articulations to update.

        Raises:
            RuntimeError: If friction_properties array is on GPU.
        """
        if self._noop_setters:
            return
        self._check_cpu_array(friction_properties, "dof_friction_properties")
        if self._dof_friction_properties is None:
            self._dof_friction_properties = wp.zeros((self._count, self._num_dofs, 3), dtype=wp.float32, device="cpu")
        if indices is not None:
            self._dof_friction_properties.numpy()[indices.numpy()] = friction_properties.numpy()
        else:
            wp.copy(self._dof_friction_properties, friction_properties)

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
        if self._noop_setters:
            return
        self._check_cpu_array(masses, "masses")
        if self._masses is None:
            self._masses = wp.ones((self._count, self._num_links), dtype=wp.float32, device="cpu")
        if indices is not None:
            self._masses.numpy()[indices.numpy()] = masses.numpy()
        else:
            wp.copy(self._masses, masses)

    def set_coms(
        self,
        coms: wp.array,
        indices: wp.array | None = None,
    ) -> None:
        """Set centers of mass of all links.

        Args:
            coms: Warp array of shape (N, L, 7) with dtype=wp.float32. Must be on CPU.
            indices: Optional indices of articulations to update.

        Raises:
            RuntimeError: If coms array is on GPU.
        """
        if self._noop_setters:
            return
        self._check_cpu_array(coms, "coms")
        if self._coms is None:
            self._coms = self._create_identity_transforms_2d(self._count, self._num_links, device="cpu")
        if indices is not None:
            self._coms.numpy()[indices.numpy()] = coms.numpy()
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
        if self._noop_setters:
            return
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
            self._inertias.numpy()[indices.numpy()] = inertias.numpy()
        else:
            wp.copy(self._inertias, inertias)

    # -- Mock setters (direct test data injection) --

    def set_mock_root_transforms(self, transforms: wp.array) -> None:
        """Set mock root transform data directly for testing.

        Args:
            transforms: Warp array of shape (N, 7) with dtype=wp.float32.
        """
        self._root_transforms = wp.clone(transforms)
        if self._root_transforms.device.alias != self._device:
            self._root_transforms = self._root_transforms.to(self._device)

    def set_mock_root_velocities(self, velocities: wp.array) -> None:
        """Set mock root velocity data directly for testing.

        Args:
            velocities: Warp array of shape (N, 6) with dtype=wp.float32.
        """
        self._root_velocities = wp.clone(velocities)
        if self._root_velocities.device.alias != self._device:
            self._root_velocities = self._root_velocities.to(self._device)

    def set_mock_link_transforms(self, transforms: wp.array) -> None:
        """Set mock link transform data directly for testing.

        Args:
            transforms: Warp array of shape (N, L, 7) with dtype=wp.float32.
        """
        self._link_transforms = wp.clone(transforms)
        if self._link_transforms.device.alias != self._device:
            self._link_transforms = self._link_transforms.to(self._device)

    def set_mock_link_velocities(self, velocities: wp.array) -> None:
        """Set mock link velocity data directly for testing.

        Args:
            velocities: Warp array of shape (N, L, 6) with dtype=wp.float32.
        """
        self._link_velocities = wp.clone(velocities)
        if self._link_velocities.device.alias != self._device:
            self._link_velocities = self._link_velocities.to(self._device)

    def set_mock_link_accelerations(self, accelerations: wp.array) -> None:
        """Set mock link acceleration data directly for testing.

        Args:
            accelerations: Warp array of shape (N, L, 6) with dtype=wp.float32.
        """
        self._link_accelerations = wp.clone(accelerations)
        if self._link_accelerations.device.alias != self._device:
            self._link_accelerations = self._link_accelerations.to(self._device)

    def set_mock_link_incoming_joint_force(self, forces: wp.array) -> None:
        """Set mock link incoming joint force data directly for testing.

        Args:
            forces: Warp array of shape (N, L, 6) with dtype=wp.float32.
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
            coms: Warp array of shape (N, L, 7) with dtype=wp.float32.
        """
        self._coms = wp.clone(coms)

    def set_mock_inertias(self, inertias: wp.array) -> None:
        """Set mock inertia data directly for testing.

        Args:
            inertias: Warp array of shape (N, L, 9) with dtype=wp.float32.
        """
        self._inertias = wp.clone(inertias)

    # -- Benchmark Utilities --

    def set_random_mock_data(self) -> None:
        """Set all internal state to random values for benchmarking.

        This method initializes all mock data with random warp arrays,
        useful for benchmarking where the actual values don't matter.
        """
        import numpy as np

        N = self._count
        L = self._num_links
        J = self._num_dofs

        # Root state - on device
        root_tf = np.random.randn(N, 7).astype(np.float32)
        root_tf[:, 3:7] /= np.linalg.norm(root_tf[:, 3:7], axis=-1, keepdims=True)
        self._root_transforms = wp.array(root_tf, dtype=wp.float32, device=self._device)
        self._root_velocities = wp.array(
            np.random.randn(N, 6).astype(np.float32), dtype=wp.float32, device=self._device
        )

        # Link state - on device
        link_tf = np.random.randn(N, L, 7).astype(np.float32)
        link_tf[:, :, 3:7] /= np.linalg.norm(link_tf[:, :, 3:7], axis=-1, keepdims=True)
        self._link_transforms = wp.array(link_tf, dtype=wp.float32, device=self._device)
        self._link_velocities = wp.array(
            np.random.randn(N, L, 6).astype(np.float32), dtype=wp.float32, device=self._device
        )
        self._link_accelerations = wp.array(
            np.random.randn(N, L, 6).astype(np.float32), dtype=wp.float32, device=self._device
        )
        self._link_incoming_joint_force = wp.array(
            np.random.randn(N, L, 6).astype(np.float32), dtype=wp.float32, device=self._device
        )

        # DOF state - on device
        self._dof_positions = wp.array(np.random.randn(N, J).astype(np.float32), dtype=wp.float32, device=self._device)
        self._dof_velocities = wp.array(np.random.randn(N, J).astype(np.float32), dtype=wp.float32, device=self._device)
        self._dof_projected_joint_forces = wp.array(
            np.random.randn(N, J).astype(np.float32), dtype=wp.float32, device=self._device
        )

        # DOF properties - on CPU (PhysX requirement)
        self._dof_limits = wp.array(np.random.randn(N, J, 2).astype(np.float32), dtype=wp.float32, device="cpu")
        self._dof_stiffnesses = wp.array(
            (np.random.rand(N, J) * 100).astype(np.float32), dtype=wp.float32, device="cpu"
        )
        self._dof_dampings = wp.array((np.random.rand(N, J) * 10).astype(np.float32), dtype=wp.float32, device="cpu")
        self._dof_max_forces = wp.array((np.random.rand(N, J) * 100).astype(np.float32), dtype=wp.float32, device="cpu")
        self._dof_max_velocities = wp.array(
            (np.random.rand(N, J) * 10).astype(np.float32), dtype=wp.float32, device="cpu"
        )
        self._dof_armatures = wp.array((np.random.rand(N, J) * 0.1).astype(np.float32), dtype=wp.float32, device="cpu")
        self._dof_friction_coefficients = wp.array(
            np.random.rand(N, J).astype(np.float32), dtype=wp.float32, device="cpu"
        )
        self._dof_friction_properties = wp.array(
            np.random.rand(N, J, 3).astype(np.float32), dtype=wp.float32, device="cpu"
        )

        # Mass properties - on CPU (PhysX requirement)
        self._masses = wp.array((np.random.rand(N, L) * 10).astype(np.float32), dtype=wp.float32, device="cpu")
        coms = np.random.randn(N, L, 7).astype(np.float32)
        coms[:, :, 3:7] /= np.linalg.norm(coms[:, :, 3:7], axis=-1, keepdims=True)
        self._coms = wp.array(coms, dtype=wp.float32, device="cpu")
        inertias = np.zeros((N, L, 9), dtype=np.float32)
        diag = np.random.rand(N, L, 3) + 0.1
        inertias[:, :, 0] = diag[:, :, 0]
        inertias[:, :, 4] = diag[:, :, 1]
        inertias[:, :, 8] = diag[:, :, 2]
        self._inertias = wp.array(inertias, dtype=wp.float32, device="cpu")
