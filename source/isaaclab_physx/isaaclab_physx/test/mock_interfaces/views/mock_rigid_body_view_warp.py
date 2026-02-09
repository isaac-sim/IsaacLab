# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mock implementation of PhysX RigidBodyView using Warp arrays."""

from __future__ import annotations

import warp as wp

from ..utils.kernels_mock import (
    copy_spatial_vectors_1d,
    copy_transforms_1d,
    init_identity_inertias_1d,
    init_identity_transforms_1d,
    init_zero_spatial_vectors_1d,
    scatter_spatial_vectors_1d,
    scatter_transforms_1d,
)


class MockRigidBodyViewWarp:
    """Mock implementation of physx.RigidBodyView using Warp arrays for unit testing.

    This class mimics the interface of the PhysX TensorAPI RigidBodyView using
    Warp structured types, allowing tests to run without Isaac Sim or GPU simulation.

    Data Shapes (using Warp types):
        - transforms: (N,) dtype=wp.transformf - [pos(3), quat_xyzw(4)]
        - velocities: (N,) dtype=wp.spatial_vectorf - [ang_vel(3), lin_vel(3)]
        - accelerations: (N,) dtype=wp.spatial_vectorf - [ang_acc(3), lin_acc(3)]
        - masses: (N, 1) dtype=wp.float32
        - coms: (N,) dtype=wp.transformf - center of mass [pos(3), quat_xyzw(4)]
        - inertias: (N, 9) dtype=wp.float32 - flattened 3x3 inertia matrix (row-major)

    Note:
        wp.spatial_vectorf stores [angular(3), linear(3)] which differs from
        torch's [linear(3), angular(3)] convention.
    """

    def __init__(
        self,
        count: int = 1,
        prim_paths: list[str] | None = None,
        device: str = "cpu",
    ):
        """Initialize the mock rigid body view.

        Args:
            count: Number of rigid body instances.
            prim_paths: USD prim paths for each instance.
            device: Device for array allocation ("cpu" or "cuda:N").
        """
        self._count = count
        self._prim_paths = prim_paths or [f"/World/RigidBody_{i}" for i in range(count)]
        self._device = device
        self._backend = "warp"

        # Internal state (lazily initialized)
        self._transforms: wp.array | None = None
        self._velocities: wp.array | None = None
        self._accelerations: wp.array | None = None
        self._masses: wp.array | None = None
        self._coms: wp.array | None = None
        self._inertias: wp.array | None = None

    # -- Helper Methods --

    def _check_cpu_array(self, arr: wp.array, name: str) -> None:
        """Check that array is on CPU, raise RuntimeError if on GPU.

        This mimics PhysX behavior where body properties must be on CPU.
        """
        if arr.device.is_cuda:
            raise RuntimeError(
                f"Expected CPU array for {name}, but got array on {arr.device}. "
                "Body properties must be set with CPU arrays."
            )

    def _create_identity_transforms(self, count: int) -> wp.array:
        """Create array of identity transforms."""
        arr = wp.zeros(count, dtype=wp.transformf, device=self._device)
        wp.launch(init_identity_transforms_1d, dim=count, inputs=[arr], device=self._device)
        return arr

    def _create_zero_spatial_vectors(self, count: int) -> wp.array:
        """Create array of zero spatial vectors."""
        arr = wp.zeros(count, dtype=wp.spatial_vectorf, device=self._device)
        wp.launch(init_zero_spatial_vectors_1d, dim=count, inputs=[arr], device=self._device)
        return arr

    def _clone_transforms(self, src: wp.array) -> wp.array:
        """Clone a transform array."""
        dst = wp.zeros_like(src)
        wp.launch(copy_transforms_1d, dim=src.shape[0], inputs=[src, dst], device=self._device)
        return dst

    def _clone_spatial_vectors(self, src: wp.array) -> wp.array:
        """Clone a spatial vector array."""
        dst = wp.zeros_like(src)
        wp.launch(copy_spatial_vectors_1d, dim=src.shape[0], inputs=[src, dst], device=self._device)
        return dst

    # -- Properties --

    @property
    def count(self) -> int:
        """Number of rigid body instances."""
        return self._count

    @property
    def prim_paths(self) -> list[str]:
        """USD prim paths for each instance."""
        return self._prim_paths

    # -- Getters --

    def get_transforms(self) -> wp.array:
        """Get world transforms of all rigid bodies.

        Returns:
            Warp array of shape (N,) with dtype=wp.transformf.
            Each transform contains [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w].
        """
        if self._transforms is None:
            self._transforms = self._create_identity_transforms(self._count)
        return self._clone_transforms(self._transforms)

    def get_velocities(self) -> wp.array:
        """Get velocities of all rigid bodies.

        Returns:
            Warp array of shape (N,) with dtype=wp.spatial_vectorf.
            Each spatial vector contains [ang_vel(3), lin_vel(3)].
        """
        if self._velocities is None:
            self._velocities = self._create_zero_spatial_vectors(self._count)
        return self._clone_spatial_vectors(self._velocities)

    def get_accelerations(self) -> wp.array:
        """Get accelerations of all rigid bodies.

        Returns:
            Warp array of shape (N,) with dtype=wp.spatial_vectorf.
            Each spatial vector contains [ang_acc(3), lin_acc(3)].
        """
        if self._accelerations is None:
            self._accelerations = self._create_zero_spatial_vectors(self._count)
        return self._clone_spatial_vectors(self._accelerations)

    def get_masses(self) -> wp.array:
        """Get masses of all rigid bodies.

        Returns:
            Warp array of shape (N, 1) with dtype=wp.float32. Always on CPU.
        """
        if self._masses is None:
            self._masses = wp.ones((self._count, 1), dtype=wp.float32, device="cpu")
        return wp.clone(self._masses)

    def get_coms(self) -> wp.array:
        """Get centers of mass of all rigid bodies.

        Returns:
            Warp array of shape (N,) with dtype=wp.transformf. Always on CPU.
            Each transform contains [pos(3), quat_xyzw(4)].
        """
        if self._coms is None:
            self._coms = wp.zeros(self._count, dtype=wp.transformf, device="cpu")
            wp.launch(init_identity_transforms_1d, dim=self._count, inputs=[self._coms], device="cpu")
        dst = wp.zeros_like(self._coms)
        wp.launch(copy_transforms_1d, dim=self._count, inputs=[self._coms, dst], device="cpu")
        return dst

    def get_inertias(self) -> wp.array:
        """Get inertia tensors of all rigid bodies.

        Returns:
            Warp array of shape (N, 9) with dtype=wp.float32 - flattened 3x3 matrices (row-major).
            Always on CPU.
        """
        if self._inertias is None:
            self._inertias = wp.zeros((self._count, 9), dtype=wp.float32, device="cpu")
            wp.launch(init_identity_inertias_1d, dim=self._count, inputs=[self._inertias], device="cpu")
        return wp.clone(self._inertias)

    # -- Setters (simulation interface) --

    def set_transforms(
        self,
        transforms: wp.array,
        indices: wp.array | None = None,
    ) -> None:
        """Set world transforms of rigid bodies.

        Args:
            transforms: Warp array of shape (N,) or (len(indices),) with dtype=wp.transformf.
            indices: Optional Warp array of indices (dtype=wp.int32) of bodies to update.
        """
        if self._transforms is None:
            self._transforms = self._create_identity_transforms(self._count)
        if indices is not None:
            wp.launch(
                scatter_transforms_1d,
                dim=indices.shape[0],
                inputs=[transforms, indices, self._transforms],
                device=self._device,
            )
        else:
            wp.copy(self._transforms, transforms)

    def set_velocities(
        self,
        velocities: wp.array,
        indices: wp.array | None = None,
    ) -> None:
        """Set velocities of rigid bodies.

        Args:
            velocities: Warp array of shape (N,) or (len(indices),) with dtype=wp.spatial_vectorf.
            indices: Optional Warp array of indices (dtype=wp.int32) of bodies to update.
        """
        if self._velocities is None:
            self._velocities = self._create_zero_spatial_vectors(self._count)
        if indices is not None:
            wp.launch(
                scatter_spatial_vectors_1d,
                dim=indices.shape[0],
                inputs=[velocities, indices, self._velocities],
                device=self._device,
            )
        else:
            wp.copy(self._velocities, velocities)

    def set_masses(
        self,
        masses: wp.array,
        indices: wp.array | None = None,
    ) -> None:
        """Set masses of rigid bodies.

        Args:
            masses: Warp array of shape (N, 1) or (len(indices), 1) with dtype=wp.float32. Must be on CPU.
            indices: Optional Warp array of indices (dtype=wp.int32) of bodies to update.

        Raises:
            RuntimeError: If masses array is on GPU.
        """
        self._check_cpu_array(masses, "masses")
        if self._masses is None:
            self._masses = wp.ones((self._count, 1), dtype=wp.float32, device="cpu")
        if indices is not None:
            # Manual scatter for 2D CPU arrays
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
        """Set centers of mass of rigid bodies.

        Args:
            coms: Warp array of shape (N,) or (len(indices),) with dtype=wp.transformf. Must be on CPU.
            indices: Optional Warp array of indices (dtype=wp.int32) of bodies to update.

        Raises:
            RuntimeError: If coms array is on GPU.
        """
        self._check_cpu_array(coms, "coms")
        if self._coms is None:
            self._coms = wp.zeros(self._count, dtype=wp.transformf, device="cpu")
            wp.launch(init_identity_transforms_1d, dim=self._count, inputs=[self._coms], device="cpu")
        if indices is not None:
            wp.launch(
                scatter_transforms_1d, dim=indices.shape[0], inputs=[coms, indices, self._coms], device="cpu"
            )
        else:
            wp.copy(self._coms, coms)

    def set_inertias(
        self,
        inertias: wp.array,
        indices: wp.array | None = None,
    ) -> None:
        """Set inertia tensors of rigid bodies.

        Args:
            inertias: Warp array of shape (N, 9) or (len(indices), 9) - flattened 3x3 matrices. Must be on CPU.
            indices: Optional Warp array of indices (dtype=wp.int32) of bodies to update.

        Raises:
            RuntimeError: If inertias array is on GPU.
        """
        self._check_cpu_array(inertias, "inertias")
        if self._inertias is None:
            self._inertias = wp.zeros((self._count, 9), dtype=wp.float32, device="cpu")
            wp.launch(init_identity_inertias_1d, dim=self._count, inputs=[self._inertias], device="cpu")
        if indices is not None:
            # Manual scatter for 2D CPU arrays
            inertias_np = inertias.numpy()
            indices_np = indices.numpy()
            self_inertias_np = self._inertias.numpy()
            self_inertias_np[indices_np] = inertias_np
            self._inertias = wp.array(self_inertias_np, dtype=wp.float32, device="cpu")
        else:
            wp.copy(self._inertias, inertias)

    # -- Mock setters (direct test data injection) --

    def set_mock_transforms(self, transforms: wp.array) -> None:
        """Set mock transform data directly for testing.

        Args:
            transforms: Warp array of shape (N,) with dtype=wp.transformf.
        """
        self._transforms = wp.clone(transforms)
        if self._transforms.device.alias != self._device:
            self._transforms = self._transforms.to(self._device)

    def set_mock_velocities(self, velocities: wp.array) -> None:
        """Set mock velocity data directly for testing.

        Args:
            velocities: Warp array of shape (N,) with dtype=wp.spatial_vectorf.
        """
        self._velocities = wp.clone(velocities)
        if self._velocities.device.alias != self._device:
            self._velocities = self._velocities.to(self._device)

    def set_mock_accelerations(self, accelerations: wp.array) -> None:
        """Set mock acceleration data directly for testing.

        Args:
            accelerations: Warp array of shape (N,) with dtype=wp.spatial_vectorf.
        """
        self._accelerations = wp.clone(accelerations)
        if self._accelerations.device.alias != self._device:
            self._accelerations = self._accelerations.to(self._device)

    def set_mock_masses(self, masses: wp.array) -> None:
        """Set mock mass data directly for testing.

        Args:
            masses: Warp array of shape (N, 1) with dtype=wp.float32.
        """
        self._masses = wp.clone(masses)

    def set_mock_coms(self, coms: wp.array) -> None:
        """Set mock center of mass data directly for testing.

        Args:
            coms: Warp array of shape (N,) with dtype=wp.transformf.
        """
        self._coms = wp.clone(coms)

    def set_mock_inertias(self, inertias: wp.array) -> None:
        """Set mock inertia data directly for testing.

        Args:
            inertias: Warp array of shape (N, 9) with dtype=wp.float32 - flattened 3x3 matrices.
        """
        self._inertias = wp.clone(inertias)

    # -- Actions (no-op for testing) --

    def apply_forces_and_torques_at_position(
        self,
        forces: wp.array | None = None,
        torques: wp.array | None = None,
        positions: wp.array | None = None,
        indices: wp.array | None = None,
        is_global: bool = True,
    ) -> None:
        """Apply forces and torques at positions (no-op in mock).

        Args:
            forces: Forces to apply, shape (N,) or (len(indices),) with dtype=wp.vec3f.
            torques: Torques to apply, shape (N,) or (len(indices),) with dtype=wp.vec3f.
            positions: Positions to apply forces at, shape (N,) or (len(indices),) with dtype=wp.vec3f.
            indices: Optional indices of bodies to apply to.
            is_global: Whether forces/torques are in global frame.
        """
        pass  # No-op for mock
