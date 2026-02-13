# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mock implementation of PhysX RigidBodyView using Warp arrays."""

from __future__ import annotations

import warp as wp

from ..utils.kernels_mock import (
    init_identity_inertias_1d,
    init_identity_transforms_1d_flat,
    scatter_floats_2d,
)


class MockRigidBodyViewWarp:
    """Mock implementation of physx.RigidBodyView using Warp arrays for unit testing.

    This class mimics the interface of the PhysX TensorAPI RigidBodyView using
    flat float32 arrays (matching real PhysX behavior), allowing tests to run
    without Isaac Sim or GPU simulation.

    Data Shapes (flat float32, matching real PhysX views):
        - transforms: (N, 7) dtype=wp.float32 - [pos(3), quat_xyzw(4)]
        - velocities: (N, 6) dtype=wp.float32 - [ang_vel(3), lin_vel(3)]
        - accelerations: (N, 6) dtype=wp.float32 - [ang_acc(3), lin_acc(3)]
        - masses: (N, 1) dtype=wp.float32
        - coms: (N, 7) dtype=wp.float32 - center of mass [pos(3), quat_xyzw(4)]
        - inertias: (N, 9) dtype=wp.float32 - flattened 3x3 inertia matrix (row-major)
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
        self._noop_setters = False

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

    def _create_identity_transforms(self, count: int, device: str | None = None) -> wp.array:
        """Create array of identity transforms as (count, 7) float32."""
        dev = device or self._device
        arr = wp.zeros((count, 7), dtype=wp.float32, device=dev)
        wp.launch(init_identity_transforms_1d_flat, dim=count, inputs=[arr], device=dev)
        return arr

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
            Warp array of shape (N, 7) with dtype=wp.float32.
            Each row contains [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w].
        """
        if self._transforms is None:
            self._transforms = self._create_identity_transforms(self._count)
        return wp.clone(self._transforms)

    def get_velocities(self) -> wp.array:
        """Get velocities of all rigid bodies.

        Returns:
            Warp array of shape (N, 6) with dtype=wp.float32.
            Each row contains [ang_vel(3), lin_vel(3)].
        """
        if self._velocities is None:
            self._velocities = wp.zeros((self._count, 6), dtype=wp.float32, device=self._device)
        return wp.clone(self._velocities)

    def get_accelerations(self) -> wp.array:
        """Get accelerations of all rigid bodies.

        Returns:
            Warp array of shape (N, 6) with dtype=wp.float32.
            Each row contains [ang_acc(3), lin_acc(3)].
        """
        if self._accelerations is None:
            self._accelerations = wp.zeros((self._count, 6), dtype=wp.float32, device=self._device)
        return wp.clone(self._accelerations)

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
            Warp array of shape (N, 7) with dtype=wp.float32. Always on CPU.
            Each row contains [pos(3), quat_xyzw(4)].
        """
        if self._coms is None:
            self._coms = self._create_identity_transforms(self._count, device="cpu")
        return wp.clone(self._coms)

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
            transforms: Warp array of shape (N, 7) or (len(indices), 7) with dtype=wp.float32.
            indices: Optional Warp array of indices (dtype=wp.int32) of bodies to update.
        """
        if self._noop_setters:
            return
        if self._transforms is None:
            self._transforms = self._create_identity_transforms(self._count)
        if indices is not None:
            wp.launch(
                scatter_floats_2d,
                dim=(indices.shape[0], 7),
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
            velocities: Warp array of shape (N, 6) or (len(indices), 6) with dtype=wp.float32.
            indices: Optional Warp array of indices (dtype=wp.int32) of bodies to update.
        """
        if self._noop_setters:
            return
        if self._velocities is None:
            self._velocities = wp.zeros((self._count, 6), dtype=wp.float32, device=self._device)
        if indices is not None:
            wp.launch(
                scatter_floats_2d,
                dim=(indices.shape[0], 6),
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
        if self._noop_setters:
            return
        self._check_cpu_array(masses, "masses")
        if self._masses is None:
            self._masses = wp.ones((self._count, 1), dtype=wp.float32, device="cpu")
        if indices is not None:
            self._masses.numpy()[indices.numpy()] = masses.numpy()
        else:
            wp.copy(self._masses, masses)

    def set_coms(
        self,
        coms: wp.array,
        indices: wp.array | None = None,
    ) -> None:
        """Set centers of mass of rigid bodies.

        Args:
            coms: Warp array of shape (N, 7) or (len(indices), 7) with dtype=wp.float32. Must be on CPU.
            indices: Optional Warp array of indices (dtype=wp.int32) of bodies to update.

        Raises:
            RuntimeError: If coms array is on GPU.
        """
        if self._noop_setters:
            return
        self._check_cpu_array(coms, "coms")
        if self._coms is None:
            self._coms = self._create_identity_transforms(self._count, device="cpu")
        if indices is not None:
            self._coms.numpy()[indices.numpy()] = coms.numpy()
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
        if self._noop_setters:
            return
        self._check_cpu_array(inertias, "inertias")
        if self._inertias is None:
            self._inertias = wp.zeros((self._count, 9), dtype=wp.float32, device="cpu")
            wp.launch(init_identity_inertias_1d, dim=self._count, inputs=[self._inertias], device="cpu")
        if indices is not None:
            self._inertias.numpy()[indices.numpy()] = inertias.numpy()
        else:
            wp.copy(self._inertias, inertias)

    # -- Mock setters (direct test data injection) --

    def set_mock_transforms(self, transforms: wp.array) -> None:
        """Set mock transform data directly for testing.

        Args:
            transforms: Warp array of shape (N, 7) with dtype=wp.float32.
        """
        self._transforms = wp.clone(transforms)
        if self._transforms.device.alias != self._device:
            self._transforms = self._transforms.to(self._device)

    def set_mock_velocities(self, velocities: wp.array) -> None:
        """Set mock velocity data directly for testing.

        Args:
            velocities: Warp array of shape (N, 6) with dtype=wp.float32.
        """
        self._velocities = wp.clone(velocities)
        if self._velocities.device.alias != self._device:
            self._velocities = self._velocities.to(self._device)

    def set_mock_accelerations(self, accelerations: wp.array) -> None:
        """Set mock acceleration data directly for testing.

        Args:
            accelerations: Warp array of shape (N, 6) with dtype=wp.float32.
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
            coms: Warp array of shape (N, 7) with dtype=wp.float32.
        """
        self._coms = wp.clone(coms)

    def set_mock_inertias(self, inertias: wp.array) -> None:
        """Set mock inertia data directly for testing.

        Args:
            inertias: Warp array of shape (N, 9) with dtype=wp.float32 - flattened 3x3 matrices.
        """
        self._inertias = wp.clone(inertias)

    # -- Convenience method for benchmarking --

    def set_random_mock_data(self) -> None:
        """Set all internal state to random values for benchmarking.

        This method initializes all mock data with random warp arrays,
        useful for benchmarking where the actual values don't matter.
        """
        import numpy as np

        N = self._count

        # Transforms with normalized quaternions - on device
        tf = np.random.randn(N, 7).astype(np.float32)
        tf[:, 3:7] /= np.linalg.norm(tf[:, 3:7], axis=-1, keepdims=True)
        self._transforms = wp.array(tf, dtype=wp.float32, device=self._device)

        # Velocities and accelerations - on device
        self._velocities = wp.array(np.random.randn(N, 6).astype(np.float32), dtype=wp.float32, device=self._device)
        self._accelerations = wp.array(np.random.randn(N, 6).astype(np.float32), dtype=wp.float32, device=self._device)

        # Mass properties - stored on CPU (PhysX requirement)
        self._masses = wp.array((np.random.rand(N, 1) * 10).astype(np.float32), dtype=wp.float32, device="cpu")

        # Center of mass with normalized quaternions - stored on CPU (PhysX requirement)
        c = np.random.randn(N, 7).astype(np.float32)
        c[:, 3:7] /= np.linalg.norm(c[:, 3:7], axis=-1, keepdims=True)
        self._coms = wp.array(c, dtype=wp.float32, device="cpu")

        # Inertia tensors (positive definite diagonal) - flattened (N, 9) - stored on CPU (PhysX requirement)
        diag_values = np.random.rand(N, 3).astype(np.float32) + 0.1
        inertias = np.zeros((N, 9), dtype=np.float32)
        inertias[:, 0] = diag_values[:, 0]
        inertias[:, 4] = diag_values[:, 1]
        inertias[:, 8] = diag_values[:, 2]
        self._inertias = wp.array(inertias, dtype=wp.float32, device="cpu")

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
            forces: Forces to apply, shape (N, 3) or (len(indices), 3) with dtype=wp.float32.
            torques: Torques to apply, shape (N, 3) or (len(indices), 3) with dtype=wp.float32.
            positions: Positions to apply forces at, shape (N, 3) or (len(indices), 3) with dtype=wp.float32.
            indices: Optional indices of bodies to apply to.
            is_global: Whether forces/torques are in global frame.
        """
        pass  # No-op for mock
