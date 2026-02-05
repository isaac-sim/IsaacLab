# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mock implementation of PhysX RigidBodyView."""

from __future__ import annotations

import torch


class MockRigidBodyView:
    """Mock implementation of physx.RigidBodyView for unit testing.

    This class mimics the interface of the PhysX TensorAPI RigidBodyView,
    allowing tests to run without Isaac Sim or GPU simulation.

    Data Shapes:
        - transforms: (N, 7) - [pos(3), quat_xyzw(4)]
        - velocities: (N, 6) - [lin_vel(3), ang_vel(3)]
        - accelerations: (N, 6) - [lin_acc(3), ang_acc(3)]
        - masses: (N, 1)
        - coms: (N, 7) - center of mass [pos(3), quat_xyzw(4)]
        - inertias: (N, 9) - flattened 3x3 inertia matrix (row-major)
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
            device: Device for tensor allocation ("cpu" or "cuda").
        """
        self._count = count
        self._prim_paths = prim_paths or [f"/World/RigidBody_{i}" for i in range(count)]
        self._device = device
        self._backend = "torch"

        # Internal state (lazily initialized)
        self._transforms: torch.Tensor | None = None
        self._velocities: torch.Tensor | None = None
        self._accelerations: torch.Tensor | None = None
        self._masses: torch.Tensor | None = None
        self._coms: torch.Tensor | None = None
        self._inertias: torch.Tensor | None = None

    # -- Helper Methods --

    def _check_cpu_tensor(self, tensor: torch.Tensor, name: str) -> None:
        """Check that tensor is on CPU, raise RuntimeError if on GPU.

        This mimics PhysX behavior where body properties must be on CPU.
        """
        if tensor.is_cuda:
            raise RuntimeError(
                f"Expected CPU tensor for {name}, but got tensor on {tensor.device}. "
                "Body properties must be set with CPU tensors."
            )

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

    def get_transforms(self) -> torch.Tensor:
        """Get world transforms of all rigid bodies.

        Returns:
            Tensor of shape (N, 7) with [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w].
        """
        if self._transforms is None:
            # Default: origin position with identity quaternion (xyzw format)
            self._transforms = torch.zeros(self._count, 7, device=self._device)
            self._transforms[:, 6] = 1.0  # w=1 for identity quaternion
        return self._transforms.clone()

    def get_velocities(self) -> torch.Tensor:
        """Get velocities of all rigid bodies.

        Returns:
            Tensor of shape (N, 6) with [lin_vel(3), ang_vel(3)].
        """
        if self._velocities is None:
            self._velocities = torch.zeros(self._count, 6, device=self._device)
        return self._velocities.clone()

    def get_accelerations(self) -> torch.Tensor:
        """Get accelerations of all rigid bodies.

        Returns:
            Tensor of shape (N, 6) with [lin_acc(3), ang_acc(3)].
        """
        if self._accelerations is None:
            self._accelerations = torch.zeros(self._count, 6, device=self._device)
        return self._accelerations.clone()

    def get_masses(self) -> torch.Tensor:
        """Get masses of all rigid bodies.

        Returns:
            Tensor of shape (N, 1) with mass values. Always on CPU.
        """
        if self._masses is None:
            self._masses = torch.ones(self._count, 1, device="cpu")
        return self._masses.clone()

    def get_coms(self) -> torch.Tensor:
        """Get centers of mass of all rigid bodies.

        Returns:
            Tensor of shape (N, 7) with [pos(3), quat_xyzw(4)]. Always on CPU.
        """
        if self._coms is None:
            # Default: local origin with identity quaternion - stored on CPU
            self._coms = torch.zeros(self._count, 7, device="cpu")
            self._coms[:, 6] = 1.0  # w=1 for identity quaternion
        return self._coms.clone()

    def get_inertias(self) -> torch.Tensor:
        """Get inertia tensors of all rigid bodies.

        Returns:
            Tensor of shape (N, 9) with flattened 3x3 inertia matrices (row-major). Always on CPU.
        """
        if self._inertias is None:
            # Default: identity inertia (unit sphere) - flattened [1,0,0,0,1,0,0,0,1] - stored on CPU
            self._inertias = torch.zeros(self._count, 9, device="cpu")
            self._inertias[:, 0] = 1.0  # [0,0]
            self._inertias[:, 4] = 1.0  # [1,1]
            self._inertias[:, 8] = 1.0  # [2,2]
        return self._inertias.clone()

    # -- Setters (simulation interface) --

    def set_transforms(
        self,
        transforms: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> None:
        """Set world transforms of rigid bodies.

        Args:
            transforms: Tensor of shape (N, 7) or (len(indices), 7).
            indices: Optional indices of bodies to update.
        """
        transforms = transforms.to(self._device)
        if self._transforms is None:
            self._transforms = torch.zeros(self._count, 7, device=self._device)
            self._transforms[:, 6] = 1.0
        if indices is not None:
            self._transforms[indices] = transforms
        else:
            self._transforms = transforms

    def set_velocities(
        self,
        velocities: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> None:
        """Set velocities of rigid bodies.

        Args:
            velocities: Tensor of shape (N, 6) or (len(indices), 6).
            indices: Optional indices of bodies to update.
        """
        velocities = velocities.to(self._device)
        if self._velocities is None:
            self._velocities = torch.zeros(self._count, 6, device=self._device)
        if indices is not None:
            self._velocities[indices] = velocities
        else:
            self._velocities = velocities

    def set_masses(
        self,
        masses: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> None:
        """Set masses of rigid bodies.

        Args:
            masses: Tensor of shape (N, 1) or (len(indices), 1). Must be on CPU.
            indices: Optional indices of bodies to update.

        Raises:
            RuntimeError: If masses tensor is on GPU.
        """
        self._check_cpu_tensor(masses, "masses")
        if self._masses is None:
            self._masses = torch.ones(self._count, 1, device="cpu")
        if indices is not None:
            self._masses[indices] = masses
        else:
            self._masses = masses

    def set_coms(
        self,
        coms: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> None:
        """Set centers of mass of rigid bodies.

        Args:
            coms: Tensor of shape (N, 7) or (len(indices), 7). Must be on CPU.
            indices: Optional indices of bodies to update.

        Raises:
            RuntimeError: If coms tensor is on GPU.
        """
        self._check_cpu_tensor(coms, "coms")
        if self._coms is None:
            self._coms = torch.zeros(self._count, 7, device="cpu")
            self._coms[:, 6] = 1.0
        if indices is not None:
            self._coms[indices] = coms
        else:
            self._coms = coms

    def set_inertias(
        self,
        inertias: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> None:
        """Set inertia tensors of rigid bodies.

        Args:
            inertias: Tensor of shape (N, 9) or (len(indices), 9) - flattened 3x3 matrices. Must be on CPU.
            indices: Optional indices of bodies to update.

        Raises:
            RuntimeError: If inertias tensor is on GPU.
        """
        self._check_cpu_tensor(inertias, "inertias")
        if self._inertias is None:
            self._inertias = torch.zeros(self._count, 9, device="cpu")
            self._inertias[:, 0] = 1.0
            self._inertias[:, 4] = 1.0
            self._inertias[:, 8] = 1.0
        if indices is not None:
            self._inertias[indices] = inertias
        else:
            self._inertias = inertias

    # -- Mock setters (direct test data injection) --

    def set_mock_transforms(self, transforms: torch.Tensor) -> None:
        """Set mock transform data directly for testing.

        Args:
            transforms: Tensor of shape (N, 7).
        """
        self._transforms = transforms.to(self._device)

    def set_mock_velocities(self, velocities: torch.Tensor) -> None:
        """Set mock velocity data directly for testing.

        Args:
            velocities: Tensor of shape (N, 6).
        """
        self._velocities = velocities.to(self._device)

    def set_mock_accelerations(self, accelerations: torch.Tensor) -> None:
        """Set mock acceleration data directly for testing.

        Args:
            accelerations: Tensor of shape (N, 6).
        """
        self._accelerations = accelerations.to(self._device)

    def set_mock_masses(self, masses: torch.Tensor) -> None:
        """Set mock mass data directly for testing.

        Args:
            masses: Tensor of shape (N, 1).
        """
        self._masses = masses.to(self._device)

    def set_mock_coms(self, coms: torch.Tensor) -> None:
        """Set mock center of mass data directly for testing.

        Args:
            coms: Tensor of shape (N, 7).
        """
        self._coms = coms.to(self._device)

    def set_mock_inertias(self, inertias: torch.Tensor) -> None:
        """Set mock inertia data directly for testing.

        Args:
            inertias: Tensor of shape (N, 9) - flattened 3x3 matrices.
        """
        self._inertias = inertias.to(self._device)

    # -- Actions (no-op for testing) --

    def apply_forces_and_torques_at_position(
        self,
        forces: torch.Tensor | None = None,
        torques: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
        is_global: bool = True,
    ) -> None:
        """Apply forces and torques at positions (no-op in mock).

        Args:
            forces: Forces to apply, shape (N, 3) or (len(indices), 3).
            torques: Torques to apply, shape (N, 3) or (len(indices), 3).
            positions: Positions to apply forces at, shape (N, 3) or (len(indices), 3).
            indices: Optional indices of bodies to apply to.
            is_global: Whether forces/torques are in global frame.
        """
        pass  # No-op for mock

    # -- Convenience method for benchmarking --

    def set_random_mock_data(self) -> None:
        """Set all internal state to random values for benchmarking.

        This method initializes all mock data with random values,
        useful for benchmarking where the actual values don't matter.
        """
        # Transforms with normalized quaternions - on device
        self._transforms = torch.randn(self._count, 7, device=self._device)
        self._transforms[:, 3:7] = torch.nn.functional.normalize(self._transforms[:, 3:7], dim=-1)

        # Velocities and accelerations - on device
        self._velocities = torch.randn(self._count, 6, device=self._device)
        self._accelerations = torch.randn(self._count, 6, device=self._device)

        # Mass properties - stored on CPU (PhysX requirement)
        self._masses = torch.rand(self._count, 1, device="cpu") * 10

        # Center of mass with normalized quaternions - stored on CPU (PhysX requirement)
        self._coms = torch.randn(self._count, 7, device="cpu")
        self._coms[:, 3:7] = torch.nn.functional.normalize(self._coms[:, 3:7], dim=-1)

        # Inertia tensors (positive definite diagonal) - flattened (N, 9) - stored on CPU (PhysX requirement)
        # Create diagonal inertia matrices and flatten
        diag_values = torch.rand(self._count, 3, device="cpu") + 0.1  # Ensure positive
        self._inertias = torch.zeros(self._count, 9, device="cpu")
        self._inertias[:, 0] = diag_values[:, 0]  # [0,0]
        self._inertias[:, 4] = diag_values[:, 1]  # [1,1]
        self._inertias[:, 8] = diag_values[:, 2]  # [2,2]
