# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mock implementations of ovphysx TensorBinding objects for unit testing."""

from __future__ import annotations

import numpy as np


class MockTensorBinding:
    """Mock of ovphysx.TensorBinding that stores data in numpy arrays.

    Mimics the real TensorBinding API: ``read(tensor)`` fills the tensor from
    the internal buffer, ``write(tensor, indices, mask)`` copies into it.
    """

    def __init__(
        self,
        tensor_type: int,
        shape: tuple[int, ...],
        count: int,
        dof_count: int = 0,
        body_count: int = 0,
        joint_count: int = 0,
        is_fixed_base: bool = False,
        dof_names: list[str] | None = None,
        body_names: list[str] | None = None,
        joint_names: list[str] | None = None,
    ):
        self.tensor_type = tensor_type
        self._shape = shape
        self._count = count
        self._dof_count = dof_count
        self._body_count = body_count
        self._joint_count = joint_count
        self._is_fixed_base = is_fixed_base
        self._dof_names = dof_names or []
        self._body_names = body_names or []
        self._joint_names = joint_names or []
        self._data = np.zeros(shape, dtype=np.float32)

    # -- Properties matching TensorBinding --

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def count(self) -> int:
        return self._count

    @property
    def dof_count(self) -> int:
        return self._dof_count

    @property
    def body_count(self) -> int:
        return self._body_count

    @property
    def joint_count(self) -> int:
        return self._joint_count

    @property
    def is_fixed_base(self) -> bool:
        return self._is_fixed_base

    @property
    def dof_names(self) -> list[str]:
        return self._dof_names

    @property
    def body_names(self) -> list[str]:
        return self._body_names

    @property
    def joint_names(self) -> list[str]:
        return self._joint_names

    # -- I/O --

    def read(self, tensor) -> None:
        """Copy internal data into the provided array (numpy or warp)."""
        try:
            import warp as wp
            if isinstance(tensor, wp.array):
                tmp = wp.from_numpy(self._data, dtype=wp.float32, device=tensor.device)
                wp.copy(tensor, tmp)
                return
        except ImportError:
            pass
        np_dst = np.asarray(tensor)
        np.copyto(np_dst, self._data.reshape(np_dst.shape))

    @staticmethod
    def _to_numpy(arr) -> np.ndarray:
        """Convert warp/torch/numpy array to numpy, handling GPU arrays."""
        try:
            import warp as wp
            if isinstance(arr, wp.array):
                return arr.numpy()
        except ImportError:
            pass
        try:
            import torch
            if isinstance(arr, torch.Tensor):
                return arr.detach().cpu().numpy()
        except ImportError:
            pass
        return np.asarray(arr)

    def write(self, tensor, indices=None, mask=None) -> None:
        """Copy from the provided array (numpy or warp) into internal data."""
        np_src = self._to_numpy(tensor).astype(np.float32)
        if indices is not None:
            idx = self._to_numpy(indices)
            self._data[idx] = np_src
        elif mask is not None:
            np_mask = self._to_numpy(mask).astype(bool)
            self._data[np_mask] = np_src[np_mask]
        else:
            np.copyto(self._data, np_src.reshape(self._data.shape))

    def destroy(self) -> None:
        pass

    def set_random_data(self, low: float = -1.0, high: float = 1.0) -> None:
        """Fill internal buffer with random data."""
        self._data = np.random.uniform(low, high, self._shape).astype(np.float32)


class MockOvPhysxBindingSet:
    """Factory that creates a full set of MockTensorBinding objects
    for a given articulation configuration.

    Mirrors the tensor types that ``Articulation._initialize_impl`` creates.
    """

    # Tensor type constants (matching ovphysx._bindings values).
    ROOT_POSE = 10
    ROOT_VELOCITY = 11
    LINK_POSE = 20
    LINK_VELOCITY = 21
    LINK_ACCELERATION = 22
    DOF_POSITION = 30
    DOF_VELOCITY = 31
    DOF_POSITION_TARGET = 32
    DOF_VELOCITY_TARGET = 33
    DOF_ACTUATION_FORCE = 34
    DOF_STIFFNESS = 35
    DOF_DAMPING = 36
    DOF_LIMIT = 37
    DOF_MAX_VELOCITY = 38
    DOF_MAX_FORCE = 39
    DOF_ARMATURE = 40
    DOF_FRICTION_PROPERTIES = 41
    LINK_WRENCH = 52
    BODY_MASS = 60
    BODY_COM_POSE = 61
    BODY_INERTIA = 62
    LINK_INCOMING_JOINT_FORCE = 74

    def __init__(
        self,
        num_instances: int,
        num_joints: int,
        num_bodies: int,
        is_fixed_base: bool = False,
        joint_names: list[str] | None = None,
        body_names: list[str] | None = None,
    ):
        N = num_instances
        D = num_joints
        L = num_bodies

        if joint_names is None:
            joint_names = [f"joint_{i}" for i in range(D)]
        if body_names is None:
            body_names = [f"body_{i}" for i in range(L)]

        common = dict(
            count=N, dof_count=D, body_count=L, joint_count=D,
            is_fixed_base=is_fixed_base, dof_names=joint_names,
            body_names=body_names, joint_names=joint_names,
        )

        self.bindings: dict[int, MockTensorBinding] = {
            self.ROOT_POSE: MockTensorBinding(self.ROOT_POSE, (N, 7), **common),
            self.ROOT_VELOCITY: MockTensorBinding(self.ROOT_VELOCITY, (N, 6), **common),
            self.LINK_POSE: MockTensorBinding(self.LINK_POSE, (N, L, 7), **common),
            self.LINK_VELOCITY: MockTensorBinding(self.LINK_VELOCITY, (N, L, 6), **common),
            self.LINK_ACCELERATION: MockTensorBinding(self.LINK_ACCELERATION, (N, L, 6), **common),
            self.DOF_POSITION: MockTensorBinding(self.DOF_POSITION, (N, D), **common),
            self.DOF_VELOCITY: MockTensorBinding(self.DOF_VELOCITY, (N, D), **common),
            self.DOF_POSITION_TARGET: MockTensorBinding(self.DOF_POSITION_TARGET, (N, D), **common),
            self.DOF_VELOCITY_TARGET: MockTensorBinding(self.DOF_VELOCITY_TARGET, (N, D), **common),
            self.DOF_ACTUATION_FORCE: MockTensorBinding(self.DOF_ACTUATION_FORCE, (N, D), **common),
            self.DOF_STIFFNESS: MockTensorBinding(self.DOF_STIFFNESS, (N, D), **common),
            self.DOF_DAMPING: MockTensorBinding(self.DOF_DAMPING, (N, D), **common),
            self.DOF_LIMIT: MockTensorBinding(self.DOF_LIMIT, (N, D, 2), **common),
            self.DOF_MAX_VELOCITY: MockTensorBinding(self.DOF_MAX_VELOCITY, (N, D), **common),
            self.DOF_MAX_FORCE: MockTensorBinding(self.DOF_MAX_FORCE, (N, D), **common),
            self.DOF_ARMATURE: MockTensorBinding(self.DOF_ARMATURE, (N, D), **common),
            self.DOF_FRICTION_PROPERTIES: MockTensorBinding(self.DOF_FRICTION_PROPERTIES, (N, D, 3), **common),
            self.BODY_MASS: MockTensorBinding(self.BODY_MASS, (N, L), **common),
            self.BODY_COM_POSE: MockTensorBinding(self.BODY_COM_POSE, (N, L, 7), **common),
            self.BODY_INERTIA: MockTensorBinding(self.BODY_INERTIA, (N, L, 9), **common),
            self.LINK_INCOMING_JOINT_FORCE: MockTensorBinding(self.LINK_INCOMING_JOINT_FORCE, (N, L, 6), **common),
        }

    def set_random_data(self) -> None:
        """Fill all bindings with random data."""
        for b in self.bindings.values():
            b.set_random_data()
        # Set sensible defaults for limits (lower < upper).
        lim = self.bindings[self.DOF_LIMIT]
        lim._data[..., 0] = -3.14
        lim._data[..., 1] = 3.14
        # Set unit quaternions for poses.
        for tt in (self.ROOT_POSE, self.LINK_POSE, self.BODY_COM_POSE):
            b = self.bindings[tt]
            b._data[..., 3:6] = 0.0
            b._data[..., 6] = 1.0
        # Set positive masses.
        self.bindings[self.BODY_MASS]._data = np.abs(self.bindings[self.BODY_MASS]._data) + 0.1
        # Set positive max velocity / force.
        self.bindings[self.DOF_MAX_VELOCITY]._data = np.abs(self.bindings[self.DOF_MAX_VELOCITY]._data) + 1.0
        self.bindings[self.DOF_MAX_FORCE]._data = np.abs(self.bindings[self.DOF_MAX_FORCE]._data) + 1.0
