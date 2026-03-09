# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mock implementation of Newton ArticulationView using structured Warp types."""

from __future__ import annotations

import numpy as np
import warp as wp


class MockNewtonCollectionView:
    """Mock Newton ArticulationView for rigid object collection testing.

    This class mimics the interface of a single ``ArticulationView`` whose combined
    fnmatch pattern matches **B** body types per world. Its data is shaped
    ``(N, B, ...)`` rather than the ``(N, 1, ...)`` convention used for single
    articulations or single rigid bodies.

    Data Shapes:
        - root_transforms: ``(N, B)`` dtype=wp.transformf
        - root_velocities: ``(N, B)`` dtype=wp.spatial_vectorf
        - body_com:        ``(N, B, 1)`` dtype=wp.vec3f
        - body_mass:       ``(N, B, 1)`` dtype=wp.float32
        - body_inertia:    ``(N, B, 1)`` dtype=wp.mat33f
        - body_f:          ``(N, B, 1)`` dtype=wp.spatial_vectorf

    Where N = num_envs, B = num_bodies (body types in the collection).
    ``count`` returns ``N * B`` so that the data class can compute
    ``num_instances = count // num_bodies``.
    """

    def __init__(
        self,
        num_envs: int = 2,
        num_bodies: int = 3,
        device: str = "cpu",
        body_names: list[str] | None = None,
    ):
        self._num_envs = num_envs
        self._num_bodies = num_bodies
        self._device = device
        self._noop_setters = False

        self._body_names = body_names if body_names is not None else [f"object_{i}" for i in range(num_bodies)]

        # Internal state (lazily initialised)
        self._root_transforms: wp.array | None = None
        self._root_velocities: wp.array | None = None
        self._attributes: dict[str, wp.array | None] = {
            "body_com": None,
            "body_mass": None,
            "body_inertia": None,
            "body_f": None,
        }

    # -- Properties --------------------------------------------------------

    @property
    def count(self) -> int:
        """Total matched entities (``N * B``)."""
        return self._num_envs * self._num_bodies

    @property
    def body_names(self) -> list[str]:
        return self._body_names

    # -- Lazy init helpers -------------------------------------------------

    def _ensure_root_transforms(self) -> wp.array:
        if self._root_transforms is None:
            self._root_transforms = wp.zeros(
                (self._num_envs, self._num_bodies), dtype=wp.transformf, device=self._device
            )
        return self._root_transforms

    def _ensure_root_velocities(self) -> wp.array:
        if self._root_velocities is None:
            self._root_velocities = wp.zeros(
                (self._num_envs, self._num_bodies), dtype=wp.spatial_vectorf, device=self._device
            )
        return self._root_velocities

    def _ensure_attribute(self, name: str) -> wp.array:
        if self._attributes[name] is None:
            self._attributes[name] = self._create_default_attribute(name)
        return self._attributes[name]

    def _create_default_attribute(self, name: str) -> wp.array:
        N, B = self._num_envs, self._num_bodies
        dev = self._device
        if name == "body_com":
            return wp.zeros((N, B, 1), dtype=wp.vec3f, device=dev)
        elif name == "body_mass":
            return wp.zeros((N, B, 1), dtype=wp.float32, device=dev)
        elif name == "body_inertia":
            return wp.zeros((N, B, 1), dtype=wp.mat33f, device=dev)
        elif name == "body_f":
            return wp.zeros((N, B, 1), dtype=wp.spatial_vectorf, device=dev)
        else:
            raise KeyError(f"Unknown attribute: {name}")

    # -- Getters -----------------------------------------------------------

    def get_root_transforms(self, state) -> wp.array:
        return self._ensure_root_transforms()

    def get_root_velocities(self, state) -> wp.array:
        return self._ensure_root_velocities()

    def get_attribute(self, name: str, model_or_state) -> wp.array:
        return self._ensure_attribute(name)

    # -- Setters -----------------------------------------------------------

    def set_root_transforms(self, state, transforms: wp.array) -> None:
        if self._noop_setters:
            return
        self._ensure_root_transforms().assign(transforms)

    def set_root_velocities(self, state, velocities: wp.array) -> None:
        if self._noop_setters:
            return
        self._ensure_root_velocities().assign(velocities)

    # -- Mock data injection -----------------------------------------------

    def set_random_mock_data(self) -> None:
        """Set all internal state to random values for testing."""
        N, B = self._num_envs, self._num_bodies
        dev = self._device

        # Root transforms
        root_tf_np = np.random.randn(N, B, 7).astype(np.float32)
        root_tf_np[..., 3:7] /= np.linalg.norm(root_tf_np[..., 3:7], axis=-1, keepdims=True)
        self._root_transforms = wp.array(root_tf_np, dtype=wp.transformf, device=dev)

        # Root velocities
        root_vel_np = np.random.randn(N, B, 6).astype(np.float32)
        self._root_velocities = wp.array(root_vel_np, dtype=wp.spatial_vectorf, device=dev)

        # Attributes (all have trailing link dim of 1)
        self._attributes["body_com"] = wp.array(
            np.random.randn(N, B, 1, 3).astype(np.float32), dtype=wp.vec3f, device=dev
        )
        self._attributes["body_mass"] = wp.array(
            (np.random.rand(N, B, 1) * 10 + 0.1).astype(np.float32), dtype=wp.float32, device=dev
        )
        self._attributes["body_inertia"] = wp.array(
            np.random.randn(N, B, 1, 9).astype(np.float32), dtype=wp.mat33f, device=dev
        )
        self._attributes["body_f"] = wp.array(
            np.random.randn(N, B, 1, 6).astype(np.float32), dtype=wp.spatial_vectorf, device=dev
        )


class MockNewtonArticulationView:
    """Mock Newton ArticulationView for unit testing without simulation.

    This class mimics the interface that ``ArticulationData`` and ``RigidObjectData``
    expect from Newton's selection API. It can be used for both articulation and
    rigid object testing since Newton has no separate rigid body view.

    Data Shapes (structured Warp types with ``(N, 1, ...)`` convention):
        - root_transforms: ``(N, 1)`` dtype=wp.transformf for floating base,
          ``(N, 1, 1)`` for fixed base
        - root_velocities: ``(N, 1)`` dtype=wp.spatial_vectorf (None for fixed base)
        - link_transforms: ``(N, 1, L)`` dtype=wp.transformf
        - link_velocities: ``(N, 1, L)`` dtype=wp.spatial_vectorf (None for fixed base)
        - dof_positions: ``(N, 1, J)`` dtype=wp.float32
        - dof_velocities: ``(N, 1, J)`` dtype=wp.float32
        - body_com: ``(N, 1, L)`` dtype=wp.vec3f
        - body_mass: ``(N, 1, L)`` dtype=wp.float32
        - body_inertia: ``(N, 1, L)`` dtype=wp.mat33f

    Where N = count, L = link_count, J = joint_dof_count

    Note:
        Newton's selection API uses structured Warp types (``wp.transformf``,
        ``wp.spatial_vectorf``, ``wp.vec3f``, ``wp.mat33f``) natively, unlike PhysX
        which uses flat float32 arrays.
    """

    def __init__(
        self,
        num_instances: int = 1,
        num_bodies: int = 2,
        num_joints: int = 1,
        device: str = "cpu",
        is_fixed_base: bool = False,
        joint_names: list[str] | None = None,
        body_names: list[str] | None = None,
    ):
        """Initialize the mock Newton articulation view.

        Args:
            num_instances: Number of articulation instances.
            num_bodies: Number of bodies (links).
            num_joints: Number of joints (DOFs).
            device: Device for array allocation ("cpu" or "cuda:N").
            is_fixed_base: Whether the articulation has a fixed base.
            joint_names: Names of joints. Defaults to ["joint_0", ...].
            body_names: Names of bodies. Defaults to ["body_0", ...].
        """
        self._count = num_instances
        self._link_count = num_bodies
        self._joint_dof_count = num_joints
        self._device = device
        self._is_fixed_base = is_fixed_base
        self._noop_setters = False

        # Set joint and body names
        self._joint_dof_names = joint_names if joint_names is not None else [f"joint_{i}" for i in range(num_joints)]
        self._body_names = body_names if body_names is not None else [f"body_{i}" for i in range(num_bodies)]

        # Internal state (lazily initialized)
        self._root_transforms: wp.array | None = None
        self._root_velocities: wp.array | None = None
        self._link_transforms: wp.array | None = None
        self._link_velocities: wp.array | None = None
        self._dof_positions: wp.array | None = None
        self._dof_velocities: wp.array | None = None

        # Attributes dict (lazily initialized)
        self._attributes: dict[str, wp.array | None] = {
            "body_com": None,
            "body_mass": None,
            "body_inertia": None,
            "joint_limit_lower": None,
            "joint_limit_upper": None,
            "joint_target_ke": None,
            "joint_target_kd": None,
            "joint_armature": None,
            "joint_friction": None,
            "joint_velocity_limit": None,
            "joint_effort_limit": None,
            "body_f": None,
            "joint_f": None,
            "joint_target_pos": None,
            "joint_target_vel": None,
            "joint_limit_ke": None,
            "joint_limit_kd": None,
        }

    # -- Properties --

    @property
    def count(self) -> int:
        """Number of articulation instances."""
        return self._count

    @property
    def link_count(self) -> int:
        """Number of links (bodies) per instance."""
        return self._link_count

    @property
    def joint_dof_count(self) -> int:
        """Number of DOFs (joints) per instance."""
        return self._joint_dof_count

    @property
    def is_fixed_base(self) -> bool:
        """Whether the articulation has a fixed base."""
        return self._is_fixed_base

    @property
    def joint_dof_names(self) -> list[str]:
        """Names of the DOFs."""
        return self._joint_dof_names

    @property
    def body_names(self) -> list[str]:
        """Names of the bodies."""
        return self._body_names

    @property
    def link_names(self) -> list[str]:
        """Alias for body_names (Newton calls bodies 'links')."""
        return self._body_names

    # -- Lazy Initialization Helpers --

    def _ensure_root_transforms(self) -> wp.array:
        """Lazily create root transforms with identity quaternions."""
        if self._root_transforms is None:
            if self._is_fixed_base:
                self._root_transforms = wp.zeros((self._count, 1, 1), dtype=wp.transformf, device=self._device)
            else:
                self._root_transforms = wp.zeros((self._count, 1), dtype=wp.transformf, device=self._device)
        return self._root_transforms

    def _ensure_root_velocities(self) -> wp.array | None:
        """Lazily create root velocities (None for fixed base)."""
        if self._is_fixed_base:
            return None
        if self._root_velocities is None:
            self._root_velocities = wp.zeros((self._count, 1), dtype=wp.spatial_vectorf, device=self._device)
        return self._root_velocities

    def _ensure_link_transforms(self) -> wp.array:
        """Lazily create link transforms."""
        if self._link_transforms is None:
            self._link_transforms = wp.zeros(
                (self._count, 1, self._link_count), dtype=wp.transformf, device=self._device
            )
        return self._link_transforms

    def _ensure_link_velocities(self) -> wp.array | None:
        """Lazily create link velocities (None for fixed base)."""
        if self._is_fixed_base:
            return None
        if self._link_velocities is None:
            self._link_velocities = wp.zeros(
                (self._count, 1, self._link_count), dtype=wp.spatial_vectorf, device=self._device
            )
        return self._link_velocities

    def _ensure_dof_positions(self) -> wp.array:
        """Lazily create DOF positions."""
        if self._dof_positions is None:
            self._dof_positions = wp.zeros(
                (self._count, 1, self._joint_dof_count), dtype=wp.float32, device=self._device
            )
        return self._dof_positions

    def _ensure_dof_velocities(self) -> wp.array:
        """Lazily create DOF velocities."""
        if self._dof_velocities is None:
            self._dof_velocities = wp.zeros(
                (self._count, 1, self._joint_dof_count), dtype=wp.float32, device=self._device
            )
        return self._dof_velocities

    def _ensure_attribute(self, name: str) -> wp.array:
        """Lazily create an attribute array."""
        if self._attributes[name] is None:
            self._attributes[name] = self._create_default_attribute(name)
        return self._attributes[name]

    def _create_default_attribute(self, name: str) -> wp.array:
        """Create a default attribute array based on name."""
        N, L, J = self._count, self._link_count, self._joint_dof_count
        dev = self._device

        if name == "body_com":
            return wp.zeros((N, 1, L), dtype=wp.vec3f, device=dev)
        elif name == "body_mass":
            return wp.zeros((N, 1, L), dtype=wp.float32, device=dev)
        elif name == "body_inertia":
            return wp.zeros((N, 1, L), dtype=wp.mat33f, device=dev)
        elif name == "body_f":
            return wp.zeros((N, 1, L), dtype=wp.spatial_vectorf, device=dev)
        elif name in (
            "joint_limit_lower",
            "joint_limit_upper",
            "joint_target_ke",
            "joint_target_kd",
            "joint_armature",
            "joint_friction",
            "joint_velocity_limit",
            "joint_effort_limit",
            "joint_f",
            "joint_target_pos",
            "joint_target_vel",
            "joint_limit_ke",
            "joint_limit_kd",
        ):
            return wp.zeros((N, 1, J), dtype=wp.float32, device=dev)
        else:
            raise KeyError(f"Unknown attribute: {name}")

    # -- Root Getters --

    def get_root_transforms(self, state) -> wp.array:
        """Get world transforms of root links.

        Args:
            state: Newton state object (unused in mock).

        Returns:
            Warp array with dtype=wp.transformf. Shape ``(N, 1)`` for floating base,
            ``(N, 1, 1)`` for fixed base.
        """
        return self._ensure_root_transforms()

    def get_root_velocities(self, state) -> wp.array | None:
        """Get velocities of root links.

        Args:
            state: Newton state object (unused in mock).

        Returns:
            Warp array of shape ``(N, 1)`` with dtype=wp.spatial_vectorf,
            or None for fixed base.
        """
        return self._ensure_root_velocities()

    # -- Link Getters --

    def get_link_transforms(self, state) -> wp.array:
        """Get world transforms of all links.

        Args:
            state: Newton state object (unused in mock).

        Returns:
            Warp array of shape ``(N, 1, L)`` with dtype=wp.transformf.
        """
        return self._ensure_link_transforms()

    def get_link_velocities(self, state) -> wp.array | None:
        """Get velocities of all links.

        Args:
            state: Newton state object (unused in mock).

        Returns:
            Warp array of shape ``(N, 1, L)`` with dtype=wp.spatial_vectorf,
            or None for fixed base.
        """
        return self._ensure_link_velocities()

    # -- DOF Getters --

    def get_dof_positions(self, state) -> wp.array:
        """Get positions of all DOFs.

        Args:
            state: Newton state object (unused in mock).

        Returns:
            Warp array of shape ``(N, 1, J)`` with dtype=wp.float32.
        """
        return self._ensure_dof_positions()

    def get_dof_velocities(self, state) -> wp.array:
        """Get velocities of all DOFs.

        Args:
            state: Newton state object (unused in mock).

        Returns:
            Warp array of shape ``(N, 1, J)`` with dtype=wp.float32.
        """
        return self._ensure_dof_velocities()

    # -- Attribute Getter --

    def get_attribute(self, name: str, model_or_state) -> wp.array:
        """Get a named attribute array.

        Args:
            name: Attribute name (e.g. "body_mass", "joint_target_ke").
            model_or_state: Newton model or state object (unused in mock).

        Returns:
            Warp array for the requested attribute.

        Raises:
            KeyError: If the attribute name is unknown.
        """
        return self._ensure_attribute(name)

    # -- Root Setters --

    def set_root_transforms(self, state, transforms: wp.array) -> None:
        """Set world transforms of root links.

        Args:
            state: Newton state object (unused in mock).
            transforms: Warp array with dtype=wp.transformf matching root shape.

        Raises:
            ValueError: If the transforms shape does not match the expected root shape.
        """
        if self._noop_setters:
            return
        expected = self._ensure_root_transforms()
        if transforms.shape != expected.shape:
            raise ValueError(f"Root transforms shape mismatch: expected {expected.shape}, got {transforms.shape}")
        expected.assign(transforms)

    def set_root_velocities(self, state, velocities: wp.array) -> None:
        """Set velocities of root links.

        Args:
            state: Newton state object (unused in mock).
            velocities: Warp array of shape ``(N, 1)`` with dtype=wp.spatial_vectorf.
        """
        if self._noop_setters:
            return
        vel = self._ensure_root_velocities()
        if vel is not None:
            vel.assign(velocities)

    # -- Mock Setters (direct test data injection) --

    def set_mock_root_transforms(self, transforms: wp.array) -> None:
        """Set mock root transform data directly for testing.

        Args:
            transforms: Warp array with dtype=wp.transformf.
        """
        self._root_transforms = transforms

    def set_mock_root_velocities(self, velocities: wp.array) -> None:
        """Set mock root velocity data directly for testing.

        Args:
            velocities: Warp array with dtype=wp.spatial_vectorf.
        """
        self._root_velocities = velocities

    def set_mock_link_transforms(self, transforms: wp.array) -> None:
        """Set mock link transform data directly for testing.

        Args:
            transforms: Warp array of shape ``(N, 1, L)`` with dtype=wp.transformf.
        """
        self._link_transforms = transforms

    def set_mock_link_velocities(self, velocities: wp.array) -> None:
        """Set mock link velocity data directly for testing.

        Args:
            velocities: Warp array of shape ``(N, 1, L)`` with dtype=wp.spatial_vectorf.
        """
        self._link_velocities = velocities

    def set_mock_dof_positions(self, positions: wp.array) -> None:
        """Set mock DOF position data directly for testing.

        Args:
            positions: Warp array of shape ``(N, 1, J)`` with dtype=wp.float32.
        """
        self._dof_positions = positions

    def set_mock_dof_velocities(self, velocities: wp.array) -> None:
        """Set mock DOF velocity data directly for testing.

        Args:
            velocities: Warp array of shape ``(N, 1, J)`` with dtype=wp.float32.
        """
        self._dof_velocities = velocities

    def set_mock_masses(self, masses: wp.array) -> None:
        """Set mock body mass data directly for testing.

        Args:
            masses: Warp array of shape ``(N, 1, L)`` with dtype=wp.float32.
        """
        self._attributes["body_mass"] = masses

    def set_mock_coms(self, coms: wp.array) -> None:
        """Set mock body center-of-mass data directly for testing.

        Args:
            coms: Warp array of shape ``(N, 1, L)`` with dtype=wp.vec3f.
        """
        self._attributes["body_com"] = coms

    def set_mock_inertias(self, inertias: wp.array) -> None:
        """Set mock body inertia data directly for testing.

        Args:
            inertias: Warp array of shape ``(N, 1, L)`` with dtype=wp.mat33f.
        """
        self._attributes["body_inertia"] = inertias

    # -- Benchmark Utilities --

    def set_random_mock_data(self) -> None:
        """Set all internal state to random values for benchmarking.

        Uses numpy for random data generation (matching PhysX mock pattern).
        """
        N = self._count
        L = self._link_count
        J = self._joint_dof_count
        dev = self._device

        # Root transforms
        if self._is_fixed_base:
            root_tf_np = np.random.randn(N, 1, 1, 7).astype(np.float32)
            root_tf_np[..., 3:7] /= np.linalg.norm(root_tf_np[..., 3:7], axis=-1, keepdims=True)
            self._root_transforms = wp.array(root_tf_np, dtype=wp.transformf, device=dev)
        else:
            root_tf_np = np.random.randn(N, 1, 7).astype(np.float32)
            root_tf_np[..., 3:7] /= np.linalg.norm(root_tf_np[..., 3:7], axis=-1, keepdims=True)
            self._root_transforms = wp.array(root_tf_np, dtype=wp.transformf, device=dev)

            # Root velocities (floating base only)
            root_vel_np = np.random.randn(N, 1, 6).astype(np.float32)
            self._root_velocities = wp.array(root_vel_np, dtype=wp.spatial_vectorf, device=dev)

        # Link transforms
        link_tf_np = np.random.randn(N, 1, L, 7).astype(np.float32)
        link_tf_np[..., 3:7] /= np.linalg.norm(link_tf_np[..., 3:7], axis=-1, keepdims=True)
        self._link_transforms = wp.array(link_tf_np, dtype=wp.transformf, device=dev)

        # Link velocities (floating base only)
        if not self._is_fixed_base:
            link_vel_np = np.random.randn(N, 1, L, 6).astype(np.float32)
            self._link_velocities = wp.array(link_vel_np, dtype=wp.spatial_vectorf, device=dev)

        # DOF state
        self._dof_positions = wp.array(np.random.randn(N, 1, J).astype(np.float32), dtype=wp.float32, device=dev)
        self._dof_velocities = wp.array(np.random.randn(N, 1, J).astype(np.float32), dtype=wp.float32, device=dev)

        # Body properties
        self._attributes["body_com"] = wp.array(
            np.random.randn(N, 1, L, 3).astype(np.float32),
            dtype=wp.vec3f,
            device=dev,
        )
        self._attributes["body_mass"] = wp.array(
            (np.random.rand(N, 1, L) * 10 + 0.1).astype(np.float32),
            dtype=wp.float32,
            device=dev,
        )
        self._attributes["body_inertia"] = wp.array(
            np.random.randn(N, 1, L, 9).astype(np.float32),
            dtype=wp.mat33f,
            device=dev,
        )

        # Joint properties
        for attr_name in (
            "joint_limit_lower",
            "joint_limit_upper",
            "joint_target_ke",
            "joint_target_kd",
            "joint_armature",
            "joint_friction",
            "joint_velocity_limit",
            "joint_effort_limit",
            "joint_f",
            "joint_target_pos",
            "joint_target_vel",
            "joint_limit_ke",
            "joint_limit_kd",
        ):
            self._attributes[attr_name] = wp.array(
                np.random.randn(N, 1, J).astype(np.float32),
                dtype=wp.float32,
                device=dev,
            )

        # Body forces
        self._attributes["body_f"] = wp.array(
            np.random.randn(N, 1, L, 6).astype(np.float32),
            dtype=wp.spatial_vectorf,
            device=dev,
        )
