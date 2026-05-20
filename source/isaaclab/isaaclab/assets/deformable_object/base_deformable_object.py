# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.assets.asset_base import AssetBase
from isaaclab.utils.warp import ProxyArray

if TYPE_CHECKING:
    from .base_deformable_object_data import BaseDeformableObjectData
    from .deformable_object_cfg import DeformableObjectCfg


class BaseDeformableObject(AssetBase):
    """Abstract base class for deformable object assets.

    Deformable objects are assets that can be deformed in the simulation. They are typically used for
    soft bodies, such as stuffed animals, food items, and cloth.

    Unlike rigid object assets, deformable objects have a more complex structure and require additional
    handling for simulation. The state of a deformable object comprises of its nodal positions and
    velocities, and not the object's root position and orientation. The nodal positions and velocities
    are in the simulation frame.

    Soft bodies can be `partially kinematic`_, where some nodes are driven by kinematic targets, and
    the rest are simulated. The kinematic targets are the desired positions of the nodes, and the
    simulation drives the nodes towards these targets.

    .. _partially kinematic: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/SoftBodies.html#kinematic-soft-bodies
    """

    cfg: DeformableObjectCfg
    """Configuration instance for the deformable object."""

    __backend_name__: str = "base"
    """The name of the backend for the deformable object."""

    def __init__(self, cfg: DeformableObjectCfg):
        """Initialize the deformable object.

        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg)

    """
    Properties
    """

    @property
    @abstractmethod
    def data(self) -> BaseDeformableObjectData:
        """Data container for the deformable object."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def num_instances(self) -> int:
        """Number of instances of the asset."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def num_bodies(self) -> int:
        """Number of bodies in the asset.

        This is always 1 since each object is a single deformable body.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def max_sim_vertices_per_body(self) -> int:
        """The maximum number of simulation mesh vertices per deformable body."""
        raise NotImplementedError()

    """
    Operations.
    """

    @abstractmethod
    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None) -> None:
        """Reset the deformable object.

        Args:
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. If None, then all the instances are updated.
                Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def write_data_to_sim(self):
        """Write data to the simulator."""
        raise NotImplementedError()

    @abstractmethod
    def update(self, dt: float):
        """Update the internal buffers.

        Args:
            dt: The amount of time passed from last :meth:`update` call [s].
        """
        raise NotImplementedError()

    """
    Operations - Write to simulation (index variants).
    """

    def write_nodal_state_to_sim_index(
        self,
        nodal_state: torch.Tensor | wp.array | ProxyArray,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set the nodal state over selected environment indices into the simulation.

        The nodal state comprises of the nodal positions and velocities. Since these are nodes,
        the velocity only has a translational component. All the quantities are in the simulation frame.

        Args:
            nodal_state: Nodal state in simulation frame [m, m/s].
                Shape is (len(env_ids), max_sim_vertices_per_body, 6)
                or (num_instances, max_sim_vertices_per_body, 6).
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        # Convert array wrappers to torch for slicing into position and velocity views.
        if isinstance(nodal_state, ProxyArray):
            nodal_state = nodal_state.torch
        elif isinstance(nodal_state, wp.array):
            nodal_state = wp.to_torch(nodal_state)
        # set into simulation
        self.write_nodal_pos_to_sim_index(nodal_state[..., :3], env_ids=env_ids, full_data=full_data)
        self.write_nodal_velocity_to_sim_index(nodal_state[..., 3:], env_ids=env_ids, full_data=full_data)

    @abstractmethod
    def write_nodal_pos_to_sim_index(
        self,
        nodal_pos: torch.Tensor | wp.array | ProxyArray,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set the nodal positions over selected environment indices into the simulation.

        Args:
            nodal_pos: Nodal positions in simulation frame [m].
                Shape is (len(env_ids), max_sim_vertices_per_body, 3)
                or (num_instances, max_sim_vertices_per_body, 3).
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_nodal_velocity_to_sim_index(
        self,
        nodal_vel: torch.Tensor | wp.array | ProxyArray,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set the nodal velocity over selected environment indices into the simulation.

        Args:
            nodal_vel: Nodal velocities in simulation frame [m/s].
                Shape is (len(env_ids), max_sim_vertices_per_body, 3)
                or (num_instances, max_sim_vertices_per_body, 3).
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_nodal_kinematic_target_to_sim_index(
        self,
        targets: torch.Tensor | wp.array | ProxyArray,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set the kinematic targets of the simulation mesh for the deformable bodies using indices.

        The kinematic targets comprise of individual nodal positions of the simulation mesh
        for the deformable body and a flag indicating whether the node is kinematically driven or not.
        The positions are in the simulation frame.

        .. note::
            The flag is set to 0.0 for kinematically driven nodes and 1.0 for free nodes.

        Args:
            targets: The kinematic targets comprising of nodal positions and flags [m].
                Shape is (len(env_ids), max_sim_vertices_per_body, 4)
                or (num_instances, max_sim_vertices_per_body, 4).
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        raise NotImplementedError()

    """
    Operations - Write to simulation (mask variants).
    """

    def write_nodal_state_to_sim_mask(
        self,
        nodal_state: torch.Tensor | wp.array | ProxyArray,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the nodal state over selected environment mask into the simulation.

        Args:
            nodal_state: Nodal state in simulation frame [m, m/s].
                Shape is (num_instances, max_sim_vertices_per_body, 6).
            env_mask: Environment mask. If None, then all indices are used.
        """
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        self.write_nodal_state_to_sim_index(nodal_state, env_ids=env_ids, full_data=True)

    def write_nodal_pos_to_sim_mask(
        self,
        nodal_pos: torch.Tensor | wp.array | ProxyArray,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the nodal positions over selected environment mask into the simulation.

        Args:
            nodal_pos: Nodal positions in simulation frame [m].
                Shape is (num_instances, max_sim_vertices_per_body, 3).
            env_mask: Environment mask. If None, then all indices are used.
        """
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        self.write_nodal_pos_to_sim_index(nodal_pos, env_ids=env_ids, full_data=True)

    def write_nodal_velocity_to_sim_mask(
        self,
        nodal_vel: torch.Tensor | wp.array | ProxyArray,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the nodal velocity over selected environment mask into the simulation.

        Args:
            nodal_vel: Nodal velocities in simulation frame [m/s].
                Shape is (num_instances, max_sim_vertices_per_body, 3).
            env_mask: Environment mask. If None, then all indices are used.
        """
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        self.write_nodal_velocity_to_sim_index(nodal_vel, env_ids=env_ids, full_data=True)

    def write_nodal_kinematic_target_to_sim_mask(
        self,
        targets: torch.Tensor | wp.array | ProxyArray,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the kinematic targets of the simulation mesh for the deformable bodies using mask.

        Args:
            targets: The kinematic targets comprising of nodal positions and flags [m].
                Shape is (num_instances, max_sim_vertices_per_body, 4).
            env_mask: Environment mask. If None, then all indices are used.
        """
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        self.write_nodal_kinematic_target_to_sim_index(targets, env_ids=env_ids, full_data=True)

    """
    Operations - Deprecated wrappers.
    """

    def write_nodal_state_to_sim(
        self,
        nodal_state: torch.Tensor | wp.array | ProxyArray,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated. Please use :meth:`write_nodal_state_to_sim_index` instead."""
        warnings.warn(
            "The method 'write_nodal_state_to_sim' is deprecated. Please use 'write_nodal_state_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_nodal_state_to_sim_index(nodal_state, env_ids=env_ids)

    def write_nodal_kinematic_target_to_sim(
        self,
        targets: torch.Tensor | wp.array | ProxyArray,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated. Please use :meth:`write_nodal_kinematic_target_to_sim_index` instead."""
        warnings.warn(
            "The method 'write_nodal_kinematic_target_to_sim' is deprecated."
            " Please use 'write_nodal_kinematic_target_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_nodal_kinematic_target_to_sim_index(targets, env_ids=env_ids)

    def write_nodal_pos_to_sim(
        self,
        nodal_pos: torch.Tensor | wp.array | ProxyArray,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated. Please use :meth:`write_nodal_pos_to_sim_index` instead."""
        warnings.warn(
            "The method 'write_nodal_pos_to_sim' is deprecated. Please use 'write_nodal_pos_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_nodal_pos_to_sim_index(nodal_pos, env_ids=env_ids)

    def write_nodal_velocity_to_sim(
        self,
        nodal_vel: torch.Tensor | wp.array | ProxyArray,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated. Please use :meth:`write_nodal_velocity_to_sim_index` instead."""
        warnings.warn(
            "The method 'write_nodal_velocity_to_sim' is deprecated."
            " Please use 'write_nodal_velocity_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_nodal_velocity_to_sim_index(nodal_vel, env_ids=env_ids)

    """
    Operations - Helper.
    """

    def transform_nodal_pos(
        self, nodal_pos: torch.Tensor, pos: torch.Tensor | None = None, quat: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Transform the nodal positions based on the pose transformation.

        This function computes the transformation of the nodal positions based on the pose transformation.
        It multiplies the nodal positions with the rotation matrix of the pose and adds the translation.
        Internally, it calls the :meth:`isaaclab.utils.math.transform_points` function.

        Args:
            nodal_pos: The nodal positions in the simulation frame [m].
                Shape is (N, max_sim_vertices_per_body, 3).
            pos: The position transformation [m]. Shape is (N, 3).
                Defaults to None, in which case the position is assumed to be zero.
            quat: The orientation transformation as quaternion (x, y, z, w). Shape is (N, 4).
                Defaults to None, in which case the orientation is assumed to be identity.

        Returns:
            The transformed nodal positions [m]. Shape is (N, max_sim_vertices_per_body, 3).
        """
        # offset the nodal positions to center them around the origin
        mean_nodal_pos = nodal_pos.mean(dim=1, keepdim=True)
        nodal_pos = nodal_pos - mean_nodal_pos
        # transform the nodal positions based on the pose around the origin
        return math_utils.transform_points(nodal_pos, pos, quat) + mean_nodal_pos
