# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import warp as wp
from collections.abc import Sequence
from typing import TYPE_CHECKING
from abc import abstractmethod

from ..asset_base import AssetBase

if TYPE_CHECKING:
    from .rigid_object_cfg import RigidObjectCfg
    from .rigid_object_data import RigidObjectData


class BaseRigidObject(AssetBase):
    """A rigid object asset class.

    Rigid objects are assets comprising of rigid bodies. They can be used to represent dynamic objects
    such as boxes, spheres, etc. A rigid body is described by its pose, velocity and mass distribution.

    For an asset to be considered a rigid object, the root prim of the asset must have the `USD RigidBodyAPI`_
    applied to it. This API is used to define the simulation properties of the rigid body. On playing the
    simulation, the physics engine will automatically register the rigid body and create a corresponding
    rigid body handle. This handle can be accessed using the :attr:`root_physx_view` attribute.

    .. note::

        For users familiar with Isaac Sim, the PhysX view class API is not the exactly same as Isaac Sim view
        class API. Similar to Isaac Lab, Isaac Sim wraps around the PhysX view API. However, as of now (2023.1 release),
        we see a large difference in initializing the view classes in Isaac Sim. This is because the view classes
        in Isaac Sim perform additional USD-related operations which are slow and also not required.

    .. _`USD RigidBodyAPI`: https://openusd.org/dev/api/class_usd_physics_rigid_body_a_p_i.html
    """

    cfg: RigidObjectCfg
    """Configuration instance for the rigid object."""

    __backend_name__: str = "base"
    """The name of the backend for the rigid object."""

    def __init__(self, cfg: RigidObjectCfg):
        """Initialize the rigid object.

        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg)

    """
    Properties
    """

    @property
    @abstractmethod
    def data(self) -> RigidObjectData:
        raise NotImplementedError()

    @property
    @abstractmethod
    def num_instances(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def num_bodies(self) -> int:
        """Number of bodies in the asset.

        This is always 1 since each object is a single rigid body.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_names(self) -> list[str]:
        """Ordered names of bodies in the rigid object."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def root_view(self):
        """Root view for the asset.
        
        Note:
            Use this view with caution. It requires handling of tensors in a specific way.
        """
        raise NotImplementedError()

    """
    Operations.
    """

    @abstractmethod
    def reset(self, env_ids: Sequence[int] | None = None, mask: wp.array | torch.Tensor | None = None):
        raise NotImplementedError()

    @abstractmethod
    def write_data_to_sim(self) -> None:
        """Write external wrench to the simulation.

        Note:
            We write external wrench to the simulation here since this function is called before the simulation step.
            This ensures that the external wrench is applied at every simulation step.
        """
        raise NotImplementedError()

    @abstractmethod
    def update(self, dt: float) -> None:
        raise NotImplementedError()

    """
    Operations - Finders.
    """

    @abstractmethod
    def find_bodies(
        self,
        name_keys: str | Sequence[str],
        preserve_order: bool = False
    ) -> tuple[wp.array | torch.Tensor, list[str], list[int]]:
        """Find bodies in the rigid body based on the name keys.

        Please check the :meth:`isaaclab.utils.string_utils.resolve_matching_names` function for more
        information on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the body mask, names and indices.
        """
        raise NotImplementedError()

    """
    Operations - Write to simulation.
    """

    @abstractmethod
    def write_root_state_to_sim(
        self,
        root_state: torch.Tensor | wp.array,
        env_ids: Sequence[int] | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (x, y, z, w), and linear
        and angular velocity. All the quantities are in the simulation frame.

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids is provided, then root_state should be of shape (len(env_ids), 13). If
        env_mask is provided, then root_state should be of shape (num_instances, 13).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus 
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        Args:
            root_state: Root state in simulation frame. Shape is (len(env_ids), 13) or (num_instances, 13).
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def write_root_com_state_to_sim(
        self,
        root_state: torch.Tensor | wp.array,
        env_ids: Sequence[int] | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root center of mass state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (x, y, z, w), and linear
        and angular velocity. All the quantities are in the simulation frame.

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids is provided, then root_state should be of shape (len(env_ids), 13). If
        env_mask is provided, then root_state should be of shape (num_instances, 13).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus 
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        Args:
            root_state: Root state in simulation frame. Shape is (len(env_ids), 13) or (num_instances, 13).
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def write_root_link_state_to_sim(
        self,
        root_state: torch.Tensor | wp.array,
        env_ids: Sequence[int] | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root link state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (x, y, z, w), and linear
        and angular velocity. All the quantities are in the simulation frame.

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids is provided, then root_state should be of shape (len(env_ids), 13). If
        env_mask is provided, then root_state should be of shape (num_instances, 13).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus 
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        Args:
            root_state: Root state in simulation frame. Shape is (len(env_ids), 13) or (num_instances, 13).
            env_mask: Environment mask. Shape is (num_instances,).
            env_ids: Environment indices. If None, then all indices are used.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_root_pose_to_sim(
        self,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids is provided, then root_pose should be of shape (len(env_ids), 7). If
        env_mask is provided, then root_pose should be of shape (num_instances, 7).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus 
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        Args:
            root_pose: Root link poses in simulation frame. Shape is (len(env_ids), 7) or (num_instances, 7).
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def write_root_link_pose_to_sim(
        self,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root link pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids is provided, then root_pose should be of shape (len(env_ids), 7). If
        env_mask is provided, then root_pose should be of shape (num_instances, 7).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus 
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        Args:
            root_pose: Root link poses in simulation frame. Shape is (len(env_ids), 7) or (num_instances, 7).
            env_mask: Environment mask. Shape is (num_instances,).
            env_ids: Environment indices. If None, then all indices are used.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_root_com_pose_to_sim(
        self,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root center of mass pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).
        The orientation is the orientation of the principle axes of inertia.

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids is provided, then root_pose should be of shape (len(env_ids), 7). If
        env_mask is provided, then root_pose should be of shape (num_instances, 7).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus 
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        Args:
            root_pose: Root center of mass poses in simulation frame. Shape is (len(env_ids), 7) or (num_instances, 7).
            env_mask: Environment mask. Shape is (num_instances,).
            env_ids: Environment indices. If None, then all indices are used.
        """
        # resolve all indices
        raise NotImplementedError()

    @abstractmethod
    def write_root_velocity_to_sim(
        self,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root center of mass velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        ..note:: This sets the velocity of the root's center of mass rather than the roots frame.

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids is provided, then root_velocity should be of shape (len(env_ids), 6). If
        env_mask is provided, then root_velocity should be of shape (num_instances, 6).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus 
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (len(env_ids), 6) or (num_instances, 6).
            env_mask: Environment mask. Shape is (num_instances,).
            env_ids: Environment indices. If None, then all indices are used.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_root_com_velocity_to_sim(
        self, root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root center of mass velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        ..note:: This sets the velocity of the root's center of mass rather than the roots frame.

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids is provided, then root_velocity should be of shape (len(env_ids), 6). If
        env_mask is provided, then root_velocity should be of shape (num_instances, 6).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus 
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (len(env_ids), 6) or (num_instances, 6).
            env_mask: Environment mask. Shape is (num_instances,).
            env_ids: Environment indices. If None, then all indices are used.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_root_link_velocity_to_sim(
        self,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root link velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        ..note:: This sets the velocity of the root's frame rather than the roots center of mass.

        ..note:: When ids are provided, then partial data is expected. When masks are provided the whole of the data
        is expected. For example, if env_ids is provided, then root_velocity should be of shape (len(env_ids), 6). If
        env_mask is provided, then root_velocity should be of shape (num_instances, 6).

        ..caution:: If both env_mask and env_ids are provided, then env_mask will be used. The function will thus 
        expect the whole of the data to be provided. If none of them are provided, then the function expects the whole
        of the data to be provided.

        Args:
            root_velocity: Root frame velocities in simulation world frame. Shape is (len(env_ids), 6) or (num_instances, 6).
            env_mask: Environment mask. Shape is (num_instances,).
            env_ids: Environment indices. If None, then all indices are used.
        """
        raise NotImplementedError()

    """
    Operations - Setters.
    """

    @abstractmethod
    def set_external_force_and_torque(
        self,
        forces: torch.Tensor | wp.array,
        torques: torch.Tensor | wp.array,
        positions: torch.Tensor | wp.array | None = None,
        body_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
        body_mask: torch.Tensor | wp.array | None = None,
        env_mask: torch.Tensor | wp.array | None = None,
        is_global: bool = False,
    ) -> None:
        """Set external force and torque to apply on the asset's bodies in their local frame.

        For many applications, we want to keep the applied external force on rigid bodies constant over a period of
        time (for instance, during the policy control). This function allows us to store the external force and torque
        into buffers which are then applied to the simulation at every step. Optionally, set the position to apply the
        external wrench at (in the local link frame of the bodies).

        .. caution::
            If the function is called with empty forces and torques, then this function disables the application
            of external wrench to the simulation.

            .. code-block:: python

                # example of disabling external wrench
                asset.set_external_force_and_torque(forces=torch.zeros(0, 3), torques=torch.zeros(0, 3))

        .. caution::
            If the function is called consecutively with and with different values for ``is_global``, then the
            all the external wrenches will be applied in the frame specified by the last call.

            .. code-block:: python

                # example of setting external wrench in the global frame
                asset.set_external_force_and_torque(forces=torch.ones(1, 1, 3), env_ids=[0], is_global=True)
                # example of setting external wrench in the link frame
                asset.set_external_force_and_torque(forces=torch.ones(1, 1, 3), env_ids=[1], is_global=False)
                # Both environments will have the external wrenches applied in the link frame

        .. note::
            This function does not apply the external wrench to the simulation. It only fills the buffers with
            the desired values. To apply the external wrench, call the :meth:`write_data_to_sim` function
            right before the simulation step.

        Args:
            forces: External forces in bodies' local frame. Shape is (len(env_ids), len(body_ids), 3).
            torques: External torques in bodies' local frame. Shape is (len(env_ids), len(body_ids), 3).
            positions: External wrench positions in bodies' local frame. Shape is (len(env_ids), len(body_ids), 3). Defaults to None.
            body_ids: Body indices to apply external wrench to. Defaults to None (all bodies).
            env_ids: Environment indices to apply external wrench to. Defaults to None (all instances).
            is_global: Whether to apply the external wrench in the global frame. Defaults to False. If set to False,
                the external wrench is applied in the link frame of the bodies.
        """
        raise NotImplementedError()

    """
    Internal helper.
    """

    @abstractmethod
    def _initialize_impl(self):
        raise NotImplementedError()

    @abstractmethod
    def _create_buffers(self):
        raise NotImplementedError()

    @abstractmethod
    def _process_cfg(self) -> None:
        """Post processing of configuration parameters."""
        raise NotImplementedError()

    """
    Internal simulation callbacks.
    """

    @abstractmethod
    def _invalidate_initialize_callback(self, event) -> None:
        """Invalidates the scene elements."""
        raise NotImplementedError()
