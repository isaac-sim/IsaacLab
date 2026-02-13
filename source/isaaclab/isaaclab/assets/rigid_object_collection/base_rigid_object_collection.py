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

from isaaclab.utils.wrench_composer import WrenchComposer

from ..asset_base import AssetBase

if TYPE_CHECKING:
    from .rigid_object_collection_cfg import RigidObjectCollectionCfg
    from .rigid_object_collection_data import RigidObjectCollectionData


class BaseRigidObjectCollection(AssetBase):
    """A rigid object collection class.

    This class represents a collection of rigid objects in the simulation, where the state of the
    rigid objects can be accessed and modified using a batched ``(env_ids, object_ids)`` API.

    For each rigid body in the collection, the root prim of the asset must have the `USD RigidBodyAPI`_
    applied to it. This API is used to define the simulation properties of the rigid bodies. On playing the
    simulation, the physics engine will automatically register the rigid bodies and create a corresponding
    rigid body handle. This handle can be accessed using the :attr:`root_view` attribute.

    Rigid objects in the collection are uniquely identified via the key of the dictionary
    :attr:`~isaaclab.assets.RigidObjectCollectionCfg.rigid_objects` in the
    :class:`~isaaclab.assets.RigidObjectCollectionCfg` configuration class.
    This differs from the :class:`~isaaclab.assets.RigidObject` class, where a rigid object is identified by
    the name of the Xform where the `USD RigidBodyAPI`_ is applied. This would not be possible for the rigid
    object collection since the :attr:`~isaaclab.assets.RigidObjectCollectionCfg.rigid_objects` dictionary
    could contain the same rigid object multiple times, leading to ambiguity.

    .. _`USD RigidBodyAPI`: https://openusd.org/dev/api/class_usd_physics_rigid_body_a_p_i.html
    """

    cfg: RigidObjectCollectionCfg
    """Configuration instance for the rigid object."""

    __backend_name__: str = "base"
    """The name of the backend for the rigid object."""

    def __init__(self, cfg: RigidObjectCollectionCfg):
        """Initialize the rigid object.

        Args:
            cfg: A configuration instance.
        """
        pass

    """
    Properties
    """

    @property
    @abstractmethod
    def data(self) -> RigidObjectCollectionData:
        raise NotImplementedError()

    @property
    @abstractmethod
    def num_instances(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def num_bodies(self) -> int:
        """Number of bodies in the rigid object collection."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_names(self) -> list[str]:
        """Ordered names of bodies in the rigid object collection."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def root_view(self):
        """Root view for the rigid object collection.

        .. note::
            Use this view with caution. It requires handling of tensors in a specific way.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def instantaneous_wrench_composer(self) -> WrenchComposer:
        """Instantaneous wrench composer for the rigid object collection."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def permanent_wrench_composer(self) -> WrenchComposer:
        """Permanent wrench composer for the rigid object collection."""
        raise NotImplementedError()

    """
    Operations.
    """

    @abstractmethod
    def reset(self, env_ids: Sequence[int] | None = None, object_ids: slice | torch.Tensor | None = None) -> None:
        """Resets all internal buffers of selected environments and objects.

        Args:
            env_ids: Environment indices. If None, then all indices are used.
            object_ids: Object indices. If None, then all indices are used.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_data_to_sim(self) -> None:
        """Write external wrench to the simulation.

        .. note::
            We write external wrench to the simulation here since this function is called before the simulation step.
            This ensures that the external wrench is applied at every simulation step.
        """
        raise NotImplementedError()

    @abstractmethod
    def update(self, dt: float) -> None:
        """Updates the simulation data.

        Args:
            dt: The time step size in seconds.
        """
        raise NotImplementedError()

    """
    Operations - Finders.
    """

    @abstractmethod
    def find_bodies(
        self, name_keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[torch.Tensor, list[str], list[int]]:
        """Find bodies in the rigid body collection based on the name keys.

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
    def write_body_state_to_sim(
        self,
        body_states: torch.Tensor,
        env_ids: Sequence[int] | None = None,
        body_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Set the bodies state over selected environment indices into the simulation.

        The body state comprises of the cartesian position, quaternion orientation in (x, y, z, w), and linear
        and angular velocity. All the quantities are in the simulation frame. Shape is
        ``(len(env_ids), len(body_ids), 13)``.

        Args:
            body_states: Body states in simulation frame. Shape is (len(env_ids), len(body_ids), 13).
            env_ids: Environment indices. If None, then all indices are used.
            body_ids: Body indices. If None, then all indices are used.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_body_com_state_to_sim(
        self,
        body_states: torch.Tensor,
        env_ids: Sequence[int] | None = None,
        body_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Set the body center of mass state over selected environment and body indices into the simulation.

        The body state comprises of the cartesian position, quaternion orientation in (x, y, z, w), and linear
        and angular velocity. All the quantities are in the simulation frame.

        Args:
            body_states: Body states in simulation frame. Shape is (len(env_ids), len(body_ids), 13).
            env_ids: Environment indices. If None, then all indices are used.
            body_ids: Body indices. If None, then all indices are used.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_body_link_state_to_sim(
        self,
        body_states: torch.Tensor,
        env_ids: Sequence[int] | None = None,
        body_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Set the body link state over selected environment and body indices into the simulation.

        The body state comprises of the cartesian position, quaternion orientation in (x, y, z, w), and linear
        and angular velocity. All the quantities are in the simulation frame.

        Args:
            body_states: Body states in simulation frame. Shape is (len(env_ids), len(body_ids), 13).
            env_ids: Environment indices. If None, then all indices are used.
            body_ids: Body indices. If None, then all indices are used.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_body_pose_to_sim(
        self,
        body_poses: torch.Tensor,
        env_ids: Sequence[int] | None = None,
        body_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Set the body poses over selected environment and body indices into the simulation.

        The body pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        Args:
            body_poses: Body poses in simulation frame. Shape is (len(env_ids), len(body_ids), 7).
            env_ids: Environment indices. If None, then all indices are used.
            body_ids: Body indices. If None, then all indices are used.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_body_link_pose_to_sim(
        self,
        body_poses: torch.Tensor,
        env_ids: Sequence[int] | None = None,
        body_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Set the body link pose over selected environment and body indices into the simulation.

        The body link pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        Args:
            body_poses: Body link poses in simulation frame. Shape is (len(env_ids), len(body_ids), 7).
            env_ids: Environment indices. If None, then all indices are used.
            body_ids: Body indices. If None, then all indices are used.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_body_com_pose_to_sim(
        self,
        body_poses: torch.Tensor,
        env_ids: Sequence[int] | None = None,
        body_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Set the body center of mass pose over selected environment and body indices into the simulation.

        The body center of mass pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).
        The orientation is the orientation of the principle axes of inertia.

        Args:
            body_poses: Body center of mass poses in simulation frame. Shape is (len(env_ids), len(body_ids), 7).
            env_ids: Environment indices. If None, then all indices are used.
            body_ids: Body indices. If None, then all indices are used.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_body_velocity_to_sim(
        self,
        body_velocities: torch.Tensor,
        env_ids: Sequence[int] | None = None,
        body_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Set the body velocity over selected environment and body indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        ..note:: This sets the velocity of the body's center of mass rather than the body's frame.

        Args:
            body_velocities: Body velocities in simulation frame. Shape is (len(env_ids), len(body_ids), 6).
            env_ids: Environment indices. If None, then all indices are used.
            body_ids: Body indices. If None, then all indices are used.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_body_com_velocity_to_sim(
        self,
        body_velocities: torch.Tensor,
        env_ids: Sequence[int] | None = None,
        body_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Set the body center of mass velocity over selected environment and body indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        ..note:: This sets the velocity of the body's center of mass rather than the body's frame.

        Args:
            body_velocities: Body center of mass velocities in simulation frame. Shape is
                (len(env_ids), len(body_ids), 6).
            env_ids: Environment indices. If None, then all indices are used.
            body_ids: Body indices. If None, then all indices are used.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_body_link_velocity_to_sim(
        self,
        body_velocities: torch.Tensor,
        env_ids: Sequence[int] | None = None,
        body_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Set the body link velocity over selected environment and body indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        ..note:: This sets the velocity of the body's frame rather than the body's center of mass.

        Args:
            body_velocities: Body link velocities in simulation frame. Shape is (len(env_ids), len(body_ids), 6).
            env_ids: Environment indices. If None, then all indices are used.
            body_ids: Body indices. If None, then all indices are used.
        """
        raise NotImplementedError()

    """
    Operations - Setters.
    """

    @abstractmethod
    def set_masses(
        self,
        masses: torch.Tensor,
        body_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Set masses of all bodies.

        Args:
            masses: Masses of all bodies. Shape is (num_instances, num_bodies).
            body_ids: The body indices to set the masses for. Defaults to None (all bodies).
            env_ids: The environment indices to set the masses for. Defaults to None (all environments).
        """
        raise NotImplementedError()

    @abstractmethod
    def set_coms(
        self,
        coms: torch.Tensor,
        body_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Set center of mass positions of all bodies.

        Args:
            coms: Center of mass positions of all bodies. Shape is (num_instances, num_bodies, 3).
            body_ids: The body indices to set the center of mass positions for. Defaults to None (all bodies).
            env_ids: The environment indices to set the center of mass positions for. Defaults to None
                (all environments).
        """
        raise NotImplementedError()

    @abstractmethod
    def set_inertias(
        self,
        inertias: torch.Tensor,
        body_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Set inertias of all bodies.

        Args:
            inertias: Inertias of all bodies. Shape is (num_instances, num_bodies, 3, 3).
            body_ids: The body indices to set the inertias for. Defaults to None (all bodies).
            env_ids: The environment indices to set the inertias for. Defaults to None (all environments).
        """
        raise NotImplementedError()

    @abstractmethod
    def set_external_force_and_torque(
        self,
        forces: torch.Tensor,
        torques: torch.Tensor,
        positions: torch.Tensor | None = None,
        body_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
        is_global: bool = False,
    ) -> None:
        """Set external force and torque to apply on the rigid object collection's bodies in their local frame.

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
            positions: External wrench positions in bodies' local frame. Shape is (len(env_ids), len(body_ids), 3).
                Defaults to None.
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
    def _initialize_impl(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _create_buffers(self) -> None:
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
        super()._invalidate_initialize_callback(event)

    """
    Deprecated properties and methods.
    """

    @property
    def num_objects(self) -> int:
        """Deprecated property. Please use :attr:`num_bodies` instead."""
        warnings.warn(
            "The `num_objects` property will be deprecated in a future release. Please use `num_bodies` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.num_bodies

    @property
    def object_names(self) -> list[str]:
        """Deprecated property. Please use :attr:`body_names` instead."""
        warnings.warn(
            "The `object_names` property will be deprecated in a future release. Please use `body_names` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_names

    def write_object_state_to_sim(
        self,
        object_state: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        object_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Deprecated method. Please use :meth:`write_body_state_to_sim` instead."""
        warnings.warn(
            "The `write_object_state_to_sim` method will be deprecated in a future release. Please use"
            " `write_body_state_to_sim` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_body_state_to_sim(object_state, env_ids=env_ids, body_ids=object_ids)

    def write_object_com_state_to_sim(
        self,
        object_state: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        object_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Deprecated method. Please use :meth:`write_body_com_state_to_sim` instead."""
        warnings.warn(
            "The `write_object_com_state_to_sim` method will be deprecated in a future release. Please use"
            " `write_body_com_state_to_sim` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_body_com_state_to_sim(object_state, env_ids=env_ids, body_ids=object_ids)

    def write_object_link_state_to_sim(
        self,
        object_state: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        object_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Deprecated method. Please use :meth:`write_body_link_state_to_sim` instead."""
        warnings.warn(
            "The `write_object_link_state_to_sim` method will be deprecated in a future release. Please use"
            " `write_body_link_state_to_sim` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_body_link_state_to_sim(object_state, env_ids=env_ids, body_ids=object_ids)

    def write_object_pose_to_sim(
        self,
        object_pose: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        object_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Deprecated method. Please use :meth:`write_body_pose_to_sim` instead."""
        warnings.warn(
            "The `write_object_pose_to_sim` method will be deprecated in a future release. Please use"
            " `write_body_pose_to_sim` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_body_pose_to_sim(object_pose, env_ids=env_ids, body_ids=object_ids)

    def write_object_link_pose_to_sim(
        self,
        object_pose: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        object_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Deprecated method. Please use :meth:`write_body_link_pose_to_sim` instead."""
        warnings.warn(
            "The `write_object_link_pose_to_sim` method will be deprecated in a future release. Please use"
            " `write_body_link_pose_to_sim` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_body_link_pose_to_sim(object_pose, env_ids=env_ids, body_ids=object_ids)

    def write_object_com_pose_to_sim(
        self,
        object_pose: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        object_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Deprecated method. Please use :meth:`write_body_com_pose_to_sim` instead."""
        warnings.warn(
            "The `write_object_com_pose_to_sim` method will be deprecated in a future release. Please use"
            " `write_body_com_pose_to_sim` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_body_com_pose_to_sim(object_pose, env_ids=env_ids, body_ids=object_ids)

    def write_object_velocity_to_sim(
        self,
        object_velocity: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        object_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Deprecated method. Please use :meth:`write_body_com_velocity_to_sim` instead."""
        warnings.warn(
            "The `write_object_velocity_to_sim` method will be deprecated in a future release. Please use"
            " `write_body_com_velocity_to_sim` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_body_com_velocity_to_sim(object_velocity, env_ids=env_ids, body_ids=object_ids)

    def write_object_com_velocity_to_sim(
        self,
        object_velocity: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        object_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Deprecated method. Please use :meth:`write_body_com_velocity_to_sim` instead."""
        warnings.warn(
            "The `write_object_com_velocity_to_sim` method will be deprecated in a future release. Please use"
            " `write_body_com_velocity_to_sim` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_body_com_velocity_to_sim(object_velocity, env_ids=env_ids, body_ids=object_ids)

    def write_object_link_velocity_to_sim(
        self,
        object_velocity: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        object_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Deprecated method. Please use :meth:`write_body_link_velocity_to_sim` instead."""
        warnings.warn(
            "The `write_object_link_velocity_to_sim` method will be deprecated in a future release. Please use"
            " `write_body_link_velocity_to_sim` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_body_link_velocity_to_sim(object_velocity, env_ids=env_ids, body_ids=object_ids)

    def find_objects(
        self, name_keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[torch.Tensor, list[str], list[int]]:
        """Deprecated method. Please use :meth:`find_bodies` instead."""
        warnings.warn(
            "The `find_objects` method will be deprecated in a future release. Please use `find_bodies` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.find_bodies(name_keys, preserve_order)
