# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.utils.wrench_composer import WrenchComposer

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
    rigid body handle. This handle can be accessed using the :attr:`root_view` attribute.

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

        .. note::
            Use this view with caution. It requires handling of tensors in a specific way.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def instantaneous_wrench_composer(self) -> WrenchComposer:
        """Instantaneous wrench composer for the rigid object."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def permanent_wrench_composer(self) -> WrenchComposer:
        """Permanent wrench composer for the rigid object."""
        raise NotImplementedError()

    """
    Operations.
    """

    @abstractmethod
    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None) -> None:
        """Reset the rigid object.

        Args:
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. If None, then all indices are used.
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
    ) -> tuple[list[int], list[str], wp.array]:
        """Find bodies in the rigid body based on the name keys.

        Please check the :meth:`isaaclab.utils.string_utils.resolve_matching_names` function for more
        information on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the body indices, names, and warp mask.
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
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (x, y, z, w), and linear
        and angular velocity. All the quantities are in the simulation frame.

        When providing the environment indices, we expect the data to be partial. However, when providing the
        environment mask, we expect the data to be full. This means that if env_ids is provided, then the shape of the
        data should be (len(env_ids), 13). However, if env_mask is provided, then the shape of the data should be
        (num_instances, 13).

        .. caution::
            If both `env_ids` and `env_mask` are provided, then `env_mask` takes precedence over `env_ids`.

        .. tip::
            For maximum performance we recommend providing the environment mask instead of the environment indices.

        Args:
            root_state: Root state in simulation frame. Shape is (len(env_ids), 13) or (num_instances, 13).
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. If None, then all indices are used.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_root_com_state_to_sim(
        self,
        root_state: torch.Tensor | wp.array,
        env_ids: Sequence[int] | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root center of mass state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (x, y, z, w), and linear
        and angular velocity. All the quantities are in the simulation frame.

        When providing the environment indices, we expect the data to be partial. However, when providing the
        environment mask, we expect the data to be full. This means that if env_ids is provided, then the shape of the
        data should be (len(env_ids), 13). However, if env_mask is provided, then the shape of the data should be
        (num_instances, 13).

        .. caution::
            If both `env_ids` and `env_mask` are provided, then `env_mask` takes precedence over `env_ids`.

        .. tip::
            For maximum performance we recommend providing the environment mask instead of the environment indices.

        Args:
            root_state: Root state in simulation frame. Shape is (len(env_ids), 13) or (num_instances, 13).
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. If None, then all indices are used.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_root_link_state_to_sim(
        self,
        root_state: torch.Tensor | wp.array,
        env_ids: Sequence[int] | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root link state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (x, y, z, w), and linear
        and angular velocity. All the quantities are in the simulation frame.

        When providing the environment indices, we expect the data to be partial. However, when providing the
        environment mask, we expect the data to be full. This means that if env_ids is provided, then the shape of the
        data should be (len(env_ids), 13). However, if env_mask is provided, then the shape of the data should be
        (num_instances, 13).

        .. caution::
            If both `env_ids` and `env_mask` are provided, then `env_mask` takes precedence over `env_ids`.

        .. tip::
            For maximum performance we recommend providing the environment mask instead of the environment indices.

        Args:
            root_state: Root state in simulation frame. Shape is (len(env_ids), 13) or (num_instances, 13).
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. If None, then all indices are used.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_root_pose_to_sim(
        self,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        When providing the environment indices, we expect the data to be partial. However, when providing the
        environment mask, we expect the data to be full. This means that if env_ids is provided, then the shape of the
        data should be (len(env_ids), 7). However, if env_mask is provided, then the shape of the data should be
        (num_instances, 7).

        .. caution::
            If both `env_ids` and `env_mask` are provided, then `env_mask` takes precedence over `env_ids`.

        .. tip::
            For maximum performance we recommend providing the environment mask instead of the environment indices.

        Args:
            root_pose: Root poses in simulation frame. Shape is (len(env_ids), 7) or (num_instances, 7).
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. If None, then all indices are used.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_root_link_pose_to_sim(
        self,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root link pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        When providing the environment indices, we expect the data to be partial. However, when providing the
        environment mask, we expect the data to be full. This means that if env_ids is provided, then the shape of the
        data should be (len(env_ids), 7). However, if env_mask is provided, then the shape of the data should be
        (num_instances, 7).

        .. caution::
            If both `env_ids` and `env_mask` are provided, then `env_mask` takes precedence over `env_ids`.

        .. tip::
            For maximum performance we recommend providing the environment mask instead of the environment indices.

        Args:
            root_pose: Root link poses in simulation frame. Shape is (len(env_ids), 7) or (num_instances, 7).
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. If None, then all indices are used.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_root_com_pose_to_sim(
        self,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root center of mass pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).
        The orientation is the orientation of the principle axes of inertia.

        When providing the environment indices, we expect the data to be partial. However, when providing the
        environment mask, we expect the data to be full. This means that if env_ids is provided, then the shape of the
        data should be (len(env_ids), 7). However, if env_mask is provided, then the shape of the data should be
        (num_instances, 7).

        .. caution::
            If both `env_ids` and `env_mask` are provided, then `env_mask` takes precedence over `env_ids`.

        .. tip::
            For maximum performance we recommend providing the environment mask instead of the environment indices.

        Args:
            root_pose: Root center of mass poses in simulation frame. Shape is (len(env_ids), 7) or (num_instances, 7).
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. If None, then all indices are used.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_root_velocity_to_sim(
        self,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root center of mass velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        ..note:: This sets the velocity of the root's center of mass rather than the roots frame.

        When providing the environment indices, we expect the data to be partial. However, when providing the
        environment mask, we expect the data to be full. This means that if env_ids is provided, then the shape of the
        data should be (len(env_ids), 6). However, if env_mask is provided, then the shape of the data should be
        (num_instances, 6).

        .. caution::
            If both `env_ids` and `env_mask` are provided, then `env_mask` takes precedence over `env_ids`.

        .. tip::
            For maximum performance we recommend providing the environment mask instead of the environment indices.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (len(env_ids), 6)
                or (num_instances, 6).
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. If None, then all indices are used.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_root_com_velocity_to_sim(
        self,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root center of mass velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        ..note:: This sets the velocity of the root's center of mass rather than the roots frame.

        When providing the environment indices, we expect the data to be partial. However, when providing the
        environment mask, we expect the data to be full. This means that if env_ids is provided, then the shape of the
        data should be (len(env_ids), 6). However, if env_mask is provided, then the shape of the data should be
        (num_instances, 6).

        .. caution::
            If both `env_ids` and `env_mask` are provided, then `env_mask` takes precedence over `env_ids`.

        .. tip::
            For maximum performance we recommend providing the environment mask instead of the environment indices.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (len(env_ids), 6)
                or (num_instances, 6).
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. If None, then all indices are used.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_root_link_velocity_to_sim(
        self,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root link velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        ..note:: This sets the velocity of the root's frame rather than the roots center of mass.

        When providing the environment indices, we expect the data to be partial. However, when providing the
        environment mask, we expect the data to be full. This means that if env_ids is provided, then the shape of the
        data should be (len(env_ids), 6). However, if env_mask is provided, then the shape of the data should be
        (num_instances, 6).

        .. caution::
            If both `env_ids` and `env_mask` are provided, then `env_mask` takes precedence over `env_ids`.

        .. tip::
            For maximum performance we recommend providing the environment mask instead of the environment indices.

        Args:
            root_velocity: Root frame velocities in simulation world frame. Shape is (len(env_ids), 6)
                or (num_instances, 6).
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. If None, then all indices are used.
        """
        raise NotImplementedError()

    """
    Operations - Setters.
    """

    @abstractmethod
    def set_masses(
        self,
        masses: torch.Tensor | wp.array,
        body_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
        env_mask: wp.array | None = None,
        body_mask: wp.array | None = None,
    ) -> None:
        """Set masses of all bodies.

        When providing the environment indices, we expect the data to be partial. However, when providing the
        environment mask, we expect the data to be full. This means that if env_ids is provided, then the shape of the
        data should be (len(env_ids), len(body_ids)). However, if env_mask is provided, then the shape of the data
        should be (num_instances, num_bodies).

        .. caution::
            If both `env_ids` and `env_mask` are provided, then `env_mask` takes precedence over `env_ids`. Similarly,
            if both `body_ids` and `body_mask` are provided, then `body_mask` takes precedence over `body_ids`.

        .. tip::
            For maximum performance we recommend providing the environment mask instead of the environment indices.

        Args:
            masses: Masses of all bodies. Shape is (len(env_ids), len(body_ids)) or (num_instances, num_bodies).
            body_ids: The body indices to set the masses for. Defaults to None (all bodies).
            env_ids: The environment indices to set the masses for. Defaults to None (all environments).
            env_mask: Environment mask. If None, then all indices are used.
            body_mask: Body mask. If None, then all bodies are used.
        """
        raise NotImplementedError()

    @abstractmethod
    def set_coms(
        self,
        coms: torch.Tensor | wp.array,
        body_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        env_mask: wp.array | None = None,
        body_mask: wp.array | None = None,
    ) -> None:
        """Set center of mass positions of all bodies.

        When providing the environment indices, we expect the data to be partial. However, when providing the
        environment mask, we expect the data to be full. This means that if env_ids is provided, then the shape of the
        data should be (len(env_ids), len(body_ids), 3). However, if env_mask is provided, then the shape of the data
        should be (num_instances, num_bodies, 3).

        .. caution::
            If both `env_ids` and `env_mask` are provided, then `env_mask` takes precedence over `env_ids`. Similarly,
            if both `body_ids` and `body_mask` are provided, then `body_mask` takes precedence over `body_ids`.

        .. tip::
            For maximum performance we recommend providing the environment mask instead of the environment indices.

        Args:
            coms: Center of mass positions of all bodies. Shape is (len(env_ids), len(body_ids), 3) or
                (num_instances, num_bodies, 3).
            body_ids: The body indices to set the center of mass positions for. Defaults to None (all bodies).
            env_ids: The environment indices to set the center of mass positions for. Defaults to None
                (all environments).
            env_mask: Environment mask. If None, then all indices are used.
            body_mask: Body mask. If None, then all bodies are used.
        """
        raise NotImplementedError()

    @abstractmethod
    def set_inertias(
        self,
        inertias: torch.Tensor | wp.array,
        body_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        env_mask: wp.array | None = None,
        body_mask: wp.array | None = None,
    ) -> None:
        """Set inertias of all bodies.

        When providing the environment indices, we expect the data to be partial. However, when providing the
        environment mask, we expect the data to be full. This means that if env_ids is provided, then the shape of the
        data should be (len(env_ids), len(body_ids), 9). However, if env_mask is provided, then the shape of the data
        should be (num_instances, num_bodies, 9).

        .. caution::
            If both `env_ids` and `env_mask` are provided, then `env_mask` takes precedence over `env_ids`. Similarly,
            if both `body_ids` and `body_mask` are provided, then `body_mask` takes precedence over `body_ids`.

        .. tip::
            For maximum performance we recommend providing the environment mask instead of the environment indices.

        Args:
            inertias: Inertias of all bodies. Shape is (len(env_ids), len(body_ids), 9) or
                (num_instances, num_bodies, 9).
            body_ids: The body indices to set the inertias for. Defaults to None (all bodies).
            env_ids: The environment indices to set the inertias for. Defaults to None (all environments).
            env_mask: Environment mask. If None, then all indices are used.
            body_mask: Body mask. If None, then all bodies are used.
        """
        raise NotImplementedError()

    @abstractmethod
    def set_external_force_and_torque(
        self,
        forces: torch.Tensor | wp.array,
        torques: torch.Tensor | wp.array,
        positions: torch.Tensor | wp.array | None = None,
        body_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
        is_global: bool = False,
        env_mask: wp.array | None = None,
        body_mask: wp.array | None = None,
    ) -> None:
        """Set external force and torque to apply on the asset's bodies in their local frame.

        For many applications, we want to keep the applied external force on rigid bodies constant over a period of
        time (for instance, during the policy control). This function allows us to store the external force and torque
        into buffers which are then applied to the simulation at every step. Optionally, set the position to apply the
        external wrench at (in the local link frame of the bodies).

        When providing the environment indices, we expect the data to be partial. However, when providing the
        environment mask, we expect the data to be full. This means that if env_ids is provided, then the shape of the
        data should be (len(env_ids), len(body_ids), 3). However, if env_mask is provided, then the shape of the data
        should be (num_instances, num_bodies, 3).

        .. caution::
            If both `env_ids` and `env_mask` are provided, then `env_mask` takes precedence over `env_ids`. Similarly,
            if both `body_ids` and `body_mask` are provided, then `body_mask` takes precedence over `body_ids`.

        .. tip::
            For maximum performance we recommend providing the environment mask instead of the environment indices.

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
            forces: External forces in bodies' local frame. Shape is (len(env_ids), len(body_ids), 3) or
                (num_instances, num_bodies, 3).
            torques: External torques in bodies' local frame. Shape is (len(env_ids), len(body_ids), 3) or
                (num_instances, num_bodies, 3).
            positions: External wrench positions in bodies' local frame. Shape is (len(env_ids), len(body_ids), 3) or
                (num_instances, num_bodies, 3). Defaults to None.
            body_ids: Body indices to apply external wrench to. Defaults to None (all bodies).
            env_ids: Environment indices to apply external wrench to. Defaults to None (all instances).
            is_global: Whether to apply the external wrench in the global frame. Defaults to False. If set to False,
                the external wrench is applied in the link frame of the bodies.
            env_mask: Environment mask. If None, then all indices are used.
            body_mask: Body mask. If None, then all bodies are used.
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
