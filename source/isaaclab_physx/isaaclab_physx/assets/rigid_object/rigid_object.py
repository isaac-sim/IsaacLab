# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

import omni.physics.tensors.impl.api as physx
from pxr import UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets.rigid_object.base_rigid_object import BaseRigidObject
from isaaclab.utils.wrench_composer import WrenchComposer

from isaaclab_physx.assets import kernels as shared_kernels
from isaaclab_physx.physics import PhysxManager as SimulationManager

from .rigid_object_data import RigidObjectData

if TYPE_CHECKING:
    from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg

# import logger
logger = logging.getLogger(__name__)


class RigidObject(BaseRigidObject):
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

    __backend_name__: str = "physx"
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
    def data(self) -> RigidObjectData:
        return self._data

    @property
    def num_instances(self) -> int:
        return self.root_view.count

    @property
    def num_bodies(self) -> int:
        """Number of bodies in the asset.

        This is always 1 since each object is a single rigid body.
        """
        return 1

    @property
    def body_names(self) -> list[str]:
        """Ordered names of bodies in the rigid object."""
        prim_paths = self.root_view.prim_paths[: self.num_bodies]
        return [path.split("/")[-1] for path in prim_paths]

    @property
    def root_view(self):
        """Root view for the asset.

        .. note::
            Use this view with caution. It requires handling of tensors in a specific way.
        """
        return self._root_view

    @property
    def instantaneous_wrench_composer(self) -> WrenchComposer:
        """Instantaneous wrench composer.

        Returns a :class:`~isaaclab.utils.wrench_composer.WrenchComposer` instance. Wrenches added or set to this wrench
        composer are only valid for the current simulation step. At the end of the simulation step, the wrenches set
        to this object are discarded. This is useful to apply forces that change all the time, things like drag forces
        for instance.
        """
        return self._instantaneous_wrench_composer

    @property
    def permanent_wrench_composer(self) -> WrenchComposer:
        """Permanent wrench composer.

        Returns a :class:`~isaaclab.utils.wrench_composer.WrenchComposer` instance. Wrenches added or set to this wrench
        composer are persistent and are applied to the simulation at every step. This is useful to apply forces that
        are constant over a period of time, things like the thrust of a motor for instance.
        """
        return self._permanent_wrench_composer

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None):
        """Reset the rigid object.

        Args:
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. If None, then all indices are used.
        """
        # resolve all indices
        if (env_ids is None) or (env_ids == slice(None)):
            env_ids = slice(None)
        # reset external wrench
        self._instantaneous_wrench_composer.reset(env_ids)
        self._permanent_wrench_composer.reset(env_ids)

    def write_data_to_sim(self) -> None:
        """Write external wrench to the simulation.

        .. note::
            We write external wrench to the simulation here since this function is called before the simulation step.
            This ensures that the external wrench is applied at every simulation step.
        """
        # write external wrench
        if self._instantaneous_wrench_composer.active or self._permanent_wrench_composer.active:
            if self._instantaneous_wrench_composer.active:
                # Compose instantaneous wrench with permanent wrench
                self._instantaneous_wrench_composer.add_forces_and_torques(
                    forces=self._permanent_wrench_composer.composed_force,
                    torques=self._permanent_wrench_composer.composed_torque,
                    body_ids=self._ALL_BODY_INDICES,
                    env_ids=self._ALL_INDICES,
                )
                # Apply both instantaneous and permanent wrench to the simulation
                self.root_view.apply_forces_and_torques_at_position(
                    force_data=self._instantaneous_wrench_composer.composed_force.flatten().view(wp.float32),
                    torque_data=self._instantaneous_wrench_composer.composed_torque.flatten().view(wp.float32),
                    position_data=None,
                    indices=self._ALL_INDICES,
                    is_global=False,
                )
            else:
                # Apply permanent wrench to the simulation
                self.root_view.apply_forces_and_torques_at_position(
                    force_data=self._permanent_wrench_composer.composed_force.flatten().view(wp.float32),
                    torque_data=self._permanent_wrench_composer.composed_torque.flatten().view(wp.float32),
                    position_data=None,
                    indices=self._ALL_INDICES,
                    is_global=False,
                )
        self._instantaneous_wrench_composer.reset()

    def update(self, dt: float) -> None:
        """Updates the simulation data.

        Args:
            dt: The time step size in seconds.
        """
        self.data.update(dt)

    """
    Operations - Finders.
    """

    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        """Find bodies in the rigid body based on the name keys.

        Please check the :meth:`isaaclab.utils.string_utils.resolve_matching_names` function for more
        information on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the body indices and names.
        """
        return string_utils.resolve_matching_names(name_keys, self.body_names, preserve_order)

    """
    Operations - Write to simulation.
    """

    def write_root_pose_to_sim_index(
        self,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (w, x, y, z).

        .. note::
            This method expects partial data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            root_pose: Root poses in simulation frame. Shape is (len(env_ids), 7).
            env_ids: Environment indices. If None, then all indices are used.
        """
        self.write_root_link_pose_to_sim_index(root_pose, env_ids=env_ids)

    def write_root_pose_to_sim_mask(
        self,
        root_pose: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root pose over selected environment mask into the simulation.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            root_pose: Root poses in simulation frame. Shape is (num_instances, 7).
            env_mask: Environment mask. If None, then all indices are used.
        """
        self.write_root_link_pose_to_sim_mask(root_pose, env_mask=env_mask)

    def write_root_velocity_to_sim_index(
        self,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root center of mass velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        NOTE: This sets the velocity of the root's center of mass rather than the roots frame.

        .. note::
            This method expects partial data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (len(env_ids), 6).
            env_ids: Environment indices. If None, then all indices are used.
        """
        self.write_root_com_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)

    def write_root_velocity_to_sim_mask(
        self,
        root_velocity: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root center of mass velocity over selected environment mask into the simulation.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (num_instances, 6).
            env_mask: Environment mask. If None, then all indices are used.
        """
        self.write_root_com_velocity_to_sim_mask(root_velocity=root_velocity, env_mask=env_mask)

    def write_root_link_pose_to_sim_index(
        self,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set the root link pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (w, x, y, z).

        .. note::
            This method expects partial data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            root_pose: Root link poses in simulation frame. Shape is (len(env_ids), 7) or (num_instances, 7).
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        # resolve all indices
        env_ids = self._resolve_env_ids(env_ids)
        wp.launch(
            shared_kernels.set_root_link_pose_to_sim,
            dim=env_ids.shape[0],
            inputs=[
                root_pose,
                env_ids,
                full_data,
            ],
            outputs=[
                self.data._root_link_pose_w.data,
                None,  # self.data._root_link_state_w.data,
                None,  # self.data._root_state_w.data,
            ],
            device=self.device,
        )
        # Update the timestamps
        self.data._root_link_pose_w.timestamp = self.data._sim_timestamp
        # Invalidate dependent timestamps
        self.data._root_com_pose_w.timestamp = -1.0
        self.data._root_link_state_w.timestamp = -1.0
        self.data._root_state_w.timestamp = -1.0
        self.data._root_com_state_w.timestamp = -1.0
        # set into simulation
        self.root_view.set_transforms(self.data._root_link_pose_w.data.view(wp.float32), indices=env_ids)

    def write_root_link_pose_to_sim_mask(
        self,
        root_pose: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root link pose over selected environment mask into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (w, x, y, z).

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            root_pose: Root poses in simulation frame. Shape is (num_instances, 7).
            env_mask: Environment mask. If None, then all indices are used.
        """
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        self.write_root_link_pose_to_sim_index(root_pose, env_ids=env_ids, full_data=True)

    def write_root_com_pose_to_sim_index(
        self,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set the root center of mass pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (w, x, y, z).
        The orientation is the orientation of the principle axes of inertia.

        .. note::
            This method expects partial data or full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            root_pose: Root center of mass poses in simulation frame. Shape is (len(env_ids), 7) or (num_instances, 7).
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        # resolve all indices
        env_ids = self._resolve_env_ids(env_ids)
        wp.launch(
            shared_kernels.set_root_com_pose_to_sim,
            dim=env_ids.shape[0],
            inputs=[
                root_pose,
                self.data.body_com_pose_b,
                env_ids,
                full_data,
            ],
            outputs=[
                self.data._root_com_pose_w.data,
                self.data._root_link_pose_w.data,
                None,  # self.data._root_com_state_w.data,
                None,  # self.data._root_link_state_w.data,
                None,  # self.data._root_state_w.data,
            ],
            device=self.device,
        )
        # Update the timestamps
        self.data._root_com_pose_w.timestamp = self.data._sim_timestamp
        self.data._root_link_pose_w.timestamp = self.data._sim_timestamp
        # Invalidate dependent timestamps
        self.data._root_com_state_w.timestamp = -1.0
        self.data._root_link_state_w.timestamp = -1.0
        self.data._root_state_w.timestamp = -1.0
        # set into simulation
        self.root_view.set_transforms(self.data._root_link_pose_w.data.view(wp.float32), indices=env_ids)

    def write_root_com_pose_to_sim_mask(
        self,
        root_pose: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root center of mass pose over selected environment mask into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (w, x, y, z).
        The orientation is the orientation of the principle axes of inertia.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            root_pose: Root center of mass poses in simulation frame. Shape is (num_instances, 7).
            env_mask: Environment mask. If None, then all indices are used.
        """
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        self.write_root_com_pose_to_sim_index(root_pose, env_ids=env_ids, full_data=True)

    def write_root_com_velocity_to_sim_index(
        self,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set the root center of mass velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        NOTE: This sets the velocity of the root's center of mass rather than the roots frame.

        .. note::
            This method expects partial data or full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame.
                Shape is (len(env_ids), 6) or (num_instances, 6).
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        # resolve all indices
        env_ids = self._resolve_env_ids(env_ids)
        wp.launch(
            shared_kernels.set_root_com_velocity_to_sim,
            dim=env_ids.shape[0],
            inputs=[
                root_velocity,
                env_ids,
                1,  # num_bodies is always 1 for RigidObject
                full_data,
            ],
            outputs=[
                self.data._root_com_vel_w.data,
                self.data._body_com_acc_w.data,
                None,  # self.data._root_state_w.data,
                None,  # self.data._root_com_state_w.data,
            ],
            device=self.device,
        )
        # Update the timestamps
        self.data._root_com_vel_w.timestamp = self.data._sim_timestamp
        self.data._body_com_acc_w.timestamp = self.data._sim_timestamp
        # Invalidate dependent timestamps
        self.data._root_link_vel_w.timestamp = -1.0
        self.data._root_link_state_w.timestamp = -1.0
        self.data._root_com_state_w.timestamp = -1.0
        self.data._root_state_w.timestamp = -1.0
        # set into simulation
        self.root_view.set_velocities(self.data._root_com_vel_w.data.view(wp.float32), indices=env_ids)

    def write_root_com_velocity_to_sim_mask(
        self,
        root_velocity: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root center of mass velocity over selected environment mask into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        NOTE: This sets the velocity of the root's center of mass rather than the roots frame.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (num_instances, 6).
            env_mask: Environment mask. If None, then all indices are used.
        """
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        self.write_root_com_velocity_to_sim_index(root_velocity, env_ids=env_ids, full_data=True)

    def write_root_link_velocity_to_sim_index(
        self,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set the root link velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        NOTE: This sets the velocity of the root's frame rather than the roots center of mass.

        .. note::
            This method expects partial data or full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            root_velocity: Root frame velocities in simulation world frame.
                Shape is (len(env_ids), 6) or (num_instances, 6).
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        # resolve all indices
        env_ids = self._resolve_env_ids(env_ids)
        # Access body_com_pose_b and root_link_pose_w properties to ensure they are current.
        wp.launch(
            shared_kernels.set_root_link_velocity_to_sim,
            dim=env_ids.shape[0],
            inputs=[
                root_velocity,
                self.data.body_com_pose_b,
                self.data.root_link_pose_w,
                env_ids,
                1,  # num_bodies is always 1 for RigidObject
                full_data,
            ],
            outputs=[
                self.data._root_link_vel_w.data,
                self.data._root_com_vel_w.data,
                self.data._body_com_acc_w.data,
                None,  # self.data._root_link_state_w.data,
                None,  # self.data._root_state_w.data,
                None,  # self.data._root_com_state_w.data,
            ],
            device=self.device,
        )
        # Update the timestamps
        self.data._root_link_vel_w.timestamp = self.data._sim_timestamp
        self.data._root_com_vel_w.timestamp = self.data._sim_timestamp
        self.data._body_com_acc_w.timestamp = self.data._sim_timestamp
        # Invalidate dependent timestamps
        self.data._root_link_state_w.timestamp = -1.0
        self.data._root_state_w.timestamp = -1.0
        self.data._root_com_state_w.timestamp = -1.0
        # set into simulation
        self.root_view.set_velocities(self.data._root_com_vel_w.data.view(wp.float32), indices=env_ids)

    def write_root_link_velocity_to_sim_mask(
        self,
        root_velocity: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root link velocity over selected environment mask into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        NOTE: This sets the velocity of the root's frame rather than the roots center of mass.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend looking at the actual implementation of the method in the backend.
            Some backends may provide optimized implementations for masks / indices.

        Args:
            root_velocity: Root frame velocities in simulation world frame. Shape is (num_instances, 6).
            env_mask: Environment mask. If None, then all indices are used.
        """
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        self.write_root_link_velocity_to_sim_index(root_velocity, env_ids=env_ids, full_data=True)

    """
    Operations - Setters.
    """

    def set_masses_index(
        self,
        masses: torch.Tensor | wp.array,
        body_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set masses of all bodies using indices.

        .. note::
            This method expects partial data or full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            masses: Masses of all bodies. Shape is (len(env_ids), len(body_ids)) or (num_instances, num_bodies)
                if full_data.
            body_ids: The body indices to set the masses for. Defaults to None (all bodies).
            env_ids: The environment indices to set the masses for. Defaults to None (all environments).
            full_data: Whether to expect full data. Defaults to False.
        """
        # resolve all indices
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)
        # Warp kernels can ingest torch tensors directly, so we don't need to convert to warp arrays here.
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_indices,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[
                masses,
                env_ids,
                body_ids,
                full_data,
            ],
            outputs=[
                self.data._body_mass,
            ],
            device=self.device,
        )

        # Set into simulation, note that when updating "model" properties with PhysX we need to do it on CPU.
        if isinstance(env_ids, wp.array):
            cpu_env_ids = wp.clone(env_ids, device="cpu")
        else:
            cpu_env_ids = wp.clone(wp.from_torch(env_ids, dtype=wp.int32), device="cpu")
        self.root_view.set_masses(wp.clone(self.data._body_mass, device="cpu"), indices=cpu_env_ids)

    def set_masses_mask(
        self,
        masses: torch.Tensor | wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set masses of all bodies using masks.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            masses: Masses of all bodies. Shape is (num_instances, num_bodies).
            body_mask: Body mask. If None, then all bodies are used.
            env_mask: Environment mask. If None, then all indices are used.
        """
        # Resolve masks.
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        if body_mask is not None:
            body_ids = wp.nonzero(body_mask)
        else:
            body_ids = self._ALL_BODY_INDICES
        # Set full data to True to ensure the right code path is taken inside the kernel.
        self.set_masses_index(masses, body_ids=body_ids, env_ids=env_ids, full_data=True)

    def set_coms_index(
        self,
        coms: torch.Tensor | wp.array,
        body_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set center of mass pose of all bodies using indices.

        .. note::
            This method expects partial data or full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            coms: Center of mass pose of all bodies. Shape is (len(env_ids), len(body_ids), 7) or
                (num_instances, num_bodies, 7) if full_data.
            body_ids: The body indices to set the center of mass pose for. Defaults to None (all bodies).
            env_ids: The environment indices to set the center of mass pose for. Defaults to None (all environments).
            full_data: Whether to expect full data. Defaults to False.
        """
        # resolve all indices
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)
        # Warp kernels can ingest torch tensors directly, so we don't need to convert to warp arrays here.
        wp.launch(
            shared_kernels.write_body_com_pose_to_buffer,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[
                coms,
                env_ids,
                body_ids,
                full_data,
            ],
            outputs=[
                self.data._body_com_pose_b,
            ],
            device=self.device,
        )
        # Set into simulation, note that when updating "model" properties with PhysX we need to do it on CPU.
        if isinstance(env_ids, wp.array):
            cpu_env_ids = wp.clone(env_ids, device="cpu")
        else:
            cpu_env_ids = wp.clone(wp.from_torch(env_ids, dtype=wp.int32), device="cpu")
        self.root_view.set_coms(wp.clone(self.data._body_com_pose_b, device="cpu"), indices=cpu_env_ids)

    def set_coms_mask(
        self,
        coms: torch.Tensor | wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set center of mass pose of all bodies using masks.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            coms: Center of mass pose of all bodies. Shape is (num_instances, num_bodies, 7).
            body_mask: Body mask. If None, then all bodies are used.
            env_mask: Environment mask. If None, then all indices are used.
        """
        # Resolve masks.
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        if body_mask is not None:
            body_ids = wp.nonzero(body_mask)
        else:
            body_ids = self._ALL_BODY_INDICES
        # Set full data to True to ensure the right code path is taken inside the kernel.
        self.set_coms_index(coms, body_ids=body_ids, env_ids=env_ids, full_data=True)

    def set_inertias_index(
        self,
        inertias: torch.Tensor | wp.array,
        body_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set inertias of all bodies using indices.

        .. note::
            This method expects partial data or full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            inertias: Inertias of all bodies. Shape is (len(env_ids), len(body_ids), 9) or
                (num_instances, num_bodies, 9) if full_data.
            body_ids: The body indices to set the inertias for. Defaults to None (all bodies).
            env_ids: The environment indices to set the inertias for. Defaults to None (all environments).
            full_data: Whether to expect full data. Defaults to False.
        """
        # resolve all indices
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)
        # Warp kernels can ingest torch tensors directly, so we don't need to convert to warp arrays here.
        wp.launch(
            shared_kernels.write_single_body_inertia_to_buffer,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[
                inertias,
                env_ids,
                full_data,
            ],
            outputs=[
                self.data._body_inertia,
            ],
            device=self.device,
        )
        # Set into simulation, note that when updating "model" properties with PhysX we need to do it on CPU.
        if isinstance(env_ids, wp.array):
            cpu_env_ids = wp.clone(env_ids, device="cpu")
        else:
            cpu_env_ids = wp.clone(wp.from_torch(env_ids, dtype=wp.int32), device="cpu")
        self.root_view.set_inertias(wp.clone(self.data._body_inertia, device="cpu"), indices=cpu_env_ids)

    def set_inertias_mask(
        self,
        inertias: torch.Tensor | wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set inertias of all bodies using masks.

        .. note::
            This method expects full data.

        .. tip::
            For maximum performance we recommend using the index method. This is because in PhysX, the tensor API
            is only supporting indexing, hence masks need to be converted to indices.

        Args:
            inertias: Inertias of all bodies. Shape is (num_instances, num_bodies, 9).
            body_mask: Body mask. If None, then all bodies are used.
            env_mask: Environment mask. If None, then all indices are used.
        """
        # Resolve masks.
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        if body_mask is not None:
            body_ids = wp.nonzero(body_mask)
        else:
            body_ids = self._ALL_BODY_INDICES
        # Set full data to True to ensure the right code path is taken inside the kernel.
        self.set_inertias_index(inertias, body_ids=body_ids, env_ids=env_ids, full_data=True)

    """
    Internal helper.
    """

    def _initialize_impl(self):
        # obtain global simulation view
        self._physics_sim_view = SimulationManager.get_physics_sim_view()
        # obtain the first prim in the regex expression (all others are assumed to be a copy of this)
        template_prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
        if template_prim is None:
            raise RuntimeError(f"Failed to find prim for expression: '{self.cfg.prim_path}'.")
        template_prim_path = template_prim.GetPath().pathString

        # find rigid root prims
        root_prims = sim_utils.get_all_matching_child_prims(
            template_prim_path,
            predicate=lambda prim: prim.HasAPI(UsdPhysics.RigidBodyAPI),
            traverse_instance_prims=False,
        )
        if len(root_prims) == 0:
            raise RuntimeError(
                f"Failed to find a rigid body when resolving '{self.cfg.prim_path}'."
                " Please ensure that the prim has 'USD RigidBodyAPI' applied."
            )
        if len(root_prims) > 1:
            raise RuntimeError(
                f"Failed to find a single rigid body when resolving '{self.cfg.prim_path}'."
                f" Found multiple '{root_prims}' under '{template_prim_path}'."
                " Please ensure that there is only one rigid body in the prim path tree."
            )

        articulation_prims = sim_utils.get_all_matching_child_prims(
            template_prim_path,
            predicate=lambda prim: prim.HasAPI(UsdPhysics.ArticulationRootAPI),
            traverse_instance_prims=False,
        )
        if len(articulation_prims) != 0:
            if articulation_prims[0].GetAttribute("physxArticulation:articulationEnabled").Get():
                raise RuntimeError(
                    f"Found an articulation root when resolving '{self.cfg.prim_path}' for rigid objects. These are"
                    f" located at: '{articulation_prims}' under '{template_prim_path}'. Please disable the articulation"
                    " root in the USD or from code by setting the parameter"
                    " 'ArticulationRootPropertiesCfg.articulation_enabled' to False in the spawn configuration."
                )

        # resolve root prim back into regex expression
        root_prim_path = root_prims[0].GetPath().pathString
        root_prim_path_expr = self.cfg.prim_path + root_prim_path[len(template_prim_path) :]
        # -- object view
        self._root_view = self._physics_sim_view.create_rigid_body_view(root_prim_path_expr.replace(".*", "*"))

        # check if the rigid body was created
        if self.root_view._backend is None:
            raise RuntimeError(f"Failed to create rigid body at: {self.cfg.prim_path}. Please check PhysX logs.")

        # log information about the rigid body
        logger.info(f"Rigid body initialized at: {self.cfg.prim_path} with root '{root_prim_path_expr}'.")
        logger.info(f"Number of instances: {self.num_instances}")
        logger.info(f"Number of bodies: {self.num_bodies}")
        logger.info(f"Body names: {self.body_names}")

        # container for data access
        self._data = RigidObjectData(self.root_view, self.device)

        # create buffers
        self._create_buffers()
        # process configuration
        self._process_cfg()
        # update the rigid body data
        self.update(0.0)
        # Let the rigid object data know that it is fully instantiated and ready to use.
        self.data.is_primed = True

    def _create_buffers(self):
        """Create buffers for storing data."""
        # constants
        self._ALL_INDICES = wp.array(np.arange(self.num_instances, dtype=np.int32), device=self.device)
        self._ALL_BODY_INDICES = wp.array(np.arange(self.num_bodies, dtype=np.int32), device=self.device)

        # external wrench composer
        self._instantaneous_wrench_composer = WrenchComposer(self)
        self._permanent_wrench_composer = WrenchComposer(self)

        # set information about rigid body into data
        self._data.body_names = self.body_names

    def _process_cfg(self) -> None:
        """Post processing of configuration parameters."""
        # default state
        # -- root state
        # note: we cast to tuple to avoid torch/numpy type mismatch.
        default_root_pose = tuple(self.cfg.init_state.pos) + tuple(self.cfg.init_state.rot)
        default_root_vel = tuple(self.cfg.init_state.lin_vel) + tuple(self.cfg.init_state.ang_vel)
        default_root_pose = np.tile(np.array(default_root_pose, dtype=np.float32), (self.num_instances, 1))
        default_root_vel = np.tile(np.array(default_root_vel, dtype=np.float32), (self.num_instances, 1))
        self._data.default_root_pose = wp.array(default_root_pose, dtype=wp.transformf, device=self.device)
        self._data.default_root_vel = wp.array(default_root_vel, dtype=wp.spatial_vectorf, device=self.device)

    def _resolve_env_ids(self, env_ids: Sequence[int] | torch.Tensor | wp.array | None) -> wp.array | torch.Tensor:
        """Resolve environment indices to a warp array or tensor.

        .. note::
            We need to convert torch tensors to warp arrays since the TensorAPI views only support warp arrays.

        Args:
            env_ids: Environment indices. If None, then all indices are used.

        Returns:
            A warp array of environment indices or a tensor of environment indices.
        """
        if (env_ids is None) or (env_ids == slice(None)):
            return self._ALL_INDICES
        elif isinstance(env_ids, list):
            return wp.array(env_ids, dtype=wp.int32, device=self.device)
        if isinstance(env_ids, torch.Tensor):
            return wp.from_torch(env_ids.to(torch.int32), dtype=wp.int32)
        return env_ids

    def _resolve_body_ids(self, body_ids: Sequence[int] | torch.Tensor | wp.array | None) -> wp.array | torch.Tensor:
        """Resolve body indices to a warp array or tensor.

        .. note::
            We do not need to convert torch tensors to warp arrays since they never get passed to the TensorAPI views.

        Args:
            body_ids: Body indices. If None, then all indices are used.

        Returns:
            A warp array of body indices or a tensor of body indices.
        """
        if (body_ids is None) or (body_ids == slice(None)):
            return self._ALL_BODY_INDICES
        elif isinstance(body_ids, list):
            return wp.array(body_ids, dtype=wp.int32, device=self.device)
        return body_ids

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._root_view = None

    @property
    def root_physx_view(self) -> physx.RigidBodyView:
        """Deprecated property. Please use :attr:`root_view` instead."""
        warnings.warn(
            "The `root_physx_view` property will be deprecated in a future release. Please use `root_view` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.root_view

    def write_root_state_to_sim(
        self, root_state: torch.Tensor | wp.array, env_ids: Sequence[int] | torch.Tensor | wp.array | None = None
    ):
        """Deprecated, same as :meth:`write_root_link_pose_to_sim_index` and
        :meth:`write_root_com_velocity_to_sim_index`."""
        warnings.warn(
            "The function 'write_root_state_to_sim' will be deprecated in a future release. Please"
            " use 'write_root_link_pose_to_sim_index' and 'write_root_com_velocity_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_root_link_pose_to_sim_index(root_state[:, :7], env_ids=env_ids)
        self.write_root_com_velocity_to_sim_index(root_state[:, 7:], env_ids=env_ids)

    def write_root_com_state_to_sim(
        self, root_state: torch.Tensor | wp.array, env_ids: Sequence[int] | torch.Tensor | wp.array | None = None
    ):
        """Deprecated, same as :meth:`write_root_com_pose_to_sim_index` and
        :meth:`write_root_com_velocity_to_sim_index`."""
        warnings.warn(
            "The function 'write_root_com_state_to_sim' will be deprecated in a future release. Please"
            " use 'write_root_com_pose_to_sim_index' and 'write_root_com_velocity_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_root_com_pose_to_sim_index(root_state[:, :7], env_ids=env_ids)
        self.write_root_com_velocity_to_sim_index(root_state[:, 7:], env_ids=env_ids)

    def write_root_link_state_to_sim(
        self, root_state: torch.Tensor | wp.array, env_ids: Sequence[int] | torch.Tensor | wp.array | None = None
    ):
        """Deprecated, same as :meth:`write_root_link_pose_to_sim_index` and
        :meth:`write_root_link_velocity_to_sim_index`."""
        warnings.warn(
            "The function 'write_root_link_state_to_sim' will be deprecated in a future release. Please"
            " use 'write_root_link_pose_to_sim_index' and 'write_root_link_velocity_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_root_link_pose_to_sim_index(root_state[:, :7], env_ids=env_ids)
        self.write_root_link_velocity_to_sim_index(root_state[:, 7:], env_ids=env_ids)
