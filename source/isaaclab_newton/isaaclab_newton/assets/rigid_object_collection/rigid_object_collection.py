# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp
from newton.selection import ArticulationView
from newton.solvers import SolverNotifyFlags

from pxr import UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets.rigid_object_collection.base_rigid_object_collection import BaseRigidObjectCollection
from isaaclab.utils.wrench_composer import WrenchComposer

from isaaclab_newton.assets import kernels as shared_kernels
from isaaclab_newton.physics import NewtonManager as SimulationManager

from .rigid_object_collection_data import RigidObjectCollectionData

if TYPE_CHECKING:
    from isaaclab.assets.rigid_object_collection.rigid_object_collection_cfg import RigidObjectCollectionCfg


class RigidObjectCollection(BaseRigidObjectCollection):
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

    __backend_name__: str = "newton"
    """The name of the backend for the rigid object."""

    def __init__(self, cfg: RigidObjectCollectionCfg):
        """Initialize the rigid object collection.

        Args:
            cfg: A configuration instance.
        """
        # Note: We never call the parent constructor as it tries to call its own spawning which we don't want.
        # check that the config is valid
        cfg.validate()
        # store inputs
        self.cfg = cfg.copy()
        # flag for whether the asset is initialized
        self._is_initialized = False
        # spawn the rigid objects
        for rigid_body_cfg in self.cfg.rigid_objects.values():
            # spawn the asset
            if rigid_body_cfg.spawn is not None:
                rigid_body_cfg.spawn.func(
                    rigid_body_cfg.prim_path,
                    rigid_body_cfg.spawn,
                    translation=rigid_body_cfg.init_state.pos,
                    orientation=rigid_body_cfg.init_state.rot,
                )
            # check that spawn was successful
            matching_prims = sim_utils.find_matching_prims(rigid_body_cfg.prim_path)
            if len(matching_prims) == 0:
                raise RuntimeError(f"Could not find prim with path {rigid_body_cfg.prim_path}.")
        # stores object names
        self._body_names_list = []

        # register various callback functions
        self._register_callbacks()
        self._debug_vis_handle = None

    """
    Properties
    """

    @property
    def data(self) -> RigidObjectCollectionData:
        return self._data

    @property
    def num_instances(self) -> int:
        return self._root_view.count // self.num_bodies

    @property
    def num_bodies(self) -> int:
        """Number of bodies in the rigid object collection."""
        return len(self.body_names)

    @property
    def body_names(self) -> list[str]:
        """Ordered names of bodies in the rigid object collection."""
        return self._body_names_list

    @property
    def root_view(self) -> ArticulationView:
        """Root view for the rigid object collection.

        A single :class:`ArticulationView` matching all body types. The 2nd dimension
        (matches per world) corresponds to the different body types.

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

    def reset(
        self,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        object_ids: slice | torch.Tensor | None = None,
        env_mask: wp.array | None = None,
        object_mask: wp.array | None = None,
    ) -> None:
        """Resets all internal buffers of selected environments and objects.

        Args:
            env_ids: Environment indices. If None, then all indices are used.
            object_ids: Object indices. If None, then all indices are used.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
            object_mask: Object mask. Not used currently.
        """
        # resolve all indices
        if env_ids is None:
            env_ids = self._ALL_ENV_INDICES
        if object_ids is None:
            object_ids = self._ALL_BODY_INDICES
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
                self._instantaneous_wrench_composer.add_forces_and_torques_index(
                    forces=self._permanent_wrench_composer.composed_force,
                    torques=self._permanent_wrench_composer.composed_torque,
                    body_ids=self._ALL_BODY_INDICES,
                    env_ids=self._ALL_ENV_INDICES,
                )
                # Apply both instantaneous and permanent wrench to a consolidated 2D buffer
                wp.launch(
                    shared_kernels.update_wrench_array_with_force_and_torque,
                    dim=(self.num_instances, self.num_bodies),
                    device=self.device,
                    inputs=[
                        self._instantaneous_wrench_composer.composed_force,
                        self._instantaneous_wrench_composer.composed_torque,
                        self._wrench_buffer,
                        self._ALL_ENV_MASK,
                        self._ALL_BODY_MASK,
                    ],
                )
            else:
                # Apply permanent wrench to a consolidated 2D buffer
                wp.launch(
                    shared_kernels.update_wrench_array_with_force_and_torque,
                    dim=(self.num_instances, self.num_bodies),
                    device=self.device,
                    inputs=[
                        self._permanent_wrench_composer.composed_force,
                        self._permanent_wrench_composer.composed_torque,
                        self._wrench_buffer,
                        self._ALL_ENV_MASK,
                        self._ALL_BODY_MASK,
                    ],
                )
            # Write the wrench buffer directly to the Newton binding (already 2D)
            wp.copy(self._data._sim_bind_body_external_wrench, self._wrench_buffer)
        self._instantaneous_wrench_composer.reset()

    def update(self, dt: float) -> None:
        """Updates the simulation data.

        Args:
            dt: The time step size [s].
        """
        self.data.update(dt)

    """
    Operations - Finders.
    """

    def find_bodies(
        self, name_keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[torch.Tensor, list[str]]:
        """Find bodies in the rigid body collection based on the name keys.

        Please check the :meth:`isaaclab.utils.string_utils.resolve_matching_names` function for more
        information on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the body indices and names.
        """
        obj_ids, obj_names = string_utils.resolve_matching_names(name_keys, self.body_names, preserve_order)
        return torch.tensor(obj_ids, device=self.device, dtype=torch.int32), obj_names

    """
    Operations - Write to simulation.
    """

    def write_body_pose_to_sim_index(
        self,
        *,
        body_poses: torch.Tensor | wp.array,
        body_ids: Sequence[int] | torch.Tensor | wp.array | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the body pose over selected environment and body indices into the simulation.

        The body pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        .. note::
            This method expects partial data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations. Performance is similar for both.
            However, to allow graphed pipelines, the mask method must be used.

        Args:
            body_poses: Body poses in simulation frame. Shape is (len(env_ids), len(body_ids), 7)
                or (len(env_ids), len(body_ids)) with dtype wp.transformf.
            body_ids: Body indices. If None, then all indices are used.
            env_ids: Environment indices. If None, then all indices are used.
        """
        self.write_body_link_pose_to_sim_index(body_poses=body_poses, env_ids=env_ids, body_ids=body_ids)

    def write_body_pose_to_sim_mask(
        self,
        *,
        body_poses: torch.Tensor | wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the body pose over selected environment mask into the simulation.

        The body pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations. Performance is similar for both.
            However, to allow graphed pipelines, the mask method must be used.

        Args:
            body_poses: Body poses in simulation frame. Shape is (num_instances, num_bodies, 7)
                or (num_instances, num_bodies) with dtype wp.transformf.
            body_mask: Body mask. If None, then all bodies are updated. Shape is (num_bodies,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        if env_mask is not None:
            env_ids = self._resolve_env_mask(env_mask)
        else:
            env_ids = self._ALL_ENV_INDICES
        if body_mask is not None:
            body_ids = self._resolve_body_mask(body_mask)
        else:
            body_ids = self._ALL_BODY_INDICES
        self.write_body_link_pose_to_sim_index(
            body_poses=body_poses, env_ids=env_ids, body_ids=body_ids, full_data=True
        )

    def write_body_velocity_to_sim_index(
        self,
        *,
        body_velocities: torch.Tensor | wp.array,
        body_ids: Sequence[int] | torch.Tensor | wp.array | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the body velocity over selected environment and body indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.

        .. note::
            This sets the velocity of the body's center of mass rather than the body's frame.

        .. note::
            This method expects partial data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations. Performance is similar for both.
            However, to allow graphed pipelines, the mask method must be used.

        Args:
            body_velocities: Body velocities in simulation frame.
                Shape is (len(env_ids), len(body_ids), 6) or (num_instances, num_bodies, 6),
                or (len(env_ids), len(body_ids)) / (num_instances, num_bodies) with dtype wp.spatial_vectorf.
            body_ids: Body indices. If None, then all indices are used.
            env_ids: Environment indices. If None, then all indices are used.
        """
        self.write_body_com_velocity_to_sim_index(body_velocities=body_velocities, env_ids=env_ids, body_ids=body_ids)

    def write_body_velocity_to_sim_mask(
        self,
        *,
        body_velocities: torch.Tensor | wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the body velocity over selected environment mask into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.

        .. note::
            This sets the velocity of the body's center of mass rather than the body's frame.

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations. Performance is similar for both.
            However, to allow graphed pipelines, the mask method must be used.

        Args:
            body_velocities: Body velocities in simulation frame.
                Shape is (num_instances, num_bodies, 6)
                or (num_instances, num_bodies) with dtype wp.spatial_vectorf.
            body_mask: Body mask. If None, then all bodies are updated. Shape is (num_bodies,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        if env_mask is not None:
            env_ids = self._resolve_env_mask(env_mask)
        else:
            env_ids = self._ALL_ENV_INDICES
        if body_mask is not None:
            body_ids = self._resolve_body_mask(body_mask)
        else:
            body_ids = self._ALL_BODY_INDICES
        self.write_body_com_velocity_to_sim_index(
            body_velocities=body_velocities, env_ids=env_ids, body_ids=body_ids, full_data=True
        )

    def write_body_link_pose_to_sim_index(
        self,
        *,
        body_poses: torch.Tensor | wp.array,
        body_ids: Sequence[int] | torch.Tensor | wp.array | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set the body link pose over selected environment and body indices into the simulation.

        The body link pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        .. note::
            This method expects partial data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations. Performance is similar for both.
            However, to allow graphed pipelines, the mask method must be used.

        Args:
            body_poses: Body link poses in simulation frame.
                Shape is (len(env_ids), len(body_ids), 7) or (num_instances, num_bodies, 7),
                or (len(env_ids), len(body_ids)) / (num_instances, num_bodies) with dtype wp.transformf.
            body_ids: Body indices. If None, then all indices are used.
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)
        if full_data:
            self.assert_shape_and_dtype(body_poses, (self.num_instances, self.num_bodies), wp.transformf, "body_poses")
        else:
            self.assert_shape_and_dtype(body_poses, (env_ids.shape[0], body_ids.shape[0]), wp.transformf, "body_poses")
        # Write to consolidated buffer
        wp.launch(
            shared_kernels.set_body_link_pose_to_sim,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[
                body_poses,
                env_ids,
                body_ids,
                full_data,
            ],
            outputs=[
                self.data.body_link_pose_w,
                None,  # body_link_state_w
                None,  # body_state_w
            ],
            device=self.device,
        )
        # Invalidate dependent timestamps
        self.data._body_com_pose_w.timestamp = -1.0
        self.data._body_com_state_w.timestamp = -1.0
        self.data._body_link_state_w.timestamp = -1.0
        self.data._body_state_w.timestamp = -1.0

    def write_body_link_pose_to_sim_mask(
        self,
        *,
        body_poses: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
        body_ids: Sequence[int] | torch.Tensor | wp.array | slice | None = None,
    ) -> None:
        """Set the body link pose over selected environment mask into the simulation.

        The body link pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations. Performance is similar for both.
            However, to allow graphed pipelines, the mask method must be used.

        Args:
            body_poses: Body link poses in simulation frame. Shape is (num_instances, num_bodies, 7)
                or (num_instances, num_bodies) with dtype wp.transformf.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
            body_ids: Body indices. If None, then all indices are used.
        """
        if env_mask is not None:
            env_ids = self._resolve_env_mask(env_mask)
        else:
            env_ids = self._ALL_ENV_INDICES
        self.write_body_link_pose_to_sim_index(
            body_poses=body_poses, env_ids=env_ids, body_ids=body_ids, full_data=True
        )

    def write_body_com_pose_to_sim_index(
        self,
        *,
        body_poses: torch.Tensor | wp.array,
        body_ids: Sequence[int] | torch.Tensor | wp.array | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set the body center of mass pose over selected environment and body indices into the simulation.

        The body center of mass pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).
        The orientation is the orientation of the principal axes of inertia.

        .. note::
            This method expects partial data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations. Performance is similar for both.
            However, to allow graphed pipelines, the mask method must be used.

        Args:
            body_poses: Body center of mass poses in simulation frame.
                Shape is (len(env_ids), len(body_ids), 7) or (num_instances, num_bodies, 7),
                or (len(env_ids), len(body_ids)) / (num_instances, num_bodies) with dtype wp.transformf.
            body_ids: Body indices. If None, then all indices are used.
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)
        if full_data:
            self.assert_shape_and_dtype(body_poses, (self.num_instances, self.num_bodies), wp.transformf, "body_poses")
        else:
            self.assert_shape_and_dtype(body_poses, (env_ids.shape[0], body_ids.shape[0]), wp.transformf, "body_poses")
        # Write to consolidated buffers (updates both com_pose_w and link_pose_w)
        wp.launch(
            shared_kernels.set_body_com_pose_to_sim,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[
                body_poses,
                self.data.body_com_pos_b,
                env_ids,
                body_ids,
                full_data,
            ],
            outputs=[
                self.data.body_com_pose_w,
                self.data.body_link_pose_w,
                None,  # body_com_state_w
                None,  # body_link_state_w
                None,  # body_state_w
            ],
            device=self.device,
        )
        # Invalidate dependent timestamps
        self.data._body_link_state_w.timestamp = -1.0
        self.data._body_state_w.timestamp = -1.0
        self.data._body_com_state_w.timestamp = -1.0

    def write_body_com_pose_to_sim_mask(
        self,
        *,
        body_poses: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
        body_ids: Sequence[int] | torch.Tensor | wp.array | slice | None = None,
    ) -> None:
        """Set the body center of mass pose over selected environment mask into the simulation.

        The body center of mass pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).
        The orientation is the orientation of the principal axes of inertia.

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations. Performance is similar for both.
            However, to allow graphed pipelines, the mask method must be used.

        Args:
            body_poses: Body center of mass poses in simulation frame. Shape is (num_instances, num_bodies, 7)
                or (num_instances, num_bodies) with dtype wp.transformf.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
            body_ids: Body indices. If None, then all indices are used.
        """
        if env_mask is not None:
            env_ids = self._resolve_env_mask(env_mask)
        else:
            env_ids = self._ALL_ENV_INDICES
        self.write_body_com_pose_to_sim_index(body_poses=body_poses, env_ids=env_ids, body_ids=body_ids, full_data=True)

    def write_body_com_velocity_to_sim_index(
        self,
        *,
        body_velocities: torch.Tensor | wp.array,
        body_ids: Sequence[int] | torch.Tensor | wp.array | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set the body center of mass velocity over selected environment and body indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.

        .. note::
            This sets the velocity of the body's center of mass rather than the body's frame.

        .. note::
            This method expects partial data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations. Performance is similar for both.
            However, to allow graphed pipelines, the mask method must be used.

        Args:
            body_velocities: Body center of mass velocities in simulation frame.
                Shape is (len(env_ids), len(body_ids), 6) or (num_instances, num_bodies, 6),
                or (len(env_ids), len(body_ids)) / (num_instances, num_bodies) with dtype wp.spatial_vectorf.
            body_ids: Body indices. If None, then all indices are used.
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)
        if full_data:
            self.assert_shape_and_dtype(
                body_velocities, (self.num_instances, self.num_bodies), wp.spatial_vectorf, "body_velocities"
            )
        else:
            self.assert_shape_and_dtype(
                body_velocities, (env_ids.shape[0], body_ids.shape[0]), wp.spatial_vectorf, "body_velocities"
            )
        # Write to consolidated buffer
        wp.launch(
            shared_kernels.set_body_com_velocity_to_sim,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[
                body_velocities,
                env_ids,
                body_ids,
                full_data,
            ],
            outputs=[
                self.data.body_com_vel_w,
                self.data.body_com_acc_w,
                None,  # body_state_w
                None,  # body_com_state_w
            ],
            device=self.device,
        )
        # Invalidate dependent timestamps
        self.data._body_link_vel_w.timestamp = -1.0
        self.data._body_state_w.timestamp = -1.0
        self.data._body_com_state_w.timestamp = -1.0
        self.data._body_link_state_w.timestamp = -1.0

    def write_body_com_velocity_to_sim_mask(
        self,
        *,
        body_velocities: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
        body_ids: Sequence[int] | torch.Tensor | wp.array | slice | None = None,
    ) -> None:
        """Set the body center of mass velocity over selected environment mask into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.

        .. note::
            This sets the velocity of the body's center of mass rather than the body's frame.

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations. Performance is similar for both.
            However, to allow graphed pipelines, the mask method must be used.

        Args:
            body_velocities: Body center of mass velocities in simulation frame.
                Shape is (num_instances, num_bodies, 6)
                or (num_instances, num_bodies) with dtype wp.spatial_vectorf.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
            body_ids: Body indices. If None, then all indices are used.
        """
        if env_mask is not None:
            env_ids = self._resolve_env_mask(env_mask)
        else:
            env_ids = self._ALL_ENV_INDICES
        self.write_body_com_velocity_to_sim_index(
            body_velocities=body_velocities, env_ids=env_ids, body_ids=body_ids, full_data=True
        )

    def write_body_link_velocity_to_sim_index(
        self,
        *,
        body_velocities: torch.Tensor | wp.array,
        body_ids: Sequence[int] | torch.Tensor | wp.array | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set the body link velocity over selected environment and body indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.

        .. note::
            This sets the velocity of the body's frame rather than the body's center of mass.

        .. note::
            This method expects partial data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations. Performance is similar for both.
            However, to allow graphed pipelines, the mask method must be used.

        Args:
            body_velocities: Body link velocities in simulation frame.
                Shape is (len(env_ids), len(body_ids), 6) or (num_instances, num_bodies, 6),
                or (len(env_ids), len(body_ids)) / (num_instances, num_bodies) with dtype wp.spatial_vectorf.
            body_ids: Body indices. If None, then all indices are used.
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)
        if full_data:
            self.assert_shape_and_dtype(
                body_velocities, (self.num_instances, self.num_bodies), wp.spatial_vectorf, "body_velocities"
            )
        else:
            self.assert_shape_and_dtype(
                body_velocities, (env_ids.shape[0], body_ids.shape[0]), wp.spatial_vectorf, "body_velocities"
            )
        # Access body_com_pos_b and body_link_pose_w to ensure they are current.
        wp.launch(
            shared_kernels.set_body_link_velocity_to_sim,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[
                body_velocities,
                self.data.body_com_pos_b,
                self.data.body_link_pose_w,
                env_ids,
                body_ids,
                full_data,
            ],
            outputs=[
                self.data.body_link_vel_w,
                self.data.body_com_vel_w,
                self.data.body_com_acc_w,
                None,  # body_link_state_w
                None,  # body_state_w
                None,  # body_com_state_w
            ],
            device=self.device,
        )
        # Invalidate dependent timestamps
        self.data._body_link_state_w.timestamp = -1.0
        self.data._body_state_w.timestamp = -1.0
        self.data._body_com_state_w.timestamp = -1.0

    def write_body_link_velocity_to_sim_mask(
        self,
        *,
        body_velocities: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
        body_ids: Sequence[int] | torch.Tensor | wp.array | slice | None = None,
    ) -> None:
        """Set the body link velocity over selected environment mask into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.

        .. note::
            This sets the velocity of the body's frame rather than the body's center of mass.

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations. Performance is similar for both.
            However, to allow graphed pipelines, the mask method must be used.

        Args:
            body_velocities: Body link velocities in simulation frame. Shape is (num_instances, num_bodies, 6)
                or (num_instances, num_bodies) with dtype wp.spatial_vectorf.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
            body_ids: Body indices. If None, then all indices are used.
        """
        if env_mask is not None:
            env_ids = self._resolve_env_mask(env_mask)
        else:
            env_ids = self._ALL_ENV_INDICES
        self.write_body_link_velocity_to_sim_index(
            body_velocities=body_velocities, env_ids=env_ids, body_ids=body_ids, full_data=True
        )

    """
    Operations - Setters.
    """

    def set_masses_index(
        self,
        *,
        masses: torch.Tensor | wp.array,
        body_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set masses of all bodies using indices.

        .. note::
            This method expects partial data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations. Performance is similar for both.
            However, to allow graphed pipelines, the mask method must be used.

        Args:
            masses: Masses of all bodies. Shape is (len(env_ids), len(body_ids)).
            body_ids: The body indices to set the masses for. Defaults to None (all bodies).
            env_ids: The environment indices to set the masses for. Defaults to None (all environments).
        """
        # resolve all indices
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)
        self.assert_shape_and_dtype(masses, (env_ids.shape[0], body_ids.shape[0]), wp.float32, "masses")
        # Write to consolidated buffer
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_indices,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[
                masses,
                env_ids,
                body_ids,
            ],
            outputs=[
                self.data._sim_bind_body_mass,
            ],
            device=self.device,
        )
        # No copy-back needed — writes go directly to Newton's state via the 2D binding
        # Tell the physics engine that some of the body properties have been updated
        SimulationManager.add_model_change(SolverNotifyFlags.BODY_INERTIAL_PROPERTIES)

    def set_masses_mask(
        self,
        *,
        masses: torch.Tensor | wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set masses of all bodies using masks.

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations. Performance is similar for both.
            However, to allow graphed pipelines, the mask method must be used.

        Args:
            masses: Masses of all bodies. Shape is (num_instances, num_bodies).
            body_mask: Body mask. If None, then all bodies are used.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        # resolve masks
        if env_mask is None:
            env_mask = self._ALL_ENV_MASK
        if body_mask is None:
            body_mask = self._ALL_BODY_MASK
        self.assert_shape_and_dtype_mask(masses, (env_mask, body_mask), wp.float32, "masses")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_mask,
            dim=(env_mask.shape[0], body_mask.shape[0]),
            inputs=[
                masses,
                env_mask,
                body_mask,
            ],
            outputs=[
                self.data._sim_bind_body_mass,
            ],
            device=self.device,
        )
        # No copy-back needed — writes go directly to Newton's state via the 2D binding
        # Tell the physics engine that some of the body properties have been updated
        SimulationManager.add_model_change(SolverNotifyFlags.BODY_INERTIAL_PROPERTIES)

    def set_coms_index(
        self,
        *,
        coms: torch.Tensor | wp.array,
        body_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set center of mass position of all bodies using indices.

        .. note::
            This method expects partial data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations. Performance is similar for both.
            However, to allow graphed pipelines, the mask method must be used.

        .. caution::
            Unlike the PhysX version of this method, this method does not set the center of mass orientation.
            Only the position is set. This is because Newton considers the center of mass orientation to always be
            aligned with the body frame.

        Args:
            coms: Center of mass position of all bodies. Shape is (len(env_ids), len(body_ids), 3).
            body_ids: The body indices to set the center of mass pose for. Defaults to None (all bodies).
            env_ids: The environment indices to set the center of mass pose for. Defaults to None (all environments).
        """
        # resolve all indices
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)
        self.assert_shape_and_dtype(coms, (env_ids.shape[0], body_ids.shape[0]), wp.vec3f, "coms")
        # Write to consolidated buffer
        wp.launch(
            shared_kernels.write_body_com_position_to_buffer_index,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[
                coms,
                env_ids,
                body_ids,
            ],
            outputs=[
                self.data._sim_bind_body_com_pos_b,
            ],
            device=self.device,
        )
        # Invalidate derived buffers that depend on com position
        self.data._body_com_pose_b.timestamp = -1.0
        self.data._body_com_pose_w.timestamp = -1.0
        # Tell the physics engine that some of the body properties have been updated
        SimulationManager.add_model_change(SolverNotifyFlags.BODY_INERTIAL_PROPERTIES)

    def set_coms_mask(
        self,
        *,
        coms: torch.Tensor | wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set center of mass position of all bodies using masks.

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations. Performance is similar for both.
            However, to allow graphed pipelines, the mask method must be used.

        .. caution::
            Unlike the PhysX version of this method, this method does not set the center of mass orientation.
            Only the position is set. This is because Newton considers the center of mass orientation to always be
            aligned with the body frame.

        Args:
            coms: Center of mass position of all bodies. Shape is (num_instances, num_bodies, 3).
            body_mask: Body mask. If None, then all bodies are used.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        # resolve masks
        if env_mask is None:
            env_mask = self._ALL_ENV_MASK
        if body_mask is None:
            body_mask = self._ALL_BODY_MASK
        self.assert_shape_and_dtype_mask(coms, (env_mask, body_mask), wp.vec3f, "coms")
        wp.launch(
            shared_kernels.write_body_com_position_to_buffer_mask,
            dim=(env_mask.shape[0], body_mask.shape[0]),
            inputs=[
                coms,
                env_mask,
                body_mask,
            ],
            outputs=[
                self.data._sim_bind_body_com_pos_b,
            ],
            device=self.device,
        )
        # Invalidate derived buffers that depend on com position
        self.data._body_com_pose_b.timestamp = -1.0
        self.data._body_com_pose_w.timestamp = -1.0
        # Tell the physics engine that some of the body properties have been updated
        SimulationManager.add_model_change(SolverNotifyFlags.BODY_INERTIAL_PROPERTIES)

    def set_inertias_index(
        self,
        *,
        inertias: torch.Tensor | wp.array,
        body_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set inertias of all bodies using indices.

        .. note::
            This method expects partial data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations. Performance is similar for both.
            However, to allow graphed pipelines, the mask method must be used.

        Args:
            inertias: Inertias of all bodies. Shape is (len(env_ids), len(body_ids), 9).
            body_ids: The body indices to set the inertias for. Defaults to None (all bodies).
            env_ids: The environment indices to set the inertias for. Defaults to None (all environments).
        """
        # resolve all indices
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)
        self.assert_shape_and_dtype(inertias, (env_ids.shape[0], body_ids.shape[0], 9), wp.float32, "inertias")
        # Write to consolidated buffer
        wp.launch(
            shared_kernels.write_body_inertia_to_buffer_index,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[
                inertias,
                env_ids,
                body_ids,
            ],
            outputs=[
                self.data._body_inertia,
            ],
            device=self.device,
        )
        # No copy-back needed — writes go directly to Newton's state via the 2D binding
        # Tell the physics engine that some of the body properties have been updated
        SimulationManager.add_model_change(SolverNotifyFlags.BODY_INERTIAL_PROPERTIES)

    def set_inertias_mask(
        self,
        *,
        inertias: torch.Tensor | wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set inertias of all bodies using masks.

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations. Performance is similar for both.
            However, to allow graphed pipelines, the mask method must be used.

        Args:
            inertias: Inertias of all bodies. Shape is (num_instances, num_bodies, 9).
            body_mask: Body mask. If None, then all bodies are used.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        # resolve masks
        if env_mask is None:
            env_mask = self._ALL_ENV_MASK
        if body_mask is None:
            body_mask = self._ALL_BODY_MASK
        self.assert_shape_and_dtype_mask(inertias, (env_mask, body_mask), wp.float32, "inertias", trailing_dims=(9,))
        wp.launch(
            shared_kernels.write_body_inertia_to_buffer_mask,
            dim=(env_mask.shape[0], body_mask.shape[0]),
            inputs=[
                inertias,
                env_mask,
                body_mask,
            ],
            outputs=[
                self.data._body_inertia,
            ],
            device=self.device,
        )
        # No copy-back needed — writes go directly to Newton's state via the 2D binding
        # Tell the physics engine that some of the body properties have been updated
        SimulationManager.add_model_change(SolverNotifyFlags.BODY_INERTIAL_PROPERTIES)

    """
    Internal helper.
    """

    def _initialize_impl(self):
        # clear body names list to prevent double counting on re-initialization
        self._body_names_list.clear()
        root_prim_path_exprs: list[str] = []

        for name, rigid_body_cfg in self.cfg.rigid_objects.items():
            # obtain the first prim in the regex expression (all others are assumed to be a copy of this)
            template_prim = sim_utils.find_first_matching_prim(rigid_body_cfg.prim_path)
            if template_prim is None:
                raise RuntimeError(f"Failed to find prim for expression: '{rigid_body_cfg.prim_path}'.")
            template_prim_path = template_prim.GetPath().pathString

            # find rigid root prims
            root_prims = sim_utils.get_all_matching_child_prims(
                template_prim_path,
                predicate=lambda prim: prim.HasAPI(UsdPhysics.RigidBodyAPI),
                traverse_instance_prims=False,
            )
            if len(root_prims) == 0:
                raise RuntimeError(
                    f"Failed to find a rigid body when resolving '{rigid_body_cfg.prim_path}'."
                    " Please ensure that the prim has 'USD RigidBodyAPI' applied."
                )
            if len(root_prims) > 1:
                raise RuntimeError(
                    f"Failed to find a single rigid body when resolving '{rigid_body_cfg.prim_path}'."
                    f" Found multiple '{root_prims}' under '{template_prim_path}'."
                    " Please ensure that there is only one rigid body in the prim path tree."
                )

            # check that no rigid object has an articulation root API, which decreases simulation performance
            articulation_prims = sim_utils.get_all_matching_child_prims(
                template_prim_path,
                predicate=lambda prim: prim.HasAPI(UsdPhysics.ArticulationRootAPI),
                traverse_instance_prims=False,
            )
            if len(articulation_prims) != 0:
                if articulation_prims[0].GetAttribute("physxArticulation:articulationEnabled").Get():
                    raise RuntimeError(
                        f"Found an articulation root when resolving '{rigid_body_cfg.prim_path}' in the rigid object"
                        f" collection. These are located at: '{articulation_prims}' under '{template_prim_path}'."
                        " Please disable the articulation root in the USD or from code by setting the parameter"
                        " 'ArticulationRootPropertiesCfg.articulation_enabled' to False in the spawn configuration."
                    )

            # resolve root prim back into regex expression
            root_prim_path = root_prims[0].GetPath().pathString
            root_prim_path_expr = rigid_body_cfg.prim_path + root_prim_path[len(template_prim_path) :]
            root_prim_path_exprs.append(root_prim_path_expr.replace(".*", "*"))
            self._body_names_list.append(name)

        # Build a single pattern that matches ALL body types by wildcarding the differing path segment
        combined_pattern = self._build_combined_pattern(root_prim_path_exprs)

        # Create a single ArticulationView matching all body types.
        # The 2nd dimension (matches per world) corresponds to the body types.
        self._root_view = ArticulationView(
            SimulationManager.get_model(),
            combined_pattern,
            verbose=False,
        )

        # container for data access
        self._data = RigidObjectCollectionData(self._root_view, self.num_bodies, self.device)

        # create buffers
        self._create_buffers()
        # process configuration
        self._process_cfg()
        # update the rigid body data
        self.update(0.0)
        # Let the rigid object collection data know that it is fully instantiated and ready to use.
        self.data.is_primed = True

    def _create_buffers(self):
        """Create buffers for storing data."""
        # constants
        self._ALL_ENV_INDICES = wp.array(
            np.arange(self.num_instances, dtype=np.int32), device=self.device, dtype=wp.int32
        )
        self._ALL_BODY_INDICES = wp.array(
            np.arange(self.num_bodies, dtype=np.int32), device=self.device, dtype=wp.int32
        )
        self._ALL_ENV_MASK = wp.ones((self.num_instances,), dtype=wp.bool, device=self.device)
        self._ALL_BODY_MASK = wp.ones((self.num_bodies,), dtype=wp.bool, device=self.device)

        # external wrench composer
        self._instantaneous_wrench_composer = WrenchComposer(self)
        self._permanent_wrench_composer = WrenchComposer(self)

        # Temporary 2D wrench buffer for write_data_to_sim
        self._wrench_buffer = wp.zeros(
            (self.num_instances, self.num_bodies), dtype=wp.spatial_vectorf, device=self.device
        )

        # set information about rigid body into data
        self._data.body_names = self.body_names

    def _process_cfg(self) -> None:
        """Post processing of configuration parameters."""
        # default state
        # -- body state
        default_body_poses = []
        default_body_vels = []
        for rigid_object_cfg in self.cfg.rigid_objects.values():
            default_body_pose = tuple(rigid_object_cfg.init_state.pos) + tuple(rigid_object_cfg.init_state.rot)
            default_body_vel = tuple(rigid_object_cfg.init_state.lin_vel) + tuple(rigid_object_cfg.init_state.ang_vel)
            default_body_pose = np.tile(np.array(default_body_pose, dtype=np.float32), (self.num_instances, 1))
            default_body_vel = np.tile(np.array(default_body_vel, dtype=np.float32), (self.num_instances, 1))
            default_body_poses.append(default_body_pose)
            default_body_vels.append(default_body_vel)
        # Stack: each has shape (num_instances, data_size) -> (num_instances, num_bodies, data_size)
        default_body_poses = np.stack(default_body_poses, axis=1)
        default_body_vels = np.stack(default_body_vels, axis=1)
        self.data.default_body_pose = wp.array(default_body_poses, dtype=wp.transformf, device=self.device)
        self.data.default_body_vel = wp.array(default_body_vels, dtype=wp.spatial_vectorf, device=self.device)

    def _resolve_env_ids(self, env_ids) -> wp.array:
        """Resolve environment indices to a warp array."""
        if isinstance(env_ids, list):
            return wp.array(env_ids, dtype=wp.int32, device=self.device)
        if (env_ids is None) or (env_ids == slice(None)):
            return self._ALL_ENV_INDICES
        if isinstance(env_ids, torch.Tensor):
            return wp.from_torch(env_ids.to(torch.int32), dtype=wp.int32)
        return env_ids

    def _resolve_body_ids(self, body_ids) -> wp.array:
        """Resolve body indices to a warp array."""
        if body_ids is None or (body_ids == slice(None)):
            return self._ALL_BODY_INDICES
        if isinstance(body_ids, slice):
            return wp.from_torch(
                torch.arange(self.num_bodies, dtype=torch.int32, device=self.device)[body_ids], dtype=wp.int32
            )
        if isinstance(body_ids, list):
            return wp.array(body_ids, dtype=wp.int32, device=self.device)
        if isinstance(body_ids, torch.Tensor):
            return wp.from_torch(body_ids.to(torch.int32), dtype=wp.int32)
        return body_ids

    def _resolve_env_mask(self, env_mask: wp.array | None) -> wp.array | torch.Tensor:
        """Resolve environment mask to indices via torch.nonzero."""
        if env_mask is not None:
            if isinstance(env_mask, wp.array):
                env_mask = wp.to_torch(env_mask)
            env_ids = torch.nonzero(env_mask)[:, 0].to(torch.int32)
        else:
            env_ids = self._ALL_ENV_INDICES
        return env_ids

    def _resolve_body_mask(self, body_mask: wp.array | None) -> wp.array | torch.Tensor:
        """Resolve body mask to indices via torch.nonzero."""
        if body_mask is not None:
            if isinstance(body_mask, wp.array):
                body_mask = wp.to_torch(body_mask)
            body_ids = torch.nonzero(body_mask)[:, 0].to(torch.int32)
        else:
            body_ids = self._ALL_BODY_INDICES
        return body_ids

    @staticmethod
    def _build_combined_pattern(prim_path_exprs: list[str]) -> str:
        """Build a single fnmatch pattern that matches all body types.

        Compares path segments across all expressions and wildcards the segments that differ.
        For example, given::

            ["/World/Env_*/DexCube/Cube", "/World/Env_*/DexSphere/Sphere"]

        produces ``"/World/Env_*/*/*"``.

        Args:
            prim_path_exprs: List of prim path expressions, one per body type.

        Returns:
            A single fnmatch pattern string.

        Raises:
            ValueError: If the expressions have different numbers of path segments.
        """
        if len(prim_path_exprs) == 1:
            return prim_path_exprs[0]

        split_paths = [p.split("/") for p in prim_path_exprs]
        lengths = {len(s) for s in split_paths}
        if len(lengths) != 1:
            raise ValueError(
                f"Cannot build combined pattern: path expressions have different segment counts: {prim_path_exprs}"
            )

        combined_segments = []
        for segments in zip(*split_paths):
            unique = set(segments)
            if len(unique) == 1:
                combined_segments.append(segments[0])
            else:
                combined_segments.append("*")
        return "/".join(combined_segments)

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event) -> None:
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._root_view = None

    def _on_prim_deletion(self, prim_path: str) -> None:
        """Invalidates and deletes the callbacks when the prim is deleted.

        Args:
            prim_path: The path to the prim that is being deleted.

        .. note::
            This function is called when the prim is deleted.
        """
        if prim_path == "/":
            self._clear_callbacks()
            return
        for prim_path_expr in [obj.prim_path for obj in self.cfg.rigid_objects.values()]:
            result = re.match(
                pattern="^" + "/".join(prim_path_expr.split("/")[: prim_path.count("/") + 1]) + "$", string=prim_path
            )
            if result:
                self._clear_callbacks()
                return

    """
    Deprecated properties and methods.
    """

    @property
    def root_physx_view(self):
        """Deprecated property. Please use :attr:`root_view` instead."""
        warnings.warn(
            "The `root_physx_view` property will be deprecated in a future release. Please use `root_view` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.root_view

    def write_body_state_to_sim(
        self,
        body_states: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        body_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_body_link_pose_to_sim_index` and
        :meth:`write_body_com_velocity_to_sim_index`."""
        warnings.warn(
            "The function 'write_body_state_to_sim' will be deprecated in a future release. Please"
            " use 'write_body_link_pose_to_sim_index' and 'write_body_com_velocity_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_body_link_pose_to_sim_index(body_poses=body_states[:, :, :7], env_ids=env_ids, body_ids=body_ids)
        self.write_body_com_velocity_to_sim_index(
            body_velocities=body_states[:, :, 7:], env_ids=env_ids, body_ids=body_ids
        )

    def write_body_com_state_to_sim(
        self,
        body_states: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        body_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_body_com_pose_to_sim_index` and
        :meth:`write_body_com_velocity_to_sim_index`."""
        warnings.warn(
            "The function 'write_body_com_state_to_sim' will be deprecated in a future release. Please"
            " use 'write_body_com_pose_to_sim_index' and 'write_body_com_velocity_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_body_com_pose_to_sim_index(body_poses=body_states[:, :, :7], env_ids=env_ids, body_ids=body_ids)
        self.write_body_com_velocity_to_sim_index(
            body_velocities=body_states[:, :, 7:], env_ids=env_ids, body_ids=body_ids
        )

    def write_body_link_state_to_sim(
        self,
        body_states: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        body_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_body_link_pose_to_sim_index` and
        :meth:`write_body_link_velocity_to_sim_index`."""
        warnings.warn(
            "The function 'write_body_link_state_to_sim' will be deprecated in a future release. Please"
            " use 'write_body_link_pose_to_sim_index' and 'write_body_link_velocity_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_body_link_pose_to_sim_index(body_poses=body_states[:, :, :7], env_ids=env_ids, body_ids=body_ids)
        self.write_body_link_velocity_to_sim_index(
            body_velocities=body_states[:, :, 7:], env_ids=env_ids, body_ids=body_ids
        )
