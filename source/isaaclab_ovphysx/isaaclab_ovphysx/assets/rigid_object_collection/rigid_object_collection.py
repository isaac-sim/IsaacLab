# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import warp as wp

from pxr import UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets.rigid_object_collection.base_rigid_object_collection import BaseRigidObjectCollection
from isaaclab.utils.string import resolve_matching_names
from isaaclab.utils.wrench_composer import WrenchComposer

from isaaclab_ovphysx import tensor_types as TT
from isaaclab_ovphysx.assets import kernels as shared_kernels
from isaaclab_ovphysx.assets.kernels import _body_wrench_to_world, resolve_view_ids
from isaaclab_ovphysx.cloner import queue_ovphysx_replication
from isaaclab_ovphysx.physics import OvPhysxManager

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

    __backend_name__: str = "ovphysx"
    """The name of the backend for the rigid object."""

    def __init__(self, cfg: RigidObjectCollectionCfg):
        """Initialize the rigid object.

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
        for rigid_body_name, rigid_body_cfg in self.cfg.rigid_objects.items():
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
            queue_ovphysx_replication(cfg.rigid_objects[rigid_body_name])
        # stores object names
        self._body_names_list: list[str] = []
        # one fused TensorBinding per tensor type, populated in _initialize_impl
        self._bindings: dict[int, Any] = {}

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
        return self._num_instances

    @property
    def num_bodies(self) -> int:
        """Number of bodies in the rigid object collection."""
        return self._num_bodies

    @property
    def body_names(self) -> list[str]:
        """Ordered names of bodies in the rigid object collection."""
        return list(self._body_names_list)

    @property
    def root_view(self):
        """Root view for the rigid object collection.

        Dictionary keyed by TensorType constant, each value a single fused
        :class:`~isaaclab_ovphysx.TensorBinding` spanning all bodies in the collection.

        .. note::
            Use this view with caution. It requires handling of tensors in a specific way.
        """
        return self._bindings

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
        env_ids: Sequence[int] | wp.array | None = None,
        object_ids: slice | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Resets all internal buffers of selected environments and objects.

        Args:
            env_ids: Environment indices. If None, then all indices are used.
            object_ids: Object indices. If None, then all indices are used.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        # resolve all indices
        if (env_ids is None) or (env_ids == slice(None)):
            env_ids = slice(None)
        # reset external wrench
        self._instantaneous_wrench_composer.reset(env_ids, env_mask)
        self._permanent_wrench_composer.reset(env_ids, env_mask)

    def write_data_to_sim(self) -> None:
        """Write external wrench to the simulation.

        .. note::
            We write external wrench to the simulation here since this function is called before the simulation step.
            This ensures that the external wrench is applied at every simulation step.
        """
        inst = self._instantaneous_wrench_composer
        perm = self._permanent_wrench_composer
        if not inst.active and not perm.active:
            return
        if inst.active:
            if perm.active:
                inst.add_raw_buffers_from(perm)
            force_b = inst.out_force_b.warp
            torque_b = inst.out_torque_b.warp
        else:
            force_b = perm.out_force_b.warp
            torque_b = perm.out_torque_b.warp

        poses = self._data.body_link_pose_w.warp  # (N, B) wp.transformf
        wp.launch(
            _body_wrench_to_world,
            dim=(self._num_instances, self._num_bodies),
            inputs=[force_b, torque_b, poses],
            outputs=[self._wrench_buf],
            device=self._device,
        )
        binding = self._get_binding(TT.LINK_WRENCH)
        if binding is not None:
            # The articulation-mode mock used by iface tests exposes an instance-major
            # ``(N, B, 9)`` view directly; the native fused binding lays elements body-
            # major flat as ``(N * B, 9)``. Dispatch via the binding's exposed shape.
            if len(binding.shape) >= 2 and binding.shape[1] == self._num_bodies:
                binding.write(self._wrench_buf)
            else:
                view = self.reshape_data_to_view_3d(self._wrench_buf, 9, device=self._device)
                binding.write(view)
        inst.reset()

    def update(self, dt: float) -> None:
        """Updates the simulation data.

        Args:
            dt: The time step size in seconds.
        """
        self._data.update(dt)

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
        obj_ids, obj_names = resolve_matching_names(name_keys, self.body_names, preserve_order)
        return torch.tensor(obj_ids, device=self._device, dtype=torch.int32), obj_names

    """
    Operations - Write to simulation.
    """

    def write_body_pose_to_sim_index(
        self,
        *,
        body_poses: wp.array,
        body_ids: Sequence[int] | wp.array | None = None,
        env_ids: Sequence[int] | wp.array | None = None,
        skip_forward: bool = False,
    ) -> None:
        """Set the body pose over selected environment and body indices into the simulation.

        The body pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        .. note::
            This method expects partial data.

        Args:
            body_poses: Body poses in simulation frame [m, rad]. Shape is (len(env_ids), len(body_ids), 7)
                or (len(env_ids), len(body_ids)) with dtype wp.transformf.
            body_ids: Body indices. If None, then all indices are used.
            env_ids: Environment indices. If None, then all indices are used.
            skip_forward: Whether to skip invalidating cached data after the write. When True, the caller
                must invalidate stale cached data before reading it back. Defaults to False.
        """
        self.write_body_link_pose_to_sim_index(
            body_poses=body_poses, body_ids=body_ids, env_ids=env_ids, skip_forward=skip_forward
        )

    def write_body_pose_to_sim_mask(
        self,
        *,
        body_poses: wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
        skip_forward: bool = False,
    ) -> None:
        """Set the body pose over selected environment and body masks into the simulation.

        The body pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        .. note::
            This method expects full data.

        Args:
            body_poses: Body poses in simulation frame [m, rad]. Shape is (num_instances, num_bodies, 7)
                or (num_instances, num_bodies) with dtype wp.transformf.
            body_mask: Body mask. If None, then all bodies are updated. Shape is (num_bodies,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
            skip_forward: Whether to skip invalidating cached data after the write. When True, the caller
                must invalidate stale cached data before reading it back. Defaults to False.
        """
        self.write_body_link_pose_to_sim_mask(
            body_poses=body_poses, body_mask=body_mask, env_mask=env_mask, skip_forward=skip_forward
        )

    def write_body_velocity_to_sim_index(
        self,
        *,
        body_velocities: wp.array,
        body_ids: Sequence[int] | wp.array | None = None,
        env_ids: Sequence[int] | wp.array | None = None,
        skip_forward: bool = False,
    ) -> None:
        """Set the body velocity over selected environment and body indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.

        .. note::
            This sets the velocity of the body's center of mass rather than the body's frame.

        .. note::
            This method expects partial data.

        Args:
            body_velocities: Body velocities in simulation world frame [m/s, rad/s].
                Shape is (len(env_ids), len(body_ids)) with dtype wp.spatial_vectorf.
            body_ids: Body indices. If None, then all indices are used.
            env_ids: Environment indices. If None, then all indices are used.
            skip_forward: Whether to skip invalidating cached data after the write. When True, the caller
                must invalidate stale cached data before reading it back. Defaults to False.
        """
        self.write_body_com_velocity_to_sim_index(
            body_velocities=body_velocities, body_ids=body_ids, env_ids=env_ids, skip_forward=skip_forward
        )

    def write_body_velocity_to_sim_mask(
        self,
        *,
        body_velocities: wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
        skip_forward: bool = False,
    ) -> None:
        """Set the body velocity over selected environment and body masks into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.

        .. note::
            This sets the velocity of the body's center of mass rather than the body's frame.

        .. note::
            This method expects full data.

        Args:
            body_velocities: Body velocities in simulation world frame [m/s, rad/s].
                Shape is (num_instances, num_bodies) with dtype wp.spatial_vectorf.
            body_mask: Body mask. If None, then all bodies are updated. Shape is (num_bodies,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
            skip_forward: Whether to skip invalidating cached data after the write. When True, the caller
                must invalidate stale cached data before reading it back. Defaults to False.
        """
        self.write_body_com_velocity_to_sim_mask(
            body_velocities=body_velocities, body_mask=body_mask, env_mask=env_mask, skip_forward=skip_forward
        )

    def write_body_link_pose_to_sim_index(
        self,
        *,
        body_poses: wp.array,
        body_ids: Sequence[int] | wp.array | None = None,
        env_ids: Sequence[int] | wp.array | None = None,
        skip_forward: bool = False,
    ) -> None:
        """Set the body link pose over selected environment and body indices into the simulation.

        The body link pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        .. note::
            This method expects partial data.

        Args:
            body_poses: Body link poses in simulation frame [m, rad]. Shape is (len(env_ids), len(body_ids), 7)
                or (len(env_ids), len(body_ids)) with dtype wp.transformf.
            body_ids: Body indices. If None, then all indices are used.
            env_ids: Environment indices. If None, then all indices are used.
            skip_forward: Whether to skip invalidating cached data after the write. When True, the caller
                must invalidate stale cached data before reading it back. Defaults to False.
        """
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)
        self.assert_shape_and_dtype(body_poses, (env_ids.shape[0], body_ids.shape[0]), wp.transformf, "body_poses")
        wp.launch(
            shared_kernels.set_body_link_pose_to_sim,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[body_poses, env_ids, body_ids, False],
            outputs=[
                self.data._body_link_pose_w.data,
                self.data._body_link_state_w.data,
                self.data._body_state_w.data,
            ],
            device=self._device,
        )
        # Mark the link pose fresh so reads within the same step return the
        # kernel-written value rather than re-fetching the pre-step OVPhysX state.
        self.data._body_link_pose_w.timestamp = self.data._sim_timestamp
        if not skip_forward:
            self.data._reset_pose()
        # set into simulation
        self._binding_write(TT.LINK_POSE, self.data._body_link_pose_w.data, env_ids=env_ids)

    def write_body_link_pose_to_sim_mask(
        self,
        *,
        body_poses: wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
        skip_forward: bool = False,
    ) -> None:
        """Set the body link pose over selected environment and body masks into the simulation.

        The body link pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        .. note::
            This method expects full data.

        Args:
            body_poses: Body link poses in simulation frame [m, rad]. Shape is (num_instances, num_bodies, 7)
                or (num_instances, num_bodies) with dtype wp.transformf.
            body_mask: Body mask. If None, then all bodies are updated. Shape is (num_bodies,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
            skip_forward: Whether to skip invalidating cached data after the write. When True, the caller
                must invalidate stale cached data before reading it back. Defaults to False.
        """
        if env_mask is not None:
            env_mask_t = wp.to_torch(env_mask) if isinstance(env_mask, wp.array) else env_mask
            env_ids = self._resolve_env_ids(torch.nonzero(env_mask_t)[:, 0].to(torch.int32))
        else:
            env_ids = self._ALL_ENV_INDICES
        if body_mask is not None:
            body_mask_t = wp.to_torch(body_mask) if isinstance(body_mask, wp.array) else body_mask
            body_ids = self._resolve_body_ids(torch.nonzero(body_mask_t)[:, 0].to(torch.int32))
        else:
            body_ids = self._ALL_BODY_INDICES
        self.assert_shape_and_dtype(body_poses, (self._num_instances, self._num_bodies), wp.transformf, "body_poses")
        wp.launch(
            shared_kernels.set_body_link_pose_to_sim,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[body_poses, env_ids, body_ids, True],
            outputs=[
                self.data._body_link_pose_w.data,
                self.data._body_link_state_w.data,
                self.data._body_state_w.data,
            ],
            device=self._device,
        )
        # Invalidate dependent timestamps
        if not skip_forward:
            self.data._reset_pose()
        # set into simulation
        self._binding_write(TT.LINK_POSE, self.data._body_link_pose_w.data, env_ids=env_ids)

    def write_body_com_pose_to_sim_index(
        self,
        *,
        body_poses: wp.array,
        body_ids: Sequence[int] | wp.array | None = None,
        env_ids: Sequence[int] | wp.array | None = None,
        skip_forward: bool = False,
    ) -> None:
        """Set the body center of mass pose over selected environment and body indices into the simulation.

        The body center of mass pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).
        The orientation is the orientation of the principal axes of inertia.

        .. note::
            This method expects partial data.

        Args:
            body_poses: Body center of mass poses in simulation frame [m, rad].
                Shape is (len(env_ids), len(body_ids), 7) or (len(env_ids), len(body_ids)) with dtype wp.transformf.
            body_ids: Body indices. If None, then all indices are used.
            env_ids: Environment indices. If None, then all indices are used.
            skip_forward: Whether to skip invalidating cached data after the write. When True, the caller
                must invalidate stale cached data before reading it back. Defaults to False.
        """
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)
        self.assert_shape_and_dtype(body_poses, (env_ids.shape[0], body_ids.shape[0]), wp.transformf, "body_poses")
        wp.launch(
            shared_kernels.set_body_com_pose_to_sim,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[body_poses, self.data.body_com_pose_b, env_ids, body_ids, False],
            outputs=[
                self.data._body_com_pose_w.data,
                self.data._body_link_pose_w.data,
                self.data._body_com_state_w.data,
                self.data._body_link_state_w.data,
                self.data._body_state_w.data,
            ],
            device=self._device,
        )
        # Invalidate dependent timestamps
        if not skip_forward:
            self.data._reset_pose(from_link=False)
        # set into simulation (OVPhysX only exposes the link frame)
        self._binding_write(TT.LINK_POSE, self.data._body_link_pose_w.data, env_ids=env_ids)

    def write_body_com_pose_to_sim_mask(
        self,
        *,
        body_poses: wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
        skip_forward: bool = False,
    ) -> None:
        """Set the body center of mass pose over selected environment and body masks into the simulation.

        The body center of mass pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).
        The orientation is the orientation of the principal axes of inertia.

        .. note::
            This method expects full data.

        Args:
            body_poses: Body center of mass poses in simulation frame [m, rad].
                Shape is (num_instances, num_bodies, 7) or (num_instances, num_bodies) with dtype wp.transformf.
            body_mask: Body mask. If None, then all bodies are updated. Shape is (num_bodies,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
            skip_forward: Whether to skip invalidating cached data after the write. When True, the caller
                must invalidate stale cached data before reading it back. Defaults to False.
        """
        if env_mask is not None:
            env_mask_t = wp.to_torch(env_mask) if isinstance(env_mask, wp.array) else env_mask
            env_ids = self._resolve_env_ids(torch.nonzero(env_mask_t)[:, 0].to(torch.int32))
        else:
            env_ids = self._ALL_ENV_INDICES
        if body_mask is not None:
            body_mask_t = wp.to_torch(body_mask) if isinstance(body_mask, wp.array) else body_mask
            body_ids = self._resolve_body_ids(torch.nonzero(body_mask_t)[:, 0].to(torch.int32))
        else:
            body_ids = self._ALL_BODY_INDICES
        self.assert_shape_and_dtype(body_poses, (self._num_instances, self._num_bodies), wp.transformf, "body_poses")
        wp.launch(
            shared_kernels.set_body_com_pose_to_sim,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[body_poses, self.data.body_com_pose_b, env_ids, body_ids, True],
            outputs=[
                self.data._body_com_pose_w.data,
                self.data._body_link_pose_w.data,
                self.data._body_com_state_w.data,
                self.data._body_link_state_w.data,
                self.data._body_state_w.data,
            ],
            device=self._device,
        )
        # Invalidate dependent timestamps
        if not skip_forward:
            self.data._reset_pose(from_link=False)
        # set into simulation (OVPhysX only exposes the link frame)
        self._binding_write(TT.LINK_POSE, self.data._body_link_pose_w.data, env_ids=env_ids)

    def write_body_com_velocity_to_sim_index(
        self,
        *,
        body_velocities: wp.array,
        body_ids: Sequence[int] | wp.array | None = None,
        env_ids: Sequence[int] | wp.array | None = None,
        skip_forward: bool = False,
    ) -> None:
        """Set the body center of mass velocity over selected environment and body indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.

        .. note::
            This sets the velocity of the body's center of mass rather than the body's frame.

        .. note::
            This method expects partial data.

        Args:
            body_velocities: Body center of mass velocities in simulation world frame [m/s, rad/s].
                Shape is (len(env_ids), len(body_ids)) with dtype wp.spatial_vectorf.
            body_ids: Body indices. If None, then all indices are used.
            env_ids: Environment indices. If None, then all indices are used.
            skip_forward: Whether to skip invalidating cached data after the write. When True, the caller
                must invalidate stale cached data before reading it back. Defaults to False.
        """
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)
        self.assert_shape_and_dtype(
            body_velocities, (env_ids.shape[0], body_ids.shape[0]), wp.spatial_vectorf, "body_velocities"
        )
        wp.launch(
            shared_kernels.set_body_com_velocity_to_sim,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[body_velocities, env_ids, body_ids, False],
            outputs=[
                self.data._body_com_vel_w.data,
                self.data._body_com_acc_w.data,
                self.data._body_state_w.data,
                self.data._body_com_state_w.data,
            ],
            device=self._device,
        )
        # Mark the COM velocity fresh so reads within the same step return the
        # kernel-written value rather than re-fetching the pre-step OVPhysX state.
        self.data._body_com_vel_w.timestamp = self.data._sim_timestamp
        if not skip_forward:
            self.data._reset_velocity()
        # set into simulation
        self._binding_write(TT.LINK_VELOCITY, self.data._body_com_vel_w.data, env_ids=env_ids)

    def write_body_com_velocity_to_sim_mask(
        self,
        *,
        body_velocities: wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
        skip_forward: bool = False,
    ) -> None:
        """Set the body center of mass velocity over selected environment and body masks into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.

        .. note::
            This sets the velocity of the body's center of mass rather than the body's frame.

        .. note::
            This method expects full data.

        Args:
            body_velocities: Body center of mass velocities in simulation world frame [m/s, rad/s].
                Shape is (num_instances, num_bodies) with dtype wp.spatial_vectorf.
            body_mask: Body mask. If None, then all bodies are updated. Shape is (num_bodies,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
            skip_forward: Whether to skip invalidating cached data after the write. When True, the caller
                must invalidate stale cached data before reading it back. Defaults to False.
        """
        if env_mask is not None:
            env_mask_t = wp.to_torch(env_mask) if isinstance(env_mask, wp.array) else env_mask
            env_ids = self._resolve_env_ids(torch.nonzero(env_mask_t)[:, 0].to(torch.int32))
        else:
            env_ids = self._ALL_ENV_INDICES
        if body_mask is not None:
            body_mask_t = wp.to_torch(body_mask) if isinstance(body_mask, wp.array) else body_mask
            body_ids = self._resolve_body_ids(torch.nonzero(body_mask_t)[:, 0].to(torch.int32))
        else:
            body_ids = self._ALL_BODY_INDICES
        self.assert_shape_and_dtype(
            body_velocities, (self._num_instances, self._num_bodies), wp.spatial_vectorf, "body_velocities"
        )
        wp.launch(
            shared_kernels.set_body_com_velocity_to_sim,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[body_velocities, env_ids, body_ids, True],
            outputs=[
                self.data._body_com_vel_w.data,
                self.data._body_com_acc_w.data,
                self.data._body_state_w.data,
                self.data._body_com_state_w.data,
            ],
            device=self._device,
        )
        # Invalidate dependent timestamps
        if not skip_forward:
            self.data._reset_velocity()
        # set into simulation
        self._binding_write(TT.LINK_VELOCITY, self.data._body_com_vel_w.data, env_ids=env_ids)

    def write_body_link_velocity_to_sim_index(
        self,
        *,
        body_velocities: wp.array,
        body_ids: Sequence[int] | wp.array | None = None,
        env_ids: Sequence[int] | wp.array | None = None,
        skip_forward: bool = False,
    ) -> None:
        """Set the body link velocity over selected environment and body indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.

        .. note::
            This sets the velocity of the body's frame rather than the body's center of mass.

        .. note::
            This method expects partial data.

        Args:
            body_velocities: Body link velocities in simulation world frame [m/s, rad/s].
                Shape is (len(env_ids), len(body_ids)) with dtype wp.spatial_vectorf.
            body_ids: Body indices. If None, then all indices are used.
            env_ids: Environment indices. If None, then all indices are used.
            skip_forward: Whether to skip invalidating cached data after the write. When True, the caller
                must invalidate stale cached data before reading it back. Defaults to False.
        """
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)
        self.assert_shape_and_dtype(
            body_velocities, (env_ids.shape[0], body_ids.shape[0]), wp.spatial_vectorf, "body_velocities"
        )
        wp.launch(
            shared_kernels.set_body_link_velocity_to_sim,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[
                body_velocities,
                self.data.body_com_pose_b,
                self.data.body_link_pose_w,
                env_ids,
                body_ids,
                False,
            ],
            outputs=[
                self.data._body_link_vel_w.data,
                self.data._body_com_vel_w.data,
                self.data._body_com_acc_w.data,
                self.data._body_link_state_w.data,
                self.data._body_state_w.data,
                self.data._body_com_state_w.data,
            ],
            device=self._device,
        )
        # Invalidate dependent timestamps
        if not skip_forward:
            self.data._reset_velocity(from_com=False)
        # set into simulation
        self._binding_write(TT.LINK_VELOCITY, self.data._body_com_vel_w.data, env_ids=env_ids)

    def write_body_link_velocity_to_sim_mask(
        self,
        *,
        body_velocities: wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
        skip_forward: bool = False,
    ) -> None:
        """Set the body link velocity over selected environment and body masks into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.

        .. note::
            This sets the velocity of the body's frame rather than the body's center of mass.

        .. note::
            This method expects full data.

        Args:
            body_velocities: Body link velocities in simulation world frame [m/s, rad/s].
                Shape is (num_instances, num_bodies) with dtype wp.spatial_vectorf.
            body_mask: Body mask. If None, then all bodies are updated. Shape is (num_bodies,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
            skip_forward: Whether to skip invalidating cached data after the write. When True, the caller
                must invalidate stale cached data before reading it back. Defaults to False.
        """
        if env_mask is not None:
            env_mask_t = wp.to_torch(env_mask) if isinstance(env_mask, wp.array) else env_mask
            env_ids = self._resolve_env_ids(torch.nonzero(env_mask_t)[:, 0].to(torch.int32))
        else:
            env_ids = self._ALL_ENV_INDICES
        if body_mask is not None:
            body_mask_t = wp.to_torch(body_mask) if isinstance(body_mask, wp.array) else body_mask
            body_ids = self._resolve_body_ids(torch.nonzero(body_mask_t)[:, 0].to(torch.int32))
        else:
            body_ids = self._ALL_BODY_INDICES
        self.assert_shape_and_dtype(
            body_velocities, (self._num_instances, self._num_bodies), wp.spatial_vectorf, "body_velocities"
        )
        wp.launch(
            shared_kernels.set_body_link_velocity_to_sim,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[
                body_velocities,
                self.data.body_com_pose_b,
                self.data.body_link_pose_w,
                env_ids,
                body_ids,
                True,
            ],
            outputs=[
                self.data._body_link_vel_w.data,
                self.data._body_com_vel_w.data,
                self.data._body_com_acc_w.data,
                self.data._body_link_state_w.data,
                self.data._body_state_w.data,
                self.data._body_com_state_w.data,
            ],
            device=self._device,
        )
        # Invalidate dependent timestamps
        if not skip_forward:
            self.data._reset_velocity(from_com=False)
        # set into simulation
        self._binding_write(TT.LINK_VELOCITY, self.data._body_com_vel_w.data, env_ids=env_ids)

    """
    Operations - Setters.
    """

    def set_masses_index(
        self,
        *,
        masses: wp.array,
        body_ids: Sequence[int] | wp.array | None = None,
        env_ids: Sequence[int] | wp.array | None = None,
    ) -> None:
        """Set body masses over selected env / body indices into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``BODY_MASS`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects partial data.

        Args:
            masses: Body masses [kg]. Shape is (len(env_ids), len(body_ids))
                with dtype wp.float32.
            body_ids: Body indices. If None, then all indices are used.
            env_ids: Environment indices. If None, then all indices are used.
        """
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)
        self.assert_shape_and_dtype(masses, (env_ids.shape[0], body_ids.shape[0]), wp.float32, "masses")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_indices,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[masses, env_ids, body_ids],
            outputs=[self.data._body_mass.data],
            device=self._device,
        )
        wp.copy(self.data._cpu_body_mass, self.data._body_mass.data)
        self._binding_write(
            TT.BODY_MASS, self.data._cpu_body_mass, env_ids=self._get_cpu_env_ids(env_ids), device="cpu"
        )

    def set_masses_mask(
        self,
        *,
        masses: wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set body masses over selected env / body masks into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``BODY_MASS`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects full data.

        Args:
            masses: Body masses [kg]. Shape is (num_instances, num_bodies)
                with dtype wp.float32.
            body_mask: Body mask. If None, all bodies are updated.
                Shape is (num_bodies,).
            env_mask: Environment mask. If None, all instances are updated.
                Shape is (num_instances,).
        """
        if env_mask is not None:
            env_mask_t = wp.to_torch(env_mask) if isinstance(env_mask, wp.array) else env_mask
            env_ids = self._resolve_env_ids(torch.nonzero(env_mask_t)[:, 0].to(torch.int32))
        else:
            env_ids = self._ALL_ENV_INDICES
        self.assert_shape_and_dtype(masses, (self._num_instances, self._num_bodies), wp.float32, "masses")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_mask,
            dim=(self._num_instances, self._num_bodies),
            inputs=[masses, self._resolve_env_mask(env_mask), self._resolve_body_mask(body_mask)],
            outputs=[self.data._body_mass.data],
            device=self._device,
        )
        wp.copy(self.data._cpu_body_mass, self.data._body_mass.data)
        self._binding_write(
            TT.BODY_MASS, self.data._cpu_body_mass, env_ids=self._get_cpu_env_ids(env_ids), device="cpu"
        )

    def set_coms_index(
        self,
        *,
        coms: wp.array,
        body_ids: Sequence[int] | wp.array | None = None,
        env_ids: Sequence[int] | wp.array | None = None,
    ) -> None:
        """Set body center-of-mass poses over selected env / body indices into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``BODY_COM_POSE`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects partial data.

        Args:
            coms: Body center-of-mass poses [m, quaternion (w, x, y, z)].
                Shape is (len(env_ids), len(body_ids)) with dtype wp.transformf.
            body_ids: Body indices. If None, then all indices are used.
            env_ids: Environment indices. If None, then all indices are used.
        """
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)
        self.assert_shape_and_dtype(coms, (env_ids.shape[0], body_ids.shape[0]), wp.transformf, "coms")
        wp.launch(
            shared_kernels.write_body_com_pose_to_buffer_index,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[coms, env_ids, body_ids],
            outputs=[self.data._body_com_pose_b.data],
            device=self._device,
        )
        # Invalidate derived buffers that depend on body_com_pose_b.
        self.data._body_com_pose_w.timestamp = -1.0
        wp.copy(self.data._cpu_body_coms, self.data._body_com_pose_b.data)
        self._binding_write(
            TT.BODY_COM_POSE,
            self.data._cpu_body_coms,
            env_ids=self._get_cpu_env_ids(env_ids),
            device="cpu",
            data_dim=7,
        )

    def set_coms_mask(
        self,
        *,
        coms: wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set body center-of-mass poses over selected env / body masks into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``BODY_COM_POSE`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects full data.

        Args:
            coms: Body center-of-mass poses [m, quaternion (w, x, y, z)].
                Shape is (num_instances, num_bodies) with dtype wp.transformf.
            body_mask: Body mask. If None, all bodies are updated.
                Shape is (num_bodies,).
            env_mask: Environment mask. If None, all instances are updated.
                Shape is (num_instances,).
        """
        if env_mask is not None:
            env_mask_t = wp.to_torch(env_mask) if isinstance(env_mask, wp.array) else env_mask
            env_ids = self._resolve_env_ids(torch.nonzero(env_mask_t)[:, 0].to(torch.int32))
        else:
            env_ids = self._ALL_ENV_INDICES
        self.assert_shape_and_dtype(coms, (self._num_instances, self._num_bodies), wp.transformf, "coms")
        wp.launch(
            shared_kernels.write_body_com_pose_to_buffer_mask,
            dim=(self._num_instances, self._num_bodies),
            inputs=[coms, self._resolve_env_mask(env_mask), self._resolve_body_mask(body_mask)],
            outputs=[self.data._body_com_pose_b.data],
            device=self._device,
        )
        # Invalidate derived buffers that depend on body_com_pose_b.
        self.data._body_com_pose_w.timestamp = -1.0
        wp.copy(self.data._cpu_body_coms, self.data._body_com_pose_b.data)
        self._binding_write(
            TT.BODY_COM_POSE,
            self.data._cpu_body_coms,
            env_ids=self._get_cpu_env_ids(env_ids),
            device="cpu",
            data_dim=7,
        )

    def set_inertias_index(
        self,
        *,
        inertias: wp.array,
        body_ids: Sequence[int] | wp.array | None = None,
        env_ids: Sequence[int] | wp.array | None = None,
    ) -> None:
        """Set body inertia tensors over selected env / body indices into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``BODY_INERTIA`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects partial data.

        Args:
            inertias: Body inertia tensors [kg·m²]. Shape is
                (len(env_ids), len(body_ids), 9) with dtype wp.float32.
                The 9 components are the row-major flatten of the 3×3 inertia
                matrix (Ixx, Ixy, Ixz, Iyx, Iyy, Iyz, Izx, Izy, Izz).
            body_ids: Body indices. If None, then all indices are used.
            env_ids: Environment indices. If None, then all indices are used.
        """
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)
        self.assert_shape_and_dtype(inertias, (env_ids.shape[0], body_ids.shape[0], 9), wp.float32, "inertias")
        wp.launch(
            shared_kernels.write_body_inertia_to_buffer_index,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[inertias, env_ids, body_ids],
            outputs=[self.data._body_inertia.data],
            device=self._device,
        )
        wp.copy(self.data._cpu_body_inertia, self.data._body_inertia.data)
        self._binding_write(
            TT.BODY_INERTIA,
            self.data._cpu_body_inertia,
            env_ids=self._get_cpu_env_ids(env_ids),
            device="cpu",
            data_dim=9,
        )

    def set_inertias_mask(
        self,
        *,
        inertias: wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set body inertia tensors over selected env / body masks into the simulation.

        This is a CPU-only write routed through pinned-host staging because
        ``BODY_INERTIA`` is a CPU-only OVPhysX binding.

        .. note::
            This method expects full data.

        Args:
            inertias: Body inertia tensors [kg·m²]. Shape is
                (num_instances, num_bodies, 9) with dtype wp.float32.
                The 9 components are the row-major flatten of the 3×3 inertia
                matrix (Ixx, Ixy, Ixz, Iyx, Iyy, Iyz, Izx, Izy, Izz).
            body_mask: Body mask. If None, all bodies are updated.
                Shape is (num_bodies,).
            env_mask: Environment mask. If None, all instances are updated.
                Shape is (num_instances,).
        """
        if env_mask is not None:
            env_mask_t = wp.to_torch(env_mask) if isinstance(env_mask, wp.array) else env_mask
            env_ids = self._resolve_env_ids(torch.nonzero(env_mask_t)[:, 0].to(torch.int32))
        else:
            env_ids = self._ALL_ENV_INDICES
        self.assert_shape_and_dtype(inertias, (self._num_instances, self._num_bodies, 9), wp.float32, "inertias")
        wp.launch(
            shared_kernels.write_body_inertia_to_buffer_mask,
            dim=(self._num_instances, self._num_bodies),
            inputs=[inertias, self._resolve_env_mask(env_mask), self._resolve_body_mask(body_mask)],
            outputs=[self.data._body_inertia.data],
            device=self._device,
        )
        wp.copy(self.data._cpu_body_inertia, self.data._body_inertia.data)
        self._binding_write(
            TT.BODY_INERTIA,
            self.data._cpu_body_inertia,
            env_ids=self._get_cpu_env_ids(env_ids),
            device="cpu",
            data_dim=9,
        )

    def _initialize_impl(self) -> None:
        """Initialize the rigid object collection from the OVPhysX simulation backend.

        For each body in :attr:`cfg.rigid_objects`, validates the prim tree,
        converts the IsaacLab prim path to an fnmatch glob, and eagerly creates
        a single fused :class:`TensorBinding` per tensor type using the new
        ``prim_paths=[...]`` API introduced in ovphysx 0.4.3.

        Then creates the :class:`RigidObjectCollectionData` container and primes
        the asset-side buffers.
        """
        physx_instance = OvPhysxManager.get_physx_instance()
        if physx_instance is None:
            raise RuntimeError("OvPhysxManager has not been initialized yet.")
        self._ovphysx = physx_instance
        self._device = OvPhysxManager.get_device()

        self._prim_paths: list[str] = []
        self._body_names_list: list[str] = []

        def has_rigid_body_api(prim) -> bool:
            return bool(prim.HasAPI(UsdPhysics.RigidBodyAPI))

        resolve_kwargs = {"predicate": has_rigid_body_api, "expected_num_matches": 1}
        for name, obj_cfg in self.cfg.rigid_objects.items():
            # Resolve the rigid body root expression.
            root_matches = sim_utils.resolve_matching_prims_from_source(obj_cfg.prim_path, **resolve_kwargs)
            _, root_prim_path_expr = root_matches[0]
            # IsaacLab paths may use ``.*`` regex or ``{ENV_REGEX_NS}`` placeholder; ovphysx
            # ``create_tensor_binding`` expects fnmatch globs.
            pattern = re.sub(r"\{ENV_REGEX_NS\}", "*", root_prim_path_expr)
            pattern = re.sub(r"\.\*", "*", pattern)
            self._prim_paths.append(pattern)
            self._body_names_list.append(name)

        self._num_bodies = len(self._prim_paths)

        # ovphysx 0.4.3+ accepts ``prim_paths=[g0, ..., g_{B-1}]`` and returns a single
        # binding spanning N*B prims with shape ``(N*B, D)`` in body-major order
        # ``(body0_env0, body0_env1, ..., body1_env0, ...)``. Bindings are stored under
        # the ``LINK_*``/``BODY_*`` data-class keys so the same key works with the
        # articulation-mode mock used by iface tests.
        _TT_MAP = (
            (TT.LINK_POSE, TT.RIGID_BODY_POSE),
            (TT.LINK_VELOCITY, TT.RIGID_BODY_VELOCITY),
            (TT.LINK_WRENCH, TT.RIGID_BODY_WRENCH),
            (TT.BODY_MASS, TT.RIGID_BODY_MASS),
            (TT.BODY_COM_POSE, TT.RIGID_BODY_COM_POSE),
            (TT.BODY_INERTIA, TT.RIGID_BODY_INERTIA),
        )
        for store_key, rb_tt in _TT_MAP:
            try:
                self._bindings[store_key] = self._ovphysx.create_tensor_binding(
                    prim_paths=self._prim_paths, tensor_type=rb_tt
                )
            except Exception as e:
                raise RuntimeError(
                    f"OVPhysX could not create fused RIGID_BODY binding {rb_tt!r} for"
                    f" prim_paths={self._prim_paths!r}."
                    f" Check that each prim path matches at least one"
                    f" UsdPhysics.RigidBodyAPI prim."
                ) from e

        # Native fused binding has ``count == N * num_bodies`` (body-major flat).
        pose_count = self._bindings[TT.LINK_POSE].count
        if pose_count % self._num_bodies != 0:
            raise RuntimeError(
                f"Fused LINK_POSE binding count {pose_count} is not divisible by"
                f" num_bodies {self._num_bodies}. prim_paths={self._prim_paths!r}."
            )
        self._num_instances = pose_count // self._num_bodies

        self._data = RigidObjectCollectionData(
            root_view=self._bindings,
            num_bodies=self._num_bodies,
            device=self._device,
        )

        self._create_buffers()
        self._process_cfg()
        self.update(0.0)
        self._data.is_primed = True

    def _create_buffers(self) -> None:
        """Pre-allocate asset-side index arrays and CPU staging buffers."""
        N = self._num_instances
        B = self._num_bodies

        self._ALL_ENV_INDICES = wp.array(np.arange(N), dtype=wp.int32, device=self._device)
        self._ALL_BODY_INDICES = wp.array(np.arange(B), dtype=wp.int32, device=self._device)

        # CPU copy of all-env indices used when calling CPU-only binding.write().
        self._cpu_all_env_ids = wp.zeros(N, dtype=wp.int32, device="cpu", pinned=True)
        wp.copy(self._cpu_all_env_ids, self._ALL_ENV_INDICES)

        # All-true boolean masks used as defaults in mask-based kernel calls.
        self._ALL_TRUE_ENV_MASK = wp.array(np.ones(N, dtype=bool), dtype=wp.bool, device=self._device)
        self._ALL_TRUE_BODY_MASK = wp.array(np.ones(B, dtype=bool), dtype=wp.bool, device=self._device)

        # External wrench buffer: direct (N, B, 9) contiguous allocation.
        # The fused LINK_WRENCH binding writes from a single (N, B, 9) buffer.
        self._wrench_buf = wp.zeros((N, B, 9), dtype=wp.float32, device=self._device)

        self._instantaneous_wrench_composer = WrenchComposer(self)
        self._permanent_wrench_composer = WrenchComposer(self)

        # set information about rigid body into data
        self._data.body_names = self._body_names_list

    def _process_cfg(self) -> None:
        """Post-processing of configuration parameters.

        Reads the per-body initial state from :attr:`cfg.rigid_objects` and
        broadcasts it across all environment instances to produce
        ``(num_instances, num_bodies, data_size)`` default-state arrays.
        """
        default_body_poses = []
        default_body_vels = []

        for obj_cfg in self.cfg.rigid_objects.values():
            default_body_pose = tuple(obj_cfg.init_state.pos) + tuple(obj_cfg.init_state.rot)
            default_body_vel = tuple(obj_cfg.init_state.lin_vel) + tuple(obj_cfg.init_state.ang_vel)
            # Broadcast across num_instances: (data_size,) -> (num_instances, data_size)
            default_body_pose = np.tile(np.array(default_body_pose, dtype=np.float32), (self._num_instances, 1))
            default_body_vel = np.tile(np.array(default_body_vel, dtype=np.float32), (self._num_instances, 1))
            default_body_poses.append(default_body_pose)
            default_body_vels.append(default_body_vel)

        # Stack per-body arrays: each (num_instances, data_size) -> (num_instances, num_bodies, data_size)
        default_body_poses = np.stack(default_body_poses, axis=1)
        default_body_vels = np.stack(default_body_vels, axis=1)
        self._data.default_body_pose = wp.array(default_body_poses, dtype=wp.transformf, device=self._device)
        self._data.default_body_vel = wp.array(default_body_vels, dtype=wp.spatial_vectorf, device=self._device)

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event) -> None:
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)

    """
    Helper functions.
    """

    def _get_binding(self, tensor_type: int):
        """Return the cached fused :class:`TensorBinding` for *tensor_type*.

        All bindings are eagerly created in :meth:`_initialize_impl` and stored
        under the ``TT.LINK_*`` / ``TT.BODY_*`` keys that
        :class:`RigidObjectCollectionData` uses.

        Args:
            tensor_type: The TensorType constant identifying which simulation
                buffer to bind (e.g. :attr:`~isaaclab_ovphysx.tensor_types.LINK_POSE`).

        Returns:
            The cached :class:`TensorBinding`, or ``None`` if not found.
        """
        return self._bindings.get(tensor_type)

    def reshape_data_to_view_2d(self, data: wp.array, device: str | None = None) -> wp.array:
        """Reshape instance-major ``(num_instances, num_bodies)`` data to body-major view order.

        The native fused multi-prim binding lays data out as
        ``(body0_env0, body0_env1, ..., body1_env0, body1_env1, ...)`` with shape
        ``(num_bodies * num_instances,)``.  This helper builds a strided view of the
        instance-major buffer with the transposed layout and clones it into a
        contiguous body-major flat array.

        Args:
            data: Source buffer with shape ``(num_instances, num_bodies)`` (any single-element dtype).
            device: Optional target device for the cloned output.  Defaults to ``data.device``.

        Returns:
            Contiguous body-major flat buffer with shape ``(num_bodies * num_instances,)``.
        """
        if device is None:
            device = str(data.device)
        element_size = wp.types.type_size_in_bytes(data.dtype)
        strided_view = wp.array(
            ptr=data.ptr,
            shape=(self.num_bodies, self.num_instances),
            dtype=data.dtype,
            strides=(element_size, self.num_bodies * element_size),
            device=str(data.device),
        )
        return wp.clone(strided_view, device=device).reshape((self.num_bodies * self.num_instances,))

    def reshape_data_to_view_3d(
        self, data: wp.array | torch.Tensor, data_dim: int, device: str | None = None
    ) -> wp.array | torch.Tensor:
        """Reshape instance-major ``(num_instances, num_bodies, data_dim)`` data to body-major view order.

        Companion of :meth:`reshape_data_to_view_2d` for 3D buffers (e.g. inertia
        tensors).  Output shape is ``(num_bodies * num_instances, data_dim)``.

        Args:
            data: Source buffer with shape ``(num_instances, num_bodies, data_dim)``.
                Supports Warp arrays and torch tensors.
            data_dim: Trailing per-element dimension size.
            device: Optional target device for the output. Defaults to ``data.device``.

        Returns:
            Contiguous body-major buffer with shape ``(num_bodies * num_instances, data_dim)``.
            Torch inputs return torch tensors, and Warp inputs return Warp arrays.
        """
        if isinstance(data, torch.Tensor):
            if device is None:
                device = data.device
            return data.transpose(0, 1).reshape(self.num_bodies * self.num_instances, data_dim).to(device).contiguous()

        if device is None:
            device = str(data.device)
        element_size = wp.types.type_size_in_bytes(data.dtype)
        row_size = element_size * data_dim
        strided_view = wp.array(
            ptr=data.ptr,
            shape=(self.num_bodies, self.num_instances, data_dim),
            dtype=data.dtype,
            strides=(row_size, self.num_bodies * row_size, element_size),
            device=str(data.device),
        )
        return wp.clone(strided_view, device=device).reshape((self.num_bodies * self.num_instances, data_dim))

    def _binding_write(
        self,
        tensor_type: int,
        instance_major_data: wp.array,
        env_ids: wp.array,
        device: str | None = None,
        data_dim: int | None = None,
    ) -> None:
        """Write an instance-major buffer through a fused binding.

        Dispatches to one of two paths depending on the binding's layout:

        * **Native fused binding** (``count == num_instances * num_bodies``,
          body-major flat layout): the instance-major buffer is reshaped via
          :meth:`reshape_data_to_view_2d` / :meth:`reshape_data_to_view_3d` to a
          contiguous body-major view, then written with body-major view indices
          ``view_id = body_id * num_instances + env_id``.
        * **Articulation-mode mock** (``count == num_instances``, instance-major
          ``(N, B[, D])`` shape): the buffer is written directly with the
          environment indices, matching the existing mock contract.

        Args:
            tensor_type: TensorType key identifying the cached binding.
            instance_major_data: Instance-major buffer of shape ``(N, B)`` or
                ``(N, B, data_dim)``.  May use ``wp.float32`` or a structured dtype.
            env_ids: Environment indices (1D ``wp.int32`` on ``self._device`` or
                ``"cpu"`` for CPU-only bindings).
            device: Destination device for the body-major clone (only used on the
                fused-binding path).  Defaults to ``self._device``.
            data_dim: When provided, treat the buffer as 3D and use
                :meth:`reshape_data_to_view_3d`.  When ``None`` (default), use
                :meth:`reshape_data_to_view_2d`.
        """
        binding = self._get_binding(tensor_type)
        if binding is None:
            return
        if device is None:
            device = self._device
        # Disambiguate via the binding's exposed shape: the articulation-mode
        # mock returns a directly instance-major view ``(N, B[, D])`` while the
        # native fused multi-prim binding lays elements body-major-flat with
        # ``shape == (N * B[, D])``.
        is_mock_layout = len(binding.shape) >= 2 and binding.shape[1] == self._num_bodies
        if is_mock_layout:
            float32_data = (
                instance_major_data if instance_major_data.dtype == wp.float32 else instance_major_data.view(wp.float32)
            )
            binding.write(float32_data, indices=env_ids)
            return
        # Native fused path: body-major flat (N*B[, D]); reshape and use view_ids.
        if data_dim is None:
            view = self.reshape_data_to_view_2d(instance_major_data, device=device).view(wp.float32)
        else:
            view = self.reshape_data_to_view_3d(instance_major_data, data_dim, device=device)
        view_ids = self._env_body_ids_to_view_ids(env_ids, self._ALL_BODY_INDICES, device=device)
        binding.write(view, indices=view_ids)

    """
    Internal helper.
    """

    def _resolve_env_ids(self, env_ids) -> wp.array:
        """Resolve environment indices to a warp int32 array on ``self._device``.

        Tests sometimes hand us indices on CPU even when the sim runs on GPU; we move the
        resolved array onto ``self._device`` so kernel launches don't fail on a device
        mismatch.
        """
        if env_ids is None or env_ids == slice(None):
            return self._ALL_ENV_INDICES
        if isinstance(env_ids, list):
            return wp.array(env_ids, dtype=wp.int32, device=self._device)
        if isinstance(env_ids, torch.Tensor):
            return wp.from_torch(env_ids.to(torch.int32), dtype=wp.int32)
        if isinstance(env_ids, wp.array) and str(env_ids.device) != self._device:
            env_ids = wp.clone(env_ids, device=self._device)
        return env_ids

    def _resolve_body_ids(self, body_ids) -> wp.array:
        """Resolve body indices to a warp int32 array on ``self._device``."""
        if body_ids is None or body_ids == slice(None):
            return self._ALL_BODY_INDICES
        if isinstance(body_ids, list):
            return wp.array(body_ids, dtype=wp.int32, device=self._device)
        return body_ids

    def _env_body_ids_to_view_ids(
        self, env_ids: torch.Tensor | wp.array, body_ids: torch.Tensor | wp.array, device: str = "cuda:0"
    ) -> wp.array:
        """Convert environment and body indices to flat view indices (body-major ordering).

        Computes ``view_id = body_id * num_instances + env_id`` for each
        (env_id, body_id) pair.  The output array is laid out column-major over
        the (env, body) grid, matching the PhysX ``root_view`` ordering.

        Args:
            env_ids: Environment indices.
            body_ids: Body indices.
            device: Target device for the returned array.

        Returns:
            A :class:`wp.array` of shape ``(len(env_ids) * len(body_ids),)`` with
            flat view indices on *device*.
        """
        if isinstance(env_ids, torch.Tensor):
            env_ids = wp.from_torch(env_ids.to(torch.int32), dtype=wp.int32)
        if isinstance(body_ids, torch.Tensor):
            body_ids = wp.from_torch(body_ids.to(torch.int32), dtype=wp.int32)
        if str(env_ids.device) != device:
            env_ids = wp.clone(env_ids, device=device)
        if str(body_ids.device) != device:
            body_ids = wp.clone(body_ids, device=device)
        num_query_envs = env_ids.shape[0]
        view_ids = wp.zeros(num_query_envs * body_ids.shape[0], dtype=wp.int32, device=device)
        wp.launch(
            resolve_view_ids,
            dim=(num_query_envs, body_ids.shape[0]),
            inputs=[env_ids, body_ids, num_query_envs, self.num_instances],
            outputs=[view_ids],
            device=device,
        )
        return view_ids

    def _resolve_env_mask(self, env_mask: wp.array | None) -> wp.array:
        """Resolve an environment mask to a ``wp.bool`` array on ``self._device``.

        ``None`` returns the pre-allocated all-true mask.

        Args:
            env_mask: Boolean environment mask or None. Shape is (num_instances,).

        Returns:
            A ``wp.bool`` array of shape (num_instances,) on ``self._device``.
        """
        if env_mask is None:
            return self._ALL_TRUE_ENV_MASK
        if isinstance(env_mask, torch.Tensor):
            return wp.from_torch(env_mask.to(torch.bool), dtype=wp.bool)
        if isinstance(env_mask, wp.array) and str(env_mask.device) != self._device:
            env_mask = wp.clone(env_mask, device=self._device)
        return env_mask

    def _resolve_body_mask(self, body_mask: wp.array | None) -> wp.array:
        """Resolve a body mask to a ``wp.bool`` array on ``self._device``.

        ``None`` returns the pre-allocated all-true mask.

        Args:
            body_mask: Boolean body mask or None. Shape is (num_bodies,).

        Returns:
            A ``wp.bool`` array of shape (num_bodies,) on ``self._device``.
        """
        if body_mask is None:
            return self._ALL_TRUE_BODY_MASK
        if isinstance(body_mask, torch.Tensor):
            return wp.from_torch(body_mask.to(torch.bool), dtype=wp.bool)
        if isinstance(body_mask, wp.array) and str(body_mask.device) != self._device:
            body_mask = wp.clone(body_mask, device=self._device)
        return body_mask

    def _get_cpu_env_ids(self, env_ids: wp.array) -> wp.array:
        """Return CPU int32 env indices for CPU-only binding writes.

        Uses the pre-allocated pinned ``_cpu_all_env_ids`` fast path when
        *env_ids* covers all instances, otherwise clones to CPU.

        Args:
            env_ids: A warp int32 array of environment indices on any device.

        Returns:
            A warp int32 array guaranteed to live on ``"cpu"``.
        """
        if env_ids.ptr == self._ALL_ENV_INDICES.ptr:
            return self._cpu_all_env_ids
        return wp.clone(env_ids, device="cpu")

    """
    Deprecated properties and methods.
    """

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
        if isinstance(body_states, wp.array):
            body_states = wp.to_torch(body_states)
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
        if isinstance(body_states, wp.array):
            body_states = wp.to_torch(body_states)
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
        if isinstance(body_states, wp.array):
            body_states = wp.to_torch(body_states)
        self.write_body_link_pose_to_sim_index(body_poses=body_states[:, :, :7], env_ids=env_ids, body_ids=body_ids)
        self.write_body_link_velocity_to_sim_index(
            body_velocities=body_states[:, :, 7:], env_ids=env_ids, body_ids=body_ids
        )
