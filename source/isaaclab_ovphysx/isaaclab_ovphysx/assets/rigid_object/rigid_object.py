# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVPhysX-backed RigidObject implementation."""

from __future__ import annotations

import re
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import warp as wp

from pxr import UsdPhysics

from isaaclab.assets.rigid_object.base_rigid_object import BaseRigidObject
from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from isaaclab.sim.utils.queries import resolve_matching_prims_from_source
from isaaclab.utils.string import resolve_matching_names
from isaaclab.utils.wrench_composer import WrenchComposer

from isaaclab_ovphysx import tensor_types as TT
from isaaclab_ovphysx.assets import kernels as shared_kernels
from isaaclab_ovphysx.assets.kernels import _body_wrench_to_world
from isaaclab_ovphysx.cloner import queue_ovphysx_replication
from isaaclab_ovphysx.physics import OvPhysxManager

from .rigid_object_data import RigidObjectData


class RigidObject(BaseRigidObject):
    """A rigid object asset class.

    Rigid objects are assets comprising of rigid bodies. They can be used to represent dynamic objects
    such as boxes, spheres, etc. A rigid body is described by its pose, velocity and mass distribution.

    For an asset to be considered a rigid object, the root prim of the asset must have the `USD RigidBodyAPI`_
    applied to it. This API is used to define the simulation properties of the rigid body. On playing the
    simulation, the physics engine will automatically register the rigid body and create a corresponding
    rigid body handle. State is read and written through ovphysx ``TensorBinding`` objects acquired from
    the :class:`~isaaclab_ovphysx.physics.OvPhysxManager`. Only free (non-articulated) rigid bodies are
    supported; prims under an ``ArticulationRootAPI`` should use
    :class:`~isaaclab_ovphysx.assets.articulation.Articulation` instead.

    .. _`USD RigidBodyAPI`: https://openusd.org/dev/api/class_usd_physics_rigid_body_a_p_i.html
    """

    cfg: RigidObjectCfg
    """Configuration instance for the rigid object."""

    __backend_name__: str = "ovphysx"
    """The name of the backend for the rigid object."""

    def __init__(self, cfg: RigidObjectCfg):
        """Initialize the rigid object.

        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg)
        queue_ovphysx_replication(cfg)
        # Bindings are created lazily (on first access) to avoid allocating
        # handles for tensor types the user never queries.
        self._bindings: dict[int, Any] = {}

    """
    Properties
    """

    @property
    def data(self) -> RigidObjectData:
        return self._data

    @property
    def num_instances(self) -> int:
        return self._num_instances

    @property
    def num_bodies(self) -> int:
        """Number of bodies in the asset.

        This is always 1 since each object is a single rigid body.
        """
        return self._num_bodies

    @property
    def body_names(self) -> list[str]:
        """Ordered names of bodies in the rigid object."""
        return self._body_names

    @property
    def root_view(self) -> dict[int, Any]:
        """Root view for the asset.

        OVPhysX exposes per-tensor-type bindings rather than a single opaque view object
        as used by the PhysX and Newton backends. Callers that need low-level binding
        access should call :meth:`_get_binding` rather than iterating this dict directly.
        For high-level state access (instance counts, prim paths, transforms), use the
        :attr:`num_instances`, :attr:`body_names`, and
        :attr:`~RigidObjectData.root_link_pose_w` accessors instead.

        .. note::
            Use this view with caution. It requires handling of tensors in a specific way.
        """
        return self._bindings

    @property
    def instantaneous_wrench_composer(self) -> WrenchComposer | None:
        """Instantaneous wrench composer.

        Returns a :class:`~isaaclab.utils.wrench_composer.WrenchComposer` instance. Wrenches added or set to this wrench
        composer are only valid for the current simulation step. At the end of the simulation step, the wrenches set
        to this object are discarded. This is useful to apply forces that change all the time, things like drag forces
        for instance.
        """
        return self._instantaneous_wrench_composer

    @property
    def permanent_wrench_composer(self) -> WrenchComposer | None:
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
        self, env_ids: Sequence[int] | torch.Tensor | wp.array | None = None, env_mask: wp.array | None = None
    ) -> None:
        """Reset the rigid object.

        Args:
            env_ids: Environment indices. If None, then all indices are used.
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

        poses = self._data.body_link_pose_w.warp  # (N, 1) wp.transformf
        wp.launch(
            _body_wrench_to_world,
            dim=(self._num_instances, 1),
            inputs=[force_b, torque_b, poses],
            outputs=[self._wrench_buf],
            device=self._device,
        )
        binding = self._get_binding(TT.RIGID_BODY_WRENCH)
        binding.write(self._wrench_buf_flat)
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

    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        """Find bodies in the rigid body based on the name keys.

        Please check the :meth:`isaaclab.utils.string.resolve_matching_names` function for more
        information on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the body indices and names.
        """
        return resolve_matching_names(name_keys, self._body_names, preserve_order)

    """
    Operations - Write to simulation.
    """

    def write_root_pose_to_sim_index(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        .. note::
            This method expects partial data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations. Performance is similar for both.
            However, to allow graphed pipelines, the mask method must be used.

        Args:
            root_pose: Root poses in simulation frame. Shape is (len(env_ids), 7)
                or (len(env_ids),) with dtype wp.transformf.
            env_ids: Environment indices. If None, then all indices are used.
        """
        self.write_root_link_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)

    def write_root_pose_to_sim_mask(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root pose over selected environment mask into the simulation.

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations. Performance is similar for both.
            However, to allow graphed pipelines, the mask method must be used.

        Args:
            root_pose: Root poses in simulation frame. Shape is (num_instances, 7)
                or (num_instances,) with dtype wp.transformf.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        self.write_root_link_pose_to_sim_mask(root_pose=root_pose, env_mask=env_mask)

    def write_root_velocity_to_sim_index(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root center of mass velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.

        .. note::
            This sets the velocity of the root's center of mass rather than the root's frame.

        .. note::
            This method expects partial data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations. Performance is similar for both.
            However, to allow graphed pipelines, the mask method must be used.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (len(env_ids), 6)
                or (len(env_ids),) with dtype wp.spatial_vectorf.
            env_ids: Environment indices. If None, then all indices are used.
        """
        self.write_root_com_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)

    def write_root_velocity_to_sim_mask(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root center of mass velocity over selected environment mask into the simulation.

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations. Performance is similar for both.
            However, to allow graphed pipelines, the mask method must be used.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (num_instances, 6)
                or (num_instances,) with dtype wp.spatial_vectorf.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        self.write_root_com_velocity_to_sim_mask(root_velocity=root_velocity, env_mask=env_mask)

    def write_root_link_pose_to_sim_index(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root link pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        .. note::
            This method expects partial data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations. Performance is similar for both.
            However, to allow graphed pipelines, the mask method must be used.

        Args:
            root_pose: Root link poses in simulation frame. Shape is (len(env_ids), 7)
                or (len(env_ids),) with dtype wp.transformf.
            env_ids: Environment indices. If None, then all indices are used.
        """
        env_ids = self._resolve_env_ids(env_ids)
        self.assert_shape_and_dtype(root_pose, (env_ids.shape[0],), wp.transformf, "root_pose")
        wp.launch(
            shared_kernels.set_root_link_pose_to_sim_index,
            dim=env_ids.shape[0],
            inputs=[root_pose, env_ids],
            outputs=[self.data.root_link_pose_w],
            device=self._device,
        )
        # Invalidate dependent root_com_pose timestamp so the next read recomposes it.
        self.data._root_com_pose_w.timestamp = -1.0
        # Push cache to the wheel via an indexed write.
        binding = self._get_binding(TT.RIGID_BODY_POSE)
        binding.write(self.data._root_link_pose_w.data.view(wp.float32), indices=env_ids)

    def write_root_link_pose_to_sim_mask(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root link pose over selected environment mask into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations. Performance is similar for both.
            However, to allow graphed pipelines, the mask method must be used.

        Args:
            root_pose: Root poses in simulation frame. Shape is (num_instances, 7)
                or (num_instances,) with dtype wp.transformf.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        env_mask_wp = self._resolve_env_mask(env_mask)
        self.assert_shape_and_dtype(root_pose, (self._num_instances,), wp.transformf, "root_pose")
        wp.launch(
            shared_kernels.set_root_link_pose_to_sim_mask,
            dim=self._num_instances,
            inputs=[root_pose, env_mask_wp],
            outputs=[self.data.root_link_pose_w],
            device=self._device,
        )
        self.data._root_com_pose_w.timestamp = -1.0
        binding = self._get_binding(TT.RIGID_BODY_POSE)
        binding.write(self.data._root_link_pose_w.data.view(wp.float32), mask=env_mask_wp)

    def write_root_com_pose_to_sim_index(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root center of mass pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).
        The orientation is the orientation of the principal axes of inertia.

        .. note::
            This method expects partial data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations. Performance is similar for both.
            However, to allow graphed pipelines, the mask method must be used.

        Args:
            root_pose: Root center of mass poses in simulation frame. Shape is (len(env_ids), 7)
                or (len(env_ids),) with dtype wp.transformf.
            env_ids: Environment indices. If None, then all indices are used.
        """
        env_ids = self._resolve_env_ids(env_ids)
        self.assert_shape_and_dtype(root_pose, (env_ids.shape[0],), wp.transformf, "root_pose")
        wp.launch(
            shared_kernels.set_root_com_pose_to_sim_index,
            dim=env_ids.shape[0],
            inputs=[root_pose, self.data.body_com_pose_b, env_ids],
            outputs=[self.data.root_com_pose_w, self.data.root_link_pose_w],
            device=self._device,
        )
        binding = self._get_binding(TT.RIGID_BODY_POSE)
        binding.write(self.data._root_link_pose_w.data.view(wp.float32), indices=env_ids)

    def write_root_com_pose_to_sim_mask(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root center of mass pose over selected environment mask into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).
        The orientation is the orientation of the principal axes of inertia.

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations. Performance is similar for both.
            However, to allow graphed pipelines, the mask method must be used.

        Args:
            root_pose: Root center of mass poses in simulation frame. Shape is (num_instances, 7)
                or (num_instances,) with dtype wp.transformf.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        env_mask_wp = self._resolve_env_mask(env_mask)
        self.assert_shape_and_dtype(root_pose, (self._num_instances,), wp.transformf, "root_pose")
        wp.launch(
            shared_kernels.set_root_com_pose_to_sim_mask,
            dim=self._num_instances,
            inputs=[root_pose, self.data.body_com_pose_b, env_mask_wp],
            outputs=[self.data.root_com_pose_w, self.data.root_link_pose_w],
            device=self._device,
        )
        binding = self._get_binding(TT.RIGID_BODY_POSE)
        binding.write(self.data._root_link_pose_w.data.view(wp.float32), mask=env_mask_wp)

    def write_root_com_velocity_to_sim_index(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root center of mass velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.

        .. note::
            This sets the velocity of the root's center of mass rather than the root's frame.

        .. note::
            This method expects partial data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations. Performance is similar for both.
            However, to allow graphed pipelines, the mask method must be used.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (len(env_ids), 6)
                or (len(env_ids),) with dtype wp.spatial_vectorf.
            env_ids: Environment indices. If None, then all indices are used.
        """
        env_ids = self._resolve_env_ids(env_ids)
        self.assert_shape_and_dtype(root_velocity, (env_ids.shape[0],), wp.spatial_vectorf, "root_velocity")
        wp.launch(
            shared_kernels.set_root_com_velocity_to_sim_index,
            dim=env_ids.shape[0],
            inputs=[root_velocity, env_ids, 1],
            outputs=[self.data.root_com_vel_w, self.data.body_com_acc_w],
            device=self._device,
        )
        # Invalidate dependent root_link_vel timestamp.
        self.data._root_link_vel_w.timestamp = -1.0
        binding = self._get_binding(TT.RIGID_BODY_VELOCITY)
        binding.write(self.data._root_com_vel_w.data.view(wp.float32), indices=env_ids)

    def write_root_com_velocity_to_sim_mask(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root center of mass velocity over selected environment mask into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.

        .. note::
            This sets the velocity of the root's center of mass rather than the root's frame.

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations. Performance is similar for both.
            However, to allow graphed pipelines, the mask method must be used.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (num_instances, 6)
                or (num_instances,) with dtype wp.spatial_vectorf.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        env_mask_wp = self._resolve_env_mask(env_mask)
        self.assert_shape_and_dtype(root_velocity, (self._num_instances,), wp.spatial_vectorf, "root_velocity")
        wp.launch(
            shared_kernels.set_root_com_velocity_to_sim_mask,
            dim=self._num_instances,
            inputs=[root_velocity, env_mask_wp, 1],
            outputs=[self.data.root_com_vel_w, self.data.body_com_acc_w],
            device=self._device,
        )
        self.data._root_link_vel_w.timestamp = -1.0
        binding = self._get_binding(TT.RIGID_BODY_VELOCITY)
        binding.write(self.data._root_com_vel_w.data.view(wp.float32), mask=env_mask_wp)

    def write_root_link_velocity_to_sim_index(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root link velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.

        .. note::
            This sets the velocity of the root's frame rather than the root's center of mass.

        .. note::
            This method expects partial data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations. Performance is similar for both.
            However, to allow graphed pipelines, the mask method must be used.

        Args:
            root_velocity: Root frame velocities in simulation world frame. Shape is (len(env_ids), 6)
                or (len(env_ids),) with dtype wp.spatial_vectorf.
            env_ids: Environment indices. If None, then all indices are used.
        """
        env_ids = self._resolve_env_ids(env_ids)
        self.assert_shape_and_dtype(root_velocity, (env_ids.shape[0],), wp.spatial_vectorf, "root_velocity")
        wp.launch(
            shared_kernels.set_root_link_velocity_to_sim_index,
            dim=env_ids.shape[0],
            inputs=[
                root_velocity,
                self.data.body_com_pose_b,
                self.data.root_link_pose_w,
                env_ids,
                1,  # num_bodies is always 1 for RigidObject
            ],
            outputs=[self.data.root_link_vel_w, self.data.root_com_vel_w, self.data.body_com_acc_w],
            device=self._device,
        )
        binding = self._get_binding(TT.RIGID_BODY_VELOCITY)
        binding.write(self.data._root_com_vel_w.data.view(wp.float32), indices=env_ids)

    def write_root_link_velocity_to_sim_mask(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root link velocity over selected environment mask into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.

        .. note::
            This sets the velocity of the root's frame rather than the root's center of mass.

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations. Performance is similar for both.
            However, to allow graphed pipelines, the mask method must be used.

        Args:
            root_velocity: Root frame velocities in simulation world frame. Shape is (num_instances, 6)
                or (num_instances,) with dtype wp.spatial_vectorf.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        env_mask_wp = self._resolve_env_mask(env_mask)
        self.assert_shape_and_dtype(root_velocity, (self._num_instances,), wp.spatial_vectorf, "root_velocity")
        wp.launch(
            shared_kernels.set_root_link_velocity_to_sim_mask,
            dim=self._num_instances,
            inputs=[root_velocity, self.data.body_com_pose_b, self.data.root_link_pose_w, env_mask_wp, 1],
            outputs=[self.data.root_link_vel_w, self.data.root_com_vel_w, self.data.body_com_acc_w],
            device=self._device,
        )
        binding = self._get_binding(TT.RIGID_BODY_VELOCITY)
        binding.write(self.data._root_com_vel_w.data.view(wp.float32), mask=env_mask_wp)

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
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)
        self.assert_shape_and_dtype(masses, (env_ids.shape[0], body_ids.shape[0]), wp.float32, "masses")
        # Scatter user data into the cached _body_mass at (env_ids, body_ids).
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_indices,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[masses, env_ids, body_ids],
            outputs=[self.data._body_mass],
            device=self._device,
        )
        # Push cache to the wheel via pinned-CPU staging (RIGID_BODY_MASS is CPU-only).
        cpu_env_ids = self._get_cpu_env_ids(env_ids)
        wp.copy(self._cpu_body_mass, self.data._body_mass)
        binding = self._get_binding(TT.RIGID_BODY_MASS)
        binding.write(self._cpu_body_mass.flatten(), indices=cpu_env_ids)

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
        env_mask_wp = self._resolve_env_mask(env_mask)
        body_mask_wp = self._resolve_body_mask(body_mask)
        self.assert_shape_and_dtype(masses, (self._num_instances, self._num_bodies), wp.float32, "masses")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_mask,
            dim=(self._num_instances, self._num_bodies),
            inputs=[masses, env_mask_wp, body_mask_wp],
            outputs=[self.data._body_mass],
            device=self._device,
        )
        wp.copy(self._cpu_body_mass, self.data._body_mass)
        binding = self._get_binding(TT.RIGID_BODY_MASS)
        binding.write(self._cpu_body_mass.flatten(), mask=self._get_cpu_env_mask(env_mask_wp))

    def set_coms_index(
        self,
        *,
        coms: torch.Tensor | wp.array,
        body_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set center of mass pose of all bodies using indices.

        .. note::
            This method expects partial data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations. Performance is similar for both.
            However, to allow graphed pipelines, the mask method must be used.

        Args:
            coms: Center of mass pose of all bodies. Shape is (len(env_ids), len(body_ids), 7).
            body_ids: The body indices to set the center of mass pose for. Defaults to None (all bodies).
            env_ids: The environment indices to set the center of mass pose for. Defaults to None
                (all environments).
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
        # Invalidate dependent root_com_pose timestamp -- it's derived from body_com_pose_b.
        self.data._root_com_pose_w.timestamp = -1.0
        # Push cache to the wheel via pinned-CPU staging (RIGID_BODY_COM_POSE is CPU-only).
        cpu_env_ids = self._get_cpu_env_ids(env_ids)
        wp.copy(self._cpu_body_coms, self.data._body_com_pose_b.data)
        binding = self._get_binding(TT.RIGID_BODY_COM_POSE)
        # Wheel binding shape is (N, 7); squeeze singleton body dim with a flat float32 view.
        binding.write(self._cpu_body_coms.reshape((self._num_instances, 7)), indices=cpu_env_ids)

    def set_coms_mask(
        self,
        *,
        coms: torch.Tensor | wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set center of mass pose of all bodies using masks.

        .. note::
            This method expects full data.

        .. tip::
            Both the index and mask methods have dedicated optimized implementations. Performance is similar for both.
            However, to allow graphed pipelines, the mask method must be used.

        Args:
            coms: Center of mass pose of all bodies. Shape is (num_instances, num_bodies, 7).
            body_mask: Body mask. If None, then all bodies are used.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        env_mask_wp = self._resolve_env_mask(env_mask)
        body_mask_wp = self._resolve_body_mask(body_mask)
        self.assert_shape_and_dtype(coms, (self._num_instances, self._num_bodies), wp.transformf, "coms")
        wp.launch(
            shared_kernels.write_body_com_pose_to_buffer_mask,
            dim=(self._num_instances, self._num_bodies),
            inputs=[coms, env_mask_wp, body_mask_wp],
            outputs=[self.data._body_com_pose_b.data],
            device=self._device,
        )
        self.data._root_com_pose_w.timestamp = -1.0
        wp.copy(self._cpu_body_coms, self.data._body_com_pose_b.data)
        binding = self._get_binding(TT.RIGID_BODY_COM_POSE)
        binding.write(self._cpu_body_coms.reshape((self._num_instances, 7)), mask=self._get_cpu_env_mask(env_mask_wp))

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
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)
        self.assert_shape_and_dtype(inertias, (env_ids.shape[0], body_ids.shape[0], 9), wp.float32, "inertias")
        wp.launch(
            shared_kernels.write_body_inertia_to_buffer_index,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[inertias, env_ids, body_ids],
            outputs=[self.data._body_inertia],
            device=self._device,
        )
        # Push cache to the wheel via pinned-CPU staging (RIGID_BODY_INERTIA is CPU-only).
        cpu_env_ids = self._get_cpu_env_ids(env_ids)
        wp.copy(self._cpu_body_inertia, self.data._body_inertia)
        binding = self._get_binding(TT.RIGID_BODY_INERTIA)
        # Wheel binding shape is (N, 9); flatten the singleton body dim.
        binding.write(self._cpu_body_inertia.reshape((self._num_instances, 9)), indices=cpu_env_ids)

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
        env_mask_wp = self._resolve_env_mask(env_mask)
        body_mask_wp = self._resolve_body_mask(body_mask)
        self.assert_shape_and_dtype(inertias, (self._num_instances, self._num_bodies, 9), wp.float32, "inertias")
        wp.launch(
            shared_kernels.write_body_inertia_to_buffer_mask,
            dim=(self._num_instances, self._num_bodies),
            inputs=[inertias, env_mask_wp, body_mask_wp],
            outputs=[self.data._body_inertia],
            device=self._device,
        )
        wp.copy(self._cpu_body_inertia, self.data._body_inertia)
        binding = self._get_binding(TT.RIGID_BODY_INERTIA)
        binding.write(
            self._cpu_body_inertia.reshape((self._num_instances, 9)), mask=self._get_cpu_env_mask(env_mask_wp)
        )

    """
    Internal helper.
    """

    def _initialize_impl(self) -> None:
        # acquire ovphysx instance
        physx_instance = OvPhysxManager.get_physx_instance()
        if physx_instance is None:
            raise RuntimeError("OvPhysxManager has not been initialized yet.")
        self._ovphysx = physx_instance
        # Derive the device from PhysicsManager (which mirrors SimulationContext.cfg.device).
        # The ovphysx PhysX object does not expose a .device property; reading it would
        # raise AttributeError (masked by hasattr) and fall back to "cuda:0" even when the
        # simulation is running on CPU, causing a device mismatch in binding.read().
        self._device = OvPhysxManager.get_device()

        # Resolve the rigid body root expression.
        def has_rigid_body_api(prim) -> bool:
            return bool(prim.HasAPI(UsdPhysics.RigidBodyAPI))

        resolve_kwargs = {"predicate": has_rigid_body_api, "expected_num_matches": 1}
        _, root_prim_path_expr = resolve_matching_prims_from_source(self.cfg.prim_path, **resolve_kwargs)[0]

        # IsaacLab paths may use ``.*`` regex or ``{ENV_REGEX_NS}`` placeholder; ovphysx
        # ``create_tensor_binding`` expects fnmatch globs.
        pattern = re.sub(r"\{ENV_REGEX_NS\}", "*", root_prim_path_expr)
        pattern = re.sub(r"\.\*", "*", pattern)
        self._binding_pattern = pattern

        # Eagerly create every binding the data container reads at init, so failures
        # surface here with a helpful message rather than as a raw wheel exception
        # (or a KeyError) at first writer call.
        for tt in (
            TT.RIGID_BODY_POSE,
            TT.RIGID_BODY_VELOCITY,
            TT.RIGID_BODY_WRENCH,
            TT.RIGID_BODY_MASS,
            TT.RIGID_BODY_COM_POSE,
            TT.RIGID_BODY_INERTIA,
        ):
            try:
                self._get_binding(tt)
            except Exception as e:
                raise RuntimeError(
                    f"OVPhysX could not create rigid-body binding {tt!r}. "
                    f"Check that prim_path={self._binding_pattern!r} matches "
                    f"at least one UsdPhysics.RigidBodyAPI prim and that the "
                    f"ovphysx wheel exposes the RIGID_BODY_* TensorType. "
                    f"Note: pattern resolution may currently include articulation "
                    f"links; an explicit selection policy is on the wheel-side roadmap."
                ) from e

        # read counts and body names from the root-pose binding
        root_pose = self._bindings[TT.RIGID_BODY_POSE]
        self._num_instances = root_pose.count
        self._num_bodies = 1
        try:
            body_names_value = root_pose.body_names
            # body_names may be an empty list for non-articulation bindings; fall back to
            # the documented single-body default in that case.
            self._body_names = list(body_names_value) if body_names_value else ["base_link"]
        except (AttributeError, TypeError):
            # ovphysx TensorBinding raises TypeError (not AttributeError) when body_names
            # is queried on a non-articulation tensor type such as RIGID_BODY_POSE:
            # "Articulation metadata … is not available for tensor type 'RIGID_BODY_POSE'."
            # For a single-body rigid object the default ["base_link"] is always correct.
            self._body_names = ["base_link"]

        # container for data access
        self._data = RigidObjectData(self._bindings, self._device, check_shapes=self._check_shapes)

        # create buffers
        self._create_buffers()
        # process configuration
        self._process_cfg()
        # update the rigid body data
        self.update(0.0)
        # Let the rigid object data know that it is fully instantiated and ready to use.
        self._data.is_primed = True

    def _create_buffers(self) -> None:
        """Create buffers for storing data."""
        N = self._num_instances
        B = 1  # rigid object always has a single body
        device = self._device

        # constants
        self._ALL_INDICES = wp.array(np.arange(N, dtype=np.int32), device=device)
        self._ALL_BODY_INDICES = wp.array(np.arange(B, dtype=np.int32), device=device)
        # All-true masks for default mask paths. These let ``binding.write(..., mask=...)``
        # cover all instances when no env_mask is supplied, without converting back to indices.
        self._ALL_TRUE_ENV_MASK = wp.array(np.ones(N, dtype=bool), dtype=wp.bool, device=device)
        self._ALL_TRUE_BODY_MASK = wp.array(np.ones(B, dtype=bool), dtype=wp.bool, device=device)

        # external wrench composer
        # The kernel writes into the (N, 1, 9) view; the binding consumes the (N, 9) view --
        # both alias the same allocation, so we cache the flat reshape once.
        self._wrench_buf = wp.zeros((N, 1, 9), dtype=wp.float32, device=device)
        self._wrench_buf_flat = wp.array(
            ptr=self._wrench_buf.ptr,
            shape=(N, 9),
            dtype=wp.float32,
            device=device,
            copy=False,
        )
        self._instantaneous_wrench_composer = WrenchComposer(self)
        self._permanent_wrench_composer = WrenchComposer(self)

        # set information about rigid body into data
        self._data.body_names = self._body_names

        # Pre-allocated pinned CPU buffers for OVPhysX TensorBinding writes. The wheel
        # requires CPU arrays for "model" property updates (mass / coms / inertia); pinned
        # host memory enables DMA fast path and avoids per-call ``wp.clone`` allocation.
        self._cpu_env_ids_all = wp.zeros(N, dtype=wp.int32, device="cpu", pinned=True)
        wp.copy(self._cpu_env_ids_all, self._ALL_INDICES)
        self._cpu_body_mass = wp.zeros((N, B), dtype=wp.float32, device="cpu", pinned=True)
        self._cpu_body_coms = wp.zeros((N, B, 7), dtype=wp.float32, device="cpu", pinned=True)
        self._cpu_body_inertia = wp.zeros((N, B, 9), dtype=wp.float32, device="cpu", pinned=True)
        # Pinned-host mask staging for CPU-only binding writes (mass/coms/inertia).
        self._cpu_env_mask = wp.zeros(N, dtype=wp.bool, device="cpu", pinned=True)

    def _process_cfg(self) -> None:
        """Post processing of configuration parameters."""
        # default state
        # -- root state
        # note: we cast to tuple to avoid torch/numpy type mismatch.
        default_root_pose = tuple(self.cfg.init_state.pos) + tuple(self.cfg.init_state.rot)
        default_root_vel = tuple(self.cfg.init_state.lin_vel) + tuple(self.cfg.init_state.ang_vel)
        default_root_pose = np.tile(np.array(default_root_pose, dtype=np.float32), (self._num_instances, 1))
        default_root_vel = np.tile(np.array(default_root_vel, dtype=np.float32), (self._num_instances, 1))
        self._data.default_root_pose = wp.array(default_root_pose, dtype=wp.transformf, device=self._device)
        self._data.default_root_vel = wp.array(default_root_vel, dtype=wp.spatial_vectorf, device=self._device)

    def _resolve_env_ids(self, env_ids: Sequence[int] | torch.Tensor | wp.array | None) -> wp.array:
        """Resolve environment indices to a warp array.

        Args:
            env_ids: Environment indices. If None, then all indices are used.

        Returns:
            A warp array of environment indices on ``self._device``.
        """
        if env_ids is None or env_ids == slice(None):
            return self._ALL_INDICES
        if isinstance(env_ids, list):
            return wp.array(env_ids, dtype=wp.int32, device=self._device)
        if isinstance(env_ids, torch.Tensor):
            return wp.from_torch(env_ids.to(torch.int32), dtype=wp.int32)
        if isinstance(env_ids, wp.array) and str(env_ids.device) != self._device:
            env_ids = wp.clone(env_ids, device=self._device)
        return env_ids

    def _resolve_body_ids(self, body_ids: Sequence[int] | torch.Tensor | wp.array | None) -> wp.array:
        """Resolve body indices to a warp array.

        Args:
            body_ids: Body indices. If None, then all indices are used.

        Returns:
            A warp array of body indices on ``self._device``.
        """
        if body_ids is None or body_ids == slice(None):
            return self._ALL_BODY_INDICES
        if isinstance(body_ids, list):
            return wp.array(body_ids, dtype=wp.int32, device=self._device)
        return body_ids

    def _resolve_env_mask(self, env_mask: wp.array | None) -> wp.array:
        """Resolve an environment mask to a ``wp.bool`` array.

        Args:
            env_mask: Environment mask. If None, then the pre-allocated all-true mask is used.

        Returns:
            A warp ``wp.bool`` array on ``self._device``.
        """
        if env_mask is None:
            return self._ALL_TRUE_ENV_MASK
        if isinstance(env_mask, torch.Tensor):
            return wp.from_torch(env_mask.to(torch.bool), dtype=wp.bool)
        if isinstance(env_mask, wp.array) and str(env_mask.device) != self._device:
            env_mask = wp.clone(env_mask, device=self._device)
        return env_mask

    def _resolve_body_mask(self, body_mask: wp.array | None) -> wp.array:
        """Resolve a body mask to a ``wp.bool`` array.

        Args:
            body_mask: Body mask. If None, then the pre-allocated all-true mask is used.

        Returns:
            A warp ``wp.bool`` array on ``self._device``.
        """
        if body_mask is None:
            return self._ALL_TRUE_BODY_MASK
        if isinstance(body_mask, torch.Tensor):
            return wp.from_torch(body_mask.to(torch.bool), dtype=wp.bool)
        if isinstance(body_mask, wp.array) and str(body_mask.device) != self._device:
            body_mask = wp.clone(body_mask, device=self._device)
        return body_mask

    def _get_cpu_env_mask(self, env_mask: wp.array) -> wp.array:
        """Return a pinned-host CPU copy of *env_mask* for a CPU-only binding write.

        The wheel's ``binding.write(mask=...)`` requires the mask on the binding's
        device, which is CPU for mass / coms / inertia. Reuses the pre-allocated
        ``_cpu_env_mask`` pinned buffer.
        """
        wp.copy(self._cpu_env_mask, env_mask)
        return self._cpu_env_mask

    def _get_cpu_env_ids(self, env_ids: wp.array | torch.Tensor) -> wp.array:
        """Return CPU int32 indices, using the pre-allocated pinned ``_cpu_env_ids_all``
        fast path when *env_ids* matches ``_ALL_INDICES``.
        """
        if isinstance(env_ids, torch.Tensor):
            env_ids = wp.from_torch(env_ids, dtype=wp.int32)
        if env_ids.ptr == self._ALL_INDICES.ptr:
            return self._cpu_env_ids_all
        return wp.clone(env_ids, device="cpu")

    def _get_binding(self, tensor_type: int):
        """Return a cached TensorBinding, creating it on first access.

        Bindings are lightweight handles (a pointer + shape metadata into PhysX's
        shared GPU buffer). Creating one does NOT allocate new GPU memory -- the
        underlying simulation buffers are allocated once by PhysX regardless of how
        many bindings point into them. Still, we defer creation so that tensor types
        the user never queries are never looked up.

        Args:
            tensor_type: The TensorType constant identifying which simulation buffer
                to bind (e.g. :attr:`~isaaclab_ovphysx.tensor_types.RIGID_BODY_POSE`).

        Returns:
            The cached TensorBinding for ``tensor_type``.
        """
        binding = self._bindings.get(tensor_type)
        if binding is not None:
            return binding
        binding = self._ovphysx.create_tensor_binding(pattern=self._binding_pattern, tensor_type=tensor_type)
        self._bindings[tensor_type] = binding
        return binding

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event) -> None:
        """Invalidates the scene elements."""
        super()._invalidate_initialize_callback(event)

    def write_root_state_to_sim(
        self,
        root_state: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_root_link_pose_to_sim_index` and
        :meth:`write_root_com_velocity_to_sim_index`."""
        warnings.warn(
            "The function 'write_root_state_to_sim' will be deprecated in a future release. Please"
            " use 'write_root_link_pose_to_sim_index' and 'write_root_com_velocity_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_root_link_pose_to_sim_index(root_pose=root_state[:, :7], env_ids=env_ids)
        self.write_root_com_velocity_to_sim_index(root_velocity=root_state[:, 7:], env_ids=env_ids)

    def write_root_com_state_to_sim(
        self,
        root_state: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_root_com_pose_to_sim_index` and
        :meth:`write_root_com_velocity_to_sim_index`."""
        warnings.warn(
            "The function 'write_root_com_state_to_sim' will be deprecated in a future release. Please"
            " use 'write_root_com_pose_to_sim_index' and 'write_root_com_velocity_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_root_com_pose_to_sim_index(root_pose=root_state[:, :7], env_ids=env_ids)
        self.write_root_com_velocity_to_sim_index(root_velocity=root_state[:, 7:], env_ids=env_ids)

    def write_root_link_state_to_sim(
        self,
        root_state: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_root_link_pose_to_sim_index` and
        :meth:`write_root_link_velocity_to_sim_index`."""
        warnings.warn(
            "The function 'write_root_link_state_to_sim' will be deprecated in a future release. Please"
            " use 'write_root_link_pose_to_sim_index' and 'write_root_link_velocity_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_root_link_pose_to_sim_index(root_pose=root_state[:, :7], env_ids=env_ids)
        self.write_root_link_velocity_to_sim_index(root_velocity=root_state[:, 7:], env_ids=env_ids)
