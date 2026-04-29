# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVPhysX-backed RigidObject implementation."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import warp as wp

from pxr import UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets.rigid_object.base_rigid_object import BaseRigidObject
from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from isaaclab.utils.string import resolve_matching_names
from isaaclab.utils.wrench_composer import WrenchComposer

from isaaclab_ovphysx import tensor_types as TT
from isaaclab_ovphysx.assets import kernels as shared_kernels
from isaaclab_ovphysx.assets.kernels import (  # noqa: F401
    _body_wrench_to_world,
    _compose_root_link_pose_from_com,
    _scatter_rows_partial,
)
from isaaclab_ovphysx.physics import OvPhysxManager

from .rigid_object_data import RigidObjectData

logger = logging.getLogger(__name__)


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
        # Bindings are created lazily (on first access) to avoid allocating
        # handles for tensor types the user never queries.
        self._bindings: dict[int, Any] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

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
        # Reshape (N, 1, 9) → (N, 9) zero-copy for the binding write.
        flat_view = wp.array(
            ptr=self._wrench_buf.ptr,
            shape=(self._num_instances, 9),
            dtype=wp.float32,
            device=self._device,
            copy=False,
        )
        binding = self._get_binding(TT.RIGID_BODY_WRENCH)
        if binding is not None:
            binding.write(flat_view)
        inst.reset()

    def update(self, dt: float) -> None:
        """Updates the simulation data.

        Args:
            dt: The time step size in seconds.
        """
        self._data.update(dt)

    # ------------------------------------------------------------------
    # Operations - Finders
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Operations - Write to simulation
    # ------------------------------------------------------------------

    def write_root_pose_to_sim_index(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root pose over selected environment indices (alias for link pose; mirrors PhysX)."""
        self.write_root_link_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)

    def write_root_pose_to_sim_mask(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root pose over selected environment mask (alias for link pose; mirrors PhysX)."""
        self.write_root_link_pose_to_sim_mask(root_pose=root_pose, env_mask=env_mask)

    def write_root_velocity_to_sim_index(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root velocity over selected environment indices (alias for COM velocity; mirrors PhysX)."""
        self.write_root_com_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)

    def write_root_velocity_to_sim_mask(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root velocity over selected environment mask (alias for COM velocity; mirrors PhysX)."""
        self.write_root_com_velocity_to_sim_mask(root_velocity=root_velocity, env_mask=env_mask)

    def write_root_link_pose_to_sim_index(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set the root link pose into the simulation. Mirrors PhysX:
        scatter into the cached ``root_link_pose_w`` buffer, then push it to the
        ``RIGID_BODY_POSE`` binding via an indexed write.
        """
        env_ids = self._resolve_env_ids(env_ids)
        if full_data:
            self.assert_shape_and_dtype(root_pose, (self._num_instances,), wp.transformf, "root_pose")
        else:
            self.assert_shape_and_dtype(root_pose, (env_ids.shape[0],), wp.transformf, "root_pose")
        wp.launch(
            shared_kernels.set_root_link_pose_to_sim,
            dim=env_ids.shape[0],
            inputs=[root_pose, env_ids, full_data],
            outputs=[
                self.data.root_link_pose_w,
                None,  # self.data._root_link_state_w.data,
                None,  # self.data._root_state_w.data,
            ],
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
        """Set the root link pose using a native mask (Newton-style).

        Scatters ``root_pose`` into the cached ``root_link_pose_w`` only where ``env_mask[i]``
        is True, then pushes the cache to the ``RIGID_BODY_POSE`` binding via the wheel's
        native ``binding.write(mask=...)`` -- no ``torch.nonzero`` round-trip.
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
        full_data: bool = False,
    ) -> None:
        """Set the root COM pose into the simulation (mirrors PhysX).

        The kernel scatters the user COM pose into ``root_com_pose_w`` and derives the
        equivalent ``root_link_pose_w`` from the body-frame COM offset; the latter is
        what we push to the ``RIGID_BODY_POSE`` binding.
        """
        env_ids = self._resolve_env_ids(env_ids)
        if full_data:
            self.assert_shape_and_dtype(root_pose, (self._num_instances,), wp.transformf, "root_pose")
        else:
            self.assert_shape_and_dtype(root_pose, (env_ids.shape[0],), wp.transformf, "root_pose")
        wp.launch(
            shared_kernels.set_root_com_pose_to_sim,
            dim=env_ids.shape[0],
            inputs=[root_pose, self.data.body_com_pose_b, env_ids, full_data],
            outputs=[
                self.data.root_com_pose_w,
                self.data.root_link_pose_w,
                None,  # self.data._root_com_state_w.data,
                None,  # self.data._root_link_state_w.data,
                None,  # self.data._root_state_w.data,
            ],
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
        """Set the root COM pose using a native mask (Newton-style)."""
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
        full_data: bool = False,
    ) -> None:
        """Set the root COM velocity into the simulation (mirrors PhysX)."""
        env_ids = self._resolve_env_ids(env_ids)
        if full_data:
            self.assert_shape_and_dtype(root_velocity, (self._num_instances,), wp.spatial_vectorf, "root_velocity")
        else:
            self.assert_shape_and_dtype(root_velocity, (env_ids.shape[0],), wp.spatial_vectorf, "root_velocity")
        wp.launch(
            shared_kernels.set_root_com_velocity_to_sim,
            dim=env_ids.shape[0],
            inputs=[root_velocity, env_ids, 1, full_data],
            outputs=[
                self.data.root_com_vel_w,
                self.data.body_com_acc_w,
                None,  # self.data._root_state_w.data,
                None,  # self.data._root_com_state_w.data,
            ],
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
        """Set the root COM velocity using a native mask (Newton-style)."""
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
        full_data: bool = False,
    ) -> None:
        """Set the root link velocity into the simulation (mirrors PhysX).

        The kernel converts user link velocity to COM velocity via the lever-arm transform
        and writes both into the data caches; we push the COM velocity to the binding.
        """
        env_ids = self._resolve_env_ids(env_ids)
        if full_data:
            self.assert_shape_and_dtype(root_velocity, (self._num_instances,), wp.spatial_vectorf, "root_velocity")
        else:
            self.assert_shape_and_dtype(root_velocity, (env_ids.shape[0],), wp.spatial_vectorf, "root_velocity")
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
                self.data.root_link_vel_w,
                self.data.root_com_vel_w,
                self.data.body_com_acc_w,
                None,  # self.data._root_link_state_w.data,
                None,  # self.data._root_state_w.data,
                None,  # self.data._root_com_state_w.data,
            ],
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
        """Set the root link velocity using a native mask (Newton-style)."""
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

    # ------------------------------------------------------------------
    # Operations - Setters
    # ------------------------------------------------------------------

    def set_masses_index(
        self,
        *,
        masses: torch.Tensor | wp.array,
        body_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set masses of all bodies using indices.

        Mirrors :meth:`isaaclab_physx.assets.RigidObject.set_masses_index`: scatter the
        user-provided rows into the cached ``_body_mass`` buffer, then push the (now
        consistent) cache to the ``RIGID_BODY_MASS`` binding via an indexed write.
        The cache is the single source of truth -- no separate invalidation needed.

        Args:
            masses: Masses of all bodies. Shape is (len(env_ids), len(body_ids)) or
                (num_instances, num_bodies) if ``full_data``.
            body_ids: The body indices to set the masses for. Defaults to None (all bodies).
            env_ids: The environment indices to set the masses for. Defaults to None
                (all environments).
            full_data: Whether ``masses`` covers all instances. Defaults to False.
        """
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)
        # Normalise (K,) input from single-body callers to (K, 1) so the 2-D scatter kernel works.
        if hasattr(masses, "shape") and len(masses.shape) == 1:
            if isinstance(masses, torch.Tensor):
                masses = masses.unsqueeze(-1)
            else:
                masses = wp.array(
                    ptr=masses.ptr, shape=(masses.shape[0], 1), dtype=wp.float32, device=str(masses.device), copy=False
                )
        # Scatter user data into the cached _body_mass at (env_ids, body_ids).
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_indices,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[masses, env_ids, body_ids, full_data],
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
        """Set masses of all bodies using a native mask (Newton-style)."""
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
        full_data: bool = False,
    ) -> None:
        """Set center of mass pose of all bodies using indices (mirrors PhysX).

        Args:
            coms: Center of mass pose of all bodies. Shape is (len(env_ids), len(body_ids), 7) or
                (num_instances, num_bodies, 7) if ``full_data``.
            body_ids: The body indices to set the center of mass pose for. Defaults to None (all bodies).
            env_ids: The environment indices to set the center of mass pose for. Defaults to None
                (all environments).
            full_data: Whether to expect full data. Defaults to False.
        """
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)
        if full_data:
            self.assert_shape_and_dtype(coms, (self._num_instances, self._num_bodies), wp.transformf, "coms")
        else:
            self.assert_shape_and_dtype(coms, (env_ids.shape[0], body_ids.shape[0]), wp.transformf, "coms")
        wp.launch(
            shared_kernels.write_body_com_pose_to_buffer,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[coms, env_ids, body_ids, full_data],
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
        """Set center of mass pose using a native mask (Newton-style)."""
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
        full_data: bool = False,
    ) -> None:
        """Set inertias of all bodies using indices (mirrors PhysX).

        Args:
            inertias: Inertias of all bodies. Shape is (len(env_ids), len(body_ids), 9) or
                (num_instances, num_bodies, 9) if ``full_data``.
            body_ids: The body indices to set the inertias for. Defaults to None (all bodies).
            env_ids: The environment indices to set the inertias for. Defaults to None (all environments).
            full_data: Whether to expect full data. Defaults to False.
        """
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)
        if full_data:
            self.assert_shape_and_dtype(inertias, (self._num_instances, self._num_bodies, 9), wp.float32, "inertias")
        else:
            self.assert_shape_and_dtype(inertias, (env_ids.shape[0], body_ids.shape[0], 9), wp.float32, "inertias")
        wp.launch(
            shared_kernels.write_body_inertia_to_buffer,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[inertias, env_ids, self._ALL_BODY_INDICES, full_data],
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
        """Set inertias using a native mask (Newton-style)."""
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

    # ------------------------------------------------------------------
    # Deprecated writers
    # ------------------------------------------------------------------

    def write_root_state_to_sim(
        self,
        root_state: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_root_link_pose_to_sim_index` and
        :meth:`write_root_com_velocity_to_sim_index`."""
        import warnings

        warnings.warn(
            "The function 'write_root_state_to_sim' will be deprecated in a future release. Please"
            " use 'write_root_link_pose_to_sim_index' and 'write_root_com_velocity_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._write_body_state(TT.RIGID_BODY_POSE, root_state[..., :7], env_ids)
        self._write_body_state(TT.RIGID_BODY_VELOCITY, root_state[..., 7:], env_ids)

    def write_root_com_state_to_sim(
        self,
        root_state: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_root_com_pose_to_sim_index` and
        :meth:`write_root_com_velocity_to_sim_index`."""
        import warnings

        warnings.warn(
            "The function 'write_root_com_state_to_sim' will be deprecated in a future release. Please"
            " use 'write_root_com_pose_to_sim_index' and 'write_root_com_velocity_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        link_pose = self._com_pose_to_link_pose(root_state[..., :7])
        self._write_body_state(TT.RIGID_BODY_POSE, link_pose, env_ids)
        self._write_body_state(TT.RIGID_BODY_VELOCITY, root_state[..., 7:], env_ids)

    def write_root_link_state_to_sim(
        self,
        root_state: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_root_link_pose_to_sim_index` and
        :meth:`write_root_link_velocity_to_sim_index`."""
        import warnings

        warnings.warn(
            "The function 'write_root_link_state_to_sim' will be deprecated in a future release. Please"
            " use 'write_root_link_pose_to_sim_index' and 'write_root_link_velocity_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._write_body_state(TT.RIGID_BODY_POSE, root_state[..., :7], env_ids)
        self._write_body_state(TT.RIGID_BODY_VELOCITY, root_state[..., 7:], env_ids)

    # ------------------------------------------------------------------
    # Internal helpers -- Write
    # ------------------------------------------------------------------

    def _n_envs_index(self, env_ids) -> int:
        """Return the number of environments from an env_ids argument."""
        if env_ids is None:
            return self._num_instances
        if isinstance(env_ids, (list, tuple)):
            return len(env_ids)
        return env_ids.shape[0] if hasattr(env_ids, "shape") else len(env_ids)

    def _to_flat_f32(self, data, target_shape: tuple[int, ...] | None = None) -> wp.array | np.ndarray:
        """Ensure data is a contiguous float32 tensor suitable for binding I/O.

        State tensor bindings (positions, velocities, poses) live on the
        simulation device (GPU in GPU mode).  We always return data on
        ``self._device`` so the binding device check passes.

        For structured warp dtypes (``transformf``, ``spatial_vectorf``, etc.) a
        zero-copy flat float32 view is created instead of roundtripping through
        CPU numpy.

        Args:
            data: Input data as a warp array, torch tensor, numpy array, or scalar.
            target_shape: Optional expected shape for validation (unused; reserved for future use).

        Returns:
            A float32 warp array on ``self._device``.
        """
        dev = self._device
        if isinstance(data, wp.array):
            if str(data.device) != dev:
                data = wp.clone(data, device=dev)
            if data.dtype == wp.float32:
                return data
            # Structured dtype: zero-copy flat float32 view.
            # transformf -> [N, 7], spatial_vectorf -> [N, 6], etc.
            floats_per_elem = data.strides[0] // 4
            return wp.array(
                ptr=data.ptr,
                shape=(data.shape[0], floats_per_elem),
                dtype=wp.float32,
                device=dev,
                copy=False,
            )
        elif isinstance(data, torch.Tensor):
            if data.is_cuda and dev.startswith("cuda"):
                return wp.from_torch(data.detach().contiguous().float())
            np_data = data.detach().cpu().numpy().astype(np.float32)
            return wp.from_numpy(np_data, dtype=wp.float32, device=dev)
        elif isinstance(data, np.ndarray):
            return wp.from_numpy(data.astype(np.float32), dtype=wp.float32, device=dev)
        elif isinstance(data, (int, float)):
            return wp.from_numpy(np.array(data, dtype=np.float32), dtype=wp.float32, device=dev)
        return wp.from_numpy(np.asarray(data, dtype=np.float32), dtype=wp.float32, device=dev)

    def _as_gpu_f32_2d(self, data, cols: int) -> wp.array:
        """View/convert data as 2-D ``[rows, cols]`` float32 on ``self._device``.

        For warp arrays with structured dtypes (``transformf``, ``spatial_vectorf``),
        creates a zero-copy flat float32 view.  For torch/numpy, converts to warp
        on the simulation device.

        Args:
            data: Input data.
            cols: Number of float32 columns per row.

        Returns:
            A 2-D float32 warp array on ``self._device``.
        """
        dev = self._device
        if isinstance(data, wp.array):
            if str(data.device) != dev:
                data = wp.clone(data, device=dev)
            if data.dtype == wp.float32 and data.ndim == 2:
                return data
            n = data.shape[0]
            return wp.array(
                ptr=data.ptr,
                shape=(n, cols),
                dtype=wp.float32,
                device=dev,
                copy=False,
            )
        if isinstance(data, torch.Tensor) and data.is_cuda and dev.startswith("cuda"):
            return wp.from_torch(data.detach().contiguous().float().reshape(-1, cols))
        np_data = self._to_cpu_numpy(data).reshape(-1, cols)
        return wp.from_numpy(np_data, dtype=wp.float32, device=dev)

    def _get_write_scratch(self, tensor_type: int, binding) -> wp.array:
        """Return a cached GPU scratch buffer for read-modify-write operations.

        Args:
            tensor_type: Tensor type key used to cache the scratch buffer.
            binding: The binding whose shape the scratch buffer should match.

        Returns:
            A float32 warp array of shape ``binding.shape`` on ``self._device``.
        """
        if not hasattr(self, "_write_scratch"):
            self._write_scratch: dict = {}
        buf = self._write_scratch.get(tensor_type)
        if buf is None:
            buf = wp.zeros(binding.shape, dtype=wp.float32, device=self._device)
            self._write_scratch[tensor_type] = buf
        return buf

    def _stage_to_pinned_cpu(self, tensor_type: int, role: str, src: wp.array) -> wp.array:
        """Copy *src* into a lazily-allocated pinned-host :class:`wp.array` keyed by
        ``(tensor_type, role)``.

        Used to bridge GPU sources to CPU-only TensorBindings (e.g. ``RIGID_BODY_COM_POSE``
        in GPU mode).  The staging buffer is reused across calls; reallocation only
        happens when the shape or dtype changes.  Mirrors PhysX's pinned-staging pattern.
        """
        if not hasattr(self, "_cpu_staging"):
            self._cpu_staging: dict = {}
        key = (tensor_type, role)
        staging = self._cpu_staging.get(key)
        if staging is None or staging.shape != src.shape or staging.dtype != src.dtype:
            staging = wp.zeros(src.shape, dtype=src.dtype, device="cpu", pinned=True)
            self._cpu_staging[key] = staging
        wp.copy(staging, src)
        return staging

    def _binding_write(
        self, tensor_type: int, binding, src: wp.array, *, indices: wp.array | None = None, mask: wp.array | None = None
    ) -> None:
        """Write *src* to *binding*, staging through pinned-host buffers for CPU-only bindings."""
        if tensor_type not in TT._CPU_ONLY_TYPES or self._device == "cpu":
            binding.write(src, indices=indices, mask=mask)
            return
        src_cpu = self._stage_to_pinned_cpu(tensor_type, "data", src)
        idx_cpu = self._stage_to_pinned_cpu(tensor_type, "indices", indices) if indices is not None else None
        mask_cpu = self._stage_to_pinned_cpu(tensor_type, "mask", mask) if mask is not None else None
        binding.write(src_cpu, indices=idx_cpu, mask=mask_cpu)

    def _binding_read(self, tensor_type: int, binding, dst: wp.array) -> None:
        """Read *binding* into *dst*, staging through a pinned-host buffer for CPU-only bindings."""
        if tensor_type not in TT._CPU_ONLY_TYPES or self._device == "cpu":
            binding.read(dst)
            return
        # Allocate or reuse the staging buffer; we only copy back to dst, not into staging first.
        if not hasattr(self, "_cpu_staging"):
            self._cpu_staging: dict = {}
        key = (tensor_type, "data")
        staging = self._cpu_staging.get(key)
        if staging is None or staging.shape != dst.shape or staging.dtype != dst.dtype:
            staging = wp.zeros(dst.shape, dtype=dst.dtype, device="cpu", pinned=True)
            self._cpu_staging[key] = staging
        binding.read(staging)
        wp.copy(dst, staging)

    def _write_body_state(self, tensor_type: int, data, env_ids=None, mask=None, _ids_gpu=None) -> None:
        """GPU-native write for the single-body state of a rigid object.

        Routes pose ``[N, 7]``, velocity ``[N, 6]``, scalar mass ``[N]``,
        COM-pose ``[N, 7]``, or inertia ``[N, 9]`` data to the matching
        OVPhysX binding via one of four paths, fastest first:

        - Full write (no env_ids, no mask): zero-copy DLPack.
        - Indexed write with full-size data: zero-copy view + indices.
          The binding API only copies the indexed rows from the full buffer,
          so no read-modify-write is needed when data is already ``[N, ...]``.
        - Indexed write with partial data ``[K, ...]``: scatter kernel into a
          GPU scratch buffer, then write with indices.
        - Masked write: data is always full ``[N, ...]``, pass directly with mask.

        1-D bindings (e.g. ``RIGID_BODY_MASS`` of shape ``(N,)``) are handled
        by treating them as ``(N, 1)`` internally.

        Args:
            tensor_type: The TensorType constant (e.g. ``RIGID_BODY_POSE``).
            data: State data to write.
            env_ids: Optional environment indices.
            mask: Optional boolean environment mask.
            _ids_gpu: Pre-converted GPU warp int32 array of env indices. When
                provided, skips the per-call GPU->CPU->GPU conversion of env_ids.
        """
        binding = self._get_binding(tensor_type)
        if binding is None:
            return
        if len(binding.shape) == 1:
            N, C = binding.shape[0], 1
        else:
            N, C = binding.shape[0], binding.shape[1]

        is_1d = len(binding.shape) == 1

        if env_ids is None and _ids_gpu is None and mask is None:
            # Full write: data must cover all N instances.
            data_rows = data.shape[0] if hasattr(data, "shape") and len(data.shape) > 0 else 1
            if data_rows != N:
                raise RuntimeError(
                    f"Shape mismatch: binding has {N} rows (num_instances) but data"
                    f" has {data_rows} rows. Expected data.shape[0] == {N}."
                )
            self._binding_write(tensor_type, binding, self._to_flat_f32(data))
            self._invalidate_root_caches(tensor_type)
            return

        if is_1d:
            # 1-D binding: ensure the source array is 1-D so that the binding's
            # index/mask scatter operates on a flat buffer.  The caller may pass
            # data as (K,) or (K, 1); normalise to (K,) here.
            _src_raw = self._to_flat_f32(data)
            n_elems = _src_raw.shape[0]
            src = wp.array(
                ptr=_src_raw.ptr,
                shape=(n_elems,),
                dtype=wp.float32,
                device=self._device,
                copy=False,
            )
        else:
            src = self._as_gpu_f32_2d(data, C)

        if env_ids is not None or _ids_gpu is not None:
            if _ids_gpu is None:
                _ids_gpu = self._env_ids_to_gpu_warp(env_ids)
            K = _ids_gpu.shape[0]
            if is_1d:
                # 1-D binding (e.g. RIGID_BODY_MASS): pass data flat; the
                # binding write() handles index scatter natively.
                self._binding_write(tensor_type, binding, src, indices=_ids_gpu)
            elif src.shape[0] == N:
                self._binding_write(tensor_type, binding, src, indices=_ids_gpu)
            else:
                scratch = self._get_write_scratch(tensor_type, binding)
                self._binding_read(tensor_type, binding, scratch)
                wp.launch(
                    _scatter_rows_partial,
                    dim=(K, C),
                    inputs=[scratch, src, _ids_gpu],
                    device=self._device,
                )
                self._binding_write(tensor_type, binding, scratch, indices=_ids_gpu)
        else:
            mask_u8 = wp.from_numpy(
                self._to_cpu_numpy(mask).astype(np.uint8),
                device=self._device,
            )
            self._binding_write(tensor_type, binding, src, mask=mask_u8)

    def _com_pose_to_link_pose(self, com_pose_w) -> wp.array:
        """Convert a world-frame COM pose to a world-frame link (actor) pose.

        Reads the body-frame COM offset from the ``RIGID_BODY_COM_POSE`` binding
        and launches :func:`_compose_root_link_pose_from_com` to compute:
        ``link_pose = com_pose_w * inverse(com_pose_b)``.

        Args:
            com_pose_w: World-frame COM poses. Shape is (N,) or (N, 7).

        Returns:
            A warp array of shape (N,) with dtype ``wp.transformf`` containing
            the equivalent world-frame link (actor) poses.
        """
        # Force a fresh read of the COM offset: the caller may have mutated the
        # RIGID_BODY_COM_POSE binding (e.g. via set_coms_index) since the last read.
        self._data._body_com_pose_b.timestamp = -1.0
        com_pose_b = self._data.body_com_pose_b.warp  # (N, 1) wp.transformf
        N = self._num_instances
        dev = self._device
        com_flat = self._to_flat_f32(com_pose_w)
        com_wp = wp.array(
            ptr=com_flat.ptr,
            shape=(N,),
            dtype=wp.transformf,
            device=dev,
            copy=False,
        )
        link_pose_wp = wp.zeros(N, dtype=wp.transformf, device=dev)
        wp.launch(
            _compose_root_link_pose_from_com,
            dim=N,
            inputs=[com_wp, com_pose_b],
            outputs=[link_pose_wp],
            device=dev,
        )
        return link_pose_wp

    def _resolve_env_ids(self, env_ids) -> wp.array:
        """Resolve environment indices to a warp int32 array on ``self._device`` (mirrors PhysX).

        Tests sometimes hand us indices on CPU even when the sim runs on GPU; we move the
        resolved array onto ``self._device`` so kernel launches don't fail on a device
        mismatch.
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

    def _resolve_body_ids(self, body_ids) -> wp.array:
        """Resolve body indices to a warp int32 array on ``self._device`` (mirrors PhysX)."""
        if body_ids is None or body_ids == slice(None):
            return self._ALL_BODY_INDICES
        if isinstance(body_ids, list):
            return wp.array(body_ids, dtype=wp.int32, device=self._device)
        return body_ids

    def _resolve_env_mask(self, env_mask: wp.array | None) -> wp.array:
        """Resolve an environment mask to a ``wp.bool`` array on ``self._device``.

        OVPhysX (like Newton) uses the wheel's native ``binding.write(mask=...)`` path,
        so the mask is preserved end-to-end -- no ``torch.nonzero`` conversion needed.
        ``None`` returns the pre-allocated all-true mask.
        """
        if env_mask is None:
            return self._ALL_TRUE_ENV_MASK
        if isinstance(env_mask, torch.Tensor):
            return wp.from_torch(env_mask.to(torch.bool), dtype=wp.bool)
        if isinstance(env_mask, wp.array) and str(env_mask.device) != self._device:
            env_mask = wp.clone(env_mask, device=self._device)
        return env_mask

    def _resolve_body_mask(self, body_mask: wp.array | None) -> wp.array:
        """Resolve a body mask to a ``wp.bool`` array on ``self._device`` (Newton-style)."""
        if body_mask is None:
            return self._ALL_TRUE_BODY_MASK
        if isinstance(body_mask, torch.Tensor):
            return wp.from_torch(body_mask.to(torch.bool), dtype=wp.bool)
        if isinstance(body_mask, wp.array) and str(body_mask.device) != self._device:
            body_mask = wp.clone(body_mask, device=self._device)
        return body_mask

    def _get_cpu_env_mask(self, env_mask: wp.array) -> wp.array:
        """Return a pinned-host CPU copy of *env_mask* for a CPU-only binding write.

        ``env_mask`` is normally on ``self._device``; the wheel's ``binding.write(mask=...)``
        requires the mask on the binding's device, which is CPU for mass / coms / inertia.
        Reuses the pre-allocated ``_cpu_env_mask`` pinned buffer.
        """
        wp.copy(self._cpu_env_mask, env_mask)
        return self._cpu_env_mask

    def _get_cpu_env_ids(self, env_ids: wp.array | torch.Tensor) -> wp.array:
        """Return CPU int32 indices, using the pre-allocated pinned ``_cpu_env_ids_all``
        fast path when *env_ids* matches ``_ALL_INDICES`` (PR #5329 pattern).
        """
        if isinstance(env_ids, torch.Tensor):
            env_ids = wp.from_torch(env_ids, dtype=wp.int32)
        if env_ids.ptr == self._ALL_INDICES.ptr:
            return self._cpu_env_ids_all
        return wp.clone(env_ids, device="cpu")

    @staticmethod
    def _to_cpu_numpy(data) -> np.ndarray:
        """Convert data (warp, torch, numpy, scalar) to a CPU numpy array."""
        if isinstance(data, wp.array):
            return data.numpy().astype(np.float32)
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy().astype(np.float32)
        return np.asarray(data, dtype=np.float32)

    @staticmethod
    def _to_cpu_indices(data, dtype=np.int32) -> np.ndarray:
        """Convert index array (warp, torch, list, numpy) to CPU numpy int array."""
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy().astype(dtype)
        if isinstance(data, wp.array):
            return data.numpy().astype(dtype)
        return np.asarray(data, dtype=dtype)

    def _env_ids_to_gpu_warp(self, env_ids) -> wp.array:
        """Convert env_ids to a GPU int32 warp array, with single-entry caching.

        The cache avoids repeated GPU->CPU->GPU round-trips when the same
        ``env_ids`` object is passed to multiple binding writes in a single step.
        A new object identity (``id()``) or shape change invalidates the cache.

        Args:
            env_ids: Environment indices as a torch tensor, warp array, list, or numpy array.

        Returns:
            A GPU int32 warp array containing the indices.
        """
        if hasattr(env_ids, "data_ptr"):
            key = (env_ids.data_ptr(), env_ids.shape[0])
        elif isinstance(env_ids, wp.array):
            key = (env_ids.ptr, env_ids.shape[0])
        else:
            key = None

        if key is not None and hasattr(self, "_ids_cache_key") and self._ids_cache_key == key:
            return self._ids_cache_val

        result = wp.array(self._to_cpu_indices(env_ids, np.int32), device=self._device)
        if key is not None:
            self._ids_cache_key = key
            self._ids_cache_val = result
        return result

    # ------------------------------------------------------------------
    # Internal helpers -- Lifecycle
    # ------------------------------------------------------------------

    def _initialize_impl(self) -> None:
        """Initialize the rigid object from the OVPhysX simulation backend.

        Creates tensor bindings for the rigid-body state tensors, reads counts
        and body names, creates the :class:`RigidObjectData` container, and
        primes the data buffers.
        """
        # Step 1-3: Acquire PhysX instance and build binding pattern.
        physx_instance = OvPhysxManager.get_physx_instance()
        if physx_instance is None:
            raise RuntimeError("OvPhysxManager has not been initialized yet.")
        self._ovphysx = physx_instance
        # Derive the device from PhysicsManager (which mirrors SimulationContext.cfg.device).
        # The ovphysx PhysX object does not expose a .device property; reading it would
        # raise AttributeError (masked by hasattr) and fall back to "cuda:0" even when the
        # simulation is running on CPU, causing a device mismatch in binding.read().
        self._device = OvPhysxManager.get_device()
        self._binding_pattern = self.cfg.prim_path

        # Validate the prim tree before creating tensor bindings -- the wheel
        # silently produces a 0-prim binding when the pattern matches nothing,
        # which surfaces as an obscure ``TypeError`` deep in property accessors.
        # Mirror PhysX's prim-scan validation so failures surface here with a
        # clear message.
        template_prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
        if template_prim is None:
            raise RuntimeError(f"Failed to find prim for expression: '{self.cfg.prim_path}'.")
        template_prim_path = template_prim.GetPath().pathString
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
                    f"Found an articulation root when resolving '{self.cfg.prim_path}' for rigid"
                    f" objects. These are located at: '{articulation_prims}' under"
                    f" '{template_prim_path}'. Please disable the articulation root in the USD"
                    " or from code by setting the parameter"
                    " 'ArticulationRootPropertiesCfg.articulation_enabled' to False in the spawn"
                    " configuration."
                )

        # Step 4: Eagerly create every binding the data container reads at init,
        # so failures surface here rather than as KeyError downstream.
        for tt in (
            TT.RIGID_BODY_POSE,
            TT.RIGID_BODY_VELOCITY,
            TT.RIGID_BODY_WRENCH,
            TT.RIGID_BODY_MASS,
            TT.RIGID_BODY_COM_POSE,
            TT.RIGID_BODY_INERTIA,
        ):
            if self._get_binding(tt) is None:
                raise RuntimeError(
                    f"OVPhysX could not create rigid-body binding {tt!r}. "
                    f"Check that prim_path={self._binding_pattern!r} matches "
                    f"at least one UsdPhysics.RigidBodyAPI prim and that the "
                    f"ovphysx wheel exposes the RIGID_BODY_* TensorType. "
                    f"Note: pattern resolution may currently include articulation "
                    f"links; an explicit selection policy is on the wheel-side roadmap."
                )

        # Step 5: Read counts and body names from the root-pose binding.
        root_pose = self._bindings[TT.RIGID_BODY_POSE]
        self._num_instances = root_pose.count
        self._num_bodies = 1
        try:
            body_names_value = root_pose.body_names
            # body_names may be an empty list for non-articulation bindings; fall
            # back to the documented single-body default in that case.
            self._body_names = list(body_names_value) if body_names_value else ["base_link"]
        except (AttributeError, TypeError):
            # ovphysx TensorBinding raises TypeError (not AttributeError) when
            # body_names is queried on a non-articulation tensor type such as
            # RIGID_BODY_POSE: "Articulation metadata … is not available for
            # tensor type 'RIGID_BODY_POSE'."  For a single-body rigid object
            # the default ["base_link"] is always correct.
            self._body_names = ["base_link"]

        # Step 6: Create the data container (mirrors PhysX: takes bindings + device).
        self._data = RigidObjectData(self._bindings, self._device)

        # Allocate asset-side buffers and apply the initial state from the configuration.
        self._create_buffers()
        self._process_cfg()

        # Step 9: Prime the data by performing the first read.
        self.update(0.0)

        # Step 10: Mark data as ready.
        self._data.is_primed = True

    def _create_buffers(self) -> None:
        """Create buffers for storing data (mirrors PhysX)."""
        N = self._num_instances
        B = 1  # rigid object always has a single body
        device = self._device

        # constants
        self._ALL_INDICES = wp.array(np.arange(N, dtype=np.int32), device=device)
        self._ALL_BODY_INDICES = wp.array(np.arange(B, dtype=np.int32), device=device)
        # All-true masks for default mask paths (mirrors Newton). These let
        # ``binding.write(..., mask=...)`` cover all instances when no env_mask is supplied,
        # without converting back to indices.
        self._ALL_TRUE_ENV_MASK = wp.array(np.ones(N, dtype=bool), dtype=wp.bool, device=device)
        self._ALL_TRUE_BODY_MASK = wp.array(np.ones(B, dtype=bool), dtype=wp.bool, device=device)

        # external wrench composer
        self._wrench_buf = wp.zeros((N, 1, 9), dtype=wp.float32, device=device)
        self._instantaneous_wrench_composer = WrenchComposer(self)
        self._permanent_wrench_composer = WrenchComposer(self)

        # Set information about rigid body into data (mirrors PhysX).
        self._data.body_names = self._body_names

        # Pre-allocated pinned CPU buffers for OVPhysX TensorBinding writes (PR #5329 pattern).
        # The wheel requires CPU arrays for "model" property updates (mass / coms / inertia);
        # pinned host memory enables DMA fast path and avoids per-call ``wp.clone`` allocation.
        self._cpu_env_ids_all = wp.zeros(N, dtype=wp.int32, device="cpu", pinned=True)
        wp.copy(self._cpu_env_ids_all, self._ALL_INDICES)
        self._cpu_body_mass = wp.zeros((N, B), dtype=wp.float32, device="cpu", pinned=True)
        self._cpu_body_coms = wp.zeros((N, B, 7), dtype=wp.float32, device="cpu", pinned=True)
        self._cpu_body_inertia = wp.zeros((N, B, 9), dtype=wp.float32, device="cpu", pinned=True)
        # Pinned-host mask staging for CPU-only binding writes (mass/coms/inertia).
        self._cpu_env_mask = wp.zeros(N, dtype=wp.bool, device="cpu", pinned=True)

    def _process_cfg(self) -> None:
        """Post-processing of configuration parameters (mirrors PhysX)."""
        # default state
        # -- root state
        # note: we cast to tuple to avoid torch/numpy type mismatch.
        default_root_pose = tuple(self.cfg.init_state.pos) + tuple(self.cfg.init_state.rot)
        default_root_vel = tuple(self.cfg.init_state.lin_vel) + tuple(self.cfg.init_state.ang_vel)
        default_root_pose = np.tile(np.array(default_root_pose, dtype=np.float32), (self._num_instances, 1))
        default_root_vel = np.tile(np.array(default_root_vel, dtype=np.float32), (self._num_instances, 1))
        self._data.default_root_pose = wp.array(default_root_pose, dtype=wp.transformf, device=self._device)
        self._data.default_root_vel = wp.array(default_root_vel, dtype=wp.spatial_vectorf, device=self._device)

    def _get_binding(self, tensor_type: int):
        """Return a cached TensorBinding, creating it on first access.

        Bindings are lightweight handles (a pointer + shape metadata into
        PhysX's shared GPU buffer).  Creating one does NOT allocate new GPU
        memory -- the underlying simulation buffers are allocated once by PhysX
        regardless of how many bindings point into them.  Still, we defer
        creation so that tensor types the user never queries are never looked up.

        Args:
            tensor_type: The TensorType constant identifying which simulation
                buffer to bind (e.g. :attr:`~isaaclab_ovphysx.tensor_types.RIGID_BODY_POSE`).

        Returns:
            A TensorBinding object, or ``None`` if the binding could not be created.
        """
        binding = self._bindings.get(tensor_type)
        if binding is not None:
            return binding
        try:
            binding = self._ovphysx.create_tensor_binding(pattern=self._binding_pattern, tensor_type=tensor_type)
            self._bindings[tensor_type] = binding
            return binding
        except Exception:
            logger.debug("Could not create tensor binding for type %s", tensor_type)
            return None

    def _invalidate_initialize_callback(self, event) -> None:
        """Invalidates the scene elements."""
        super()._invalidate_initialize_callback(event)
