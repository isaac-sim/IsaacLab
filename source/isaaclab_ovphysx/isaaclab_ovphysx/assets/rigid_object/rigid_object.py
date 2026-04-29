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

from isaaclab.assets.rigid_object.base_rigid_object import BaseRigidObject
from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from isaaclab.utils.string import resolve_matching_names
from isaaclab.utils.wrench_composer import WrenchComposer

from isaaclab_ovphysx import tensor_types as TT
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
        """Set the root pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        .. note::
            This method expects partial data.

        Args:
            root_pose: Root poses in simulation frame. Shape is (len(env_ids), 7)
                or (len(env_ids),) with dtype wp.transformf.
            env_ids: Environment indices. If None, then all indices are used.
        """
        self._write_body_state(TT.RIGID_BODY_POSE, root_pose, env_ids)

    def write_root_pose_to_sim_mask(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root pose over selected environment mask into the simulation.

        .. note::
            This method expects full data.

        Args:
            root_pose: Root poses in simulation frame. Shape is (num_instances, 7)
                or (num_instances,) with dtype wp.transformf.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        self._write_body_state(TT.RIGID_BODY_POSE, root_pose, mask=env_mask)

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

        Args:
            root_pose: Root link poses in simulation frame. Shape is (len(env_ids), 7)
                or (len(env_ids),) with dtype wp.transformf.
            env_ids: Environment indices. If None, then all indices are used.
        """
        self._write_body_state(TT.RIGID_BODY_POSE, root_pose, env_ids)

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

        Args:
            root_pose: Root poses in simulation frame. Shape is (num_instances, 7)
                or (num_instances,) with dtype wp.transformf.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        self._write_body_state(TT.RIGID_BODY_POSE, root_pose, mask=env_mask)

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

        Args:
            root_pose: Root center of mass poses in simulation frame. Shape is (len(env_ids), 7)
                or (len(env_ids),) with dtype wp.transformf.
            env_ids: Environment indices. If None, then all indices are used.
        """
        N = self._num_instances
        if env_ids is None and hasattr(root_pose, "shape") and len(root_pose.shape) > 0:
            if root_pose.shape[0] != N:
                raise RuntimeError(
                    f"Shape mismatch: expected {N} rows (num_instances) but data has"
                    f" {root_pose.shape[0]} rows. Expected data.shape[0] == {N}."
                )
        link_pose = self._com_pose_to_link_pose(root_pose)
        self._write_body_state(TT.RIGID_BODY_POSE, link_pose, env_ids)

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

        Args:
            root_pose: Root center of mass poses in simulation frame. Shape is (num_instances, 7)
                or (num_instances,) with dtype wp.transformf.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        N = self._num_instances
        if hasattr(root_pose, "shape") and len(root_pose.shape) > 0:
            if root_pose.shape[0] != N:
                raise RuntimeError(
                    f"Shape mismatch: expected {N} rows (num_instances) but data has"
                    f" {root_pose.shape[0]} rows. Expected data.shape[0] == {N}."
                )
        link_pose = self._com_pose_to_link_pose(root_pose)
        self._write_body_state(TT.RIGID_BODY_POSE, link_pose, mask=env_mask)

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

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (len(env_ids), 6)
                or (len(env_ids),) with dtype wp.spatial_vectorf.
            env_ids: Environment indices. If None, then all indices are used.
        """
        self._write_body_state(TT.RIGID_BODY_VELOCITY, root_velocity, env_ids)

    def write_root_velocity_to_sim_mask(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root center of mass velocity over selected environment mask into the simulation.

        .. note::
            This method expects full data.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (num_instances, 6)
                or (num_instances,) with dtype wp.spatial_vectorf.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        self._write_body_state(TT.RIGID_BODY_VELOCITY, root_velocity, mask=env_mask)

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

        Args:
            root_velocity: Root center of mass velocities in simulation world frame.
                Shape is (len(env_ids), 6) or (len(env_ids),) with dtype wp.spatial_vectorf.
            env_ids: Environment indices. If None, then all indices are used.
        """
        self._write_body_state(TT.RIGID_BODY_VELOCITY, root_velocity, env_ids)

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

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (num_instances, 6)
                or (num_instances,) with dtype wp.spatial_vectorf.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        self._write_body_state(TT.RIGID_BODY_VELOCITY, root_velocity, mask=env_mask)

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

        Args:
            root_velocity: Root frame velocities in simulation world frame.
                Shape is (len(env_ids), 6) or (len(env_ids),) with dtype wp.spatial_vectorf.
            env_ids: Environment indices. If None, then all indices are used.
        """
        self._write_body_state(TT.RIGID_BODY_VELOCITY, root_velocity, env_ids)

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

        Args:
            root_velocity: Root frame velocities in simulation world frame. Shape is (num_instances, 6)
                or (num_instances,) with dtype wp.spatial_vectorf.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        self._write_body_state(TT.RIGID_BODY_VELOCITY, root_velocity, mask=env_mask)

    # ------------------------------------------------------------------
    # Operations - Setters
    # ------------------------------------------------------------------

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

        Args:
            masses: Masses of all bodies. Shape is (len(env_ids),).
            body_ids: Accepted for contract parity with :class:`BaseRigidObject`;
                ignored because a rigid object has a single body.
            env_ids: The environment indices to set the masses for. Defaults to None (all environments).
        """
        self._write_body_state(TT.RIGID_BODY_MASS, masses, env_ids=env_ids)
        self._data._invalidate_caches(env_ids)

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

        Args:
            masses: Masses of all bodies. Shape is (num_instances,).
            body_mask: Accepted for contract parity with :class:`BaseRigidObject`;
                ignored because a rigid object has a single body.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        self._write_body_state(TT.RIGID_BODY_MASS, masses, mask=env_mask)
        self._data._invalidate_caches()

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

        Args:
            coms: Center of mass pose of all bodies. Shape is (len(env_ids), len(body_ids), 7).
                For a rigid object ``len(body_ids) == 1``.
            body_ids: Accepted for contract parity with :class:`BaseRigidObject`;
                ignored because a rigid object has a single body.
            env_ids: The environment indices to set the center of mass pose for.
                Defaults to None (all environments).
        """
        # The RIGID_BODY_COM_POSE binding is (N, 7); squeeze the singleton body dim.
        if isinstance(coms, wp.array) and coms.ndim == 3:
            K = coms.shape[0]
            coms = wp.array(ptr=coms.ptr, shape=(K, 7), dtype=wp.float32, device=coms.device, copy=False)
        elif isinstance(coms, torch.Tensor) and coms.ndim == 3:
            coms = coms.reshape(coms.shape[0], 7)
        self._write_body_state(TT.RIGID_BODY_COM_POSE, coms, env_ids=env_ids)
        self._data._invalidate_caches(env_ids)

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

        Args:
            coms: Center of mass pose of all bodies. Shape is (num_instances, num_bodies, 7).
                For a rigid object ``num_bodies == 1``.
            body_mask: Accepted for contract parity with :class:`BaseRigidObject`;
                ignored because a rigid object has a single body.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        # The RIGID_BODY_COM_POSE binding is (N, 7); squeeze the singleton body dim.
        if isinstance(coms, wp.array) and coms.ndim == 3:
            N = coms.shape[0]
            coms = wp.array(ptr=coms.ptr, shape=(N, 7), dtype=wp.float32, device=coms.device, copy=False)
        elif isinstance(coms, torch.Tensor) and coms.ndim == 3:
            coms = coms.reshape(coms.shape[0], 7)
        self._write_body_state(TT.RIGID_BODY_COM_POSE, coms, mask=env_mask)
        self._data._invalidate_caches()

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

        Args:
            inertias: Inertias of all bodies. Shape is (len(env_ids), len(body_ids), 9).
                For a rigid object ``len(body_ids) == 1``.
            body_ids: Accepted for contract parity with :class:`BaseRigidObject`;
                ignored because a rigid object has a single body.
            env_ids: The environment indices to set the inertias for.
                Defaults to None (all environments).
        """
        self._write_body_state(TT.RIGID_BODY_INERTIA, inertias, env_ids=env_ids)
        self._data._invalidate_caches(env_ids)

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

        Args:
            inertias: Inertias of all bodies. Shape is (num_instances, num_bodies, 9).
                For a rigid object ``num_bodies == 1``.
            body_mask: Accepted for contract parity with :class:`BaseRigidObject`;
                ignored because a rigid object has a single body.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        self._write_body_state(TT.RIGID_BODY_INERTIA, inertias, mask=env_mask)
        self._data._invalidate_caches()

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
            binding.write(self._to_flat_f32(data))
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
                binding.write(src, indices=_ids_gpu)
            elif src.shape[0] == N:
                binding.write(src, indices=_ids_gpu)
            else:
                scratch = self._get_write_scratch(tensor_type, binding)
                binding.read(scratch)
                wp.launch(
                    _scatter_rows_partial,
                    dim=(K, C),
                    inputs=[scratch, src, _ids_gpu],
                    device=self._device,
                )
                binding.write(scratch, indices=_ids_gpu)
        else:
            mask_u8 = wp.from_numpy(
                self._to_cpu_numpy(mask).astype(np.uint8),
                device=self._device,
            )
            binding.write(src, mask=mask_u8)
        self._invalidate_root_caches(tensor_type)

    def _invalidate_root_caches(self, tensor_type: int) -> None:
        """Force re-read from GPU on next property access after a binding write.

        Args:
            tensor_type: The TensorType that was written, used to select
                which data buffers to invalidate.
        """
        if tensor_type == TT.RIGID_BODY_POSE:
            if self._data._root_link_pose_w_buf is not None:
                self._data._root_link_pose_w_buf.timestamp = -1.0
            if self._data._root_com_pose_w_buf is not None:
                self._data._root_com_pose_w_buf.timestamp = -1.0
        elif tensor_type == TT.RIGID_BODY_VELOCITY:
            if self._data._root_link_vel_w_buf is not None:
                self._data._root_link_vel_w_buf.timestamp = -1.0
            if self._data._root_com_vel_w_buf is not None:
                self._data._root_com_vel_w_buf.timestamp = -1.0

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
        # Ensure the COM-offset buffer is populated.
        self._data._ensure_root_buffers()
        # Force a fresh read: the caller may have mutated the RIGID_BODY_COM_POSE binding
        # after the last lazy read (e.g. via set_coms_index), so we cannot rely on the
        # cached buffer being current.  The frame-conversion result is only correct if it
        # uses the binding value that is current at write time.
        self._data._body_com_pose_b_buf.timestamp = -1.0
        self._data._read_transform_binding(TT.RIGID_BODY_COM_POSE, self._data._body_com_pose_b_buf)
        # Convert the user-supplied com_pose_w to a warp transformf array on device.
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
            inputs=[com_wp, self._data._body_com_pose_b_buf.data],
            outputs=[link_pose_wp],
            device=dev,
        )
        return link_pose_wp

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

        # Step 4: Eagerly create the GPU bindings so failures surface at init.
        for tt in (TT.RIGID_BODY_POSE, TT.RIGID_BODY_VELOCITY, TT.RIGID_BODY_WRENCH):
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

        # Step 6: Create the data container.
        self._data = RigidObjectData(self._bindings, self._device)
        self._data.num_instances = self._num_instances
        self._data.num_bodies = 1
        self._data.body_names = self._body_names

        # Allocate buffers and apply the initial state from the configuration.
        self._create_buffers()
        self._process_cfg()

        # Step 9: Prime the data by performing the first read.
        self.update(0.0)

        # Step 10: Mark data as ready.
        self._data.is_primed = True

    def _create_buffers(self) -> None:
        """Allocate index arrays, Warp views, wrench staging buffer, and wrench composers."""
        N = self._num_instances
        device = self._device

        self._ALL_INDICES = torch.arange(N, dtype=torch.int32, device=device)
        self._ALL_BODY_INDICES = torch.arange(1, dtype=torch.int32, device=device)
        self._ALL_INDICES_WP = wp.from_torch(self._ALL_INDICES, dtype=wp.int32)
        self._ALL_BODY_INDICES_WP = wp.from_torch(self._ALL_BODY_INDICES, dtype=wp.int32)

        self._wrench_buf = wp.zeros((N, 1, 9), dtype=wp.float32, device=device)

        self._instantaneous_wrench_composer = WrenchComposer(self)
        self._permanent_wrench_composer = WrenchComposer(self)

    def _process_cfg(self) -> None:
        """Delegate initial-state application to the data container."""
        self._data._process_cfg(self.cfg)

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
