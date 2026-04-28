# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVPhysX-backed RigidObject implementation."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np  # noqa: F401  -- reserved for future buffer init helpers
import torch
import warp as wp

from isaaclab.assets.rigid_object.base_rigid_object import BaseRigidObject
from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from isaaclab.utils.wrench_composer import WrenchComposer

from isaaclab_ovphysx import tensor_types as TT
from isaaclab_ovphysx.assets.kernels import _body_wrench_to_world, _scatter_rows_partial  # noqa: F401
from isaaclab_ovphysx.physics import OvPhysxManager

from .rigid_object_data import RigidObjectData

logger = logging.getLogger(__name__)


class RigidObject(BaseRigidObject):
    """RigidObject backed by the ovphysx TensorBindingsAPI.

    Reads and writes simulation state through ovphysx TensorBinding objects
    created from the OvPhysxManager's PhysX instance.  Only free (non-articulated)
    rigid bodies are supported; prims under an ArticulationRootAPI should use
    :class:`~isaaclab_ovphysx.assets.articulation.Articulation` instead.
    """

    __backend_name__ = "ovphysx"

    cfg: RigidObjectCfg
    """The configuration of the asset."""

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
        """Data container with simulation state for this rigid object."""
        return self._data

    @property
    def num_instances(self) -> int:
        """Number of rigid-body instances (environments)."""
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
        """Bindings dict in lieu of a single opaque PhysX view.

        OVPhysX exposes per-tensor-type bindings rather than a monolithic view
        object.  Callers that need raw binding access should prefer
        :meth:`_get_binding` instead of iterating this dict directly.
        """
        return self._bindings

    @property
    def instantaneous_wrench_composer(self) -> WrenchComposer | None:
        """Wrench composer for forces applied only during the current step."""
        return self._instantaneous_wrench_composer

    @property
    def permanent_wrench_composer(self) -> WrenchComposer | None:
        """Wrench composer for forces applied persistently every step."""
        return self._permanent_wrench_composer

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def reset(
        self, env_ids: Sequence[int] | torch.Tensor | wp.array | None = None, env_mask: wp.array | None = None
    ) -> None:
        """Reset the rigid object.

        .. caution::
            If both `env_ids` and `env_mask` are provided, then `env_mask` takes precedence over `env_ids`.

        Args:
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError("reset() is implemented in Task 13.")

    def write_data_to_sim(self) -> None:
        """Write external wrench to the simulation.

        .. note::
            We write external wrench to the simulation here since this function is called before the simulation step.
            This ensures that the external wrench is applied at every simulation step.
        """
        raise NotImplementedError("write_data_to_sim() is implemented in Task 12.")

    def update(self, dt: float) -> None:
        """Update internal data buffers after a simulation step.

        Args:
            dt: The simulation time step [s] used for finite-difference quantities.
        """
        raise NotImplementedError("update() is implemented in Task 13.")

    # ------------------------------------------------------------------
    # Operations - Finders
    # ------------------------------------------------------------------

    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        """Find bodies in the rigid body based on the name keys.

        Please check the :func:`isaaclab.utils.string.resolve_matching_names` function for more
        information on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the body indices and names.
        """
        raise NotImplementedError("find_bodies() is implemented in Task 13.")

    # ------------------------------------------------------------------
    # Operations - Write to simulation (Task 10)
    # ------------------------------------------------------------------

    def write_root_pose_to_sim_index(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root pose over selected environment indices into the simulation.

        Args:
            root_pose: Root poses in simulation frame. Shape is (len(env_ids), 7) or
                (len(env_ids),) with dtype wp.transformf.
            env_ids: Environment indices. If None, then all indices are used.
        """
        raise NotImplementedError("write_root_pose_to_sim_index() is implemented in Task 10.")

    def write_root_pose_to_sim_mask(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root pose over selected environment mask into the simulation.

        Args:
            root_pose: Root poses in simulation frame. Shape is (num_instances, 7) or
                (num_instances,) with dtype wp.transformf.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError("write_root_pose_to_sim_mask() is implemented in Task 10.")

    def write_root_link_pose_to_sim_index(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root link pose over selected environment indices into the simulation.

        Args:
            root_pose: Root link poses in simulation frame. Shape is (len(env_ids), 7) or
                (len(env_ids),) with dtype wp.transformf.
            env_ids: Environment indices. If None, then all indices are used.
        """
        raise NotImplementedError("write_root_link_pose_to_sim_index() is implemented in Task 10.")

    def write_root_link_pose_to_sim_mask(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root link pose over selected environment mask into the simulation.

        Args:
            root_pose: Root link poses in simulation frame. Shape is (num_instances, 7) or
                (num_instances,) with dtype wp.transformf.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError("write_root_link_pose_to_sim_mask() is implemented in Task 10.")

    def write_root_com_pose_to_sim_index(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root center of mass pose over selected environment indices into the simulation.

        Args:
            root_pose: Root center of mass poses in simulation frame. Shape is (len(env_ids), 7) or
                (len(env_ids),) with dtype wp.transformf.
            env_ids: Environment indices. If None, then all indices are used.
        """
        raise NotImplementedError("write_root_com_pose_to_sim_index() is implemented in Task 10.")

    def write_root_com_pose_to_sim_mask(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root center of mass pose over selected environment mask into the simulation.

        Args:
            root_pose: Root center of mass poses in simulation frame. Shape is (num_instances, 7) or
                (num_instances,) with dtype wp.transformf.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError("write_root_com_pose_to_sim_mask() is implemented in Task 10.")

    def write_root_velocity_to_sim_index(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root center of mass velocity over selected environment indices into the simulation.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (len(env_ids), 6)
                or (len(env_ids),) with dtype wp.spatial_vectorf.
            env_ids: Environment indices. If None, then all indices are used.
        """
        raise NotImplementedError("write_root_velocity_to_sim_index() is implemented in Task 10.")

    def write_root_velocity_to_sim_mask(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root center of mass velocity over selected environment mask into the simulation.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (num_instances, 6)
                or (num_instances,) with dtype wp.spatial_vectorf.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError("write_root_velocity_to_sim_mask() is implemented in Task 10.")

    def write_root_com_velocity_to_sim_index(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root center of mass velocity over selected environment indices into the simulation.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (len(env_ids), 6)
                or (len(env_ids),) with dtype wp.spatial_vectorf.
            env_ids: Environment indices. If None, then all indices are used.
        """
        raise NotImplementedError("write_root_com_velocity_to_sim_index() is implemented in Task 10.")

    def write_root_com_velocity_to_sim_mask(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root center of mass velocity over selected environment mask into the simulation.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (num_instances, 6)
                or (num_instances,) with dtype wp.spatial_vectorf.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError("write_root_com_velocity_to_sim_mask() is implemented in Task 10.")

    def write_root_link_velocity_to_sim_index(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root link velocity over selected environment indices into the simulation.

        Args:
            root_velocity: Root frame velocities in simulation world frame. Shape is (len(env_ids), 6)
                or (len(env_ids),) with dtype wp.spatial_vectorf.
            env_ids: Environment indices. If None, then all indices are used.
        """
        raise NotImplementedError("write_root_link_velocity_to_sim_index() is implemented in Task 10.")

    def write_root_link_velocity_to_sim_mask(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root link velocity over selected environment mask into the simulation.

        Args:
            root_velocity: Root frame velocities in simulation world frame. Shape is (num_instances, 6)
                or (num_instances,) with dtype wp.spatial_vectorf.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError("write_root_link_velocity_to_sim_mask() is implemented in Task 10.")

    # ------------------------------------------------------------------
    # Operations - Setters (Task 11)
    # ------------------------------------------------------------------

    def set_masses_index(
        self,
        *,
        masses: torch.Tensor | wp.array,
        body_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set masses of all bodies.

        Args:
            masses: Masses of all bodies [kg]. Shape is (len(env_ids), len(body_ids)).
            body_ids: The body indices to set the masses for. Defaults to None (all bodies).
            env_ids: The environment indices to set the masses for. Defaults to None (all environments).
        """
        raise NotImplementedError("set_masses_index() is implemented in Task 11.")

    def set_masses_mask(
        self,
        *,
        masses: torch.Tensor | wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set masses of all bodies.

        Args:
            masses: Masses of all bodies [kg]. Shape is (num_instances, num_bodies).
            body_mask: Body mask. If None, then all bodies are used. Shape is (num_bodies,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError("set_masses_mask() is implemented in Task 11.")

    def set_coms_index(
        self,
        *,
        coms: torch.Tensor | wp.array,
        body_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set center of mass positions of all bodies.

        Args:
            coms: Center of mass positions of all bodies. Shape is (len(env_ids), len(body_ids), 3).
            body_ids: The body indices to set the center of mass positions for. Defaults to None (all bodies).
            env_ids: The environment indices to set the center of mass positions for. Defaults to None
                (all environments).
        """
        raise NotImplementedError("set_coms_index() is implemented in Task 11.")

    def set_coms_mask(
        self,
        *,
        coms: torch.Tensor | wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set center of mass positions of all bodies.

        Args:
            coms: Center of mass positions of all bodies. Shape is (num_instances, num_bodies, 3)
                or (num_instances, num_bodies) with dtype wp.vec3f.
            body_mask: Body mask. If None, then all bodies are used. Shape is (num_bodies,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError("set_coms_mask() is implemented in Task 11.")

    def set_inertias_index(
        self,
        *,
        inertias: torch.Tensor | wp.array,
        body_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set inertias of all bodies.

        Args:
            inertias: Inertias of all bodies. Shape is (len(env_ids), len(body_ids), 9).
            body_ids: The body indices to set the inertias for. Defaults to None (all bodies).
            env_ids: The environment indices to set the inertias for. Defaults to None (all environments).
        """
        raise NotImplementedError("set_inertias_index() is implemented in Task 11.")

    def set_inertias_mask(
        self,
        *,
        inertias: torch.Tensor | wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set inertias of all bodies.

        Args:
            inertias: Inertias of all bodies. Shape is (num_instances, num_bodies, 9).
            body_mask: Body mask. If None, then all bodies are used. Shape is (num_bodies,).
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError("set_inertias_mask() is implemented in Task 11.")

    # ------------------------------------------------------------------
    # Deprecated writers (Task 10 will implement via the new index/mask API)
    # ------------------------------------------------------------------

    def write_root_state_to_sim(
        self,
        root_state: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_root_pose_to_sim_index` and :meth:`write_root_velocity_to_sim_index`."""
        raise NotImplementedError("write_root_state_to_sim() is implemented in Task 10.")

    def write_root_com_state_to_sim(
        self,
        root_state: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated, same as :meth:`write_root_com_pose_to_sim_index` and :meth:`write_root_velocity_to_sim_index`."""
        raise NotImplementedError("write_root_com_state_to_sim() is implemented in Task 10.")

    def write_root_link_state_to_sim(
        self,
        root_state: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated.

        Use :meth:`write_root_pose_to_sim_index` and :meth:`write_root_link_velocity_to_sim_index` instead.
        """
        raise NotImplementedError("write_root_link_state_to_sim() is implemented in Task 10.")

    # ------------------------------------------------------------------
    # Internal helpers
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
        self._device = str(self._ovphysx.device) if hasattr(self._ovphysx, "device") else "cuda:0"
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
        self._body_names = list(root_pose.body_names) if hasattr(root_pose, "body_names") else ["base_link"]

        # Step 6: Create the data container.
        self._data = RigidObjectData(self._bindings, self._device)
        self._data._num_instances = self._num_instances
        self._data._num_bodies = 1

        # Steps 7-8: Placeholder methods (Task 9 fills them in).
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
