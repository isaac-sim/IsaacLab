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
    # Operations - Write to simulation
    # ------------------------------------------------------------------

    def write_root_pose_to_sim_index(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root pose over selected environment indices into the simulation.

        The root pose is in the actor (link) frame.

        Args:
            root_pose: Root poses in simulation frame [m, -]. Shape is (len(env_ids), 7) or
                (len(env_ids),) with dtype ``wp.transformf``.
            env_ids: Environment indices. If None, then all indices are used.
        """
        self._write_root_state(TT.RIGID_BODY_POSE, root_pose, env_ids)

    def write_root_pose_to_sim_mask(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root pose over selected environment mask into the simulation.

        The root pose is in the actor (link) frame.

        Args:
            root_pose: Root poses in simulation frame [m, -]. Shape is (num_instances, 7) or
                (num_instances,) with dtype ``wp.transformf``.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        self._write_root_state(TT.RIGID_BODY_POSE, root_pose, mask=env_mask)

    def write_root_link_pose_to_sim_index(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root link pose over selected environment indices into the simulation.

        The root link pose is the canonical actor-frame pose written directly to
        the ``RIGID_BODY_POSE`` binding.

        Args:
            root_pose: Root link poses in simulation frame [m, -]. Shape is (len(env_ids), 7) or
                (len(env_ids),) with dtype ``wp.transformf``.
            env_ids: Environment indices. If None, then all indices are used.
        """
        self._write_root_state(TT.RIGID_BODY_POSE, root_pose, env_ids)

    def write_root_link_pose_to_sim_mask(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root link pose over selected environment mask into the simulation.

        The root link pose is the canonical actor-frame pose written directly to
        the ``RIGID_BODY_POSE`` binding.

        Args:
            root_pose: Root link poses in simulation frame [m, -]. Shape is (num_instances, 7) or
                (num_instances,) with dtype ``wp.transformf``.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        self._write_root_state(TT.RIGID_BODY_POSE, root_pose, mask=env_mask)

    def write_root_com_pose_to_sim_index(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root center of mass pose over selected environment indices into the simulation.

        The user supplies the world-frame COM pose. This method converts it to
        the actor (link) frame using the body-frame COM offset from the
        ``RIGID_BODY_COM_POSE`` binding before writing to the simulation.

        Args:
            root_pose: Root center of mass poses in simulation frame [m, -]. Shape is (len(env_ids), 7) or
                (len(env_ids),) with dtype ``wp.transformf``.
            env_ids: Environment indices. If None, then all indices are used.
        """
        link_pose = self._com_pose_to_link_pose(root_pose)
        self._write_root_state(TT.RIGID_BODY_POSE, link_pose, env_ids)

    def write_root_com_pose_to_sim_mask(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root center of mass pose over selected environment mask into the simulation.

        The user supplies the world-frame COM pose. This method converts it to
        the actor (link) frame using the body-frame COM offset from the
        ``RIGID_BODY_COM_POSE`` binding before writing to the simulation.

        Args:
            root_pose: Root center of mass poses in simulation frame [m, -]. Shape is (num_instances, 7) or
                (num_instances,) with dtype ``wp.transformf``.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        link_pose = self._com_pose_to_link_pose(root_pose)
        self._write_root_state(TT.RIGID_BODY_POSE, link_pose, mask=env_mask)

    def write_root_velocity_to_sim_index(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z)
        in the simulation world frame.

        Args:
            root_velocity: Root velocities in simulation world frame [m/s, rad/s]. Shape is (len(env_ids), 6)
                or (len(env_ids),) with dtype ``wp.spatial_vectorf``.
            env_ids: Environment indices. If None, then all indices are used.
        """
        self._write_root_state(TT.RIGID_BODY_VELOCITY, root_velocity, env_ids)

    def write_root_velocity_to_sim_mask(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root velocity over selected environment mask into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z)
        in the simulation world frame.

        Args:
            root_velocity: Root velocities in simulation world frame [m/s, rad/s]. Shape is (num_instances, 6)
                or (num_instances,) with dtype ``wp.spatial_vectorf``.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        self._write_root_state(TT.RIGID_BODY_VELOCITY, root_velocity, mask=env_mask)

    def write_root_com_velocity_to_sim_index(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root center of mass velocity over selected environment indices into the simulation.

        For a single rigid body the COM velocity and the link velocity share the same
        ``RIGID_BODY_VELOCITY`` binding. The data is written directly with no frame
        conversion.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame [m/s, rad/s].
                Shape is (len(env_ids), 6) or (len(env_ids),) with dtype ``wp.spatial_vectorf``.
            env_ids: Environment indices. If None, then all indices are used.
        """
        self._write_root_state(TT.RIGID_BODY_VELOCITY, root_velocity, env_ids)

    def write_root_com_velocity_to_sim_mask(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root center of mass velocity over selected environment mask into the simulation.

        For a single rigid body the COM velocity and the link velocity share the same
        ``RIGID_BODY_VELOCITY`` binding. The data is written directly with no frame
        conversion.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame [m/s, rad/s].
                Shape is (num_instances, 6) or (num_instances,) with dtype ``wp.spatial_vectorf``.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        self._write_root_state(TT.RIGID_BODY_VELOCITY, root_velocity, mask=env_mask)

    def write_root_link_velocity_to_sim_index(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set the root link velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z)
        in the simulation world frame, evaluated at the actor (link) frame.

        Args:
            root_velocity: Root frame velocities in simulation world frame [m/s, rad/s].
                Shape is (len(env_ids), 6) or (len(env_ids),) with dtype ``wp.spatial_vectorf``.
            env_ids: Environment indices. If None, then all indices are used.
        """
        self._write_root_state(TT.RIGID_BODY_VELOCITY, root_velocity, env_ids)

    def write_root_link_velocity_to_sim_mask(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the root link velocity over selected environment mask into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z)
        in the simulation world frame, evaluated at the actor (link) frame.

        Args:
            root_velocity: Root frame velocities in simulation world frame [m/s, rad/s].
                Shape is (num_instances, 6) or (num_instances,) with dtype ``wp.spatial_vectorf``.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        self._write_root_state(TT.RIGID_BODY_VELOCITY, root_velocity, mask=env_mask)

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
        """Set rigid actor masses for a subset of environments.

        Args:
            masses: Mass values [kg]. Shape is ``(len(env_ids),)``.
            body_ids: Accepted for contract parity with :class:`BaseRigidObject`;
                ignored because a rigid object has a single body.
            env_ids: Indices of environments to write to. ``None`` writes to
                all environments.
        """
        self._write_root_state(TT.RIGID_BODY_MASS, masses, env_ids=env_ids)
        self._data._invalidate_caches(env_ids)

    def set_masses_mask(
        self,
        *,
        masses: torch.Tensor | wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set rigid actor masses for environments selected by mask.

        Args:
            masses: Mass values [kg]. Shape is ``(num_instances,)``.
            body_mask: Accepted for contract parity with :class:`BaseRigidObject`;
                ignored because a rigid object has a single body.
            env_mask: Boolean environment mask. ``None`` writes to all
                environments. Shape is ``(num_instances,)``.
        """
        self._write_root_state(TT.RIGID_BODY_MASS, masses, mask=env_mask)
        self._data._invalidate_caches()

    def set_coms_index(
        self,
        *,
        coms: torch.Tensor | wp.array,
        body_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set rigid actor center-of-mass poses for a subset of environments.

        Args:
            coms: Center-of-mass poses in the body frame [m, -].
                Shape is ``(len(env_ids), 7)``.
            body_ids: Accepted for contract parity with :class:`BaseRigidObject`;
                ignored because a rigid object has a single body.
            env_ids: Indices of environments to write to. ``None`` writes to
                all environments.
        """
        self._write_root_state(TT.RIGID_BODY_COM_POSE, coms, env_ids=env_ids)
        self._data._invalidate_caches(env_ids)

    def set_coms_mask(
        self,
        *,
        coms: torch.Tensor | wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set rigid actor center-of-mass poses for environments selected by mask.

        Args:
            coms: Center-of-mass poses in the body frame [m, -].
                Shape is ``(num_instances, 7)``.
            body_mask: Accepted for contract parity with :class:`BaseRigidObject`;
                ignored because a rigid object has a single body.
            env_mask: Boolean environment mask. ``None`` writes to all
                environments. Shape is ``(num_instances,)``.
        """
        self._write_root_state(TT.RIGID_BODY_COM_POSE, coms, mask=env_mask)
        self._data._invalidate_caches()

    def set_inertias_index(
        self,
        *,
        inertias: torch.Tensor | wp.array,
        body_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set rigid actor inertia tensors for a subset of environments.

        Args:
            inertias: Inertia tensors [kg·m²], row-major flattened.
                Shape is ``(len(env_ids), 9)``.
            body_ids: Accepted for contract parity with :class:`BaseRigidObject`;
                ignored because a rigid object has a single body.
            env_ids: Indices of environments to write to. ``None`` writes to
                all environments.
        """
        self._write_root_state(TT.RIGID_BODY_INERTIA, inertias, env_ids=env_ids)
        self._data._invalidate_caches(env_ids)

    def set_inertias_mask(
        self,
        *,
        inertias: torch.Tensor | wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set rigid actor inertia tensors for environments selected by mask.

        Args:
            inertias: Inertia tensors [kg·m²], row-major flattened.
                Shape is ``(num_instances, 9)``.
            body_mask: Accepted for contract parity with :class:`BaseRigidObject`;
                ignored because a rigid object has a single body.
            env_mask: Boolean environment mask. ``None`` writes to all
                environments. Shape is ``(num_instances,)``.
        """
        self._write_root_state(TT.RIGID_BODY_INERTIA, inertias, mask=env_mask)
        self._data._invalidate_caches()

    # ------------------------------------------------------------------
    # Deprecated writers
    # ------------------------------------------------------------------

    def write_root_state_to_sim(
        self,
        root_state: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated. Use :meth:`write_root_pose_to_sim_index` and
        :meth:`write_root_velocity_to_sim_index` instead."""
        import warnings

        warnings.warn(
            "write_root_state_to_sim() is deprecated. Use write_root_pose_to_sim_index() and"
            " write_root_velocity_to_sim_index() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._write_root_state(TT.RIGID_BODY_POSE, root_state[..., :7], env_ids)
        self._write_root_state(TT.RIGID_BODY_VELOCITY, root_state[..., 7:], env_ids)

    def write_root_com_state_to_sim(
        self,
        root_state: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated. Use :meth:`write_root_com_pose_to_sim_index` and
        :meth:`write_root_com_velocity_to_sim_index` instead."""
        import warnings

        warnings.warn(
            "write_root_com_state_to_sim() is deprecated. Use write_root_com_pose_to_sim_index() and"
            " write_root_com_velocity_to_sim_index() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        link_pose = self._com_pose_to_link_pose(root_state[..., :7])
        self._write_root_state(TT.RIGID_BODY_POSE, link_pose, env_ids)
        self._write_root_state(TT.RIGID_BODY_VELOCITY, root_state[..., 7:], env_ids)

    def write_root_link_state_to_sim(
        self,
        root_state: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated. Use :meth:`write_root_link_pose_to_sim_index` and
        :meth:`write_root_link_velocity_to_sim_index` instead."""
        import warnings

        warnings.warn(
            "write_root_link_state_to_sim() is deprecated. Use write_root_link_pose_to_sim_index() and"
            " write_root_link_velocity_to_sim_index() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._write_root_state(TT.RIGID_BODY_POSE, root_state[..., :7], env_ids)
        self._write_root_state(TT.RIGID_BODY_VELOCITY, root_state[..., 7:], env_ids)

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

    def _write_root_state(self, tensor_type: int, data, env_ids=None, mask=None, _ids_gpu=None) -> None:
        """GPU-native write for root pose [N,7] or velocity [N,6].

        Three paths, fastest first:

        - Full write (no env_ids, no mask): zero-copy DLPack.
        - Indexed write with full-size data: zero-copy view + indices.
          The binding API only copies the indexed rows from the full buffer,
          so no read-modify-write is needed when data is already ``[N,...]``.
        - Indexed write with partial data ``[K,...]``: scatter kernel into a
          GPU scratch buffer, then write with indices.
        - Masked write: data is always full ``[N,...]``, pass directly with mask.

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
            binding.write(self._to_flat_f32(data))
            self._invalidate_root_caches(tensor_type)
            return

        src = self._to_flat_f32(data) if is_1d else self._as_gpu_f32_2d(data, C)

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
