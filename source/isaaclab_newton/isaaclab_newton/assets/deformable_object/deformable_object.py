# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

from isaaclab.assets.deformable_object.base_deformable_object import BaseDeformableObject
from isaaclab.markers import VisualizationMarkers
from isaaclab.physics import PhysicsEvent
from isaaclab.utils.warp import ProxyArray

from ...physics.newton_manager import NewtonManager as SimulationManager
from .deformable_object_data import DeformableObjectData
from .kernels import (
    compute_nodal_state_w,
    enforce_kinematic_targets,
    scatter_particles_state_vec6f_mask,
    scatter_particles_vec3f_index,
    scatter_particles_vec3f_mask,
    set_kinematic_flags_to_one,
    vec6f,
    write_nodal_kinematic_target_index,
    write_nodal_kinematic_target_mask,
)

if TYPE_CHECKING:
    from isaaclab.assets.deformable_object.deformable_object_cfg import DeformableObjectCfg

logger = logging.getLogger(__name__)


class DeformableObject(BaseDeformableObject):
    """A deformable object asset class (Newton backend).

    This class manages cloth/deformable bodies in the Newton physics engine. Newton stores all
    particles in flat arrays (``state.particle_q``, ``state.particle_qd``). This class builds
    a per-instance indexing layer on top of those flat arrays, enabling the standard
    :class:`BaseDeformableObject` interface for reading/writing nodal state.

    Newton cloning imports the authored prototypes and replicates their native element ranges.
    Initialization selects this asset's ranges from the completed backend, without reading USD.
    """

    cfg: DeformableObjectCfg
    """Configuration instance for the deformable object."""

    __backend_name__: str = "newton"
    """The name of the backend for the deformable object."""

    def __init__(self, cfg: DeformableObjectCfg):
        """Initialize the deformable object.

        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg)

        # Register custom vec6f type for nodal state validation.
        self._DTYPE_TO_TORCH_TRAILING_DIMS = {**self._DTYPE_TO_TORCH_TRAILING_DIMS, vec6f: (6,)}

    """
    Properties
    """

    @property
    def data(self) -> DeformableObjectData:
        return self._data

    @property
    def num_instances(self) -> int:
        return self._num_instances

    @property
    def num_bodies(self) -> int:
        """Number of bodies in the asset.

        This is always 1 since each object is a single deformable body.
        """
        return 1

    @property
    def max_sim_vertices_per_body(self) -> int:
        """The maximum number of simulation mesh vertices per deformable body."""
        return self._particles_per_body

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None) -> None:
        """Reset the deformable object.

        No-op to match the PhysX deformable object convention.

        Args:
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. If None, then all the instances are updated.
                Shape is (num_instances,).
        """
        pass

    def write_data_to_sim(self):
        """Apply kinematic targets to the Newton simulation.

        Reads the stored kinematic target buffer and enforces it on particles:
        kinematic particles (flag=0) get inv_mass=0, particle_flags=0, target position,
        and zero velocity; free particles (flag=1) get their original inv_mass and
        particle_flags=1 (ACTIVE) restored.

        Writes to both ``state_0`` and ``state_1`` so kinematic positions survive
        the state swaps that happen between substeps.
        """
        if (
            self._data.nodal_kinematic_target is None
            or self._default_particle_inv_mass is None
            or self._default_particle_flags is None
        ):
            return

        model = SimulationManager.get_model()
        if model is None:
            return

        for state in self._iter_particle_states():
            wp.launch(
                enforce_kinematic_targets,
                dim=(self._num_instances, self._particles_per_body),
                inputs=[
                    self._data.nodal_kinematic_target.warp,
                    self._particle_offsets,
                    self._default_particle_inv_mass,
                    self._default_particle_flags,
                ],
                outputs=[
                    state.particle_q,
                    state.particle_qd,
                    model.particle_inv_mass,
                    model.particle_flags,
                ],
                device=self.device,
            )

    def update(self, dt: float):
        self._data.update(dt)

    """
    Operations - Write to simulation.
    """

    def write_nodal_pos_to_sim_index(
        self,
        nodal_pos: torch.Tensor | wp.array | ProxyArray,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set the nodal positions over selected environment indices into the simulation.

        Args:
            nodal_pos: Nodal positions in simulation frame [m].
                Shape is (len(env_ids), max_sim_vertices_per_body, 3)
                or (num_instances, max_sim_vertices_per_body, 3).
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        env_ids = self._resolve_env_ids(env_ids)
        if isinstance(nodal_pos, ProxyArray):
            nodal_pos = nodal_pos.warp
        if full_data:
            self.assert_shape_and_dtype(
                nodal_pos, (self.num_instances, self._particles_per_body), wp.vec3f, "nodal_pos"
            )
        else:
            self.assert_shape_and_dtype(nodal_pos, (env_ids.shape[0], self._particles_per_body), wp.vec3f, "nodal_pos")
        if isinstance(nodal_pos, torch.Tensor):
            nodal_pos = wp.from_torch(nodal_pos.contiguous(), dtype=wp.vec3f)

        for state in self._iter_particle_states():
            wp.launch(
                scatter_particles_vec3f_index,
                dim=(env_ids.shape[0], self._particles_per_body),
                inputs=[nodal_pos, env_ids, self._particle_offsets, full_data],
                outputs=[state.particle_q],
                device=self.device,
            )

        SimulationManager._mark_particles_dirty()
        self._invalidate_nodal_pos_cache()

    def write_nodal_velocity_to_sim_index(
        self,
        nodal_vel: torch.Tensor | wp.array | ProxyArray,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set the nodal velocity over selected environment indices into the simulation.

        Args:
            nodal_vel: Nodal velocities in simulation frame [m/s].
                Shape is (len(env_ids), max_sim_vertices_per_body, 3)
                or (num_instances, max_sim_vertices_per_body, 3).
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        env_ids = self._resolve_env_ids(env_ids)
        if isinstance(nodal_vel, ProxyArray):
            nodal_vel = nodal_vel.warp
        if full_data:
            self.assert_shape_and_dtype(
                nodal_vel, (self.num_instances, self._particles_per_body), wp.vec3f, "nodal_vel"
            )
        else:
            self.assert_shape_and_dtype(nodal_vel, (env_ids.shape[0], self._particles_per_body), wp.vec3f, "nodal_vel")
        if isinstance(nodal_vel, torch.Tensor):
            nodal_vel = wp.from_torch(nodal_vel.contiguous(), dtype=wp.vec3f)

        for state in self._iter_particle_states():
            wp.launch(
                scatter_particles_vec3f_index,
                dim=(env_ids.shape[0], self._particles_per_body),
                inputs=[nodal_vel, env_ids, self._particle_offsets, full_data],
                outputs=[state.particle_qd],
                device=self.device,
            )

        self._invalidate_nodal_vel_cache()

    def write_nodal_kinematic_target_to_sim_index(
        self,
        targets: torch.Tensor | wp.array | ProxyArray,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set the kinematic targets of the simulation mesh for the deformable bodies.

        Newton has no native kinematic target API. Instead:
        - Kinematic (flag=0.0): set ``particle_inv_mass`` to 0, write target pos, zero vel
        - Free (flag=1.0): restore original ``particle_inv_mass``

        Args:
            targets: The kinematic targets comprising of nodal positions and flags [m].
                Shape is (len(env_ids), max_sim_vertices_per_body, 4)
                or (num_instances, max_sim_vertices_per_body, 4).
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        env_ids = self._resolve_env_ids(env_ids)
        if isinstance(targets, ProxyArray):
            targets = targets.warp
        if full_data:
            self.assert_shape_and_dtype(targets, (self.num_instances, self._particles_per_body), wp.vec4f, "targets")
        else:
            self.assert_shape_and_dtype(targets, (env_ids.shape[0], self._particles_per_body), wp.vec4f, "targets")
        if isinstance(targets, torch.Tensor):
            if targets.dim() == 2:
                targets = targets.unsqueeze(0)
            targets = wp.from_torch(targets.contiguous(), dtype=wp.vec4f)

        # Store kinematic targets in our data buffer
        if self._data.nodal_kinematic_target is not None:
            wp.launch(
                write_nodal_kinematic_target_index,
                dim=(env_ids.shape[0], self._particles_per_body),
                inputs=[targets, env_ids, full_data],
                outputs=[self._data.nodal_kinematic_target.warp],
                device=self.device,
            )

    """
    Operations - Write to simulation (mask variants).
    """

    def write_nodal_state_to_sim_mask(
        self,
        nodal_state: torch.Tensor | wp.array | ProxyArray,
        env_mask: wp.array | torch.Tensor | None = None,
    ) -> None:
        """Set the nodal state over selected environment mask into the simulation.

        Args:
            nodal_state: Nodal state in simulation frame [m, m/s].
                Shape is (num_instances, max_sim_vertices_per_body, 6).
            env_mask: Environment mask. If None, then all indices are used.
                Shape is (num_instances,).
        """
        env_mask = self._resolve_mask(env_mask, self._ALL_ENV_MASK)
        if isinstance(nodal_state, ProxyArray):
            nodal_state = nodal_state.warp
        self.assert_shape_and_dtype(nodal_state, (env_mask.shape[0], self._particles_per_body), vec6f, "nodal_state")
        if isinstance(nodal_state, torch.Tensor):
            nodal_state = wp.from_torch(nodal_state.contiguous(), dtype=vec6f)

        for state in self._iter_particle_states():
            wp.launch(
                scatter_particles_state_vec6f_mask,
                dim=(env_mask.shape[0], self._particles_per_body),
                inputs=[nodal_state, env_mask, self._particle_offsets],
                outputs=[state.particle_q, state.particle_qd],
                device=self.device,
            )

        SimulationManager._mark_particles_dirty()
        self._invalidate_nodal_state_cache()

    def write_nodal_pos_to_sim_mask(
        self,
        nodal_pos: torch.Tensor | wp.array | ProxyArray,
        env_mask: wp.array | torch.Tensor | None = None,
    ) -> None:
        """Set the nodal positions over selected environment mask into the simulation.

        Args:
            nodal_pos: Nodal positions in simulation frame [m].
                Shape is (num_instances, max_sim_vertices_per_body, 3).
            env_mask: Environment mask. If None, then all indices are used.
                Shape is (num_instances,).
        """
        env_mask = self._resolve_mask(env_mask, self._ALL_ENV_MASK)
        if isinstance(nodal_pos, ProxyArray):
            nodal_pos = nodal_pos.warp
        self.assert_shape_and_dtype(nodal_pos, (env_mask.shape[0], self._particles_per_body), wp.vec3f, "nodal_pos")
        if isinstance(nodal_pos, torch.Tensor):
            nodal_pos = wp.from_torch(nodal_pos.contiguous(), dtype=wp.vec3f)

        for state in self._iter_particle_states():
            wp.launch(
                scatter_particles_vec3f_mask,
                dim=(env_mask.shape[0], self._particles_per_body),
                inputs=[nodal_pos, env_mask, self._particle_offsets],
                outputs=[state.particle_q],
                device=self.device,
            )

        SimulationManager._mark_particles_dirty()
        self._invalidate_nodal_pos_cache()

    def write_nodal_velocity_to_sim_mask(
        self,
        nodal_vel: torch.Tensor | wp.array | ProxyArray,
        env_mask: wp.array | torch.Tensor | None = None,
    ) -> None:
        """Set the nodal velocity over selected environment mask into the simulation.

        Args:
            nodal_vel: Nodal velocities in simulation frame [m/s].
                Shape is (num_instances, max_sim_vertices_per_body, 3).
            env_mask: Environment mask. If None, then all indices are used.
                Shape is (num_instances,).
        """
        env_mask = self._resolve_mask(env_mask, self._ALL_ENV_MASK)
        if isinstance(nodal_vel, ProxyArray):
            nodal_vel = nodal_vel.warp
        self.assert_shape_and_dtype(nodal_vel, (env_mask.shape[0], self._particles_per_body), wp.vec3f, "nodal_vel")
        if isinstance(nodal_vel, torch.Tensor):
            nodal_vel = wp.from_torch(nodal_vel.contiguous(), dtype=wp.vec3f)

        for state in self._iter_particle_states():
            wp.launch(
                scatter_particles_vec3f_mask,
                dim=(env_mask.shape[0], self._particles_per_body),
                inputs=[nodal_vel, env_mask, self._particle_offsets],
                outputs=[state.particle_qd],
                device=self.device,
            )

        self._invalidate_nodal_vel_cache()

    def write_nodal_kinematic_target_to_sim_mask(
        self,
        targets: torch.Tensor | wp.array | ProxyArray,
        env_mask: wp.array | torch.Tensor | None = None,
    ) -> None:
        """Set the kinematic targets over selected environment mask into the target buffer.

        Args:
            targets: The kinematic targets comprising of nodal positions and flags [m].
                Shape is (num_instances, max_sim_vertices_per_body, 4).
            env_mask: Environment mask. If None, then all indices are used.
                Shape is (num_instances,).
        """
        env_mask = self._resolve_mask(env_mask, self._ALL_ENV_MASK)
        if isinstance(targets, ProxyArray):
            targets = targets.warp
        self.assert_shape_and_dtype(targets, (env_mask.shape[0], self._particles_per_body), wp.vec4f, "targets")
        if isinstance(targets, torch.Tensor):
            targets = wp.from_torch(targets.contiguous(), dtype=wp.vec4f)

        if self._data.nodal_kinematic_target is not None:
            wp.launch(
                write_nodal_kinematic_target_mask,
                dim=(env_mask.shape[0], self._particles_per_body),
                inputs=[targets, env_mask],
                outputs=[self._data.nodal_kinematic_target.warp],
                device=self.device,
            )

    """
    Internal helper.
    """

    def _resolve_env_ids(self, env_ids):
        """Resolve environment indices to a warp int32 array."""
        if env_ids is None or (isinstance(env_ids, slice) and env_ids == slice(None)):
            return self._ALL_INDICES
        elif isinstance(env_ids, list):
            return wp.array(env_ids, dtype=wp.int32, device=self.device)
        elif isinstance(env_ids, torch.Tensor):
            return wp.from_torch(env_ids.to(torch.int32), dtype=wp.int32)
        return env_ids

    def _resolve_mask(self, mask: wp.array | torch.Tensor | None, full_mask: wp.array) -> wp.array:
        """Resolve an environment mask to a warp bool array."""
        if mask is None:
            return full_mask
        if isinstance(mask, torch.Tensor):
            if mask.dtype != torch.bool:
                mask = mask.to(torch.bool)
            return wp.from_torch(mask, dtype=wp.bool)
        return mask

    def _iter_particle_states(self):
        """Yield active Newton states."""
        for state in (SimulationManager.get_state_0(), SimulationManager.get_state_1()):
            if state is None:
                continue
            yield state

    def _invalidate_nodal_pos_cache(self) -> None:
        """Invalidate cached position-derived deformable data."""
        self._data._nodal_pos_w.timestamp = -1.0
        self._data._nodal_state_w.timestamp = -1.0
        self._data._root_pos_w.timestamp = -1.0

    def _invalidate_nodal_vel_cache(self) -> None:
        """Invalidate cached velocity-derived deformable data."""
        self._data._nodal_vel_w.timestamp = -1.0
        self._data._nodal_state_w.timestamp = -1.0
        self._data._root_vel_w.timestamp = -1.0

    def _invalidate_nodal_state_cache(self) -> None:
        """Invalidate all cached nodal state data."""
        self._invalidate_nodal_pos_cache()
        self._invalidate_nodal_vel_cache()

    def _initialize_impl(self):
        """Initialize physics handles and buffers after the Newton model is ready."""
        # Replace this selection with Newton's family views when the pinned version includes
        # https://github.com/newton-physics/newton/pull/3326.
        pattern = re.compile(self.cfg.prim_path)
        selected = [
            value for path, value in SimulationManager.backend.deformable_ranges.items() if pattern.fullmatch(path)
        ]
        if not selected:
            raise RuntimeError(f"No imported deformable matches '{self.cfg.prim_path}'.")
        self._particles_per_body, self._deformable_type = selected[0][1:]
        if any((count, kind) != selected[0][1:] for _, count, kind in selected):
            raise ValueError(f"Deformable selection '{self.cfg.prim_path}' requires equal particle counts and types.")
        self._num_instances = len(selected)
        self._recorded_particle_offsets = [offset for offset, _, _ in selected]

        logger.info("Newton deformable object initialized at: %s", self.cfg.prim_path)
        logger.info("Number of instances: %d", self._num_instances)
        logger.info("Particles per body: %d", self._particles_per_body)

        # Build particle offset array on device
        self._particle_offsets = wp.array(self._recorded_particle_offsets, dtype=wp.int32, device=self.device)

        # Create data container
        self._data = DeformableObjectData(
            particle_offsets=self._particle_offsets,
            particles_per_body=self._particles_per_body,
            num_instances=self._num_instances,
            device=self.device,
        )

        # Create buffers
        self._create_buffers()

        # Update data once
        self.update(0.0)

        # Register rebind callback for full resets
        self._physics_ready_handle = SimulationManager.register_callback(
            lambda _: self._data._create_simulation_bindings(),
            PhysicsEvent.PHYSICS_READY,
            name=f"deformable_object_rebind_{self.cfg.prim_path}",
        )

    def _create_buffers(self):
        """Create buffers for storing data."""
        # Constants
        self._ALL_INDICES = wp.array(np.arange(self._num_instances, dtype=np.int32), device=self.device)
        self._ALL_ENV_MASK = wp.ones((self._num_instances,), dtype=wp.bool, device=self.device)

        # Snapshot default positions from current state (after finalize + FK)
        state = SimulationManager.get_state_0()
        if state is not None and state.particle_q is not None:
            from .kernels import gather_particles_vec3f

            self._default_nodal_pos_w = wp.zeros(
                (self._num_instances, self._particles_per_body), dtype=wp.vec3f, device=self.device
            )
            wp.launch(
                gather_particles_vec3f,
                dim=(self._num_instances, self._particles_per_body),
                inputs=[state.particle_q, self._particle_offsets, self._particles_per_body],
                outputs=[self._default_nodal_pos_w],
                device=self.device,
            )

            # Compute default nodal state as vec6f (positions + zero velocities)
            nodal_velocities = wp.zeros(
                (self._num_instances, self._particles_per_body), dtype=wp.vec3f, device=self.device
            )
            default_nodal_state_w = wp.zeros(
                (self._num_instances, self._particles_per_body), dtype=vec6f, device=self.device
            )
            wp.launch(
                compute_nodal_state_w,
                dim=(self._num_instances, self._particles_per_body),
                inputs=[self._default_nodal_pos_w, nodal_velocities],
                outputs=[default_nodal_state_w],
                device=self.device,
            )
            self._data.default_nodal_state_w = ProxyArray(default_nodal_state_w)
        else:
            self._default_nodal_pos_w = None

        # Snapshot default particle_inv_mass for kinematic target restoration
        model = SimulationManager.get_model()
        if model is not None and hasattr(model, "particle_inv_mass") and model.particle_inv_mass is not None:
            self._default_particle_inv_mass = wp.clone(model.particle_inv_mass)
        else:
            self._default_particle_inv_mass = None
        if model is not None and hasattr(model, "particle_flags") and model.particle_flags is not None:
            self._default_particle_flags = wp.clone(model.particle_flags)
        else:
            self._default_particle_flags = None

        # Kinematic targets -- allocate and initialize with free flags
        nodal_kinematic_target = wp.zeros(
            (self._num_instances, self._particles_per_body), dtype=wp.vec4f, device=self.device
        )
        wp.launch(
            set_kinematic_flags_to_one,
            dim=(self._num_instances * self._particles_per_body,),
            inputs=[nodal_kinematic_target.reshape((self._num_instances * self._particles_per_body,))],
            device=self.device,
        )
        self._data.nodal_kinematic_target = ProxyArray(nodal_kinematic_target)

        # Set up the model parameters
        model = SimulationManager.get_model()
        if model is not None:
            if hasattr(model, "edge_rest_angle"):
                model.edge_rest_angle.zero_()

    """
    Internal simulation callbacks.
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "target_visualizer"):
                self.target_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            self.target_visualizer.set_visibility(True)
        else:
            if hasattr(self, "target_visualizer"):
                self.target_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        num_enabled = 0
        if self._deformable_type == "volume":
            kinematic_target_torch = self.data.nodal_kinematic_target.torch
            targets_enabled = kinematic_target_torch[:, :, 3] == 0.0
            num_enabled = int(torch.sum(targets_enabled).item())
        if num_enabled == 0:
            positions = torch.tensor([[0.0, 0.0, -10.0]], device=self.device)
        else:
            positions = kinematic_target_torch[targets_enabled][..., :3]
        self.target_visualizer.visualize(positions)

    def _clear_callbacks(self) -> None:
        """Clears all registered callbacks."""
        super()._clear_callbacks()
        if hasattr(self, "_physics_ready_handle") and self._physics_ready_handle is not None:
            self._physics_ready_handle.deregister()
            self._physics_ready_handle = None

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        super()._invalidate_initialize_callback(event)
