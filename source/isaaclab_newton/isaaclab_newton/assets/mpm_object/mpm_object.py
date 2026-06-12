# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

from isaaclab.assets.deformable_object.base_deformable_object import BaseDeformableObject
from isaaclab.physics import PhysicsEvent
from isaaclab.utils.warp import ProxyArray

from isaaclab_newton.cloner import queue_newton_physics_replication
from isaaclab_newton.physics import NewtonManager as SimulationManager
from isaaclab_newton.sim.spawners.mpm import create_mpm_particle_visualization, emit_mpm_particles

from .kernels import (
    compute_particle_state_w,
    gather_particles_vec3f,
    scatter_particles_state_vec6f_index,
    scatter_particles_state_vec6f_mask,
    scatter_particles_vec3f_index,
    scatter_particles_vec3f_mask,
    vec6f,
)
from .mpm_object_data import MPMObjectData

if TYPE_CHECKING:
    from .mpm_object_cfg import MPMObjectCfg

logger = logging.getLogger(__name__)


@dataclass
class MPMObjectRegistryEntry:
    """Particle object registration consumed by Newton builder replication."""

    cfg: MPMObjectCfg
    particle_offsets: list[int] = field(default_factory=list)
    particles_per_object: int = 0


def add_mpm_entry_to_builder(
    builder,
    entry: MPMObjectRegistryEntry,
    env_idx: int,
    env_position: list[float],
    env_rotation: list[float] | tuple[float, float, float, float],
) -> None:
    """Emit one registered MPM object into one Newton builder world."""
    if env_idx == 0:
        entry.particle_offsets.clear()
        entry.particles_per_object = 0

    before_count = builder.particle_count
    position, orientation = _compose_env_asset_pose(entry.cfg, env_position, env_rotation)
    emit_mpm_particles(builder, entry.cfg.spawn, position=position, orientation=orientation)
    delta = builder.particle_count - before_count

    entry.particle_offsets.append(before_count)
    if env_idx == 0:
        entry.particles_per_object = delta
    elif entry.particles_per_object != delta:
        raise RuntimeError(
            f"MPM object '{entry.cfg.prim_path}' produced {delta} particles in env {env_idx}, "
            f"but env 0 produced {entry.particles_per_object}."
        )


def add_registered_mpm_objects_to_builder(
    builder,
    world_idx: int,
    env_position: list[float],
    env_rotation: list[float] | tuple[float, float, float, float],
) -> None:
    """Emit all registered MPM objects into one Newton builder world."""
    for entry in SimulationManager._mpm_object_registry:
        add_mpm_entry_to_builder(builder, entry, world_idx, env_position, env_rotation)


class MPMObject(BaseDeformableObject):
    """Newton MPM particle object asset.

    The object is presented through Isaac Lab's deformable-object interface so it
    can participate in existing scene reset/update/state workflows while exposing
    particle-specific aliases on :attr:`data`.
    """

    cfg: MPMObjectCfg
    __backend_name__: str = "newton"

    _DTYPE_TO_TORCH_TRAILING_DIMS = {**BaseDeformableObject._DTYPE_TO_TORCH_TRAILING_DIMS, vec6f: (6,)}

    def __init__(self, cfg: MPMObjectCfg):
        super().__init__(cfg)
        queue_newton_physics_replication(cfg)
        self._registry_entry = MPMObjectRegistryEntry(self.cfg)
        SimulationManager._mpm_object_registry.append(self._registry_entry)
        self._physics_ready_handle = None

    @property
    def data(self) -> MPMObjectData:
        return self._data

    @property
    def num_instances(self) -> int:
        return self._num_instances

    @property
    def num_bodies(self) -> int:
        return 1

    @property
    def max_sim_vertices_per_body(self) -> int:
        return self._particles_per_object

    @property
    def particles_per_object(self) -> int:
        """Number of particles generated for each environment instance."""
        return self._particles_per_object

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None) -> None:
        """Reset selected particle instances to their default particle state."""
        if env_mask is not None:
            self.write_nodal_state_to_sim_mask(self.data.default_nodal_state_w.warp, env_mask=env_mask)
        else:
            self.write_nodal_state_to_sim_index(self.data.default_nodal_state_w.warp, env_ids=env_ids, full_data=True)

    def write_data_to_sim(self):
        """No-op; MPM particle writes are applied immediately by write methods."""

    def update(self, dt: float):
        self._data.update(dt)

    def write_nodal_state_to_sim_index(
        self,
        nodal_state: torch.Tensor | wp.array | ProxyArray,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        self._scatter_to_sim_index(
            nodal_state,
            env_ids,
            full_data,
            vec6f,
            scatter_particles_state_vec6f_index,
            ("particle_q", "particle_qd"),
            "nodal_state",
        )
        self._invalidate_caches(pos=True, vel=True)

    def write_nodal_pos_to_sim_index(
        self,
        nodal_pos: torch.Tensor | wp.array | ProxyArray,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        self._scatter_to_sim_index(
            nodal_pos, env_ids, full_data, wp.vec3f, scatter_particles_vec3f_index, ("particle_q",), "nodal_pos"
        )
        self._invalidate_caches(pos=True)

    def write_nodal_velocity_to_sim_index(
        self,
        nodal_vel: torch.Tensor | wp.array | ProxyArray,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        self._scatter_to_sim_index(
            nodal_vel, env_ids, full_data, wp.vec3f, scatter_particles_vec3f_index, ("particle_qd",), "nodal_vel"
        )
        self._invalidate_caches(vel=True)

    def write_nodal_kinematic_target_to_sim_index(
        self,
        targets: torch.Tensor | wp.array | ProxyArray,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        raise NotImplementedError("MPMObject does not support deformable kinematic targets.")

    def write_nodal_state_to_sim_mask(
        self,
        nodal_state: torch.Tensor | wp.array | ProxyArray,
        env_mask: wp.array | torch.Tensor | None = None,
    ) -> None:
        self._scatter_to_sim_mask(
            nodal_state,
            env_mask,
            vec6f,
            scatter_particles_state_vec6f_mask,
            ("particle_q", "particle_qd"),
            "nodal_state",
        )
        self._invalidate_caches(pos=True, vel=True)

    def write_nodal_pos_to_sim_mask(
        self,
        nodal_pos: torch.Tensor | wp.array | ProxyArray,
        env_mask: wp.array | torch.Tensor | None = None,
    ) -> None:
        self._scatter_to_sim_mask(
            nodal_pos, env_mask, wp.vec3f, scatter_particles_vec3f_mask, ("particle_q",), "nodal_pos"
        )
        self._invalidate_caches(pos=True)

    def write_nodal_velocity_to_sim_mask(
        self,
        nodal_vel: torch.Tensor | wp.array | ProxyArray,
        env_mask: wp.array | torch.Tensor | None = None,
    ) -> None:
        self._scatter_to_sim_mask(
            nodal_vel, env_mask, wp.vec3f, scatter_particles_vec3f_mask, ("particle_qd",), "nodal_vel"
        )
        self._invalidate_caches(vel=True)

    def write_nodal_kinematic_target_to_sim_mask(
        self,
        targets: torch.Tensor | wp.array | ProxyArray,
        env_mask: wp.array | torch.Tensor | None = None,
    ) -> None:
        raise NotImplementedError("MPMObject does not support deformable kinematic targets.")

    write_particle_state_to_sim_index = write_nodal_state_to_sim_index
    write_particle_pos_to_sim_index = write_nodal_pos_to_sim_index
    write_particle_velocity_to_sim_index = write_nodal_velocity_to_sim_index
    write_particle_state_to_sim_mask = write_nodal_state_to_sim_mask
    write_particle_pos_to_sim_mask = write_nodal_pos_to_sim_mask
    write_particle_velocity_to_sim_mask = write_nodal_velocity_to_sim_mask

    def _scatter_to_sim_index(self, data, env_ids, full_data: bool, dtype, kernel, targets, name: str) -> None:
        """Scatter per-environment particle data into the Newton state arrays in ``targets``."""
        env_ids = self._resolve_env_ids(env_ids)
        num_rows = self.num_instances if full_data else env_ids.shape[0]
        data = self._as_warp(data, dtype, (num_rows, self._particles_per_object), name)
        for state in self._iter_particle_states():
            wp.launch(
                kernel,
                dim=(env_ids.shape[0], self._particles_per_object),
                inputs=[data, env_ids, self._particle_offsets, full_data],
                outputs=[getattr(state, target) for target in targets],
                device=self.device,
            )

    def _scatter_to_sim_mask(self, data, env_mask, dtype, kernel, targets, name: str) -> None:
        """Scatter masked per-environment particle data into the Newton state arrays in ``targets``."""
        env_mask = self._resolve_mask(env_mask)
        data = self._as_warp(data, dtype, (env_mask.shape[0], self._particles_per_object), name)
        for state in self._iter_particle_states():
            wp.launch(
                kernel,
                dim=(env_mask.shape[0], self._particles_per_object),
                inputs=[data, env_mask, self._particle_offsets],
                outputs=[getattr(state, target) for target in targets],
                device=self.device,
            )

    def _as_warp(self, data, dtype, shape: tuple[int, int], name: str) -> wp.array:
        """Validate user data and return it as a Warp array of ``dtype``."""
        if isinstance(data, ProxyArray):
            data = data.warp
        self.assert_shape_and_dtype(data, shape, dtype, name)
        if isinstance(data, torch.Tensor):
            data = wp.from_torch(data.contiguous(), dtype=dtype)
        return data

    def _initialize_impl(self):
        entry = self._registry_entry
        self._num_instances = len(entry.particle_offsets)
        self._particles_per_object = entry.particles_per_object
        self._recorded_particle_offsets = entry.particle_offsets

        if self._num_instances == 0 or self._particles_per_object == 0:
            raise RuntimeError(
                f"No MPM particle instances found for '{self.cfg.prim_path}'. "
                "Ensure Newton replication processed the MPM object registry."
            )

        logger.info(
            "Newton MPM object initialized at '%s': %d instances x %d particles.",
            self.cfg.prim_path,
            self._num_instances,
            self._particles_per_object,
        )

        self._particle_offsets = wp.array(self._recorded_particle_offsets, dtype=wp.int32, device=self.device)
        self._data = MPMObjectData(
            particle_offsets=self._particle_offsets,
            particles_per_object=self._particles_per_object,
            num_instances=self._num_instances,
            device=self.device,
        )
        self._create_buffers()
        self.update(0.0)

        self._physics_ready_handle = SimulationManager.register_callback(
            lambda _: self._data._create_simulation_bindings(),
            PhysicsEvent.PHYSICS_READY,
            name=f"mpm_object_rebind_{self.cfg.prim_path}",
        )

    def _create_buffers(self):
        self._ALL_INDICES = wp.array(np.arange(self._num_instances, dtype=np.int32), device=self.device)
        self._ALL_ENV_MASK = wp.ones((self._num_instances,), dtype=wp.bool, device=self.device)

        state = SimulationManager.get_state_0()
        if state is None or state.particle_q is None or state.particle_qd is None:
            raise RuntimeError("Cannot initialize MPMObject buffers before Newton particle state exists.")

        default_pos = wp.zeros((self._num_instances, self._particles_per_object), dtype=wp.vec3f, device=self.device)
        default_vel = wp.zeros((self._num_instances, self._particles_per_object), dtype=wp.vec3f, device=self.device)
        default_state = wp.zeros((self._num_instances, self._particles_per_object), dtype=vec6f, device=self.device)
        wp.launch(
            gather_particles_vec3f,
            dim=(self._num_instances, self._particles_per_object),
            inputs=[state.particle_q, self._particle_offsets],
            outputs=[default_pos],
            device=self.device,
        )
        wp.launch(
            gather_particles_vec3f,
            dim=(self._num_instances, self._particles_per_object),
            inputs=[state.particle_qd, self._particle_offsets],
            outputs=[default_vel],
            device=self.device,
        )
        wp.launch(
            compute_particle_state_w,
            dim=(self._num_instances, self._particles_per_object),
            inputs=[default_pos, default_vel],
            outputs=[default_state],
            device=self.device,
        )
        self._data.default_nodal_state_w = ProxyArray(default_state)
        self._data.default_particle_state_w = self._data.default_nodal_state_w
        self._create_kit_points()

    def _create_kit_points(self) -> None:
        """Create Kit-visible ``UsdGeom.Points`` prims for the particles when the Kit visualizer is active."""
        from isaaclab.sim import SimulationContext  # noqa: PLC0415

        sim = SimulationContext.instance()
        if sim is None or "kit" not in sim.resolve_visualizer_types() or not self.cfg.spawn.visible:
            return

        first_offset = self._recorded_particle_offsets[0]
        radii = (
            SimulationManager.get_model()
            .particle_radius[first_offset : first_offset + self._particles_per_object]
            .numpy()
        )
        base_path = _create_kit_visualization_path(self.cfg.prim_path)
        prim_paths = create_mpm_particle_visualization(
            prim_path=base_path,
            positions=self.data.particle_pos_w.warp.numpy(),
            widths=2.0 * radii,
            color=self.cfg.spawn.visual_color,
        )
        for env_idx, prim_path in enumerate(prim_paths):
            SimulationManager.register_particle_visual_prim(
                prim_path,
                particle_offset=self._recorded_particle_offsets[env_idx],
                particle_count=self._particles_per_object,
                sync_frequency=self.cfg.spawn.visual_update_frequency,
            )
        logger.info("Kit MPM particle visualization initialized at: %s", base_path)

    def _resolve_env_ids(self, env_ids):
        if env_ids is None or (isinstance(env_ids, slice) and env_ids == slice(None)):
            return self._ALL_INDICES
        if isinstance(env_ids, torch.Tensor):
            return wp.from_torch(env_ids.to(device=self.device, dtype=torch.int32), dtype=wp.int32)
        if isinstance(env_ids, Sequence):
            return wp.array(list(env_ids), dtype=wp.int32, device=self.device)
        return env_ids

    def _resolve_mask(self, mask: wp.array | torch.Tensor | None) -> wp.array:
        if mask is None:
            return self._ALL_ENV_MASK
        if isinstance(mask, torch.Tensor):
            if mask.dtype != torch.bool:
                mask = mask.to(torch.bool)
            return wp.from_torch(mask.to(device=self.device).contiguous(), dtype=wp.bool)
        return mask

    def _iter_particle_states(self):
        """Yield the Newton states whose particle arrays must receive writes."""
        state_0 = SimulationManager.get_state_0()
        state_1 = SimulationManager.get_state_1()
        yield state_0
        if state_1 is not None and state_1 is not state_0:
            yield state_1

    def _invalidate_caches(self, pos: bool = False, vel: bool = False) -> None:
        """Invalidate gathered data buffers after a particle write and flag the render sync."""
        if pos:
            self._data._particle_pos_w.timestamp = -1.0
            self._data._root_pos_w.timestamp = -1.0
        if vel:
            self._data._particle_vel_w.timestamp = -1.0
            self._data._root_vel_w.timestamp = -1.0
        self._data._particle_state_w.timestamp = -1.0
        SimulationManager._mark_particles_dirty()

    def _set_debug_vis_impl(self, debug_vis: bool):
        raise NotImplementedError("Debug visualization is not implemented for MPMObject.")

    def _debug_vis_callback(self, event):
        raise NotImplementedError("Debug visualization is not implemented for MPMObject.")

    def _clear_callbacks(self) -> None:
        super()._clear_callbacks()
        if self._physics_ready_handle is not None:
            self._physics_ready_handle.deregister()
            self._physics_ready_handle = None
        registry = SimulationManager._mpm_object_registry
        if self._registry_entry in registry:
            registry.remove(self._registry_entry)


def _compose_env_asset_pose(
    cfg: MPMObjectCfg,
    env_position: list[float],
    env_rotation: list[float] | tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Compose the environment transform with the asset's initial pose (both ``xyzw``)."""
    env_pos = wp.vec3(*env_position)
    env_rot = wp.quat(*env_rotation)
    pos = env_pos + wp.quat_rotate(env_rot, wp.vec3(*cfg.init_state.pos))
    rot = env_rot * wp.quat(*cfg.init_state.rot)
    return (float(pos[0]), float(pos[1]), float(pos[2])), (float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3]))


def _create_kit_visualization_path(prim_path: str) -> str:
    sanitized = "".join(char if char.isalnum() else "_" for char in prim_path.strip("/"))
    return f"/World/Visuals/MPMParticles/{sanitized or 'Object'}"
