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
from isaaclab_newton.physics import NewtonManager as SimulationManager

import isaaclab.sim as sim_utils
from isaaclab.assets.deformable_object.base_deformable_object import BaseDeformableObject
from isaaclab.markers import VisualizationMarkers
from isaaclab.physics import PhysicsEvent
from isaaclab.utils.warp import ProxyArray

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


@dataclass
class DeformableRegistryEntry:
    """Entry in the deformable body registry.

    Registered by :class:`DeformableObject` during ``__init__``, consumed by
    ``newton_physics_replicate`` inside the per-world ``begin_world``/``end_world`` loop.
    After replication, ``particle_offsets`` and ``particles_per_body`` are filled in
    so the asset can bind to the correct particle ranges.
    """

    prim_path: str
    sim_mesh_prim_path: str
    vis_mesh_prim_path: str
    vertices: list
    indices: list
    init_pos: tuple[float, float, float]
    init_rot: tuple[float, float, float, float]  # (x, y, z, w)
    deformable_type: str | None = None  # "volume" or "surface"
    # Cloth params
    density: float = 1.0
    tri_ke: float = 1e4
    tri_ka: float = 1e4
    tri_kd: float = 1.5e-6
    edge_ke: float = 5.0
    edge_kd: float = 1e-2
    particle_radius: float = 0.008
    # Tet params
    k_mu: float = 1e5
    k_lambda: float = 1e5
    k_damp: float = 0.0
    # Filled by newton_physics_replicate:
    particle_offsets: list[int] = field(default_factory=list)
    particles_per_body: int = 0


if TYPE_CHECKING:
    from isaaclab.assets.deformable_object.deformable_object_cfg import DeformableObjectCfg

logger = logging.getLogger(__name__)


def add_deformable_entry_to_builder(
    builder,
    entry: DeformableRegistryEntry,
    env_idx: int,
    env_position: list[float],
    env_rotation: list[float] | tuple[float, float, float, float],
) -> None:
    """Add a deformable registry entry to a Newton ``ModelBuilder`` for one environment.

    Depending on the deformable type (``"volume"`` or ``"surface"``), calls
    ``builder.add_soft_mesh()`` or ``builder.add_cloth_mesh()`` with the mesh
    data and material properties stored in the registry entry.

    Also records the particle offset for the instance and, on the first
    environment, records the per-body particle count.

    Args:
        builder: The Newton ``ModelBuilder``.
        entry: A :class:`DeformableRegistryEntry` with mesh data and config.
        env_idx: The environment index.
        env_position: World position [x, y, z] [m] for this environment.
        env_rotation: World orientation as quaternion ``(x, y, z, w)`` for this environment.
    """
    if env_idx == 0:
        entry.particle_offsets.clear()
        entry.particles_per_body = 0

    before_count = getattr(builder, "particle_count", 0)

    env_pos = wp.vec3(float(env_position[0]), float(env_position[1]), float(env_position[2]))
    env_rot = wp.quat(
        float(env_rotation[0]),
        float(env_rotation[1]),
        float(env_rotation[2]),
        float(env_rotation[3]),
    )
    init_pos = wp.vec3(float(entry.init_pos[0]), float(entry.init_pos[1]), float(entry.init_pos[2]))
    init_rot = wp.quat(
        float(entry.init_rot[0]),
        float(entry.init_rot[1]),
        float(entry.init_rot[2]),
        float(entry.init_rot[3]),
    )
    body_pos = env_pos + wp.quat_rotate(env_rot, init_pos)
    body_rot = env_rot * init_rot

    if entry.deformable_type == "volume":
        builder.add_soft_mesh(
            pos=body_pos,
            rot=body_rot,
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=entry.vertices,
            indices=entry.indices,
            density=entry.density,
            k_mu=entry.k_mu,
            k_lambda=entry.k_lambda,
            k_damp=entry.k_damp,
            particle_radius=entry.particle_radius,
        )
    elif entry.deformable_type == "surface":
        builder.add_cloth_mesh(
            pos=body_pos,
            rot=body_rot,
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=entry.vertices,
            indices=entry.indices,
            density=entry.density,
            tri_ke=entry.tri_ke,
            tri_ka=entry.tri_ka,
            tri_kd=entry.tri_kd,
            edge_ke=entry.edge_ke,
            edge_kd=entry.edge_kd,
            particle_radius=entry.particle_radius,
        )
    else:
        raise ValueError(
            f"Invalid deformable type '{entry.deformable_type}' for registry entry with prim path '{entry.prim_path}'"
        )

    after_count = getattr(builder, "particle_count", 0)
    delta = after_count - before_count

    entry.particle_offsets.append(before_count)
    if env_idx == 0:
        entry.particles_per_body = delta
    elif entry.particles_per_body != delta:
        raise RuntimeError(
            f"Deformable body '{entry.prim_path}' produced {delta} particles in env {env_idx}, "
            f"but env 0 produced {entry.particles_per_body}."
        )


def add_registered_deformables_to_builder(
    builder,
    world_idx: int,
    env_position: list[float],
    env_rotation: list[float] | tuple[float, float, float, float],
) -> None:
    """Add all registered deformable entries to one Newton builder world."""
    for entry in SimulationManager._deformable_registry:
        add_deformable_entry_to_builder(builder, entry, world_idx, env_position, env_rotation)


def color_registered_deformables(builder) -> None:
    """Color the Newton builder when deformables were registered."""
    if SimulationManager._deformable_registry:
        builder.color()


def install_deformable_builder_hooks() -> None:
    """Install deformable builder hooks without removing hooks owned by other extensions."""
    SimulationManager._deformable_registry = []
    if not hasattr(SimulationManager, "_per_world_builder_hooks"):
        SimulationManager._per_world_builder_hooks = []
    if not hasattr(SimulationManager, "_post_replicate_hooks"):
        SimulationManager._post_replicate_hooks = []
    if add_registered_deformables_to_builder not in SimulationManager._per_world_builder_hooks:
        SimulationManager._per_world_builder_hooks.append(add_registered_deformables_to_builder)
    if color_registered_deformables not in SimulationManager._post_replicate_hooks:
        SimulationManager._post_replicate_hooks.append(color_registered_deformables)


def clear_deformable_builder_hooks() -> None:
    """Clear deformable registry state and remove only deformable-owned builder hooks."""
    SimulationManager._deformable_registry = []
    if hasattr(SimulationManager, "_per_world_builder_hooks"):
        SimulationManager._per_world_builder_hooks = [
            hook
            for hook in SimulationManager._per_world_builder_hooks
            if hook is not add_registered_deformables_to_builder
        ]
    if hasattr(SimulationManager, "_post_replicate_hooks"):
        SimulationManager._post_replicate_hooks = [
            hook for hook in SimulationManager._post_replicate_hooks if hook is not color_registered_deformables
        ]


class DeformableObject(BaseDeformableObject):
    """A deformable object asset class (Newton backend).

    This class manages cloth/deformable bodies in the Newton physics engine. Newton stores all
    particles in flat arrays (``state.particle_q``, ``state.particle_qd``). This class builds
    a per-instance indexing layer on top of those flat arrays, enabling the standard
    :class:`BaseDeformableObject` interface for reading/writing nodal state.

    The cloth mesh is added to the Newton :class:`ModelBuilder` during the ``MODEL_INIT`` phase.
    The mesh data is read from the USD prim at :attr:`cfg.prim_path`, and cloth simulation
    parameters (density, stiffness, etc.) come from :attr:`DeformableObjectCfg`.
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

        # initialize deformable type to None, should be set to either surface or volume on initialization
        self._deformable_type: str | None = None

        # Read mesh from the spawned USD prim and register in the deformable registry.
        self._registry_entry = self._register_deformable()

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

    def _register_deformable(self) -> DeformableRegistryEntry:
        """Read mesh from the spawned USD prim and register in NewtonManager's deformable registry.

        Returns:
            The registry entry (also stored on NewtonManager._deformable_registry).

        Note:
            pxr imports are deferred to this method (not module level) so that
            ``resolve_task_config`` can import the env-cfg module before Kit
            starts without polluting the ``pxr`` module cache.
        """
        from pxr import Gf, UsdGeom, UsdShade

        # Resolve the path of the actually-spawned template prim. This must mirror
        # :meth:`AssetBase.__init__`: ``spawn_path`` is set by ``InteractiveScene``
        # when the asset is part of the template-based cloning flow (the spawn
        # lives at ``/World/template/<Asset>/proto_asset_*`` and per-env clones at
        # ``/World/envs/env_*/<Asset>`` are not yet authored). For Direct envs
        # that spawn straight at the cloned regex, ``spawn_path`` is unset, so
        # we fall back to ``prim_path`` — which already matches the spawned prim.
        # The cloned-regex ``cfg.prim_path`` is still used below to build the
        # registry entry's :attr:`sim_mesh_prim_path` / :attr:`vis_mesh_prim_path`
        # so post-replicate consumers resolve all per-env clones.
        lookup_path = (
            self.cfg.spawn.spawn_path
            if self.cfg.spawn is not None and self.cfg.spawn.spawn_path is not None
            else self.cfg.prim_path
        )
        template_prim = sim_utils.find_first_matching_prim(lookup_path)
        if template_prim is None:
            raise RuntimeError(f"Failed to find prim for expression: '{lookup_path}'.")
        template_prim_path = template_prim.GetPrimPath()

        # Discover sim / visual mesh prims under the template.
        # The spawner authors a visual UsdGeom.Mesh and a separate simulation mesh
        # (UsdGeom.TetMesh for volume, UsdGeom.Mesh for surface) with a
        # ``*DeformableSimAPI`` applied, so we split candidates by that schema.
        def _is_sim_mesh(prim) -> bool:
            return any("DeformableSimAPI" in api for api in prim.GetAppliedSchemas())

        tet_prims = sim_utils.get_all_matching_child_prims(template_prim_path, lambda p: p.GetTypeName() == "TetMesh")
        mesh_prims = sim_utils.get_all_matching_child_prims(template_prim_path, lambda p: p.GetTypeName() == "Mesh")

        if len(tet_prims) > 1:
            raise ValueError(
                f"Found multiple TetMesh prims under '{template_prim_path}': "
                f"{[p.GetPrimPath() for p in tet_prims]}."
                " Deformable body schema supports only one simulation mesh per asset."
            )

        # Pick simulation and visual mesh prims.
        if len(tet_prims) == 1:
            deformable_type = "volume"
            mesh_prim = tet_prims[0]
            vis_candidates = [p for p in mesh_prims if not _is_sim_mesh(p)]
        elif len(mesh_prims) > 0:
            deformable_type = "surface"
            sim_candidates = [p for p in mesh_prims if _is_sim_mesh(p)]
            vis_candidates = [p for p in mesh_prims if not _is_sim_mesh(p)]
            if len(sim_candidates) > 1:
                raise ValueError(
                    f"Found multiple simulation Mesh prims under '{template_prim_path}': "
                    f"{[p.GetPrimPath() for p in sim_candidates]}."
                    " Deformable body schema supports only one simulation mesh per asset."
                )
            # Fall back to the single authored Mesh when no explicit sim mesh was tagged
            # (legacy / self-simulated surfaces where the visual mesh *is* the sim mesh).
            mesh_prim = sim_candidates[0] if sim_candidates else vis_candidates[0]
            if not sim_candidates:
                vis_candidates = []  # visual == sim, no separate embedding target
        else:
            raise ValueError(
                f"Could not find any surface or volume mesh in '{template_prim_path}'. Please check asset."
            )

        # Revert visual and simulation mesh prim paths back to template-relative form for registry storage,
        # since the actual prim paths will differ per world instance after replication.
        # When vis_candidates is empty the visual mesh IS the simulation mesh
        # (e.g. a plain surface cloth with no separate visual embedding).
        vis_mesh_prim = vis_candidates[0] if vis_candidates else mesh_prim
        vis_mesh_prim_path = str(vis_mesh_prim.GetPrimPath())
        vis_mesh_prim_path = self.cfg.prim_path + vis_mesh_prim_path[len(template_prim_path.pathString) :]
        sim_mesh_prim_path = str(mesh_prim.GetPrimPath())
        sim_mesh_prim_path = self.cfg.prim_path + sim_mesh_prim_path[len(template_prim_path.pathString) :]
        logger.info("Registered visual UsdGeom.Mesh at %s.", vis_mesh_prim_path)

        # Bake the template prim's xform directly into the vertex positions.
        xform_cache = UsdGeom.XformCache()
        mesh_to_parent_frame = (
            xform_cache.GetLocalToWorldTransform(mesh_prim)
            * xform_cache.GetLocalToWorldTransform(template_prim.GetParent()).GetInverse()
        )

        def _bake_points(raw_pts) -> list[wp.vec3]:
            out = []
            for p in raw_pts:
                q = mesh_to_parent_frame.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
                out.append(wp.vec3(float(q[0]), float(q[1]), float(q[2])))
            return out

        if deformable_type == "volume":
            tet_mesh = UsdGeom.TetMesh(mesh_prim)
            pts = tet_mesh.GetPointsAttr().Get()
            vertices = _bake_points(pts)
            raw_tet_indices = tet_mesh.GetTetVertexIndicesAttr().Get()
            indices = []
            for vec4i in raw_tet_indices:
                indices.extend([int(vec4i[0]), int(vec4i[1]), int(vec4i[2]), int(vec4i[3])])
            logger.info("Registered UsdGeom.TetMesh: %d vertices, %d tetrahedra.", len(pts), len(indices) // 4)
        else:  # surface
            usd_mesh = UsdGeom.Mesh(mesh_prim)
            pts = usd_mesh.GetPointsAttr().Get()
            vertices = _bake_points(pts)
            indices = list(usd_mesh.GetFaceVertexIndicesAttr().Get())
            logger.info("Registered UsdGeom.Mesh: %d vertices.", len(pts))

        # init_pos/init_rot are already baked into the vertices by the Xform
        # transform above. Setting them to identity prevents add_cloth_mesh/add_soft_mesh
        # from applying them a second time.
        # Note: add_deformable_entry_to_builder passes init_rot directly to
        # wp.quat(x, y, z, w), so identity must be (0, 0, 0, 1) not (1, 0, 0, 0).
        init_pos = (0.0, 0.0, 0.0)
        init_rot = (0.0, 0.0, 0.0, 1.0)

        # Look up the bound deformable physics material
        if not template_prim.HasAPI(UsdShade.MaterialBindingAPI):
            raise ValueError(
                f"Template prim '{template_prim_path}' must have a UsdShade.MaterialBindingAPI applied"
                " with a Newton deformable physics material target."
            )
        material_targets = UsdShade.MaterialBindingAPI(template_prim).GetDirectBindingRel("physics").GetTargets()
        stage = template_prim.GetStage()
        material_prim = None
        for mat_path in material_targets:
            mat_prim = stage.GetPrimAtPath(mat_path)
            if mat_prim.GetAttribute("newton:density").IsValid():
                material_prim = mat_prim
                break
        if material_prim is None:
            raise ValueError(
                f"Could not find a Newton deformable physics material"
                f" among the physics material targets of '{template_prim_path}'."
            )

        def _get_material_attr(name: str, default):
            attr = material_prim.GetAttribute(name)
            return attr.Get() if attr.IsValid() else default

        density = _get_material_attr("newton:density", DeformableRegistryEntry.density)
        particle_radius = _get_material_attr("newton:particleRadius", DeformableRegistryEntry.particle_radius)
        k_mu = _get_material_attr("newton:kMu", DeformableRegistryEntry.k_mu)
        k_lambda = _get_material_attr("newton:kLambda", DeformableRegistryEntry.k_lambda)
        k_damp = _get_material_attr("newton:kDamp", DeformableRegistryEntry.k_damp)

        tri_ke = _get_material_attr("newton:triKe", DeformableRegistryEntry.tri_ke)
        tri_ka = _get_material_attr("newton:triKa", DeformableRegistryEntry.tri_ka)
        tri_kd = _get_material_attr("newton:triKd", DeformableRegistryEntry.tri_kd)
        edge_ke = _get_material_attr("newton:edgeKe", DeformableRegistryEntry.edge_ke)
        edge_kd = _get_material_attr("newton:edgeKd", DeformableRegistryEntry.edge_kd)

        entry = DeformableRegistryEntry(
            prim_path=self.cfg.prim_path,
            sim_mesh_prim_path=sim_mesh_prim_path,
            vis_mesh_prim_path=vis_mesh_prim_path,
            vertices=vertices,
            indices=indices,
            deformable_type=deformable_type,
            init_pos=init_pos,
            init_rot=init_rot,
            density=density,
            tri_ke=tri_ke,
            tri_ka=tri_ka,
            tri_kd=tri_kd,
            edge_ke=edge_ke,
            edge_kd=edge_kd,
            particle_radius=particle_radius,
            k_mu=k_mu,
            k_lambda=k_lambda,
            k_damp=k_damp,
        )
        SimulationManager._deformable_registry.append(entry)
        self._deformable_type = deformable_type
        return entry

    def _initialize_impl(self):
        """Initialize physics handles and buffers after the Newton model is ready."""
        entry = self._registry_entry
        self._num_instances = len(entry.particle_offsets)
        self._particles_per_body = entry.particles_per_body
        self._recorded_particle_offsets = entry.particle_offsets

        if self._num_instances == 0:
            raise RuntimeError(
                f"No deformable body instances found for '{self.cfg.prim_path}'. "
                "Ensure newton_physics_replicate or MODEL_INIT processed the registry."
            )

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
