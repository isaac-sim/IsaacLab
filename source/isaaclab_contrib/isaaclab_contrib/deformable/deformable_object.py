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

from .deformable_object_data import DeformableObjectData
from .kernels import (
    compute_nodal_state_w,
    enforce_kinematic_targets,
    scatter_particles_vec3f_index,
    set_kinematic_flags_to_one,
    vec6f,
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
    density: float = 0.02
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
    """
    before_count = getattr(builder, "particle_count", 0)

    body_pos = wp.vec3(
        entry.init_pos[0] + env_position[0],
        entry.init_pos[1] + env_position[1],
        entry.init_pos[2] + env_position[2],
    )
    body_rot = wp.quat(entry.init_rot[0], entry.init_rot[1], entry.init_rot[2], entry.init_rot[3])

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
            f"Invalid deformable type '{entry.deformable_type}' for registry entry "
            f"with prim path '{entry.prim_path}'"
        )

    after_count = getattr(builder, "particle_count", 0)
    delta = after_count - before_count

    entry.particle_offsets.append(before_count)
    if env_idx == 0:
        entry.particles_per_body = delta


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
        if self._data.nodal_kinematic_target is None or self._default_particle_inv_mass is None:
            return

        model = SimulationManager._model
        if model is None:
            return

        for state in (SimulationManager._state_0, SimulationManager._state_1):
            if state is None:
                continue
            wp.launch(
                enforce_kinematic_targets,
                dim=(self._num_instances, self._particles_per_body),
                inputs=[
                    self._data.nodal_kinematic_target,
                    self._particle_offsets,
                    self._default_particle_inv_mass,
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
        nodal_pos: torch.Tensor | wp.array,
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
        if full_data:
            self.assert_shape_and_dtype(
                nodal_pos, (self.num_instances, self._particles_per_body), wp.vec3f, "nodal_pos"
            )
        else:
            self.assert_shape_and_dtype(nodal_pos, (env_ids.shape[0], self._particles_per_body), wp.vec3f, "nodal_pos")
        if isinstance(nodal_pos, torch.Tensor):
            nodal_pos = wp.from_torch(nodal_pos.contiguous(), dtype=wp.vec3f)

        # Scatter into both Newton states
        for state in (SimulationManager._state_0, SimulationManager._state_1):
            if state is not None and state.particle_q is not None:
                wp.launch(
                    scatter_particles_vec3f_index,
                    dim=(env_ids.shape[0], self._particles_per_body),
                    inputs=[nodal_pos, env_ids, self._particle_offsets, full_data],
                    outputs=[state.particle_q],
                    device=self.device,
                )

        # Invalidate data caches
        self._data._nodal_pos_w.timestamp = -1.0
        self._data._nodal_state_w.timestamp = -1.0
        self._data._root_pos_w.timestamp = -1.0

    def write_nodal_velocity_to_sim_index(
        self,
        nodal_vel: torch.Tensor | wp.array,
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
        if full_data:
            self.assert_shape_and_dtype(
                nodal_vel, (self.num_instances, self._particles_per_body), wp.vec3f, "nodal_vel"
            )
        else:
            self.assert_shape_and_dtype(nodal_vel, (env_ids.shape[0], self._particles_per_body), wp.vec3f, "nodal_vel")
        if isinstance(nodal_vel, torch.Tensor):
            nodal_vel = wp.from_torch(nodal_vel.contiguous(), dtype=wp.vec3f)

        # Scatter into both Newton states
        for state in (SimulationManager._state_0, SimulationManager._state_1):
            if state is not None and state.particle_qd is not None:
                wp.launch(
                    scatter_particles_vec3f_index,
                    dim=(env_ids.shape[0], self._particles_per_body),
                    inputs=[nodal_vel, env_ids, self._particle_offsets, full_data],
                    outputs=[state.particle_qd],
                    device=self.device,
                )

        # Invalidate data caches
        self._data._nodal_vel_w.timestamp = -1.0
        self._data._nodal_state_w.timestamp = -1.0
        self._data._root_vel_w.timestamp = -1.0

    def write_nodal_kinematic_target_to_sim_index(
        self,
        targets: torch.Tensor | wp.array,
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
            targets_torch = wp.to_torch(targets)
            buffer_torch = wp.to_torch(self._data.nodal_kinematic_target)
            env_ids_torch = wp.to_torch(env_ids).long()
            if full_data:
                buffer_torch[env_ids_torch] = targets_torch[env_ids_torch]
            else:
                buffer_torch[env_ids_torch] = targets_torch

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

        # Find the first spawned mesh prim in regex path
        template_prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
        if template_prim is None:
            raise RuntimeError(f"Failed to find prim for expression: '{self.cfg.prim_path}'.")
        template_prim_path = template_prim.GetPrimPath()

        # Discover sim / visual mesh prims under the template.
        def _is_sim_mesh(prim) -> bool:
            return any("DeformableSimAPI" in api for api in prim.GetAppliedSchemas())

        tet_prims = sim_utils.get_all_matching_child_prims(template_prim_path, lambda p: p.GetTypeName() == "TetMesh")
        mesh_prims = sim_utils.get_all_matching_child_prims(template_prim_path, lambda p: p.GetTypeName() == "Mesh")

        # When the spawn config uses a surface deformable material, the PhysX
        # spawner may still create a TetMesh proxy (sim_tetmesh) alongside the
        # original Mesh.  Newton reads the original Mesh directly, so ignore
        # the PhysX TetMesh proxy and treat this as a surface deformable.
        _is_surface_material = (
            self.cfg.spawn is not None
            and getattr(self.cfg.spawn, "physics_material", None) is not None
            and "Surface" in type(self.cfg.spawn.physics_material).__name__
        )

        if len(tet_prims) > 1:
            raise ValueError(
                f"Found multiple TetMesh prims under '{template_prim_path}': "
                f"{[p.GetPrimPath() for p in tet_prims]}."
                " Deformable body schema supports only one simulation mesh per asset."
            )

        # Pick simulation and visual mesh prims.
        if len(tet_prims) == 1 and not _is_surface_material:
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
            if _is_surface_material and vis_candidates:
                # Newton reads the original authored Mesh directly.  The PhysX
                # sim_mesh proxy has no points at scene-construction time, so
                # prefer the original mesh when using a surface deformable material.
                mesh_prim = vis_candidates[0]
                vis_candidates = []  # visual == sim, no separate embedding target
            elif sim_candidates:
                mesh_prim = sim_candidates[0]
            else:
                mesh_prim = vis_candidates[0]
                vis_candidates = []  # visual == sim, no separate embedding target
        else:
            raise ValueError(
                f"Could not find any surface or volume mesh in '{template_prim_path}'. Please check asset."
            )

        # BUG FIX: vis_mesh_prim fallback for empty vis_candidates.
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

        # BUG FIX: init_pos/init_rot are already baked into the vertices by the Xform
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
                " with a physics material target that has 'OmniPhysicsDeformableMaterialAPI' applied."
            )
        material_targets = UsdShade.MaterialBindingAPI(template_prim).GetDirectBindingRel("physics").GetTargets()
        stage = template_prim.GetStage()
        material_prim = None
        for mat_path in material_targets:
            mat_prim = stage.GetPrimAtPath(mat_path)
            if "OmniPhysicsDeformableMaterialAPI" in mat_prim.GetAppliedSchemas():
                material_prim = mat_prim
                break
        if material_prim is None:
            raise ValueError(
                f"Could not find a physics material with 'OmniPhysicsDeformableMaterialAPI' applied"
                f" among the physics material targets of '{template_prim_path}'."
            )
        density = material_prim.GetAttribute("omniphysics:density").Get()
        youngs_modulus = material_prim.GetAttribute("omniphysics:youngsModulus").Get()
        poissons_ratio = material_prim.GetAttribute("omniphysics:poissonsRatio").Get()
        # Convert Young's modulus and Poisson's ratio to Lame parameters for Newton
        k_mu = youngs_modulus / (2 * (1 + poissons_ratio))
        k_lambda = (youngs_modulus * poissons_ratio) / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))

        tri_ke = material_prim.GetAttribute("newton:triKe").Get()
        tri_ka = material_prim.GetAttribute("newton:triKa").Get()
        tri_kd = material_prim.GetAttribute("newton:triKd").Get()
        edge_ke = material_prim.GetAttribute("newton:edgeKe").Get()
        edge_kd = material_prim.GetAttribute("newton:edgeKd").Get()
        particle_radius = material_prim.GetAttribute("newton:particleRadius").Get()
        k_damp = material_prim.GetAttribute("newton:kDamp").Get()

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

        # Bind simulation state arrays
        state = SimulationManager._state_0
        if state is not None:
            self._data.bind_simulation_state(state.particle_q, state.particle_qd)

        # Create buffers
        self._create_buffers()

        # Update data once
        self.update(0.0)

        # Register rebind callback for full resets
        self._physics_ready_handle = SimulationManager.register_callback(
            lambda _: self._rebind_state(),
            PhysicsEvent.PHYSICS_READY,
            name=f"deformable_object_rebind_{self.cfg.prim_path}",
        )

    def _rebind_state(self) -> None:
        """Rebind state arrays after a full simulation reset."""
        state = SimulationManager._state_0
        if state is not None and hasattr(self, "_data"):
            self._data.bind_simulation_state(state.particle_q, state.particle_qd)

    def _create_buffers(self):
        """Create buffers for storing data."""
        # Constants
        self._ALL_INDICES = wp.array(np.arange(self._num_instances, dtype=np.int32), device=self.device)

        # Snapshot default positions from current state (after finalize + FK)
        state = SimulationManager._state_0
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
            self._data.default_nodal_state_w = wp.zeros(
                (self._num_instances, self._particles_per_body), dtype=vec6f, device=self.device
            )
            wp.launch(
                compute_nodal_state_w,
                dim=(self._num_instances, self._particles_per_body),
                inputs=[self._default_nodal_pos_w, nodal_velocities],
                outputs=[self._data.default_nodal_state_w],
                device=self.device,
            )
        else:
            self._default_nodal_pos_w = None

        # Snapshot default particle_inv_mass for kinematic target restoration
        model = SimulationManager._model
        if model is not None and hasattr(model, "particle_inv_mass") and model.particle_inv_mass is not None:
            self._default_particle_inv_mass = wp.clone(model.particle_inv_mass)
        else:
            self._default_particle_inv_mass = None

        # Kinematic targets -- allocate and initialize with free flags
        self._data.nodal_kinematic_target = wp.zeros(
            (self._num_instances, self._particles_per_body), dtype=wp.vec4f, device=self.device
        )
        wp.launch(
            set_kinematic_flags_to_one,
            dim=(self._num_instances * self._particles_per_body,),
            inputs=[self._data.nodal_kinematic_target.reshape((self._num_instances * self._particles_per_body,))],
            device=self.device,
        )

        # Set up the model parameters
        model = SimulationManager._model
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
            kinematic_target_torch = wp.to_torch(self.data.nodal_kinematic_target)
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
