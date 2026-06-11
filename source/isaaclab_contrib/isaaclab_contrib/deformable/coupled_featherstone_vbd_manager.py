# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Coupled Featherstone + VBD Newton manager."""

from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING

import warp as wp
from isaaclab_newton.physics.newton_manager import NewtonManager
from newton import Contacts, Control, Model, ModelBuilder, State
from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx
from newton.solvers import SolverBase, SolverFeatherstone, SolverVBD

from isaaclab.sim.utils.stage import get_current_stage

from .deformable_object import (
    add_deformable_entry_to_builder,
    clear_deformable_builder_hooks,
    install_deformable_builder_hooks,
    setup_registered_deformable_fabric_sync,
)
from .kernels import _kernel_body_particle_reaction
from .newton_manager_cfg import CoupledFeatherstoneVBDSolverCfg

if TYPE_CHECKING:
    from isaaclab.sim.simulation_context import SimulationContext

logger = logging.getLogger(__name__)


class NewtonCoupledFeatherstoneVBDManager(NewtonManager):
    """:class:`NewtonManager` specialization for the coupled Featherstone + VBD
    solver. Due to Newton deformables not being properly integrated yet, this
    manager uses the same temporary solutions from VBD Manager.

    Always uses Newton's :class:`CollisionPipeline` for contact handling.
    """

    _rigid_solver: SolverFeatherstone
    _soft_solver: SolverVBD
    _coupling_mode: str | None = None

    @classmethod
    def initialize(cls, sim_context: SimulationContext) -> None:
        """Initialize the manager with simulation context.

        Args:
            sim_context: Parent simulation context.

        TODO: Subclass should not override this method, once deformables
        supported on Newton import_usd, this can be unified with NewtonManager's
        implementation.
        """

        # Deformable body registry and extension hooks.
        # Experimental deformable support registers callbacks here so the manager
        # and cloner can invoke them without hard-coding deformable logic.
        install_deformable_builder_hooks()

        super().initialize(sim_context)

    @classmethod
    def step(cls) -> None:
        """Step the physics simulation."""
        from isaaclab.physics import PhysicsManager

        sim = PhysicsManager._sim
        if sim is None or not sim.is_playing():
            return

        # Notify solver of model changes
        if cls._model_changes:
            with wp.ScopedDevice(PhysicsManager._device):
                for change in cls._model_changes:
                    cls._rigid_solver.notify_model_changed(change)
                    cls._soft_solver.notify_model_changed(change)
                NewtonManager._model_changes = set()
        super().step()

    @classmethod
    def _solver_specific_clear(cls):
        """Clear VBD-specific state."""
        clear_deformable_builder_hooks()

    @classmethod
    def _get_deformable_ignore_paths(cls) -> list[str]:
        """Return USD prim paths to skip when calling ``builder.add_usd``.

        For each registered deformable body, both the simulation mesh (which
        carries ``UsdPhysics.CollisionAPI``) and the visual mesh are returned.
        The sim mesh must be skipped so Newton does not create a redundant
        static mesh collider alongside the particles produced by
        ``add_soft_mesh``.  The visual mesh is skipped so Newton does not
        treat it as a collider — Kit reads it directly from USD for rendering.

        Paths may contain regex patterns; Newton's ``add_usd`` matches them
        via :func:`re.match`.
        """
        paths: list[str] = []
        for entry in cls._deformable_registry:
            paths.append(entry.sim_mesh_prim_path)
            paths.append(entry.vis_mesh_prim_path)
        return paths

    @classmethod
    def start_simulation(cls) -> None:
        """Start simulation by finalizing model and initializing state.

        This function finalizes the model and initializes the simulation state.
        Note: Collision pipeline is initialized later in initialize_solver() after
        we determine whether the solver needs external collision detection.

        TODO: Subclass should not override this method, missing piece is
        having Newton bind a surface mesh to volume deformable tetrahedral mesh
        in addition to removing the deformable_registry data structure.
        """
        super().start_simulation()

        # Apply global model parameters from :class:`NewtonModelCfg` to the finalized model.
        # Sets ``soft_contact_ke/kd/mu`` and optionally overrides per-shape
        # ``shape_material_ke/kd/mu`` on the Newton model.
        from isaaclab.physics import PhysicsManager

        cfg = PhysicsManager._cfg
        if cfg is not None and hasattr(cfg, "model_cfg") and cfg.model_cfg is not None:
            model = cls._model
            if model is None:
                return

            model_cfg = cfg.model_cfg
            model.soft_contact_ke = float(model_cfg.soft_contact_ke)
            model.soft_contact_kd = float(model_cfg.soft_contact_kd)
            model.soft_contact_mu = float(model_cfg.soft_contact_mu)

            if model_cfg.shape_material_ke is not None:
                model.shape_material_ke.fill_(float(model_cfg.shape_material_ke))
            if model_cfg.shape_material_kd is not None:
                model.shape_material_kd.fill_(float(model_cfg.shape_material_kd))
            if model_cfg.shape_material_mu is not None:
                model.shape_material_mu.fill_(float(model_cfg.shape_material_mu))

        # Setup USD/Fabric sync for Kit viewport deformable rendering
        setup_registered_deformable_fabric_sync(cls)

    @classmethod
    def instantiate_builder_from_stage(cls):
        """Create builder from USD stage with special treatment for deformable
        bodies, as these are not read from USD yet.

        Detects env Xforms (e.g. ``/World/Env_0``, ``/World/Env_1``) and builds
        each as a separate Newton world via ``begin_world``/``end_world``.
        Falls back to a flat ``add_usd`` when no env Xforms are found.

        TODO: Subclass should not override this method, once deformables
        supported on Newton import_usd, this can be unified with NewtonManager's
        implementation.
        """
        import re

        from pxr import UsdGeom

        stage = get_current_stage()
        up_axis = UsdGeom.GetStageUpAxis(stage)

        # Scan /World children for env-like Xforms (Env_0, env_1, ...)
        env_pattern = re.compile(r"^[Ee]nv_(\d+)$")
        world_prim = stage.GetPrimAtPath("/World")
        env_paths: list[tuple[int, str]] = []
        if world_prim and world_prim.IsValid():
            for child in world_prim.GetChildren():
                m = env_pattern.match(child.GetName())
                if m:
                    env_paths.append((int(m.group(1)), child.GetPath().pathString))
        env_paths.sort(key=lambda x: x[0])

        builder = ModelBuilder(up_axis=up_axis)

        schema_resolvers = [SchemaResolverNewton(), SchemaResolverPhysx()]

        # Deformable sim/visual mesh paths must be skipped by ``add_usd``
        # so they don't get duplicated as static colliders.
        deformable_ignore_paths = cls._get_deformable_ignore_paths()

        if not env_paths:
            # No env Xforms — flat loading
            builder.add_usd(stage, ignore_paths=deformable_ignore_paths, schema_resolvers=schema_resolvers)

            # Add deformable bodies from the registry (single world at origin).
            for entry in cls._deformable_registry:
                add_deformable_entry_to_builder(builder, entry, 0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
        else:
            # Load everything except the env subtrees (ground plane, lights, etc.)
            ignore_paths = [path for _, path in env_paths] + deformable_ignore_paths
            builder.add_usd(stage, ignore_paths=ignore_paths, schema_resolvers=schema_resolvers)

            # Build a prototype from the first env (all envs assumed identical)
            _, proto_path = env_paths[0]
            proto = ModelBuilder(up_axis=up_axis)
            proto.add_usd(
                stage,
                root_path=proto_path,
                ignore_paths=deformable_ignore_paths,
                schema_resolvers=schema_resolvers,
            )

            # Inject registered sites into the proto before replication
            global_sites, proto_sites, world_sites = cls._cl_inject_sites(builder, {proto_path: proto})
            global_site_map: dict[str, tuple[int, None]] = {label: (idx, None) for label, idx in global_sites.items()}
            num_worlds = len(env_paths)
            local_site_map: dict[str, list[list[int]]] = {}
            site_entries = proto_sites.get(id(proto), {})

            # Add each env as a separate Newton world
            xform_cache = UsdGeom.XformCache()
            for col, (_, env_path) in enumerate(env_paths):
                builder.begin_world()
                offset = builder.shape_count
                world_xform = xform_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(env_path))
                translation = world_xform.ExtractTranslation()
                rotation = world_xform.ExtractRotationQuat()
                pos = (translation[0], translation[1], translation[2])
                quat = (
                    rotation.GetImaginary()[0],
                    rotation.GetImaginary()[1],
                    rotation.GetImaginary()[2],
                    rotation.GetReal(),
                )
                env_xform = wp.transform(pos, quat)
                builder.add_builder(proto, xform=env_xform)
                for label, xform in world_sites.items():
                    if label not in local_site_map:
                        local_site_map[label] = [[] for _ in range(num_worlds)]
                    site_idx = builder.add_site(body=-1, xform=wp.transform_multiply(env_xform, xform), label=label)
                    local_site_map[label][col].append(site_idx)
                for label, proto_shape_indices in site_entries.items():
                    if label not in local_site_map:
                        local_site_map[label] = [[] for _ in range(num_worlds)]
                    for proto_shape_idx in proto_shape_indices:
                        local_site_map[label][col].append(offset + proto_shape_idx)

                # Add deformable bodies from the registry into this world.
                for entry in cls._deformable_registry:
                    add_deformable_entry_to_builder(builder, entry, col, list(pos), quat)

                builder.end_world()

            NewtonManager._cl_site_index_map = {
                **global_site_map,
                **{label: (None, per_world) for label, per_world in local_site_map.items()},
            }
            NewtonManager._num_envs = len(env_paths)

        # Call builder.color() if any deformable entries were added (required by VBD solver)
        if cls._deformable_registry:
            builder.color()

        cls.set_builder(builder)

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: CoupledFeatherstoneVBDSolverCfg) -> None:
        """Construct a custom coupling between two solvers and populate the
        base-class slots.

        VBD always uses Newton's :class:`CollisionPipeline` and steps with
        separate input/output states, so the flags are fixed.
        """
        cls._coupling_mode = solver_cfg.coupling_mode

        valid = set(inspect.signature(SolverFeatherstone.__init__).parameters) - {"self", "model"}
        kwargs = {k: v for k, v in solver_cfg.rigid_solver_cfg.to_dict().items() if k in valid}
        cls._rigid_solver = SolverFeatherstone(model, **kwargs)

        valid = set(inspect.signature(SolverVBD.__init__).parameters) - {"self", "model"}
        kwargs = {k: v for k, v in solver_cfg.soft_solver_cfg.to_dict().items() if k in valid}
        cls._soft_solver = SolverVBD(model, **kwargs)

        # Dummy solver for the newtonmanager
        NewtonManager._solver = SolverBase(model)

        NewtonManager._use_single_state = False
        NewtonManager._needs_collision_pipeline = True

        if solver_cfg.coupling_mode == "kinematic":
            cls._gravity_zero = wp.zeros(1, dtype=wp.vec3)
            cls._gravity_saved = wp.clone(model.gravity)
            # Save original PD gains and create zeroed versions for kinematic step
            cls._ke_saved = wp.clone(model.joint_target_ke)
            cls._kd_saved = wp.clone(model.joint_target_kd)
            cls._ke_zero = wp.zeros_like(model.joint_target_ke)
            cls._kd_zero = wp.zeros_like(model.joint_target_kd)

    @classmethod
    def _step_solver(
        cls, state_in: State, state_out: State, control: Control, contacts: Contacts | None, substep_dt: float
    ) -> None:
        """One coupled substep.

        Args:
            state_in: Current state (read/write).
            state_out: Next state (write).
            control: Joint-level control inputs.
            contacts: Ignored -- the solver uses its own internal contacts.
            dt: Substep timestep [s].
        """
        if cls._coupling_mode == "kinematic":
            cls._step_kinematic(state_in, state_out, control, substep_dt)
        elif cls._coupling_mode == "one_way":
            cls._step_one_way(state_in, state_out, control, substep_dt)
        else:
            cls._step_two_way(state_in, state_out, control, substep_dt)

    @classmethod
    def _simulate_physics_only(cls) -> None:
        # Rebuild BVH once per step for solvers that require it (e.g. VBD cloth).
        if hasattr(cls._soft_solver, "rebuild_bvh"):
            cls._soft_solver.rebuild_bvh(cls._state_0)
        super()._simulate_physics_only()

    @classmethod
    def _step_kinematic(cls, state_in: State, state_out: State, control: Control, dt: float) -> None:
        """Kinematic coupling: mirrors some Newton examples (e.g. softbody_franka) exactly.

        1. Clear forces.
        2. Assign joint_qd from control targets (velocity = (target - current) / frame_dt).
        3. Disable gravity and rigid contacts for the rigid solver step.
        4. Step rigid solver as kinematic integrator (q += qd * dt).
        5. Restore gravity, collision detect, VBD step.
        """
        model = cls._model

        # 1. Clear forces
        state_in.clear_forces()
        state_out.clear_forces()

        # 2. Kinematic rigid step: assign qd, disable gravity/contacts/PD gains
        saved_particle_count = model.particle_count
        saved_shape_contact_pair_count = model.shape_contact_pair_count
        model.particle_count = 0
        model.gravity.assign(cls._gravity_zero)
        model.shape_contact_pair_count = 0

        # Zero out PD gains so rigid solver (Featherstone) acts as a pure kinematic integrator
        model.joint_target_ke.assign(cls._ke_zero)
        model.joint_target_kd.assign(cls._kd_zero)

        # Assign joint velocities from control targets
        state_in.joint_qd.assign(control.joint_target_vel)

        cls._rigid_solver.step(state_in, state_out, control, None, dt)

        # 3. Restore everything
        state_in.particle_f.zero_()
        model.particle_count = saved_particle_count
        model.gravity.assign(cls._gravity_saved)
        model.shape_contact_pair_count = saved_shape_contact_pair_count
        model.joint_target_ke.assign(cls._ke_saved)
        model.joint_target_kd.assign(cls._kd_saved)

        # 4. Collision detection
        cls._collision_pipeline.collide(state_in, cls._contacts)

        # 5. VBD step
        cls._soft_solver.step(state_in, state_out, control, cls._contacts, dt)

    @classmethod
    def _step_one_way(cls, state_in: State, state_out: State, control: Control, dt: float) -> None:
        """One-way coupling: collide, then rigid step, then VBD."""
        # 1. Clear forces
        state_in.clear_forces()
        state_out.clear_forces()

        # 2. Collision detection (cloth-body contacts)
        cls._collision_pipeline.collide(state_in, cls._contacts)

        # 3. Rigid-body step (does not read soft-contact reactions)
        cls._rigid_step(state_in, state_out, control, dt)

        # 4. Clear spurious particle forces from rigid step
        state_in.particle_f.zero_()

        # 5. VBD step -- particles only, reads updated rigid poses
        cls._soft_solver.step(state_in, state_out, control, cls._contacts, dt)

    @classmethod
    def _step_two_way(cls, state_in: State, state_out: State, control: Control, dt: float) -> None:
        """Two-way coupling: collide, inject reactions into body_f, rigid step, VBD step."""
        # 1. Clear forces
        state_in.clear_forces()
        state_out.clear_forces()

        # 2. Collision detection BEFORE rigid step
        cls._collision_pipeline.collide(state_in, cls._contacts)

        # 3. Inject contact reaction forces into body_f.
        #    state_out holds the previous substep's body_q (states swap each
        #    substep), used for finite-difference body velocity in friction.
        #    particle_q_prev is reconstructed from particle_qd inside the
        #    kernel because VBD mutates particle_q in place, so the swapped
        #    state's particle_q is not a clean prior-substep snapshot.
        if state_in.body_f is not None:
            cls._apply_reactions(state_in, state_out, dt)

        # 4. Rigid-body step (reads body_f for soft-contact reactions)
        cls._rigid_step(state_in, state_out, control, dt)

        # 5. Clear spurious particle forces from rigid step
        state_in.particle_f.zero_()

        # 6. VBD step -- uses same contacts detected in step 2
        cls._soft_solver.step(state_in, state_out, control, cls._contacts, dt)

    @classmethod
    def _rigid_step(cls, state_in: State, state_out: State, control: Control, dt: float) -> None:
        """Advance rigid bodies with the configured sub-solver."""
        model = cls._model

        # set particle_count = 0 to disable particle simulation in robot solver
        saved_particle_count = model.particle_count
        model.particle_count = 0

        cls._rigid_solver.step(state_in, state_out, control, None, dt)

        # restore original settings
        model.particle_count = saved_particle_count

    @classmethod
    def _apply_reactions(cls, state: State, state_prev: State, dt: float) -> None:
        """Launch the reaction kernel to inject normal + friction forces into body_f.

        Args:
            state: Current state with particle positions/velocities and body state.
            state_prev: Previous substep state whose ``body_q`` provides
                the reference poses for finite-difference body velocity.
            dt: Substep timestep [s].
        """
        model = cls._model
        contacts = cls._contacts

        if contacts is None:
            return

        contact_capacity = int(contacts.soft_contact_particle.shape[0])
        if contact_capacity == 0:
            return

        # The kernel reconstructs particle_q_prev from particle_qd internally:
        # state_prev.particle_q is unreliable because VBD mutates particle_q
        # in place during its iteration, so the swapped state's particle_q is
        # not a clean snapshot of the prior substep.
        wp.launch(
            _kernel_body_particle_reaction,
            dim=contact_capacity,
            inputs=[
                contacts.soft_contact_count,
                contacts.soft_contact_particle,
                contacts.soft_contact_shape,
                contacts.soft_contact_body_pos,
                contacts.soft_contact_body_vel,
                contacts.soft_contact_normal,
                state.particle_q,
                state.particle_qd,
                model.particle_radius,
                state.body_q,
                state_prev.body_q,
                state.body_qd,
                model.body_com,
                model.shape_body,
                model.shape_material_mu,
                float(model.soft_contact_ke),
                float(model.soft_contact_kd),
                float(model.soft_contact_mu),
                float(cls._soft_solver.friction_epsilon),
                float(dt),
                state.body_f,
            ],
        )
