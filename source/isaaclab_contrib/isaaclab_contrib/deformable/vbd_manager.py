# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""VBD Newton manager."""

from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING

import numpy as np
import warp as wp
from isaaclab_newton.physics.newton_manager import NewtonManager
from newton import JointType, Model, ModelBuilder, eval_fk
from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx
from newton.solvers import SolverVBD

from isaaclab.physics import PhysicsManager
from isaaclab.sim.utils.stage import get_current_stage

from isaaclab_contrib.cable.cable_object import install_cable_builder_hooks

from .deformable_object import install_deformable_builder_hooks
from .newton_manager_cfg import VBDSolverCfg

if TYPE_CHECKING:
    from isaaclab.sim.simulation_context import SimulationContext

logger = logging.getLogger(__name__)


@wp.kernel(enable_backward=False)
def _sync_cable_curve_points(
    fabric_points: wp.fabricarrayarray(dtype=wp.vec3f),
    fabric_world_matrices: wp.fabricarray(dtype=wp.mat44d),
    body_offsets: wp.fabricarray(dtype=wp.uint32),
    body_counts: wp.fabricarray(dtype=wp.uint32),
    last_edge_lengths: wp.fabricarray(dtype=wp.float32),
    body_q: wp.array(ndim=1, dtype=wp.transformf),
):
    """Reconstruct ``UsdGeomBasisCurves`` control points from cable body transforms."""
    i = wp.tid()
    offset = int(body_offsets[i])
    count = int(body_counts[i])
    inv_world = wp.inverse(wp.transpose(wp.mat44f(fabric_world_matrices[i])))

    for j in range(count):
        node_world = wp.transform_get_translation(body_q[offset + j])
        fabric_points[i][j] = wp.transform_point(inv_world, node_world)

    tail_world = wp.transform_point(body_q[offset + count - 1], wp.vec3(0.0, 0.0, float(last_edge_lengths[i])))
    fabric_points[i][count] = wp.transform_point(inv_world, tail_world)


class NewtonVBDManager(NewtonManager):
    """:class:`NewtonManager` specialization for the VBD solver.

    Always uses Newton's :class:`CollisionPipeline` for contact handling.
    """

    _newton_cable_body_offset_attr = "newton:cableBodyOffset"
    _newton_cable_body_count_attr = "newton:cableBodyCount"
    _newton_cable_last_edge_length_attr = "newton:cableLastEdgeLength"
    _curves_dirty: bool = False
    _cable_body_q_cpu = None
    _non_cable_articulation_mask: wp.array | None = None
    """(articulation_count,) wp.bool — False for articulations containing
    :attr:`newton.JointType.CABLE` joints, True elsewhere. Used to skip cable
    articulations in :meth:`forward` because Newton's ``eval_fk`` does not
    handle cable joints (their relative transform falls through to identity,
    collapsing rod segments onto their parent anchors).
    """

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
        install_cable_builder_hooks()

        super().initialize(sim_context)

    @classmethod
    def _solver_specific_clear(cls) -> None:
        """Clear VBD-specific Fabric sync state and shared builder hooks."""
        cls._curves_dirty = False
        cls._cable_body_q_cpu = None
        cls._non_cable_articulation_mask = None
        NewtonManager._cable_registry = []
        NewtonManager._deformable_registry = []
        NewtonManager._per_world_builder_hooks = []

    @classmethod
    def _mark_curves_dirty(cls) -> None:
        """Flag that cable curve points have changed and Fabric needs re-sync."""
        cls._curves_dirty = True

    @classmethod
    def _mark_state_dirty(cls) -> None:
        """Flag that all VBD state has changed and Fabric needs re-sync."""
        super()._mark_state_dirty()
        cls._mark_curves_dirty()

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

        # Newton's ``eval_fk`` has no case for :attr:`newton.JointType.CABLE`, so the unmasked
        # ``eval_fk`` at the end of :meth:`NewtonManager.start_simulation` collapsed every cable
        # capsule onto its parent joint anchor (same failure mode that motivates the mask for
        # later FK passes). Drop the corrupted states and rebuild them from ``model.body_q``
        # (untouched by ``eval_fk``), then re-run :meth:`forward` with the cable mask to seed
        # non-cable ``body_q`` from joint coordinates without touching cables.
        # NOTE: Can be removed once Newton patches cable joints in eval_fk.
        cls._build_non_cable_articulation_mask()
        if cls._non_cable_articulation_mask is not None and cls._model is not None:
            cls._state_0 = cls._model.state()
            cls._state_1 = cls._model.state()
            cls.forward()

        # Apply global model parameters from :class:`NewtonModelCfg` to the finalized model.
        # Sets ``soft_contact_ke/kd/mu`` and optionally overrides per-shape
        # ``shape_material_ke/kd/mu`` on the Newton model.
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
        if not cls._clone_physics_only and cls._deformable_registry:
            import re

            import usdrt

            if NewtonManager._usdrt_stage is None:
                NewtonManager._usdrt_stage = get_current_stage(fabric=True)

            stage = get_current_stage()
            for entry in cls._deformable_registry:
                for inst_idx, offset in enumerate(entry.particle_offsets):
                    # Resolve regex pattern to concrete instance path of visual mesh
                    resolved_vis = re.sub(r"(?<=[Ee]nv_)\.\*", str(inst_idx), entry.vis_mesh_prim_path)
                    resolved_vis = re.sub(r"\.\*", str(inst_idx), resolved_vis)
                    vis_prim = stage.GetPrimAtPath(resolved_vis)

                    if not vis_prim or not vis_prim.IsValid():
                        logger.warning("[setup_fabric_particle_sync] vis prim not found at %s", resolved_vis)
                        continue

                    # Create per-instance particle offset and count attributes on the visual mesh
                    # prim so the Fabric sync kernel can find the right slice of particle_q
                    # and iterate only over this body's particles (counts vary across bodies).
                    fab_prim = NewtonManager._usdrt_stage.GetPrimAtPath(vis_prim.GetPath().pathString)
                    fab_prim.CreateAttribute(
                        NewtonManager._newton_particle_offset_attr, usdrt.Sdf.ValueTypeNames.UInt, True
                    )
                    fab_prim.GetAttribute(NewtonManager._newton_particle_offset_attr).Set(offset)
                    fab_prim.CreateAttribute(
                        NewtonManager._newton_particle_count_attr, usdrt.Sdf.ValueTypeNames.UInt, True
                    )
                    fab_prim.GetAttribute(NewtonManager._newton_particle_count_attr).Set(entry.particles_per_body)

            cls._mark_particles_dirty()
            cls.sync_particles_to_usd()

        if not cls._clone_physics_only and cls._cable_registry:
            import re

            import usdrt

            if NewtonManager._usdrt_stage is None:
                NewtonManager._usdrt_stage = get_current_stage(fabric=True)

            stage = get_current_stage()
            curves_registered = False
            for entry in cls._cable_registry:
                curve_template_path = entry.curve_prim_path or f"{entry.prim_path}/geometry/mesh"
                for inst_idx, body_offset in enumerate(entry.body_offsets):
                    resolved = re.sub(r"(?<=[Ee]nv_)\.\*", str(inst_idx), curve_template_path)
                    resolved = re.sub(r"\.\*", str(inst_idx), resolved)
                    curve_prim = stage.GetPrimAtPath(resolved)
                    if not curve_prim or not curve_prim.IsValid():
                        logger.warning("[setup_fabric_cable_sync] curve prim not found at %s", resolved)
                        continue
                    usd_points = curve_prim.GetAttribute("points").Get()
                    expected_points = len(entry.edges) + 1
                    if usd_points is None or len(usd_points) != expected_points:
                        logger.warning(
                            "[setup_fabric_cable_sync] curve %s has %s points, expected %d; skipping.",
                            resolved,
                            0 if usd_points is None else len(usd_points),
                            expected_points,
                        )
                        continue
                    fab_prim = NewtonManager._usdrt_stage.GetPrimAtPath(curve_prim.GetPath().pathString)
                    xformable_prim = usdrt.Rt.Xformable(fab_prim)
                    if not xformable_prim.HasWorldXform():
                        xformable_prim.SetWorldXformFromUsd()
                    # Pre-seed Fabric ``points``: without this Hydra reads an empty array on frame 0.
                    fab_points_attr = fab_prim.GetAttribute("points")
                    if not fab_points_attr.IsValid():
                        fab_points_attr = fab_prim.CreateAttribute(
                            "points", usdrt.Sdf.ValueTypeNames.Point3fArray, True
                        )
                    fab_points_attr.Set(
                        usdrt.Vt.Vec3fArray([usdrt.Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in usd_points])
                    )
                    fab_prim.CreateAttribute(cls._newton_cable_body_offset_attr, usdrt.Sdf.ValueTypeNames.UInt, True)
                    fab_prim.GetAttribute(cls._newton_cable_body_offset_attr).Set(int(body_offset))
                    fab_prim.CreateAttribute(cls._newton_cable_body_count_attr, usdrt.Sdf.ValueTypeNames.UInt, True)
                    fab_prim.GetAttribute(cls._newton_cable_body_count_attr).Set(len(entry.edges))
                    fab_prim.CreateAttribute(
                        cls._newton_cable_last_edge_length_attr, usdrt.Sdf.ValueTypeNames.Float, True
                    )
                    fab_prim.GetAttribute(cls._newton_cable_last_edge_length_attr).Set(float(entry.last_edge_length))
                    curves_registered = True
            if curves_registered:
                cls._mark_curves_dirty()

    @classmethod
    def _build_non_cable_articulation_mask(cls) -> None:
        """Build :attr:`_non_cable_articulation_mask` from finalized joint topology.
        NOTE: Can be removed once Newton patches cable joints in eval_fk.

        Walks :attr:`newton.Model.joint_type` and :attr:`newton.Model.joint_articulation`
        to find articulations that contain at least one :attr:`newton.JointType.CABLE`
        joint, then allocates a device-resident boolean mask that is ``False`` for
        those articulations and ``True`` elsewhere. Leaves the mask as ``None``
        when there are no cables registered so :meth:`forward` can take the
        unmasked fast path via ``super().forward()``.

        Raises:
            RuntimeError: If cables are registered but the finalized model is
                missing the joint topology needed to build the mask, or contains
                no :attr:`newton.JointType.CABLE` joints. Falling through to
                ``super().forward()`` in those cases would corrupt cable
                ``body_q`` silently each render.
        """
        if not cls._cable_registry:
            return

        model = cls._model
        if model is None or model.joint_type is None or model.joint_articulation is None:
            raise RuntimeError(
                "Cannot build non-cable articulation mask: cables are registered but Newton model"
                " state is incomplete (missing model/joint_type/joint_articulation). Without the"
                " mask, `forward()` calls eval_fk on cable joints and silently collapses rod"
                " segments onto their parent anchors."
            )
        if model.articulation_count == 0:
            raise RuntimeError(
                "Cannot build non-cable articulation mask: cables are registered but the finalized"
                " model has zero articulations."
            )

        joint_type_np = model.joint_type.numpy()
        joint_articulation_np = model.joint_articulation.numpy()
        cable_art_ids = {
            int(joint_articulation_np[j])
            for j in range(len(joint_type_np))
            if int(joint_type_np[j]) == int(JointType.CABLE) and int(joint_articulation_np[j]) >= 0
        }
        if not cable_art_ids:
            raise RuntimeError(
                "Cannot build non-cable articulation mask: cables are registered but the finalized"
                " model has no JointType.CABLE joints. The cable replicate hook likely did not run."
            )

        mask_np = np.ones(model.articulation_count, dtype=np.bool_)
        for art_id in cable_art_ids:
            mask_np[art_id] = False
        cls._non_cable_articulation_mask = wp.array(mask_np, dtype=wp.bool, device=PhysicsManager._device)

    @classmethod
    def forward(cls) -> None:
        """Update articulation kinematics, skipping cable articulations.
        NOTE: Can be removed once Newton patches cable joints in eval_fk.

        Newton's ``eval_fk`` has no case for :attr:`newton.JointType.CABLE`, so a
        cable joint's relative transform falls through to the identity, snapping
        each child segment onto its parent's joint anchor and destroying the
        rod state that VBD integrated directly into ``body_q``. This override
        passes :attr:`_non_cable_articulation_mask` so cable articulations are
        excluded from the FK pass triggered by Kit-style visualizers (which set
        :meth:`~isaaclab.visualizers.BaseVisualizer.requires_forward_before_step`
        to ``True``).
        """
        if cls._non_cable_articulation_mask is None:
            if cls._cable_registry:
                raise RuntimeError(
                    "Cables are registered but `_non_cable_articulation_mask` is None — refusing to"
                    " fall through to the unmasked eval_fk that would corrupt cable body_q. The mask"
                    " is built in `start_simulation()`; ensure it has run."
                )
            super().forward()
            return
        eval_fk(
            cls._model,
            cls._state_0.joint_q,
            cls._state_0.joint_qd,
            cls._state_0,
            cls._non_cable_articulation_mask,
        )

    @classmethod
    def sync_curves_to_usd(cls) -> None:
        """Update cable ``UsdGeomBasisCurves.points`` from Newton ``body_q``.

        Runs on the CPU Fabric device because Kit/Hydra reads that bucket for
        runtime-spawned ``UsdGeomBasisCurves``.
        """
        if cls._usdrt_stage is None or cls._state_0 is None or cls._state_0.body_q is None:
            return
        if not getattr(cls, "_cable_registry", None) or not cls._curves_dirty:
            return
        import usdrt

        selection = cls._usdrt_stage.SelectPrims(
            require_attrs=[
                (usdrt.Sdf.ValueTypeNames.Point3fArray, "points", usdrt.Usd.Access.ReadWrite),
                (usdrt.Sdf.ValueTypeNames.UInt, cls._newton_cable_body_offset_attr, usdrt.Usd.Access.Read),
                (usdrt.Sdf.ValueTypeNames.UInt, cls._newton_cable_body_count_attr, usdrt.Usd.Access.Read),
                (usdrt.Sdf.ValueTypeNames.Float, cls._newton_cable_last_edge_length_attr, usdrt.Usd.Access.Read),
                (usdrt.Sdf.ValueTypeNames.Matrix4d, "omni:fabric:worldMatrix", usdrt.Usd.Access.Read),
            ],
            device="cpu",
        )
        if selection.GetCount() == 0:
            return

        # wp.launch requires inputs on the same device as the launch.
        if cls._cable_body_q_cpu is None or cls._cable_body_q_cpu.shape != cls._state_0.body_q.shape:
            cls._cable_body_q_cpu = wp.empty_like(cls._state_0.body_q, device="cpu")
        wp.copy(cls._cable_body_q_cpu, cls._state_0.body_q)

        wp.launch(
            _sync_cable_curve_points,
            dim=selection.GetCount(),
            inputs=[
                wp.fabricarrayarray(data=selection, attrib="points", dtype=wp.vec3f),
                wp.fabricarray(data=selection, attrib="omni:fabric:worldMatrix"),
                wp.fabricarray(data=selection, attrib=cls._newton_cable_body_offset_attr),
                wp.fabricarray(data=selection, attrib=cls._newton_cable_body_count_attr),
                wp.fabricarray(data=selection, attrib=cls._newton_cable_last_edge_length_attr),
                cls._cable_body_q_cpu,
            ],
            device="cpu",
        )
        cls._curves_dirty = False

    @classmethod
    def pre_render(cls) -> None:
        super().pre_render()
        cls.sync_curves_to_usd()

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

            # Run per-world builder hooks for the single world at origin.
            # Hooks include deformable and cable registries; each owns its own registration.
            if hasattr(cls, "_per_world_builder_hooks"):
                for hook in cls._per_world_builder_hooks:
                    hook(builder, 0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
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
            global_sites, proto_sites = cls._cl_inject_sites(builder, {proto_path: proto})
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
                builder.add_builder(proto, xform=wp.transform(pos, quat))
                for label, proto_shape_indices in site_entries.items():
                    if label not in local_site_map:
                        local_site_map[label] = [[] for _ in range(num_worlds)]
                    for proto_shape_idx in proto_shape_indices:
                        local_site_map[label][col].append(offset + proto_shape_idx)

                # Run per-world builder hooks for this world (deformables, cables, ...).
                if hasattr(cls, "_per_world_builder_hooks"):
                    for hook in cls._per_world_builder_hooks:
                        hook(builder, col, list(pos), list(quat))

                builder.end_world()

            NewtonManager._cl_site_index_map = {
                **global_site_map,
                **{label: (None, per_world) for label, per_world in local_site_map.items()},
            }
            NewtonManager._num_envs = len(env_paths)

        # run vbd builder coloring
        builder.color()

        cls.set_builder(builder)

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: VBDSolverCfg) -> None:
        """Construct :class:`SolverVBD` and populate the base-class slots.

        VBD always uses Newton's :class:`CollisionPipeline` and steps with
        separate input/output states, so the flags are fixed.
        """
        valid = set(inspect.signature(SolverVBD.__init__).parameters) - {"self", "model"}
        kwargs = {k: v for k, v in solver_cfg.to_dict().items() if k in valid}
        NewtonManager._solver = SolverVBD(model, **kwargs)
        NewtonManager._use_single_state = False
        NewtonManager._needs_collision_pipeline = True

    @classmethod
    def _simulate_physics_only(cls) -> None:
        # Rebuild BVH once per step for solvers that require it (e.g. VBD cloth).
        # Guard against Newton versions where ``SolverVBD`` did not initialize
        # ``particle_enable_self_contact`` when ``model.particle_count == 0``
        # (rigid-body-only or cable-only scenes).  In that case ``rebuild_bvh``
        # would raise ``AttributeError``; the call is a no-op anyway since there
        # are no particles to rebuild BVH for.
        if hasattr(cls._solver, "rebuild_bvh") and getattr(cls._solver, "particle_enable_self_contact", False):
            cls._solver.rebuild_bvh(cls._state_0)
        super()._simulate_physics_only()
