# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cable / 1D-rod asset class, registry entry, and replicate-hook plumbing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import newton
import warp as wp
from isaaclab_newton.assets.articulation.articulation import Articulation
from isaaclab_newton.physics import NewtonManager as SimulationManager

import isaaclab.sim as sim_utils

if TYPE_CHECKING:
    from .cable_object_cfg import CableObjectCfg


@dataclass
class CableRegistryEntry:
    """Mutable bridge between :class:`CableObject` and the per-world replicate hook."""

    prim_path: str
    node_positions: list[wp.vec3]
    edges: list[tuple[int, int]]
    radius: float
    curve_prim_path: str = ""

    init_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    init_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    stretch_stiffness: float = 1.0e9
    bend_stiffness: float = 0.0
    stretch_damping: float = 0.0
    bend_damping: float = 0.0
    density: float = 1500.0

    body_offsets: list[int] = field(default_factory=list)
    last_edge_length: float = 0.0
    # Per-env Newton body indices of each cable segment in edge order; outer list
    # indexed by ``world_idx``, inner indexed by ``CableAttachmentCfg.cable_anchor``.
    segment_body_indices: list[list[int]] = field(default_factory=list)


def add_cable_entry_to_builder(
    builder,
    entry: CableRegistryEntry,
    env_idx: int,
    env_position: list[float],
    env_rotation: list[float] | tuple[float, float, float, float],
    cable_idx: int = 0,
) -> None:
    """Add one cable to a Newton ``ModelBuilder`` for one environment.

    Args:
        builder: The Newton ``ModelBuilder``.
        entry: Registry entry describing the cable's geometry and material.
        env_idx: Zero-based environment (world) index.
        env_position: World translation ``[x, y, z]`` [m] for this environment.
        env_rotation: World orientation quaternion ``(x, y, z, w)`` for this environment.
        cable_idx: Zero-based cable index within :attr:`SimulationManager._cable_registry`.
    """
    if env_idx == 0:
        entry.body_offsets.clear()
        entry.segment_body_indices.clear()
        entry.last_edge_length = 0.0

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

    composed_pos = env_pos + wp.quat_rotate(env_rot, init_pos)
    composed_rot = env_rot * init_rot

    world_nodes: list[wp.vec3] = []
    for node in entry.node_positions:
        rotated = wp.quat_rotate(composed_rot, node)
        world_nodes.append(composed_pos + rotated)

    shape_cfg = newton.ModelBuilder.ShapeConfig()
    shape_cfg.density = float(entry.density)
    # Unique negative group disables segment-vs-segment self-collision (Newton
    # filters same-negative pairs) while keeping negative-vs-positive collisions.
    shape_cfg.collision_group = -(1 + cable_idx)

    # Pre-expand ``env_.*`` here: the cloner's ``_rename_builder_labels`` does
    # not visit builder-hook bodies, so we must produce the per-env label ourselves.
    expanded_prim_path = entry.prim_path.replace("env_.*", f"env_{env_idx}")
    entry.body_offsets.append(builder.body_count)
    rod_body_indices, _rod_joint_indices = builder.add_rod_graph(
        node_positions=world_nodes,
        edges=entry.edges,
        radius=entry.radius,
        cfg=shape_cfg,
        stretch_stiffness=entry.stretch_stiffness,
        stretch_damping=entry.stretch_damping,
        bend_stiffness=entry.bend_stiffness,
        bend_damping=entry.bend_damping,
        label=f"{expanded_prim_path}/cable",
        wrap_in_articulation=True,
    )
    entry.segment_body_indices.append(list(rod_body_indices))
    if env_idx == 0:
        u, v = entry.edges[-1]
        entry.last_edge_length = float(wp.length(entry.node_positions[v] - entry.node_positions[u]))


def add_registered_cables_to_builder(
    builder,
    world_idx: int,
    env_position: list[float],
    env_rotation: list[float] | tuple[float, float, float, float],
) -> None:
    """Per-world hook that registers all cables in :attr:`SimulationManager._cable_registry`."""
    for cable_idx, entry in enumerate(SimulationManager._cable_registry):
        add_cable_entry_to_builder(builder, entry, world_idx, env_position, env_rotation, cable_idx=cable_idx)


def apply_cable_attachments_to_builder(
    builder,
    world_idx: int,
    env_position: list[float],
    env_rotation: list[float] | tuple[float, float, float, float],
) -> None:
    """Per-world hook that realizes pending cable attachments as Newton fixed joints.

    Args:
        builder: The Newton ``ModelBuilder`` for the current scene.
        world_idx: Zero-based environment (world) index for this invocation.
        env_position: World translation ``[x, y, z]`` [m] for this environment.
        env_rotation: World orientation quaternion ``(x, y, z, w)`` for this environment.
    """
    pending = getattr(SimulationManager, "_pending_cable_attachments", None)
    if not pending:
        return

    def _to_wp_xform(pos, quat_xyzw):
        return wp.transform(
            (float(pos[0]), float(pos[1]), float(pos[2])),
            (float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2]), float(quat_xyzw[3])),
        )

    for cable_idx, attachment in pending:
        entry = SimulationManager._cable_registry[cable_idx]
        segments_in_world = entry.segment_body_indices[world_idx]
        num_segments = len(segments_in_world)
        anchor_idx = attachment.cable_anchor
        if not -num_segments <= anchor_idx < num_segments:
            raise ValueError(
                f"CableAttachmentCfg.cable_anchor={anchor_idx} is out of range for cable"
                f" '{entry.prim_path}' with {num_segments} segments;"
                f" valid range is [-{num_segments}, {num_segments - 1}]."
            )
        cable_body_idx = segments_in_world[anchor_idx]

        # Try both the unexpanded regex template and the per-env-expanded form:
        # USD-imported targets keep the template until the cloner's post-build
        # rewrite, while builder-hook targets are already per-env expanded.
        # Filter by ``body_world``: ``-1`` is Newton's "global" sentinel for
        # bodies added outside any ``begin_world``/``end_world`` block (e.g.
        # single-env flat path); world-specific matches win over global ones.
        target_path = attachment.target_prim_path
        expanded_target_path = target_path.replace("env_.*", f"env_{world_idx}")
        body_label = builder.body_label
        body_world = builder.body_world
        target_body_idx = -1
        for body_idx in range(len(body_label)):
            label = body_label[body_idx]
            if label != target_path and label != expanded_target_path:
                continue
            if body_world[body_idx] == world_idx or body_world[body_idx] == -1:
                target_body_idx = body_idx
                break
        if target_body_idx < 0:
            available_in_world = [body_label[i] for i in range(len(body_label)) if body_world[i] in (world_idx, -1)]
            searched = (
                target_path
                if target_path == expanded_target_path
                else f"{target_path!r} (also tried {expanded_target_path!r})"
            )
            raise ValueError(
                f"CableAttachmentCfg.target_prim_path {searched} did not match any body in world {world_idx}."
                f" Available body labels in this world: {available_in_world}."
            )

        parent_xform = _to_wp_xform(attachment.cable_local_pos, attachment.cable_local_quat)
        child_xform = _to_wp_xform(attachment.target_local_pos, attachment.target_local_quat)

        builder.add_joint_fixed(
            parent=cable_body_idx,
            child=target_body_idx,
            parent_xform=parent_xform,
            child_xform=child_xform,
            label=f"{entry.prim_path}/attachment_seg{anchor_idx}_w{world_idx}",
            collision_filter_parent=True,
        )


def install_cable_builder_hooks() -> None:
    """Reset the cable registry and install the per-world cable + attachment hooks."""
    SimulationManager._cable_registry = []
    SimulationManager._pending_cable_attachments = []
    if not hasattr(SimulationManager, "_per_world_builder_hooks"):
        SimulationManager._per_world_builder_hooks = []
    if add_registered_cables_to_builder not in SimulationManager._per_world_builder_hooks:
        SimulationManager._per_world_builder_hooks.append(add_registered_cables_to_builder)
    if apply_cable_attachments_to_builder not in SimulationManager._per_world_builder_hooks:
        SimulationManager._per_world_builder_hooks.append(apply_cable_attachments_to_builder)


class CableObject(Articulation):
    """Cable / 1D-rod asset (Newton backend)."""

    cfg: CableObjectCfg

    def __init__(self, cfg: CableObjectCfg):
        """Initialize the cable object.

        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg)

        self._registry_entry = self._register_cable()

        # Look up by identity via ``list.index`` rather than ``len(registry) - 1``
        # to stay robust if the base init ever mutates the registry concurrently.
        cable_idx = SimulationManager._cable_registry.index(self._registry_entry)
        for attachment in self.cfg.attachments:
            SimulationManager._pending_cable_attachments.append((cable_idx, attachment))

    def _register_cable(self) -> CableRegistryEntry:
        """Read cable geometry + material from the spawned USD prim and append to the registry.

        Returns:
            The registry entry (also appended to :attr:`SimulationManager._cable_registry`).

        Raises:
            ValueError: If the template prim has no ``UsdGeomBasisCurves``
                descendant, or the curve is missing its ``widths`` attribute.
            NotImplementedError: If more than one ``UsdGeomBasisCurves``
                descendant is found under the template prim — multi-curve
                cables under a single :class:`CableObject` are not supported.
            RuntimeError: If the template prim cannot be located, or the active
                Newton solver is not a VBD variant (only :class:`VBDSolverCfg`
                and its coupled variants register the cable builder hooks; no
                other Newton solver steps :attr:`newton.JointType.CABLE`).

        Note:
            ``pxr`` imports are deferred to this method (not module level) so
            that ``resolve_task_config`` can import the env-cfg module before
            Kit starts without polluting the ``pxr`` module cache.
        """
        # ``pxr`` import is deferred so ``resolve_task_config`` can import the
        # env-cfg module before Kit starts without polluting ``pxr`` caches.
        from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

        if not hasattr(SimulationManager, "_cable_registry"):
            raise RuntimeError(
                "CableObject can only be simulated under the Newton VBD solver"
                " (`VBDSolverCfg` or one of its coupled variants:"
                " `CoupledMJWarpVBDSolverCfg`, `CoupledFeatherstoneVBDSolverCfg`)."
                " The cable registry is installed by the VBD manager's `initialize()`"
                " hook via `install_cable_builder_hooks()`, and `JointType.CABLE`"
                " is not stepped by any other Newton solver. Switch the solver cfg"
                " or remove the CableObject from the scene."
            )

        if self.cfg.spawn is None:
            raise ValueError(
                f"CableObjectCfg(prim_path='{self.cfg.prim_path}') has no `spawn` configuration."
                " CableObject requires a `CableCfg` (or compatible USD-loading cfg) to register"
                " cable geometry; pass one via `CableObjectCfg.spawn`."
            )

        lookup_path = self.cfg.spawn.spawn_path if self.cfg.spawn.spawn_path is not None else self.cfg.prim_path
        template_prim = sim_utils.find_first_matching_prim(lookup_path)
        if template_prim is None:
            raise RuntimeError(f"Failed to find cable template prim for expression: '{lookup_path}'.")
        template_prim_path = template_prim.GetPrimPath()

        stage = template_prim.GetStage()
        curve_prims = [
            descendant for descendant in Usd.PrimRange(template_prim) if descendant.GetTypeName() == "BasisCurves"
        ]
        if not curve_prims:
            raise ValueError(f"No UsdGeomBasisCurves prim found under '{template_prim_path}'.")
        if len(curve_prims) > 1:
            paths = ", ".join(str(p.GetPrimPath()) for p in curve_prims)
            raise NotImplementedError(
                f"Found {len(curve_prims)} BasisCurves prims under '{template_prim_path}' ({paths}); "
                "multi-curve cables under a single CableObject are not supported yet."
            )
        curve_prim = curve_prims[0]
        curves = UsdGeom.BasisCurves(curve_prim)

        # Bake the curve prim's xform into node positions so the replicate hook
        # only needs to apply the env transform.
        xform_cache = UsdGeom.XformCache()
        curve_to_parent_frame = (
            xform_cache.GetLocalToWorldTransform(curve_prim)
            * xform_cache.GetLocalToWorldTransform(template_prim.GetParent()).GetInverse()
        )
        raw_points = curves.GetPointsAttr().Get()
        node_positions: list[wp.vec3] = []
        for p in raw_points:
            q = curve_to_parent_frame.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
            node_positions.append(wp.vec3(float(q[0]), float(q[1]), float(q[2])))

        raw_widths = curves.GetWidthsAttr().Get()
        if raw_widths is None or len(raw_widths) == 0:
            raise ValueError(f"UsdGeomBasisCurves at '{curve_prim.GetPrimPath()}' is missing the `widths` attribute.")
        widths_list = [float(w) for w in raw_widths]
        if max(widths_list) - min(widths_list) > 1e-9:
            raise ValueError(
                f"UsdGeomBasisCurves at '{curve_prim.GetPrimPath()}' has non-uniform `widths`"
                f" (min={min(widths_list)}, max={max(widths_list)}); tapered cables are not supported."
                " Author a constant width across all control points."
            )
        radius = widths_list[0] / 2.0

        connections_attr = curve_prim.GetAttribute("connections")
        if not connections_attr.IsValid() or connections_attr.Get() is None:
            raise ValueError(
                f"UsdGeomBasisCurves at '{curve_prim.GetPrimPath()}' is missing the `connections`"
                " attribute (expected `int2[]` listing each edge as a pair of control-point indices)."
                " Author this attribute on the curve prim — `spawn_cable` writes it automatically;"
                " user-imported curve USDs must add it explicitly."
            )
        edges = [(int(e[0]), int(e[1])) for e in connections_attr.Get()]

        # Material binding requires ``UsdPhysics.CollisionAPI`` on the curve;
        # without it the spawner's bind silently no-ops.
        material_targets = (
            UsdShade.MaterialBindingAPI(curve_prim).GetDirectBindingRel("physics").GetTargets()
            if curve_prim.HasAPI(UsdShade.MaterialBindingAPI)
            else []
        )
        material_prim = None
        for mat_path in material_targets:
            mat_prim = stage.GetPrimAtPath(mat_path)
            if mat_prim.GetAttribute("newton:density").IsValid():
                material_prim = mat_prim
                break
        if material_prim is None:
            has_collision_api = curve_prim.HasAPI(UsdPhysics.CollisionAPI)
            hint = (
                ""
                if has_collision_api
                else (
                    " Hint: the curve has no `UsdPhysics.CollisionAPI`, which `bind_physics_material`"
                    " requires; set `CableCfg.collision_props = sim_utils.CollisionPropertiesCfg()` so"
                    " `spawn_cable` applies the API (cables are currently Newton-only and the API has"
                    " no PhysX runtime effect)."
                )
            )
            raise ValueError(
                f"Could not find a Newton cable physics material bound to '{curve_prim.GetPrimPath()}'." + hint
            )

        def _get_material_attr(name: str, default):
            attr = material_prim.GetAttribute(name)
            return attr.Get() if attr.IsValid() else default

        stretch_stiffness = _get_material_attr("newton:stretchStiffness", CableRegistryEntry.stretch_stiffness)
        bend_stiffness = _get_material_attr("newton:bendStiffness", CableRegistryEntry.bend_stiffness)
        stretch_damping = _get_material_attr("newton:stretchDamping", CableRegistryEntry.stretch_damping)
        bend_damping = _get_material_attr("newton:bendDamping", CableRegistryEntry.bend_damping)
        density = _get_material_attr("newton:density", CableRegistryEntry.density)

        # init_pos/init_rot stay identity: the template xform is already baked
        # into ``node_positions``; the replicate hook only adds the env transform.
        entry = CableRegistryEntry(
            prim_path=self.cfg.prim_path,
            curve_prim_path=str(curve_prim.GetPrimPath()),
            node_positions=node_positions,
            edges=edges,
            radius=radius,
            stretch_stiffness=float(stretch_stiffness),
            bend_stiffness=float(bend_stiffness),
            stretch_damping=float(stretch_damping),
            bend_damping=float(bend_damping),
            density=float(density),
        )
        SimulationManager._cable_registry.append(entry)
        return entry
