# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cable / 1D-rod asset class, registry entry, and replicate-hook plumbing.

The structure mirrors :mod:`isaaclab_contrib.deformable.deformable_object`. Cables
differ from deformables in two respects only:

1. They subclass :class:`Articulation` (not :class:`BaseDeformableObject`) because
   ``newton.ModelBuilder.add_rod_graph`` produces a Newton articulation, and
   ``ArticulationView`` already covers state read/write.
2. Their material is consumed in-memory by the cable replicate hook (no USD
   read-back), since :class:`CableObject` always holds the source cfg.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import newton
import warp as wp


@dataclass
class CableRegistryEntry:
    """Mutable bridge between :class:`CableObject` and the replicate hook.

    Populated by :meth:`CableObject._register_cable` (reads the spawned
    ``UsdGeomBasisCurves`` and its Newton physics material) and consumed by
    :func:`add_cable_entry_to_builder`. Material-field semantics and defaults
    mirror :class:`~isaaclab_newton.sim.spawners.materials.NewtonCableMaterialCfg`.
    """

    prim_path: str
    node_positions: list[wp.vec3]
    edges: list[tuple[int, int]]
    radius: float

    init_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    init_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    stretch_stiffness: float = 1.0e9
    bend_stiffness: float = 0.0
    stretch_damping: float = 0.0
    bend_damping: float = 0.0
    density: float = 1500.0

    # Filled by :func:`add_cable_entry_to_builder`.
    body_offsets: list[int] = field(default_factory=list)
    last_edge_length: float = 0.0


from isaaclab_newton.assets.articulation.articulation import Articulation  # noqa: E402
from isaaclab_newton.physics import NewtonManager as SimulationManager  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim.spawners.shapes import CableCfg  # noqa: E402

if TYPE_CHECKING:
    from .cable_object_cfg import CableObjectCfg


def add_cable_entry_to_builder(
    builder,
    entry: CableRegistryEntry,
    env_idx: int,
    env_position: list[float],
    env_rotation: list[float] | tuple[float, float, float, float],
) -> None:
    """Add one cable to a Newton ``ModelBuilder`` for one environment.

    Composes the env transform with the cable's init transform and applies it to
    each control point, then calls :meth:`newton.ModelBuilder.add_rod_graph` with
    the explicit stiffness / damping / density fields stored on the entry.
    Density flows through :class:`newton.ModelBuilder.ShapeConfig` so Newton
    computes per-segment mass from ``density * pi * r^2 * segment_length``. The
    articulation is labelled ``"{entry.prim_path}/cable"`` so the cloner's
    ``_rename_builder_labels`` rewrites the source prefix to each env's
    destination prefix during replication.

    Args:
        builder: The Newton ``ModelBuilder``.
        entry: Registry entry describing the cable's geometry and material.
        env_idx: Zero-based environment (world) index.
        env_position: World translation ``[x, y, z]`` [m] for this environment.
        env_rotation: World orientation as quaternion ``(x, y, z, w)`` for this environment.
    """
    if env_idx == 0:
        entry.body_offsets.clear()
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

    # Compose: world = env_T ∘ init_T ∘ local
    composed_pos = env_pos + wp.quat_rotate(env_rot, init_pos)
    composed_rot = env_rot * init_rot

    world_nodes: list[wp.vec3] = []
    for node in entry.node_positions:
        rotated = wp.quat_rotate(composed_rot, node)
        world_nodes.append(composed_pos + rotated)

    shape_cfg = newton.ModelBuilder.ShapeConfig()
    shape_cfg.density = float(entry.density)

    # ``label`` is load-bearing: Newton suffixes ``_articulation`` to produce
    # ``{prim_path}/cable_articulation``, which is the path :class:`ArticulationView`
    # searches for per env after the cloner rewrites the source prefix.
    entry.body_offsets.append(builder.body_count)
    builder.add_rod_graph(
        node_positions=world_nodes,
        edges=entry.edges,
        radius=entry.radius,
        cfg=shape_cfg,
        stretch_stiffness=entry.stretch_stiffness,
        stretch_damping=entry.stretch_damping,
        bend_stiffness=entry.bend_stiffness,
        bend_damping=entry.bend_damping,
        label=f"{entry.prim_path}/cable",
        wrap_in_articulation=True,
    )
    if env_idx == 0:
        u, v = entry.edges[-1]
        entry.last_edge_length = float(wp.length(entry.node_positions[v] - entry.node_positions[u]))


def add_registered_cables_to_builder(
    builder,
    world_idx: int,
    env_position: list[float],
    env_rotation: list[float] | tuple[float, float, float, float],
) -> None:
    """Loop function for ``_per_world_builder_hooks``.

    Iterates :attr:`SimulationManager._cable_registry` and calls
    :func:`add_cable_entry_to_builder` for each registered cable.
    Mirrors :func:`isaaclab_contrib.deformable.deformable_object.add_registered_deformables_to_builder`.
    """
    for entry in SimulationManager._cable_registry:
        add_cable_entry_to_builder(builder, entry, world_idx, env_position, env_rotation)


def install_cable_builder_hooks() -> None:
    """Set up the cable registry and per-world hook on ``SimulationManager``.

    Resets ``_cable_registry`` to an empty list on each call — install is intended
    to be called once per scene setup, not per asset.

    Mirrors :func:`isaaclab_contrib.deformable.deformable_object.install_deformable_builder_hooks`
    (see ``deformable_object.py:190-201``).
    """
    SimulationManager._cable_registry = []
    if not hasattr(SimulationManager, "_per_world_builder_hooks"):
        SimulationManager._per_world_builder_hooks = []
    if add_registered_cables_to_builder not in SimulationManager._per_world_builder_hooks:
        SimulationManager._per_world_builder_hooks.append(add_registered_cables_to_builder)


class CableObject(Articulation):
    """Cable / 1D-rod asset (Newton backend).

    Subclasses :class:`Articulation` so the cable's per-segment poses and
    per-cable-joint state are exposed via :class:`ArticulationData` with no
    parallel data class.

    Override surface beyond the base:

    - :meth:`__init__` defers to the base ``__init__`` and then calls
      :meth:`_register_cable` (mirroring :meth:`DeformableObject._register_deformable`),
      which builds a :class:`CableRegistryEntry` from cfg and appends it to the
      cable registry. Caller must have called :func:`install_cable_builder_hooks`
      before constructing any :class:`CableObject` (typical: from a solver manager
      init, mirroring how the deformable contrib package wires things up).
    """

    cfg: CableObjectCfg

    def __init__(self, cfg: CableObjectCfg):
        """Initialize the cable object.

        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg)

        # Read the cable's centerline / material from cfg and register in the
        # cable registry. Mirrors :meth:`DeformableObject._register_deformable`.
        self._registry_entry = self._register_cable()

    def _register_cable(self) -> CableRegistryEntry:
        """Read cable geometry + material from the spawned USD prim and register on
        :attr:`SimulationManager._cable_registry`.

        Mirrors :meth:`DeformableObject._register_deformable`:

        1. Locate the spawned template prim (via ``cfg.spawn.spawn_path`` or
           ``cfg.prim_path``).
        2. Find the single ``UsdGeomBasisCurves`` child authored by
           :func:`spawn_cable` and read its ``points`` and ``widths`` attributes.
        3. Bake the template prim's xform into the per-node positions so the
           replicate hook only needs to apply the env transform.
        4. Look up the bound Newton cable physics material and read each
           ``newton:*`` attribute into the entry, falling back to the
           :class:`CableRegistryEntry` field defaults when an attribute is
           missing.

        Returns:
            The registry entry (also appended to ``SimulationManager._cable_registry``).

        Raises:
            ValueError: If ``cfg.spawn`` is not a :class:`~isaaclab.sim.spawners.shapes.CableCfg`,
                the template prim has no ``UsdGeomBasisCurves`` child, the curve
                is missing its ``widths`` attribute, or no Newton cable physics
                material is bound to the curve prim (commonly because
                :class:`UsdPhysics.CollisionAPI` was not applied — set
                ``CableCfg.collision_props`` so :func:`spawn_cable` applies it).
            RuntimeError: If the template prim cannot be located, or
                :func:`install_cable_builder_hooks` has not been called before
                constructing the :class:`CableObject`.

        Note:
            ``pxr`` imports are deferred to this method (not module level) so
            that ``resolve_task_config`` can import the env-cfg module before
            Kit starts without polluting the ``pxr`` module cache.
        """
        from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

        if not isinstance(self.cfg.spawn, CableCfg):
            raise ValueError(
                f"CableObjectCfg requires `spawn` to be a CableCfg instance, got {type(self.cfg.spawn).__name__}."
            )
        if not hasattr(SimulationManager, "_cable_registry"):
            raise RuntimeError(
                "CableObject requires `install_cable_builder_hooks()` to have been called"
                " before constructing any CableObject instance (typically from the solver"
                " manager init, mirroring the deformable contrib pattern)."
            )

        # Resolve the spawned template prim. ``spawn_path`` is set by InteractiveScene's
        # template-based cloning flow; falls back to ``prim_path`` for direct envs that
        # spawn straight at the cloned regex.
        lookup_path = self.cfg.spawn.spawn_path if self.cfg.spawn.spawn_path is not None else self.cfg.prim_path
        template_prim = sim_utils.find_first_matching_prim(lookup_path)
        if template_prim is None:
            raise RuntimeError(f"Failed to find cable template prim for expression: '{lookup_path}'.")
        template_prim_path = template_prim.GetPrimPath()

        # Find the single UsdGeomBasisCurves child authored by spawn_cable.
        curve_prims = sim_utils.get_all_matching_child_prims(
            template_prim_path, lambda p: p.GetTypeName() == "BasisCurves"
        )
        if len(curve_prims) != 1:
            raise ValueError(
                f"Expected exactly one UsdGeomBasisCurves prim under '{template_prim_path}', found {len(curve_prims)}."
            )
        curve_prim = curve_prims[0]
        curves = UsdGeom.BasisCurves(curve_prim)

        # Bake the curve prim's xform into the per-node positions so the replicate
        # hook only needs to apply the env transform.
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

        # Read the capsule width (per-control-point but broadcast equal by spawn_cable).
        raw_widths = curves.GetWidthsAttr().Get()
        if raw_widths is None or len(raw_widths) == 0:
            raise ValueError(f"UsdGeomBasisCurves at '{curve_prim.GetPrimPath()}' is missing the `widths` attribute.")
        radius = float(raw_widths[0]) / 2.0

        # Linear edge chain.
        edges = [(i, i + 1) for i in range(len(node_positions) - 1)]

        # Look up the bound Newton cable physics material via the standard
        # MaterialBindingAPI on the curve prim. The material binding requires
        # :class:`UsdPhysics.CollisionAPI` on the curve prim (see
        # :func:`bind_physics_material`); the most common reason no material is
        # found is that the user omitted ``CableCfg.collision_props`` so the
        # spawner's bind silently no-op'd.
        material_targets = (
            UsdShade.MaterialBindingAPI(curve_prim).GetDirectBindingRel("physics").GetTargets()
            if curve_prim.HasAPI(UsdShade.MaterialBindingAPI)
            else []
        )
        stage = curve_prim.GetStage()
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
                    " `spawn_cable` applies the API (cables are currently Newton-only, and the API has"
                    " no PhysX runtime effect since the cable is in the cloner's `_cable_ignore_paths`)."
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

        # init_pos/init_rot default to identity — the template xform is already baked
        # into ``node_positions`` above, so the replicate hook only applies the env
        # transform. Matches DeformableObject._register_deformable.
        entry = CableRegistryEntry(
            prim_path=self.cfg.prim_path,
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
