# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export a finalized Newton model back to USD.

Isaac Lab applies most configuration directly to the solver rather than to the stage, so a scene
that has been loaded and overridden no longer has a USD file describing what is actually being
simulated. This module writes that state back out.

The exporter is the inverse of :meth:`newton.ModelBuilder.add_usd`. Because the importer parses
through the official ``UsdPhysics.LoadUsdPhysicsFromRange`` utility, core physics is authored with
standard ``UsdPhysics`` schemas; Newton-specific extras (armature, joint friction, contact
parameters) are authored as ``newton:*`` attributes, matching what ``SchemaResolverNewton`` reads.

The correctness contract is *model idempotence*, not USD fidelity::

    m1 = load(source.usda)
    export(m1) -> out.usda
    m2 = load(out.usda)
    assert m1 == m2

The importer normalizes as it reads -- it converts units, bakes shape scale into geometry, and
collapses fixed-joint chains -- so the exported stage differs from the source stage by
construction. What must hold is that re-importing the export changes nothing further.

.. note::
    The export records **what is being simulated**, not the source art asset. Newton reduces convex
    collision meshes to a hull (capped at ``Mesh.MAX_HULL_VERTICES``) while *importing*, and the
    original geometry is not retained in the model, so it cannot be recovered on export. Model
    idempotence still holds -- re-hulling a hull is a fixed point -- but an exported collision mesh
    is coarser than the asset it came from. Full-detail visual meshes are exported losslessly.

Known gaps (tracked, not silently ignored):

* Joint types beyond revolute, prismatic and fixed (D6, ball, distance) raise
  :class:`NotImplementedError` rather than exporting something wrong.
* Geometry beyond box, sphere, capsule, cylinder, plane and meshes (heightfields, SDF) likewise.
* Shape ordering follows USD stage traversal, which need not match the source model's shape order.
  Values round-trip, but array indices may be permuted for assets that interleave visual and
  collision prims.
* A cloned scene holds one world per environment, all sharing the source asset's prim paths, so a
  single export describes one of them -- see the ``world`` argument of :func:`export_model_to_usd`.
* Procedurally added bodies have no source prim path. Their absence is caught by the entity counts
  rather than silently exported, so an export that cannot describe the whole world fails instead.
* Friction on shapes that do not collide is exported but not read back: Newton's importer takes
  ``mu`` from the bound material only for collision shapes and gives visual shapes its default. The
  file is complete; the reimported model differs only where friction can have no effect.
* Free-joint degrees of freedom carry no exportable state. A floating body is expressed in USD by
  the *absence* of a joint, so per-DOF values written to a free joint (drive gains, armature) have
  nowhere to be authored and do not survive the round-trip.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import newton
import numpy as np

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

if TYPE_CHECKING:
    import warp as wp
    from newton import Model

# USD authors angular quantities in degrees; Newton stores radians. The importer multiplies angles
# by this factor and divides angular gains by it, so the exporter applies the inverse.
_DEGREES_TO_RADIANS = math.pi / 180.0

# Joint limits at or beyond this magnitude [rad or m] are Newton's "unlimited" sentinel and are not
# authored back, so that a reimport regenerates the same sentinel rather than a huge finite limit.
_LIMIT_SENTINEL = 1.0e9

# Effort limits at or above this magnitude [N or N·m] are Newton's "unlimited" default.
_EFFORT_SENTINEL = 1.0e6

__all__ = ["export_model_to_usd"]


def _quat_to_gf(q) -> Gf.Quatf:
    """Convert a Newton ``(x, y, z, w)`` quaternion to a USD ``Gf.Quatf`` (real part first).

    The sign is canonicalized so the real part is non-negative. A quaternion and its negation encode
    the same rotation, but USD's matrix round-trip may return either, which would otherwise surface
    as a spurious difference when comparing model arrays after a reimport.
    """
    sign = -1.0 if float(q[3]) < 0.0 else 1.0
    return Gf.Quatf(sign * float(q[3]), Gf.Vec3f(sign * float(q[0]), sign * float(q[1]), sign * float(q[2])))


def _author_transform(prim: Usd.Prim, transform, scale=None) -> None:
    """Author a Newton ``wp.transform`` as USD translate/orient ops, plus an optional scale.

    The scale is authored here rather than by the geometry writers because this function clears the
    xform op order; a scale op added beforehand would be discarded.

    Args:
        prim: The prim to author on.
        transform: Newton transform as position [m] followed by an ``(x, y, z, w)`` quaternion.
        scale: Optional per-axis scale to author after the rotation.
    """
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(float(transform[0]), float(transform[1]), float(transform[2])))
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(_quat_to_gf(transform[3:7]))
    if scale is not None:
        xform.AddScaleOp().Set(Gf.Vec3f(float(scale[0]), float(scale[1]), float(scale[2])))


def _geo_type_name(shape_type: int) -> str:
    """Return the :class:`newton.GeoType` member name for ``shape_type``."""
    try:
        return newton.GeoType(int(shape_type)).name
    except (ValueError, AttributeError):
        return str(int(shape_type))


def _author_mesh_geometry(stage: Usd.Stage, path: str, source, convex: bool, collides: bool = True) -> Usd.Prim:
    """Define a ``UsdGeom.Mesh`` from a Newton mesh source.

    Newton stores mesh points in unit space with the size carried separately in ``shape_scale``
    (unlike primitives, whose scale is baked into their size attributes). The points are therefore
    authored unscaled and the caller applies ``shape_scale`` as a scale op, so that a reimport
    re-derives the same ``shape_scale``.

    ``CONVEX_MESH`` shapes additionally carry their collision approximation, so that the reimport
    reproduces the same :class:`newton.GeoType` instead of a full-detail mesh.

    Args:
        stage: The stage to author on.
        path: Prim path for the shape.
        source: The :class:`newton.Mesh` held by the model for this shape.
        convex: Whether the shape is a convex-hull approximation.

    Returns:
        The defined mesh prim.

    Raises:
        NotImplementedError: If the model holds no mesh source for the shape.
    """
    if source is None or not hasattr(source, "vertices"):
        raise NotImplementedError(f"Shape at '{path}' is a mesh but the model holds no mesh source to export.")

    points = np.asarray(source.vertices, dtype=np.float32).reshape(-1, 3)
    indices = np.asarray(source.indices, dtype=np.int32).reshape(-1)

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(indices))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(indices.size // 3, 3, dtype=np.int32)))
    # Newton triangulates on import, so declaring the subdivision scheme keeps the reimport from
    # treating the mesh as a subdivision surface and altering the geometry.
    mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)

    prim = mesh.GetPrim()
    if not collides:
        # A visual-only mesh carries no collision approximation; applying one would make the
        # reimport treat it as a collider.
        return prim
    collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
    collision.CreateApproximationAttr().Set(UsdPhysics.Tokens.convexHull if convex else UsdPhysics.Tokens.none)
    if convex:
        max_hull_vertices = getattr(source, "MAX_HULL_VERTICES", None)
        if max_hull_vertices is not None:
            prim.CreateAttribute("newton:maxHullVertices", Sdf.ValueTypeNames.Int).Set(int(max_hull_vertices))
    return prim


def _define_collision_geometry(stage: Usd.Stage, path: str, shape_type: int, scale, source=None, collides: bool = True):
    """Define the collision gprim for a Newton shape.

    Args:
        stage: The stage to author on.
        path: Prim path for the shape.
        shape_type: The :class:`newton.GeoType` value.
        scale: Newton's per-shape scale, interpreted per geometry type [m].
        source: The :class:`newton.Mesh` held by the model, for mesh-typed shapes.

    Returns:
        A tuple of the defined prim and the scale the caller must author as a scale op, or ``None``
        when the scale is already baked into the geometry's size attributes.

    Raises:
        NotImplementedError: If the geometry type is outside the supported set.
    """
    name = _geo_type_name(shape_type)
    if name == "BOX":
        # Newton stores box half-extents. UsdGeom.Cube carries a single uniform size, so a
        # non-uniform box authors the extent ratio as a scale op instead.
        gprim = UsdGeom.Cube.Define(stage, path)
        gprim.GetSizeAttr().Set(float(scale[0]) * 2.0)
        if not (math.isclose(scale[0], scale[1]) and math.isclose(scale[1], scale[2])):
            ratio = (1.0, float(scale[1]) / float(scale[0]), float(scale[2]) / float(scale[0]))
            return gprim.GetPrim(), ratio
    elif name == "SPHERE":
        gprim = UsdGeom.Sphere.Define(stage, path)
        gprim.GetRadiusAttr().Set(float(scale[0]))
    elif name == "CAPSULE":
        gprim = UsdGeom.Capsule.Define(stage, path)
        gprim.GetRadiusAttr().Set(float(scale[0]))
        gprim.GetHeightAttr().Set(float(scale[1]) * 2.0)
        gprim.GetAxisAttr().Set(UsdGeom.Tokens.z)
    elif name == "CYLINDER":
        gprim = UsdGeom.Cylinder.Define(stage, path)
        gprim.GetRadiusAttr().Set(float(scale[0]))
        gprim.GetHeightAttr().Set(float(scale[1]) * 2.0)
        gprim.GetAxisAttr().Set(UsdGeom.Tokens.z)
    elif name in ("MESH", "CONVEX_MESH"):
        # Mesh points are unit-space; the scale stays a scale op rather than being baked in.
        return _author_mesh_geometry(stage, path, source, convex=name == "CONVEX_MESH", collides=collides), scale
    elif name == "PLANE":
        # Newton keeps a plane as (width, length, 0) facing +Z, with 0 meaning unbounded; the importer
        # reads exactly those two attributes back from a Z-axis UsdGeom.Plane.
        gprim = UsdGeom.Plane.Define(stage, path)
        gprim.GetAxisAttr().Set(UsdGeom.Tokens.z)
        gprim.GetWidthAttr().Set(float(scale[0]))
        gprim.GetLengthAttr().Set(float(scale[1]))
    else:
        raise NotImplementedError(
            f"Exporting shape geometry '{name}' is not supported yet; supported types are BOX,"
            " SPHERE, CAPSULE, CYLINDER, PLANE, MESH and CONVEX_MESH. Heightfields and SDF shapes are"
            " out of scope."
        )
    return gprim.GetPrim(), None


def _author_scene(stage: Usd.Stage, scene_path: str, stage_info: dict[str, Any] | None) -> None:
    """Author the physics scene prim carrying solver settings.

    Solver configuration lives on the scene prim rather than in the model, so it is recovered from
    the importer's ``stage_info`` when available.
    """
    scene = UsdPhysics.Scene.Define(stage, scene_path)
    prim = scene.GetPrim()
    if not stage_info:
        return
    physics_dt = stage_info.get("physics_dt")
    if physics_dt:
        prim.CreateAttribute("newton:timeStepsPerSecond", Sdf.ValueTypeNames.Int).Set(int(round(1.0 / physics_dt)))
    max_iters = stage_info.get("max_solver_iterations")
    if max_iters is not None and int(max_iters) >= 0:
        prim.CreateAttribute("newton:maxSolverIterations", Sdf.ValueTypeNames.Int).Set(int(max_iters))


def _author_inertia(mass_api: UsdPhysics.MassAPI, inertia) -> None:
    """Author a full inertia tensor as USD principal moments plus a principal-axes rotation.

    ``UsdPhysics.MassAPI`` stores inertia as a diagonal in a rotated frame, so a tensor carrying
    products of inertia cannot be expressed by ``diagonalInertia`` alone. Authoring only the diagonal
    silently discards the off-diagonal terms; this diagonalizes instead, which is lossless because
    an inertia tensor is symmetric and therefore orthogonally diagonalizable.

    Args:
        mass_api: The mass API to author on.
        inertia: The body inertia tensor [kg·m²], shape [3, 3].
    """
    tensor = np.asarray(inertia, dtype=np.float64)
    diagonal = np.diag(tensor)
    off_diagonal = np.max(np.abs(tensor - np.diag(diagonal)))

    # An already-diagonal tensor is authored as-is. Diagonalizing it anyway would rotate the frame
    # for no reason and lose precision, because USD stores the moments and axes at float32.
    if off_diagonal <= 1e-9 * max(float(np.max(np.abs(diagonal))), 1e-30):
        mass_api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*(float(m) for m in diagonal)))
        return

    moments, axes = np.linalg.eigh(tensor)
    # eigh may return a reflection; USD requires a proper rotation for the principal-axes frame.
    if np.linalg.det(axes) < 0.0:
        axes[:, 0] = -axes[:, 0]
    mass_api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*(float(m) for m in moments)))

    rotation = Gf.Matrix3d(*(float(v) for v in axes.T.flatten())).GetOrthonormalized()
    quat = rotation.ExtractRotation().GetQuat()
    imaginary = quat.GetImaginary()
    mass_api.GetPrincipalAxesAttr().Set(
        Gf.Quatf(float(quat.GetReal()), float(imaginary[0]), float(imaginary[1]), float(imaginary[2]))
    )


def _author_body(stage: Usd.Stage, path: str, model: Model, body_index: int) -> Usd.Prim:
    """Author one rigid body: transform, ``RigidBodyAPI`` and mass properties."""
    prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    _author_transform(prim, model.body_q.numpy()[body_index])
    UsdPhysics.RigidBodyAPI.Apply(prim)

    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.GetMassAttr().Set(float(model.body_mass.numpy()[body_index]))
    _author_inertia(mass_api, model.body_inertia.numpy()[body_index].reshape(3, 3))
    com = model.body_com.numpy()[body_index]
    mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(float(com[0]), float(com[1]), float(com[2])))
    return prim


def _author_shape(stage: Usd.Stage, path: str, model: Model, shape_index: int) -> None:
    """Author one collision shape and its physics material."""
    flags = int(model.shape_flags.numpy()[shape_index])
    collides = bool(flags & int(newton.ShapeFlags.COLLIDE_SHAPES))
    visible = bool(flags & int(newton.ShapeFlags.VISIBLE))

    source = model.shape_source[shape_index] if shape_index < len(model.shape_source) else None
    prim, extra_scale = _define_collision_geometry(
        stage,
        path,
        model.shape_type.numpy()[shape_index],
        model.shape_scale.numpy()[shape_index],
        source,
        collides=collides,
    )
    _author_transform(prim, model.shape_transform.numpy()[shape_index], scale=extra_scale)

    # Visual-only shapes must not become colliders on reimport, and collision-only shapes must not
    # become visible: Newton distinguishes the two through shape flags, and authoring every shape as
    # a visible collider doubles the collision set.
    if not visible:
        UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)

    color = model.shape_color.numpy()[shape_index]
    UsdGeom.Gprim(prim).CreateDisplayColorAttr().Set(
        Vt.Vec3fArray([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
    )
    # The physics material is authored for every shape, collider or not: Isaac Lab's randomization
    # writes per-shape friction regardless of whether a shape can contact, and the round-trip is
    # judged on the model, not on what the solver would use.
    if collides:
        UsdPhysics.CollisionAPI.Apply(prim)

    material_prim = UsdShade.Material.Define(stage, f"{path}_physicsMaterial").GetPrim()
    material = UsdPhysics.MaterialAPI.Apply(material_prim)
    mu = float(model.shape_material_mu.numpy()[shape_index])
    material.GetStaticFrictionAttr().Set(mu)
    material.GetDynamicFrictionAttr().Set(mu)
    material.GetRestitutionAttr().Set(float(model.shape_material_restitution.numpy()[shape_index]))
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        UsdShade.Material(material_prim),
        bindingStrength=UsdShade.Tokens.weakerThanDescendants,
        materialPurpose="physics",
    )


def _author_joint(stage: Usd.Stage, path: str, model: Model, joint_index: int, body_paths: dict[int, str]) -> None:
    """Author one joint with its limits, drive gains and Newton-specific extras.

    Free joints are not authored: a floating body is expressed in USD by the absence of a joint, so
    emitting one would add a joint the importer never created.
    """
    joint_type = int(model.joint_type.numpy()[joint_index])
    type_name = newton.JointType(joint_type).name
    if type_name == "FREE":
        return

    dof_start = int(model.joint_qd_start.numpy()[joint_index])
    is_angular = type_name == "REVOLUTE"

    if type_name == "REVOLUTE":
        joint = UsdPhysics.RevoluteJoint.Define(stage, path)
    elif type_name == "PRISMATIC":
        joint = UsdPhysics.PrismaticJoint.Define(stage, path)
    elif type_name == "FIXED":
        joint = UsdPhysics.FixedJoint.Define(stage, path)
    else:
        raise NotImplementedError(
            f"Exporting joint type '{type_name}' is not supported yet; supported types are REVOLUTE,"
            " PRISMATIC and FIXED."
        )

    # A joint that closes a kinematic loop is kept out of the articulation tree; Newton records that
    # as no articulation membership, USD as ``excludeFromArticulation``. Without it the reimport
    # sees a cycle and rejects the whole model.
    articulation_of = model.joint_articulation
    if articulation_of is not None and int(articulation_of.numpy()[joint_index]) < 0:
        joint.CreateExcludeFromArticulationAttr().Set(True)

    parent = int(model.joint_parent.numpy()[joint_index])
    child = int(model.joint_child.numpy()[joint_index])
    if parent >= 0 and parent in body_paths:
        joint.GetBody0Rel().SetTargets([body_paths[parent]])
    if child >= 0 and child in body_paths:
        joint.GetBody1Rel().SetTargets([body_paths[child]])

    _author_joint_frames(joint, model, joint_index)

    if type_name == "FIXED":
        return

    axis = model.joint_axis.numpy()[dof_start]
    joint.GetAxisAttr().Set(_dominant_axis(axis))

    angle_scale = 1.0 / _DEGREES_TO_RADIANS if is_angular else 1.0
    lower = float(model.joint_limit_lower.numpy()[dof_start])
    upper = float(model.joint_limit_upper.numpy()[dof_start])
    if abs(lower) < _LIMIT_SENTINEL:
        joint.GetLowerLimitAttr().Set(lower * angle_scale)
    if abs(upper) < _LIMIT_SENTINEL:
        joint.GetUpperLimitAttr().Set(upper * angle_scale)

    # Drive gains: the importer divides angular gains by DEGREES_TO_RADIANS, so multiply here.
    gain_scale = _DEGREES_TO_RADIANS if is_angular else 1.0
    target_ke = float(model.joint_target_ke.numpy()[dof_start])
    target_kd = float(model.joint_target_kd.numpy()[dof_start])
    effort = float(model.joint_effort_limit.numpy()[dof_start])
    # The effort limit lives on the drive but is independent of the gains: a torque-controlled joint
    # has zero stiffness and damping yet still caps its effort, so the drive must be authored
    # whenever any of the three is set.
    if target_ke or target_kd or effort < _EFFORT_SENTINEL:
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular" if is_angular else "linear")
        drive.GetStiffnessAttr().Set(target_ke * gain_scale)
        drive.GetDampingAttr().Set(target_kd * gain_scale)
        if effort < _EFFORT_SENTINEL:
            drive.GetMaxForceAttr().Set(effort)
        # The drive target is the joint's commanded set-point. It is authored in degrees for angular
        # joints, the inverse of the importer's radian conversion.
        target_pos = model.joint_target_pos.numpy()[dof_start] if hasattr(model, "joint_target_pos") else None
        if target_pos is not None:
            drive.GetTargetPositionAttr().Set(float(target_pos) * angle_scale)

    prim = joint.GetPrim()
    armature = float(model.joint_armature.numpy()[dof_start])
    if armature:
        prim.CreateAttribute("newton:armature", Sdf.ValueTypeNames.Float).Set(armature)
    friction = float(model.joint_friction.numpy()[dof_start])
    if friction:
        prim.CreateAttribute("newton:friction", Sdf.ValueTypeNames.Float).Set(friction)


def _author_joint_frames(joint, model: Model, joint_index: int) -> None:
    """Author the joint's attachment frames relative to its parent and child bodies.

    Newton stores these as ``joint_X_p`` (the joint frame in the parent body frame) and
    ``joint_X_c`` (in the child body frame); USD spells them ``physics:localPos0``/``localRot0`` and
    ``physics:localPos1``/``localRot1``. Leaving them unauthored collapses every joint to the body
    origin with no rotation, which silently reassembles the robot in the wrong pose.

    Args:
        joint: The USD joint to author on.
        model: The finalized Newton model.
        joint_index: Index of the joint being authored.
    """
    for transform, position_attr, rotation_attr in (
        (model.joint_X_p.numpy()[joint_index], joint.CreateLocalPos0Attr, joint.CreateLocalRot0Attr),
        (model.joint_X_c.numpy()[joint_index], joint.CreateLocalPos1Attr, joint.CreateLocalRot1Attr),
    ):
        position_attr().Set(Gf.Vec3f(float(transform[0]), float(transform[1]), float(transform[2])))
        rotation_attr().Set(_quat_to_gf(transform[3:7]))


def _dominant_axis(axis) -> str:
    """Return the USD axis token (``"X"``/``"Y"``/``"Z"``) closest to ``axis``."""
    magnitudes = [abs(float(axis[i])) for i in range(3)]
    return "XYZ"[magnitudes.index(max(magnitudes))]


def _canonical_paths(path_map: dict[str, int]) -> dict[int, str]:
    """Invert an importer path map, choosing one canonical prim path per model index.

    The importer's maps are many-to-one: nested prims (``.../mesh`` and ``.../mesh/mesh``) can refer
    to the same shape. Authoring one prim per *path* would emit more colliders than the model holds
    and the reimport would see extra shapes, so exactly one prim is authored per model index. The
    shallowest path wins, since that is the prim the deeper one is nested under.

    Args:
        path_map: An importer map of prim path to model index.

    Returns:
        A map of model index to its canonical prim path.
    """
    canonical: dict[int, str] = {}
    for path, index in path_map.items():
        current = canonical.get(index)
        if current is None or (path.count("/"), path) < (current.count("/"), current):
            canonical[index] = path
    return canonical


_SUPPORTED_GEOMETRY = frozenset({"BOX", "SPHERE", "CAPSULE", "CYLINDER", "PLANE", "MESH", "CONVEX_MESH"})


def _reject_unsupported_geometry(model: Model, shape_indices: list[int]) -> None:
    """Fail on geometry USD cannot carry before the provenance guard reports a bare count mismatch.

    Heightfield terrain is generated in memory by the terrain importer and has no prim, so it also
    has no path; without this check the shortfall surfaces as "N shapes but N-1 prim paths", which
    hides the cause.

    Raises:
        NotImplementedError: Naming each unsupported geometry type present in the world.
    """
    types = model.shape_type.numpy()[shape_indices]
    flags = model.shape_flags.numpy()[shape_indices]
    present = {_geo_type_name(t) for t, f in zip(types, flags) if not (int(f) & int(newton.ShapeFlags.SITE))}
    unsupported = sorted(present - _SUPPORTED_GEOMETRY)
    if unsupported:
        raise NotImplementedError(
            f"Exporting shape geometry {unsupported} is not supported yet; supported types are"
            f" {sorted(_SUPPORTED_GEOMETRY)}. Heightfield terrain and SDF shapes have no USD form here."
        )


def _world_member_indices(world_array: wp.array | None, entity_count: int, world: int) -> list[int]:
    """Model indices belonging to ``world``, in model order.

    Each entity carries the index of the world it was built into. Entities belonging to no world
    (a negative index, e.g. a ground plane shared by every environment) are members of all of them.
    A model built without per-world grouping reports every entity as worldless, which collapses to
    the whole model -- the single-world case, and the same code path.

    Args:
        world_array: The model's per-entity world assignment, or ``None`` when the model does not
            track one.
        entity_count: Number of entities of this kind in the model.
        world: Index of the world to select.

    Returns:
        The model indices of that world's entities, ascending. Position ``k`` holds the model index
        of the world's ``k``-th entity, which is what the importer's path maps are keyed by.
    """
    if world_array is None:
        return list(range(entity_count))
    membership = world_array.numpy()[:entity_count]
    return np.flatnonzero((membership == world) | (membership < 0)).tolist()


def _resolve_world_paths(
    path_map: dict[str, int],
    model_indices: list[int],
    kind: str,
    exempt: set[int] | None = None,
    describe=None,
) -> dict[int, str]:
    """Map each of a world's model indices to the prim path it was imported from.

    The importer's path maps are recorded against the *source* asset, so their indices are local to
    one world. This lifts them onto the model indices of the requested world, whose entities the
    builder copies in source order, and then checks coverage entity by entity rather than by count:
    a path may point at an entity that has nothing to export (a site imported from a prim) and an
    entity may lack a path (terrain generated in memory), and a count cannot tell the two apart.

    Args:
        path_map: An importer map of prim path to source-local index.
        model_indices: Model indices of the selected world, as returned by
            :func:`_world_member_indices`.
        kind: Entity kind, used in the error message.
        exempt: Model indices that legitimately have no prim (free joints, sites).
        describe: Optional callable mapping a model index to a short label for the error message.

    Returns:
        A map of model index to canonical prim path, covering every non-exempt entity of the world.

    Raises:
        ValueError: If a non-exempt entity of the world has no prim path, which would silently export
            a subset of what is being simulated.
    """
    local_paths = _canonical_paths(path_map)
    lifted = {
        model_indices[local_index]: path
        for local_index, path in local_paths.items()
        if local_index < len(model_indices)
    }
    exempt = exempt or set()
    unmapped = [index for index in model_indices if index not in lifted and index not in exempt]
    if unmapped:
        shown = ", ".join(describe(i) if describe else str(i) for i in unmapped[:6])
        raise ValueError(
            f"{len(unmapped)} {kind} in the selected world have no source prim path ({shown}"
            f"{', ...' if len(unmapped) > 6 else ''}). Exporting would silently drop them; entities the"
            f" scene adds without a prim cannot be written back to USD."
        )
    return {index: path for index, path in lifted.items() if index not in exempt}


def _articulation_root_paths(body_paths: list[str]) -> list[str]:
    """Return the prim paths that should carry ``UsdPhysics.ArticulationRootAPI``.

    Newton rejects any joint that does not belong to an articulation, so the exported stage must
    carry an articulation root that is an ancestor of the jointed bodies. The root is derived from
    the body paths rather than assumed, because source assets root their bodies anywhere (``/World``,
    ``/cartpole``, ``/env``).

    Args:
        body_paths: Source prim paths of the exported bodies.

    Returns:
        The deepest common ancestor of ``body_paths``, or -- when the bodies span several top-level
        prims and share no ancestor below the pseudo-root -- each distinct top-level prim.
    """
    if not body_paths:
        return []
    split_paths = [path.strip("/").split("/") for path in body_paths]
    common: list[str] = []
    for parts in zip(*split_paths):
        if len(set(parts)) != 1:
            break
        common.append(parts[0])
    # A body path's last element is the body itself, never a valid articulation root.
    if common and len(common) == min(len(parts) for parts in split_paths):
        common = common[:-1]
    if common:
        return ["/" + "/".join(common)]
    # No shared ancestor: fall back to each distinct top-level prim.
    return sorted({"/" + parts[0] for parts in split_paths})


@dataclass(frozen=True)
class WorldPrimPaths:
    """Prim paths at which one world's entities are authored, keyed by model index.

    Attributes:
        bodies: Body index to the prim path it was imported from.
        shapes: Shape index to the prim path its geometry is authored at.
        joints: Joint index to the prim path it was imported from.
    """

    bodies: dict[int, str]
    shapes: dict[int, str]
    joints: dict[int, str]


# ``ModelBuilder.approximate_meshes(keep_visual_shapes=True)`` -- which the importer runs on visible
# meshes with a collision approximation -- copies the mesh as a visual-only shape labelled with this
# suffix and turns the original into the collision-only shape. The copy has no prim of its own.
_VISUAL_TWIN_SUFFIX = "_visual"


def _visual_twins(model: Model, shape_indices: list[int]) -> dict[int, int]:
    """Map each visual twin among ``shape_indices`` to the shape it was copied from."""
    labels = getattr(model, "shape_label", None)
    if labels is None:
        return {}
    flags = model.shape_flags.numpy()
    bodies = model.shape_body.numpy()
    collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
    by_label = {(labels[i], int(bodies[i])): i for i in shape_indices}
    twins = {}
    for i in shape_indices:
        label = labels[i]
        if not label.endswith(_VISUAL_TWIN_SUFFIX) or int(flags[i]) & collide:
            continue
        source = by_label.get((label[: -len(_VISUAL_TWIN_SUFFIX)], int(bodies[i])))
        if source is not None and source != i:
            twins[i] = source
    return twins


def resolve_world_prim_paths(model: Model, stage_info: dict[str, Any] | None, world: int = 0) -> WorldPrimPaths:
    """Map one world's bodies, shapes and joints to the prim paths the export authors them at.

    The importer records provenance against the source asset; this lifts it onto the model indices
    of ``world`` and checks that every entity is covered, so an export cannot silently describe a
    subset of what is simulated. Entities that legitimately have no prim -- free joints, sites, and
    the visual twins the builder makes when approximating meshes -- are resolved or exempted here.

    Args:
        model: The finalized Newton model.
        stage_info: The dict returned by :meth:`newton.ModelBuilder.add_usd`.
        world: Index of the world to resolve. Defaults to ``0``.

    Returns:
        The prim paths of the world's entities, keyed by model index.

    Raises:
        ValueError: If ``stage_info`` carries no ``path_body_map``, if ``world`` is not a world of
            the model, or if an entity of the world has no prim path.
        NotImplementedError: If the world holds geometry outside the supported set.
    """
    if not stage_info or "path_body_map" not in stage_info:
        raise ValueError(
            "export_model_to_usd requires the dict returned by ModelBuilder.add_usd() to recover"
            " source prim paths; export without prim-path provenance is not supported."
        )

    if not 0 <= world < max(model.world_count, 1):
        raise ValueError(f"World {world} is out of range for a model with {model.world_count} world(s).")

    body_indices = _world_member_indices(model.body_world, model.body_count, world)
    shape_indices = _world_member_indices(model.shape_world, model.shape_count, world)
    joint_indices = _world_member_indices(model.joint_world, model.joint_count, world)
    twins = _visual_twins(model, shape_indices)
    # Two kinds of entity legitimately have no prim. A floating body is expressed in USD by the
    # *absence* of a joint, so its free joint has nowhere to be authored; and a site is a bare frame
    # the cloner adds for sensors, carrying no geometry USD could hold.
    joint_types = model.joint_type.numpy()
    free_joints = {i for i in joint_indices if int(joint_types[i]) == int(newton.JointType.FREE)}
    shape_flags = model.shape_flags.numpy()
    sites = {i for i in shape_indices if int(shape_flags[i]) & int(newton.ShapeFlags.SITE)}

    _reject_unsupported_geometry(model, [i for i in shape_indices if i not in sites and i not in twins])

    body_labels = list(model.body_label) if getattr(model, "body_label", None) is not None else None
    describe_body = (lambda i: f"{i}:{body_labels[i]}") if body_labels else None
    describe_shape = lambda i: f"{i}:{_geo_type_name(model.shape_type.numpy()[i])}"  # noqa: E731
    describe_joint = lambda i: f"{i}:{newton.JointType(int(joint_types[i])).name}"  # noqa: E731

    body_paths = _resolve_world_paths(stage_info["path_body_map"], body_indices, "bodies", describe=describe_body)
    shape_paths = _resolve_world_paths(
        stage_info.get("path_shape_map", {}),
        shape_indices,
        "shapes",
        exempt=sites | set(twins),
        describe=describe_shape,
    )
    # A shape sharing its body's prim path is authored onto a child of that prim; a visual twin is
    # authored as a sibling of the shape it was copied from.
    body_map = stage_info["path_body_map"]
    shape_paths = {i: path if path not in body_map else f"{path}/collision" for i, path in shape_paths.items()}
    shape_paths.update(
        {twin: f"{shape_paths[src]}{_VISUAL_TWIN_SUFFIX}" for twin, src in twins.items() if src in shape_paths}
    )
    joint_paths = _resolve_world_paths(
        stage_info.get("path_joint_map", {}), joint_indices, "joints", exempt=free_joints, describe=describe_joint
    )
    return WorldPrimPaths(bodies=body_paths, shapes=shape_paths, joints=joint_paths)


def export_model_to_usd(
    model: Model,
    usd_path: str,
    *,
    stage_info: dict[str, Any] | None = None,
    articulation_root_path: str | None = None,
    world: int = 0,
) -> str:
    """Export one world of a finalized Newton model to a USD stage.

    Prims are authored at the prim paths the model was imported from, recovered from ``stage_info``
    through :func:`resolve_world_prim_paths`, so that a reimport reproduces the same body, joint and
    shape ordering.

    A model cloned across environments holds one world per environment, all sharing the source
    asset's prim paths. Exactly one of them is exported, since a single set of prim paths cannot
    describe several environments at once; ``world`` selects which. Per-environment differences such
    as domain randomization mean the worlds are not interchangeable, so the export describes the
    selected environment rather than the scene as a whole.

    Args:
        model: The finalized Newton model to export.
        usd_path: Destination path for the USD file.
        stage_info: The dict returned by :meth:`newton.ModelBuilder.add_usd`, supplying the
            ``path_body_map``, ``path_joint_map`` and ``path_shape_map`` provenance and the solver
            settings. Without it there is no prim-path provenance and the export is rejected.
        articulation_root_path: Prim path receiving ``UsdPhysics.ArticulationRootAPI``. Defaults to
            ``None``, deriving the root from the exported body paths, which is required for assets
            that do not root their bodies under ``/World``.
        world: Index of the world to export. Defaults to ``0``, the only world of an uncloned model.

    Returns:
        The path the stage was written to.

    Raises:
        ValueError: If ``stage_info`` carries no ``path_body_map``, if ``world`` is not a world of
            the model, or if the selected world's entity counts disagree with the prim paths.
        NotImplementedError: If the model contains geometry or joint types outside the supported set.
    """
    paths = resolve_world_prim_paths(model, stage_info, world)

    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    body_path_list = list(stage_info["path_body_map"])
    root_paths = (
        [articulation_root_path] if articulation_root_path is not None else _articulation_root_paths(body_path_list)
    )
    for root_path in root_paths:
        UsdPhysics.ArticulationRootAPI.Apply(UsdGeom.Xform.Define(stage, root_path).GetPrim())
    _author_scene(stage, f"{root_paths[0]}/physicsScene" if root_paths else "/physicsScene", stage_info)

    for body_index, path in sorted(paths.bodies.items()):
        _author_body(stage, path, model, body_index)
    for shape_index, path in sorted(paths.shapes.items()):
        _author_shape(stage, path, model, shape_index)
    for joint_index, path in sorted(paths.joints.items()):
        _author_joint(stage, path, model, joint_index, paths.bodies)

    stage.GetRootLayer().Save()
    return str(usd_path)
