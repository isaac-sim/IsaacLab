# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native rod with rigid connectors and a physical spring-hinged RJ45 clip."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass

if TYPE_CHECKING:
    import newton

    from pxr import Usd

    from .assets import ConnectorRenderPart

CONNECTOR_SPAN_M = 0.04
CONNECTOR_DENSITY_KG_M3 = 2000.0
PATCH_CABLE_COLOR = (0.35, 0.80, 0.015)
CONNECTOR_COLORS = {"plug": (0.32, 0.34, 0.36), "latch": (0.48, 0.50, 0.52)}
# Canonical Latch prim origin, transformed into the cable-head frame [m].
LATCH_HINGE_POSITION_M = (0.0, 0.004677868448, -0.030711488256)
# The native 11.86 x 6.55 mm shank already fits the measured jack. Leave its
# height intact and provide 0.20 mm lateral clearance without shrinking the boot.
GB300_CONNECTOR_TIP_WIDTH_SCALE = 0.97


def hanging_cable_positions(
    loose_start_e: tuple[float, float, float],
    anchor_socket_e: tuple[float, float, float],
    *,
    flexible_length_m: float = 0.24,
) -> list[tuple[float, float, float]]:
    """Route a smooth patch lead from the loose plug to a seated GB300 plug [m].

    The first and last 40 mm segments are the rigid connector housings. The
    flexible length excludes those housings and is sampled at roughly 10 mm
    intervals. Both connector exits point outward from the cabinet.
    """
    from .workcell import head_goal_for_socket

    x, y, z = loose_start_e
    anchor_x, anchor_y, anchor_z = head_goal_for_socket(anchor_socket_e)
    start = np.array((x, y - CONNECTOR_SPAN_M, z))
    end = np.array((anchor_x, anchor_y - CONNECTOR_SPAN_M / 2, anchor_z))
    if not math.isfinite(flexible_length_m) or flexible_length_m <= np.linalg.norm(end - start):
        raise ValueError("The flexible cable length must exceed the distance between connector exits.")
    t = np.linspace(0.0, 1.0, 257)[:, None]
    lower, upper = 0.0, flexible_length_m
    # Fit arc length instead of changing cable length with socket placement.
    for _ in range(32):
        bow = (lower + upper) / 2
        offset = np.array((0.0, -bow, 0.0))
        curve = (
            (1 - t) ** 3 * start
            + 3 * (1 - t) ** 2 * t * (start + offset)
            + 3 * (1 - t) * t**2 * (end + offset)
            + t**3 * end
        )
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
        samples = np.linspace(0.0, arc[-1], math.ceil(flexible_length_m / 0.01) + 1)
        jacket = np.column_stack([np.interp(samples, arc, curve[:, axis]) for axis in range(3)])
        if np.linalg.norm(np.diff(jacket, axis=0), axis=1).sum() < flexible_length_m:
            lower = bow
        else:
            upper = bow
    points = np.vstack(((x, y, z), jacket, (anchor_x, anchor_y + CONNECTOR_SPAN_M / 2, anchor_z)))
    # The root rotates local +Z toward the rack-front -Y direction.
    return [(float(p[0] - x), float(p[2] - z), float(y - p[1])) for p in points]


def _gb300_connector_points(points: tuple[tuple[float, float, float], ...]) -> np.ndarray:
    """Fit the connector nose to the measured GB300 jack aperture [m]."""
    result = np.asarray(points, dtype=np.float32).copy()
    fraction = np.clip((result[:, 2] + 0.020) / 0.005, 0.0, 1.0)
    result[:, 0] *= GB300_CONNECTOR_TIP_WIDTH_SCALE + (1.0 - GB300_CONNECTOR_TIP_WIDTH_SCALE) * fraction
    return result


def _gb300_connector_normals(part: ConnectorRenderPart) -> np.ndarray:
    """Transform authored shading normals through the fitted connector's local deformation."""
    normals = np.asarray(part.normals, dtype=np.float32).reshape(-1, 3).copy()
    if not len(normals):
        return normals
    points = np.asarray(part.points)
    if part.normals_interpolation == "faceVarying":
        points = points[np.asarray(part.face_vertex_indices)]
    elif part.normals_interpolation != "vertex":
        raise ValueError(f"Unsupported connector normal interpolation: {part.normals_interpolation}")
    fraction = np.clip((points[:, 2] + 0.020) / 0.005, 0.0, 1.0)
    scale = GB300_CONNECTOR_TIP_WIDTH_SCALE + (1.0 - GB300_CONNECTOR_TIP_WIDTH_SCALE) * fraction
    derivative = ((fraction > 0) & (fraction < 1)) * (1.0 - GB300_CONNECTOR_TIP_WIDTH_SCALE) / 0.005
    normals[:, 0] /= scale
    normals[:, 2] -= points[:, 0] * derivative * normals[:, 0]
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-12)
    return normals


@lru_cache(maxsize=2)
def _connector_closed_mesh(name: str) -> newton.Mesh:
    """Close only the canonical CAD's small planar boundary loops for a signed contact volume."""
    import newton
    import trimesh

    from isaaclab.utils.mesh import convert_faces_to_triangles

    from .assets import connector_render_parts

    part = next(part for part in connector_render_parts(CONNECTOR_SPAN_M) if part.name == name)
    faces = convert_faces_to_triangles(np.asarray(part.face_vertex_indices), np.asarray(part.face_vertex_counts))
    mesh = trimesh.Trimesh(vertices=_gb300_connector_points(part.points), faces=faces, process=True)
    # Preserve all exterior CAD faces. Boundary caps seal exporter omissions,
    # rather than convexifying the plug or filling its visible surface details.
    _, inverse, counts = np.unique(mesh.edges_sorted, axis=0, return_inverse=True, return_counts=True)
    edges = mesh.edges[counts[inverse] == 1]
    remaining = {int(start): int(end) for start, end in edges}
    if len(remaining) != len(edges) or len(set(remaining.values())) != len(edges):
        raise ValueError(f"The {name} CAD has branching open boundaries; inspect it before SDF cooking.")
    vertices, faces = mesh.vertices.tolist(), mesh.faces.tolist()
    while remaining:
        start = next(iter(remaining))
        loop, current = [start], remaining.pop(start)
        while current != start:
            loop.append(current)
            current = remaining.pop(current)
        ring = mesh.vertices[loop]
        center = ring.mean(axis=0)
        _, _, axes = np.linalg.svd(ring - center)
        if np.max(np.abs((ring - center) @ axes[-1])) > 2.0e-6:
            raise ValueError(f"The {name} CAD boundary is not planar within 2 micrometers.")
        if np.linalg.norm(np.ptp(ring, axis=0)) > 0.006:
            raise ValueError(f"The {name} CAD opening exceeds the measured small exporter omissions.")
        # Ear clipping also preserves the latch's concave six-corner side caps.
        planar = (ring - center) @ axes[:2].T
        signed_area = np.sum(planar[:, 0] * np.roll(planar[:, 1], -1) - planar[:, 1] * np.roll(planar[:, 0], -1))
        direction = np.sign(signed_area)
        pending = list(range(len(loop)))
        while len(pending) > 3:
            for index, current in enumerate(pending):
                previous, following = pending[index - 1], pending[(index + 1) % len(pending)]
                triangle = planar[[previous, current, following]]
                edge = np.roll(triangle, -1, axis=0) - triangle
                turn = edge[0, 0] * edge[1, 1] - edge[0, 1] * edge[1, 0]
                if direction * turn <= 1.0e-12:
                    continue
                others = [vertex for vertex in pending if vertex not in (previous, current, following)]
                delta = planar[others, None] - triangle
                side = direction * (edge[:, 0] * delta[:, :, 1] - edge[:, 1] * delta[:, :, 0])
                if np.any(np.all(side >= -1.0e-12, axis=1)):
                    continue
                faces.append((loop[previous], loop[following], loop[current]))
                pending.pop(index)
                break
            else:
                raise ValueError(f"The {name} CAD boundary cannot be triangulated without altering its outline.")
        faces.append((loop[pending[0]], loop[pending[2]], loop[pending[1]]))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    trimesh.repair.fix_normals(mesh)
    if not mesh.is_watertight or not mesh.is_volume:
        raise ValueError(f"The {name} contact mesh must enclose a consistent signed volume.")
    return newton.Mesh(np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces, np.int32).reshape(-1))


@lru_cache(maxsize=2)
def _connector_contact_mesh(name: str) -> newton.Mesh:
    """Cook a signed field on the closed, dimension-preserving CAD surface."""
    result = _connector_closed_mesh(name)
    result.build_sdf(max_resolution=256, narrow_band_range=(-0.003, 0.003), margin=0.0003, sign_method="parity")
    return result


def _latch_hinge_damping() -> float:
    """Return inertia-derived hinge damping [N m s/rad]."""
    from .assets import LATCH_SPRING_KE

    mesh = _connector_closed_mesh("latch")
    offset = np.asarray(mesh.com) - np.asarray(LATCH_HINGE_POSITION_M)
    inertia = CONNECTOR_DENSITY_KG_M3 * (
        np.asarray(mesh.inertia).reshape(3, 3)[0, 0] + mesh.mass * np.sum(offset[1:] ** 2)
    )
    return 4.0 * math.sqrt(LATCH_SPRING_KE * inertia)


@clone
def spawn_patch_cable(
    prim_path: str,
    cfg: PatchCableCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a native cable, two connector housings, and an independently moving clip."""
    from pxr import Gf, UsdGeom, UsdPhysics, Vt

    from isaaclab.sim.spawners.shapes.shapes import spawn_cable

    from .assets import LATCH_LIMIT_LOWER, LATCH_LIMIT_UPPER, LATCH_SPRING_KE, connector_render_parts

    root = spawn_cable.__wrapped__(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    # Explicit cross-section frames keep both rectangular plugs level. Automatic
    # parallel transport along the loop otherwise rolls the seated connector.
    tangents = np.diff(np.asarray(cfg.positions), axis=0)
    normals = np.cross(tangents, (1.0, 0.0, 0.0))
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-8)
    curve = UsdGeom.BasisCurves(root.GetStage().GetPrimAtPath(f"{prim_path}/geometry/mesh"))
    curve.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*normal) for normal in np.vstack((normals, normals[-1]))]))
    curve.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    for name, start, end, reverse in (
        ("head", cfg.positions[0], cfg.positions[1], False),
        ("latch", cfg.positions[0], cfg.positions[1], False),
        ("tail", cfg.positions[-2], cfg.positions[-1], True),
    ):
        head = UsdGeom.Xform.Define(root.GetStage(), f"{prim_path}/{name}")
        head.AddTranslateOp().Set(Gf.Vec3d(*(0.5 * (a + b) for a, b in zip(start, end, strict=True))))
        if name == "head":
            UsdPhysics.RigidBodyAPI.Apply(head.GetPrim())
            points = np.asarray(connector_render_parts(CONNECTOR_SPAN_M)[0].points)
            center, half = (points.max(0) + points.min(0)) / 2, np.ptp(points, axis=0) / 2
            mass = CONNECTOR_DENSITY_KG_M3 * float(np.prod(2 * half))
            inertia = mass / 3 * (np.sum(half**2) - half**2)
            # Author the grasp-volume mass once; its hook-added collider is massless.
            mass_api = UsdPhysics.MassAPI.Apply(head.GetPrim())
            mass_api.CreateMassAttr(mass)
            mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*center))
            mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
        elif name == "latch":
            # The closed CAD volume supplies the spring clip's mass and inertia.
            UsdPhysics.RigidBodyAPI.Apply(head.GetPrim())
            mesh = _connector_closed_mesh("latch")
            inertia, axes = np.linalg.eigh(np.asarray(mesh.inertia).reshape(3, 3))
            if np.linalg.det(axes) < 0:
                axes[:, 0] *= -1
            principal = wp.quat_from_matrix(wp.mat33(axes))
            mass_api = UsdPhysics.MassAPI.Apply(head.GetPrim())
            mass_api.CreateMassAttr(CONNECTOR_DENSITY_KG_M3 * float(mesh.mass))
            mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*mesh.com))
            mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*(float(value) for value in CONNECTOR_DENSITY_KG_M3 * inertia)))
            mass_api.CreatePrincipalAxesAttr(Gf.Quatf(float(principal[3]), Gf.Vec3f(*principal[:3])))
        for part in connector_render_parts(CONNECTOR_SPAN_M):
            if (name == "head" and part.name == "latch") or (name == "latch" and part.name != "latch"):
                continue
            mesh = UsdGeom.Mesh.Define(root.GetStage(), f"{head.GetPath()}/{part.name}")
            points = [
                tuple(float(value) for value in ((x, -y, -z) if reverse else (x, y, z)))
                for x, y, z in _gb300_connector_points(part.points)
            ]
            mesh.CreatePointsAttr(Vt.Vec3fArray(points))
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray(part.face_vertex_counts))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(part.face_vertex_indices))
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            normals = _gb300_connector_normals(part)
            if len(normals):
                if reverse:
                    normals[:, 1:] *= -1
                mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals))
                mesh.SetNormalsInterpolation(part.normals_interpolation)
            mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*CONNECTOR_COLORS[part.name])]))
    if not math.isfinite(cfg.latch_damping_scale) or cfg.latch_damping_scale <= 0.0:
        raise ValueError("latch_damping_scale must be finite and positive.")
    # MJWarp needs a tree hinge to preserve its spring and limits; a loop
    # hinge becomes two connect constraints without an actuated coordinate.
    UsdPhysics.ArticulationRootAPI.Apply(root)
    joint = UsdPhysics.RevoluteJoint.Define(root.GetStage(), f"{prim_path}/spring_hinge")
    joint.CreateBody0Rel().SetTargets([f"{prim_path}/head"])
    joint.CreateBody1Rel().SetTargets([f"{prim_path}/latch"])
    joint.CreateAxisAttr("X")
    joint.CreateLocalPos0Attr(Gf.Vec3f(*LATCH_HINGE_POSITION_M))
    joint.CreateLocalPos1Attr(Gf.Vec3f(*LATCH_HINGE_POSITION_M))
    joint.CreateLocalRot0Attr(Gf.Quatf(0.0, Gf.Vec3f(0.0, 0.0, 1.0)))
    joint.CreateLocalRot1Attr(Gf.Quatf(0.0, Gf.Vec3f(0.0, 0.0, 1.0)))
    joint.CreateLowerLimitAttr(math.degrees(LATCH_LIMIT_LOWER))
    joint.CreateUpperLimitAttr(math.degrees(LATCH_LIMIT_UPPER))
    joint.CreateCollisionEnabledAttr(False)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    # USD angular gains are per degree; Newton imports them into SI units.
    drive.CreateStiffnessAttr(LATCH_SPRING_KE * math.pi / 180.0)
    drive.CreateDampingAttr(cfg.latch_damping_scale * _latch_hinge_damping() * math.pi / 180.0)
    drive.CreateTargetPositionAttr(0.0)
    drive.CreateTargetVelocityAttr(0.0)
    return root


@configclass
class PatchCableCfg(sim_utils.CableCfg):
    """An RJ45 connector owns each terminal 40 mm cable span."""

    func = spawn_patch_cable

    latch_damping_scale: float = 1.0
    """Scale the imported rigid connector's inertia-derived spring-clip damping."""


def _pin_seated_tail(builder: newton.ModelBuilder, tail: int) -> None:
    """Prescribe the seated cable boundary without redundant plug contacts."""
    import newton

    builder.body_flags[tail] = int(newton.BodyFlags.KINEMATIC)
    builder.body_mass[tail] = builder.body_inv_mass[tail] = 0.0
    builder.body_inertia[tail] = builder.body_inv_inertia[tail] = wp.mat33(0.0)
    # Its pose is fixed; neither the socket fit nor its embedded rod needs a solve.
    for shape in builder.body_shapes[tail]:
        builder.shape_flags[shape] = int(builder.shape_flags[shape]) & ~int(newton.ShapeFlags.COLLIDE_SHAPES)


def _assign_fixed_fixtures(builder: newton.ModelBuilder, world_idx: int, env_root: str, fixtures: int) -> None:
    """Put the workcell's static shapes on the rigid solver's fixture body."""
    prefix = f"{env_root}/Pedestal/"
    for shape, (label, body) in enumerate(zip(builder.shape_label, builder.shape_body, strict=True)):
        if body == -1 and int(builder.shape_world[shape]) == world_idx and str(label).startswith(prefix):
            builder.shape_body[shape] = fixtures
            builder.body_shapes[-1].remove(shape)
            builder.body_shapes[fixtures].append(shape)


def _filter_terminal_rod_contacts(
    builder: newton.ModelBuilder, body: int, grasp_body_patterns: tuple[str, ...]
) -> None:
    """Let connector housings own finger contact around terminal rods."""
    for rod_shape in tuple(builder.body_shapes[body]):
        for other, label in enumerate(builder.shape_label):
            if any(pattern in str(label) for pattern in grasp_body_patterns):
                builder.add_shape_collision_filter_pair(rod_shape, other)


def _filter_embedded_rods(builder: newton.ModelBuilder, head: int, latch: int, start: int) -> None:
    """Exclude the rods inside the housing, hinge, and strain relief."""
    # The prescribed terminal replaces the fixed joint's parent filtering.
    pairs = [(latch, start), (head, start + 1), (head, start)]
    for body, rod in pairs:
        for shape in builder.body_shapes[body]:
            for rod_shape in builder.body_shapes[rod]:
                builder.add_shape_collision_filter_pair(shape, rod_shape)


def add_connectors(
    builder: newton.ModelBuilder,
    world_idx: int,
    env_position: wp.vec3,
    env_rotation: wp.quat,
    *,
    grasp_body_patterns: tuple[str, ...] = (),
    rigid_connector_contact_stiffness: float = 1.0e5,
) -> None:
    """Add physical connector shapes to native end bodies before model finalization.

    Both registered workcell sockets own their contacts independently of the goal.
    ``rigid_connector_contact_stiffness`` tunes only the MJWarp housing's
    ordinary contacts [N/m]; the spring clip and native VBD path are unchanged.
    """
    import newton

    from isaaclab.utils.mesh import convert_faces_to_triangles

    from .assets import connector_render_parts
    from .workcell import add_rack_colliders

    if not math.isfinite(rigid_connector_contact_stiffness) or rigid_connector_contact_stiffness <= 0.0:
        raise ValueError("rigid_connector_contact_stiffness must be finite and positive.")

    # Hooks run before source labels are rewritten to the destination environment.
    groups = [
        i
        for i, (label, world) in enumerate(zip(builder.curve_label, builder.curve_world, strict=True))
        if int(world) == world_idx and str(label).endswith("/Cable/geometry/mesh")
    ]
    if len(groups) != 1:
        raise ValueError(f"Expected one native cable in world {world_idx}, found {len(groups)}.")
    group = groups[0]
    root = str(builder.curve_label[group]).removesuffix("/geometry/mesh")
    # Give the fixed workcell one owner and mirror it into VBD like the robot.
    # A shape cannot belong to two coupled entries, even when it is static.
    env_root = root.rsplit("/", 1)[0]
    fixtures = builder.add_link(label=f"{env_root}/Fixtures", mass=0.0)
    joint = builder.add_joint_fixed(parent=-1, child=fixtures, label=f"{env_root}/Fixtures/fixed")
    builder.add_articulation([joint], label=f"{env_root}/Fixtures/articulation")
    _assign_fixed_fixtures(builder, world_idx, env_root, fixtures)
    add_rack_colliders(builder, fixtures, env_position, env_rotation)
    start, end = int(builder._curve_body_start[group]), int(builder._curve_body_end[group])
    matches = [
        body
        for body, (label, world) in enumerate(zip(builder.body_label, builder.body_world, strict=True))
        if str(label) == f"{root}/head" and int(world) == world_idx
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one imported rigid connector at {root}/head, found {len(matches)}.")
    head_body = matches[0]
    # Only the root terminal is prescribed; the remaining rod stays dynamic.
    builder.body_label[start] = f"{root}/geometry/plug_endpoint"
    builder.body_flags[start] = int(newton.BodyFlags.KINEMATIC)
    builder.body_inv_mass[start] = 0.0
    builder.body_inv_inertia[start] = wp.mat33(0.0)
    _filter_terminal_rod_contacts(builder, start, grasp_body_patterns)
    latch_bodies = [
        body
        for body, (label, world) in enumerate(zip(builder.body_label, builder.body_world, strict=True))
        if str(label) == f"{root}/latch" and int(world) == world_idx
    ]
    if len(latch_bodies) != 1:
        raise ValueError(f"Expected one imported spring latch at {root}/latch, found {len(latch_bodies)}.")
    latch_body = latch_bodies[0]
    for name, body, reverse in (("head", head_body, False), ("latch", latch_body, False), ("tail", end - 1, True)):
        builder.body_label[body] = f"{root}/{name}"
        # The terminal rod is inside the connector housing. Let the rectangular
        # housing own finger contact while the rod still contacts other bodies.
        if name != "latch":
            _filter_terminal_rod_contacts(builder, body, grasp_body_patterns)
        # Kit moves the authored mesh with the body; NewtonGL uses the dynamic shapes below.
        for shape, (label, world) in enumerate(zip(builder.shape_label, builder.shape_world, strict=True)):
            if int(world) == world_idx and str(label).startswith(f"{root}/{name}/"):
                builder.shape_flags[shape] = int(builder.shape_flags[shape]) & ~int(newton.ShapeFlags.VISIBLE)
        rotation = wp.quat(1.0, 0.0, 0.0, 0.0) if reverse else wp.quat_identity()
        parts = connector_render_parts(CONNECTOR_SPAN_M)
        for part in parts:
            if (name == "head" and part.name == "latch") or (name == "latch" and part.name != "latch"):
                continue
            counts = np.asarray(part.face_vertex_counts)
            triangles = convert_faces_to_triangles(np.asarray(part.face_vertex_indices), counts)
            mesh = newton.Mesh(_gb300_connector_points(part.points), triangles.reshape(-1))
            if name in ("head", "latch"):
                # Keep keeper contact stiff after MJWarp's timestep safety clamp on the tiny hinge.
                contact_stiffness = rigid_connector_contact_stiffness if name == "head" else 1.0e6
                builder.add_shape_mesh(
                    body=body,
                    mesh=_connector_contact_mesh(part.name),
                    cfg=newton.ModelBuilder.ShapeConfig(
                        density=0.0,
                        mu=1.0,
                        ke=contact_stiffness,
                        # Hydroelastic contacts supply their own gains; damp ordinary socket contact.
                        kd=2.0 * math.sqrt(contact_stiffness * builder.body_mass[body]),
                        margin=0.0,
                        gap=0.0003,
                        is_visible=False,
                    ),
                    label=f"{root}/{name}/{'grip' if part.name == 'plug' else 'contact'}",
                )
            visual_mesh = mesh
            normals = _gb300_connector_normals(part)
            if len(normals):
                # Newton consumes vertex normals; split render corners to preserve the CAD's sharp edges.
                indices = triangles.reshape(-1)
                normal_indices = (
                    convert_faces_to_triangles(np.arange(len(part.face_vertex_indices)), counts).reshape(-1)
                    if part.normals_interpolation == "faceVarying"
                    else indices
                )
                visual_mesh = newton.Mesh(
                    mesh.vertices[indices],
                    np.arange(len(indices), dtype=np.int32),
                    normals=normals[normal_indices],
                    compute_inertia=False,
                )
            builder.add_shape_mesh(
                body=body,
                mesh=visual_mesh,
                xform=wp.transform(wp.vec3(), rotation),
                cfg=newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False),
                color=CONNECTOR_COLORS[part.name],
                label=f"{root}/{name}/{part.name}_visual",
            )
    _filter_embedded_rods(builder, head_body, latch_body, start)
    for joint in range(int(builder._curve_joint_start[group]), int(builder._curve_joint_end[group])):
        if builder.joint_type[joint] != newton.JointType.ROD:
            continue
        dof = int(builder.joint_qd_start[joint])
        # Native rod material axes: shear X/Y, stretch Z, bend X/Y, twist Z.
        builder.joint_target_kd[dof : dof + 6] = [0.01, 0.01, 0.01, 0.02, 0.02, 0.02]
    # The far plug is already latched; the loose connector and spans are dynamic.
    _pin_seated_tail(builder, end - 1)
