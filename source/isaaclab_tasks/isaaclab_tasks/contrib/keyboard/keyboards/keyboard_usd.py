# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD authoring for procedural keyboard articulations."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

from .keyboard_geometry import base_profile_visuals, generate_keyboard, inset_box
from .keyboard_labels import label_mesh_data
from .keyboard_schema import KeyboardStyle, ResolvedKey, ResolvedKeyboard

if TYPE_CHECKING:
    from .keyboard_gen_cfg import KeyboardSpawnerCfg


@lru_cache(maxsize=1)
def _usd_modules():
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, Vt  # noqa: PLC0415

    return Gf, Sdf, UsdGeom, UsdPhysics, Vt


def spawn_keyboard(
    prim_path: str,
    cfg: KeyboardSpawnerCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Any:
    """Author a procedural keyboard into a USD stage.

    ``partition_mode='single'`` authors one keyboard articulation rooted at
    ``prim_path``. ``partition_mode='fixed_dof'`` authors a container at
    ``prim_path`` and homogeneous key-bank articulations under
    ``{prim_path}/parts/part_*``.
    """

    resolved = generate_keyboard(cfg)
    return _author_keyboard(prim_path, resolved, cfg, translation=translation, orientation=orientation, **kwargs)


def _author_keyboard(
    prim_path: str,
    resolved: ResolvedKeyboard,
    cfg: KeyboardSpawnerCfg,
    *,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Any:
    """Author a resolved procedural keyboard into a USD stage."""

    _, _, UsdGeom, _, _ = _usd_modules()

    stage = kwargs.pop("stage", None)
    if stage is None:
        from isaaclab.sim.utils import get_current_stage  # noqa: PLC0415

        stage = get_current_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"Cannot spawn keyboard at existing prim path '{prim_path}'.")

    root = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
    _set_xform(root, translation or (0.0, 0.0, 0.0), orientation or (0.0, 0.0, 0.0, 1.0))
    _author_keyboard_metadata(root, resolved, prim_path)

    if resolved.partition_mode == "fixed_dof":
        _spawn_partitioned_articulations(stage, prim_path, root, resolved, cfg)
    else:
        _spawn_single_articulation(stage, prim_path, root, resolved, cfg)
    return root


def _author_keyboard_metadata(root, resolved: ResolvedKeyboard, prim_path: str) -> None:
    _, Sdf, _, _, _ = _usd_modules()

    root.CreateAttribute("keyboard:family", Sdf.ValueTypeNames.String).Set(resolved.family)
    root.CreateAttribute("keyboard:style", Sdf.ValueTypeNames.String).Set(resolved.style_name)
    root.CreateAttribute("keyboard:baseProfile", Sdf.ValueTypeNames.String).Set(resolved.style.base_profile)
    root.CreateAttribute("keyboard:deckTiltRad", Sdf.ValueTypeNames.Float).Set(float(resolved.style.deck_tilt))
    root.CreateAttribute("keyboard:seed", Sdf.ValueTypeNames.Int).Set(resolved.seed)
    root.CreateAttribute("keyboard:activeKeyCount", Sdf.ValueTypeNames.Int).Set(resolved.active_key_count)
    root.CreateAttribute("keyboard:slotCount", Sdf.ValueTypeNames.Int).Set(resolved.slot_count)
    root.CreateAttribute("keyboard:partitionMode", Sdf.ValueTypeNames.String).Set(resolved.partition_mode)
    root.CreateAttribute("keyboard:partitionDof", Sdf.ValueTypeNames.Int).Set(resolved.partition_dof)
    root.CreateAttribute("keyboard:partitionCount", Sdf.ValueTypeNames.Int).Set(resolved.partition_count)
    root.CreateAttribute("keyboard:partitionTailPolicy", Sdf.ValueTypeNames.String).Set(resolved.partition_tail_policy)
    if resolved.partition_mode == "fixed_dof":
        root.CreateAttribute("keyboard:articulationRootPattern", Sdf.ValueTypeNames.String).Set(
            f"{prim_path}/parts/part_*"
        )
    else:
        root.CreateAttribute("keyboard:articulationRootPattern", Sdf.ValueTypeNames.String).Set(prim_path)


def _disable_self_collision(prim) -> None:
    """Disable articulation self-collision (keys are independent springs; collisions are wasteful).

    Sets the flag for both the PhysX and Newton backends via applied-schema attributes.
    """
    _, Sdf, _, _, _ = _usd_modules()
    prim.AddAppliedSchema("PhysxArticulationAPI")
    prim.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(False)
    prim.AddAppliedSchema("NewtonArticulationRootAPI")
    prim.CreateAttribute("newton:selfCollisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)


def _spawn_single_articulation(
    stage, prim_path: str, root, resolved: ResolvedKeyboard, cfg: KeyboardSpawnerCfg
) -> None:
    Gf, Sdf, UsdGeom, UsdPhysics, _ = _usd_modules()

    UsdPhysics.ArticulationRootAPI.Apply(root)
    _disable_self_collision(root)
    base_path = f"{prim_path}/base_link"
    keys_scope = f"{prim_path}/keys"
    joints_scope = f"{prim_path}/joints"
    UsdGeom.Xform.Define(stage, base_path)
    UsdGeom.Scope.Define(stage, keys_scope)
    UsdGeom.Scope.Define(stage, joints_scope)

    base_prim = stage.GetPrimAtPath(base_path)
    UsdPhysics.RigidBodyAPI.Apply(base_prim)
    base_mass_api = UsdPhysics.MassAPI.Apply(base_prim)
    base_mass_api.CreateMassAttr(float(resolved.style.base_mass))
    # Explicit tiny inertia: this base_link is the fixed articulation root, so its inertia is
    # dynamically irrelevant; setting it avoids the engine's "invalid inertia tensor" warning.
    base_mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(1.0e-6, 1.0e-6, 1.0e-6))
    _define_base_geometry(stage, base_path, resolved, cfg, collision=cfg.include_case_collision)

    fixed_joint = UsdPhysics.FixedJoint.Define(stage, f"{joints_scope}/base_fixed_joint")
    fixed_joint.CreateBody1Rel().SetTargets([Sdf.Path(base_path)])

    for key in resolved.keys:
        _author_key_articulation(
            stage,
            base_path=base_path,
            link_path=f"{keys_scope}/key_{key.slot:03d}",
            joint_path=f"{joints_scope}/key_{key.slot:03d}_joint",
            key=key,
            resolved=resolved,
            cfg=cfg,
        )


def _spawn_partitioned_articulations(
    stage, prim_path: str, root, resolved: ResolvedKeyboard, cfg: KeyboardSpawnerCfg
) -> None:
    Gf, Sdf, UsdGeom, UsdPhysics, _ = _usd_modules()

    base_path = f"{prim_path}/base_link"
    parts_scope = f"{prim_path}/parts"
    UsdGeom.Xform.Define(stage, base_path)
    UsdGeom.Scope.Define(stage, parts_scope)
    _define_base_geometry(stage, base_path, resolved, cfg, collision=cfg.include_case_collision)

    keys_by_part = _partition_keys(resolved)
    for part_index, keys in enumerate(keys_by_part):
        part_name = f"part_{part_index:03d}"
        part_path = f"{parts_scope}/{part_name}"
        part_root = UsdGeom.Xform.Define(stage, part_path).GetPrim()
        UsdPhysics.ArticulationRootAPI.Apply(part_root)
        _disable_self_collision(part_root)
        part_root.CreateAttribute("keyboard:partitionIndex", Sdf.ValueTypeNames.Int).Set(part_index)
        part_root.CreateAttribute("keyboard:partitionDof", Sdf.ValueTypeNames.Int).Set(len(keys))
        part_root.CreateAttribute("keyboard:partitionName", Sdf.ValueTypeNames.String).Set(part_name)

        part_base_path = f"{part_path}/base_link"
        part_keys_scope = f"{part_path}/keys"
        part_joints_scope = f"{part_path}/joints"
        UsdGeom.Xform.Define(stage, part_base_path)
        UsdGeom.Scope.Define(stage, part_keys_scope)
        UsdGeom.Scope.Define(stage, part_joints_scope)

        part_base_prim = stage.GetPrimAtPath(part_base_path)
        UsdPhysics.RigidBodyAPI.Apply(part_base_prim)
        part_mass = max(float(resolved.style.base_mass) / max(resolved.partition_count, 1), 0.001)
        part_mass_api = UsdPhysics.MassAPI.Apply(part_base_prim)
        part_mass_api.CreateMassAttr(part_mass)
        # Explicit tiny inertia: this base_link is the fixed articulation root (no collider), so its
        # inertia is dynamically irrelevant; setting it avoids the engine's "invalid inertia" warning.
        part_mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(1.0e-6, 1.0e-6, 1.0e-6))

        fixed_joint = UsdPhysics.FixedJoint.Define(stage, f"{part_joints_scope}/base_fixed_joint")
        fixed_joint.CreateBody1Rel().SetTargets([Sdf.Path(part_base_path)])

        for key in keys:
            _, local_slot = resolved.key_partition(key.slot)
            _author_key_articulation(
                stage,
                # Local-only link/joint names so every part has an identical metatype (PhysX requires
                # this to form one articulation view across parts). Global slot is kept on the
                # ``keyboard:slot`` attribute authored inside ``_author_key_articulation``.
                base_path=part_base_path,
                link_path=f"{part_keys_scope}/key_{local_slot:03d}",
                joint_path=f"{part_joints_scope}/key_{local_slot:03d}_joint",
                key=key,
                resolved=resolved,
                cfg=cfg,
                partition_index=part_index,
                partition_local_slot=local_slot,
            )


def _partition_keys(resolved: ResolvedKeyboard) -> list[tuple[ResolvedKey, ...]]:
    if resolved.partition_mode != "fixed_dof":
        return [resolved.keys]
    return [
        resolved.keys[start : start + resolved.partition_dof]
        for start in range(0, resolved.slot_count, resolved.partition_dof)
    ]


def _define_base_geometry(
    stage,
    base_path: str,
    resolved: ResolvedKeyboard,
    cfg: KeyboardSpawnerCfg,
    *,
    collision: bool,
) -> None:
    _define_box(
        stage,
        f"{base_path}/case_collision",
        resolved.case_center,
        inset_box(resolved.case_size, cfg.case_collision_margin),
        resolved.deck_quat_xyzw,
        resolved.style.case_color,
        collision=collision,
    )
    _define_box(
        stage,
        f"{base_path}/plate_visual",
        resolved.plate_center,
        resolved.plate_size,
        resolved.deck_quat_xyzw,
        resolved.style.plate_color,
        collision=False,
    )
    for visual in base_profile_visuals(resolved):
        _define_box(
            stage,
            f"{base_path}/{visual.name}",
            visual.center,
            visual.size,
            visual.quat_xyzw,
            visual.color,
            collision=False,
        )


def _author_key_articulation(
    stage,
    *,
    base_path: str,
    link_path: str,
    joint_path: str,
    key: ResolvedKey,
    resolved: ResolvedKeyboard,
    cfg: KeyboardSpawnerCfg,
    partition_index: int = 0,
    partition_local_slot: int | None = None,
) -> None:
    Gf, Sdf, UsdGeom, UsdPhysics, _ = _usd_modules()

    if partition_local_slot is None:
        partition_local_slot = key.slot
    link = UsdGeom.Xform.Define(stage, link_path).GetPrim()
    _set_xform(link, key.center, key.quat_xyzw)
    link.CreateAttribute("keyboard:slot", Sdf.ValueTypeNames.Int).Set(key.slot)
    link.CreateAttribute("keyboard:partitionIndex", Sdf.ValueTypeNames.Int).Set(partition_index)
    link.CreateAttribute("keyboard:partitionLocalSlot", Sdf.ValueTypeNames.Int).Set(partition_local_slot)
    if not key.active:
        UsdGeom.Imageable(link).MakeInvisible()
    UsdPhysics.RigidBodyAPI.Apply(link)
    UsdPhysics.MassAPI.Apply(link).CreateMassAttr(float(key.mass))

    if key.active:
        if cfg.use_tapered_keycaps:
            _define_tapered_keycap(stage, f"{link_path}/visual", key.size, resolved.style.cap_top_scale, key.color)
        else:
            _define_box(stage, f"{link_path}/visual", (0.0, 0.0, 0.0), key.size, _identity_quat(), key.color)
        _define_box(
            stage,
            f"{link_path}/collision",
            (0.0, 0.0, 0.0),
            key.collision_size,
            _identity_quat(),
            key.color,
            collision=True,
        )
        _define_label_mesh(stage, f"{link_path}/label", key, resolved.style)

    joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(base_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(link_path)])
    joint.CreateAxisAttr("Z")
    joint.CreateLocalPos0Attr(Gf.Vec3f(*key.center))
    joint.CreateLocalRot0Attr(_gf_quat(key.quat_xyzw))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr(_gf_quat(_identity_quat()))
    joint.CreateLowerLimitAttr(float(key.lower_limit))
    joint.CreateUpperLimitAttr(float(key.upper_limit))

    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    drive.CreateTypeAttr("force")
    drive.CreateTargetPositionAttr(0.0)
    # The authored style spring is often too weak to hold the keycap against gravity, so the cap sags to
    # the bottom and reads as permanently pressed (and leaves no travel for the robot to press into).
    # Raise stiffness so the cap rests within ~10% of the top, well above the 50% actuation point. Keep
    # the spring underdamped (zeta ~0.35) so it actively springs the cap back up with a little bounce on
    # release instead of just creeping back. The rest pose and the spring-back overshoot both stay above
    # the actuation point, so releasing a key never re-triggers a keystroke.
    travel = float(key.upper_limit - key.lower_limit)
    if travel > 1.0e-6:
        hold_stiffness = key.mass * 9.81 / (0.1 * travel)
        stiffness = max(float(key.stiffness), hold_stiffness)
        damping = 0.7 * (stiffness * key.mass) ** 0.5
    else:
        stiffness, damping = float(key.stiffness), float(key.damping)
    drive.CreateStiffnessAttr(stiffness)
    drive.CreateDampingAttr(damping)
    drive.CreateMaxForceAttr(float(key.max_force))


def _identity_quat() -> tuple[float, float, float, float]:
    return (0.0, 0.0, 0.0, 1.0)


def _gf_quat(quat_xyzw: tuple[float, float, float, float]):
    Gf, _, _, _, _ = _usd_modules()

    x, y, z, w = quat_xyzw
    return Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z)))


def _set_xform(prim, translation: tuple[float, float, float], quat_xyzw: tuple[float, float, float, float]) -> None:
    Gf, _, UsdGeom, _, _ = _usd_modules()

    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
    xform.AddOrientOp().Set(_gf_quat(quat_xyzw))


def _define_box(
    stage,
    path: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    quat_xyzw: tuple[float, float, float, float],
    color: tuple[float, float, float],
    *,
    collision: bool = False,
):
    Gf, _, UsdGeom, UsdPhysics, _ = _usd_modules()

    cube = UsdGeom.Cube.Define(stage, path)
    prim = cube.GetPrim()
    cube.CreateSizeAttr(1.0)
    _set_xform(prim, center, quat_xyzw)
    UsdGeom.Xformable(prim).AddScaleOp().Set(Gf.Vec3d(*size))
    UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collision:
        UsdPhysics.CollisionAPI.Apply(prim)
    return prim


def _define_tapered_keycap(
    stage,
    path: str,
    size: tuple[float, float, float],
    top_scale: float,
    color: tuple[float, float, float],
):
    Gf, _, UsdGeom, _, Vt = _usd_modules()

    sx, sy, sz = size
    bx, by = sx * 0.5, sy * 0.5
    tx, ty = bx * top_scale, by * top_scale
    z0, z1 = -sz * 0.5, sz * 0.5
    points = [
        (-bx, -by, z0),
        (bx, -by, z0),
        (bx, by, z0),
        (-bx, by, z0),
        (-tx, -ty, z1),
        (tx, -ty, z1),
        (tx, ty, z1),
        (-tx, ty, z1),
    ]
    indices = [
        0,
        1,
        2,
        3,
        4,
        7,
        6,
        5,
        0,
        4,
        5,
        1,
        1,
        5,
        6,
        2,
        2,
        6,
        7,
        3,
        3,
        7,
        4,
        0,
    ]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*point) for point in points]))
    mesh.CreateFaceVertexCountsAttr([4, 4, 4, 4, 4, 4])
    mesh.CreateFaceVertexIndicesAttr(indices)
    UsdGeom.Gprim(mesh.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return mesh.GetPrim()


def _define_label_mesh(stage, path: str, key: ResolvedKey, style: KeyboardStyle):
    Gf, _, UsdGeom, _, Vt = _usd_modules()

    vertices, faces = label_mesh_data(key, style)
    if not vertices or not faces:
        # Author a tiny (effectively invisible) quad so every active key has exactly one label mesh.
        # This keeps the per-key shape topology identical across variants/label modes, which Newton's
        # shared ArticulationView requires (it rejects non-uniform per-world shape strides).
        eps = 1.0e-5
        top_z = key.size[2] * 0.5
        vertices = [(-eps, -eps, top_z), (eps, -eps, top_z), (eps, eps, top_z), (-eps, eps, top_z)]
        faces = [(0, 1, 2, 3)]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*point) for point in vertices]))
    mesh.CreateFaceVertexCountsAttr([4] * len(faces))
    mesh.CreateFaceVertexIndicesAttr([index for face in faces for index in face])
    mesh.CreateDoubleSidedAttr(True)
    UsdGeom.Gprim(mesh.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*style.label_color)])
    return mesh.GetPrim()
