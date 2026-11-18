# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Prepare the robot, convex handle contacts and coupled solver for the physical pour."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from dataclasses import MISSING
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp
from filelock import FileLock
from isaaclab_newton.physics import MJWarpSolverCfg, MPMSolverCfg
from isaaclab_newton.physics.coupling import CouplerEntryCfg, CouplerProxyCfg
from scipy.spatial.transform import Rotation

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.rizon_sharpa import (
    ARM_JOINT_NAMES,
    HAND_JOINT_NAMES,
    ROBOT_BUNDLE_SHA256,
    arm_actuators,
    local_robot_usd,
)

from .motion import ARM_HOME_POSITIONS, HAND_OPEN_POSITIONS, ROBOT_BASE_POS

if TYPE_CHECKING:
    from pxr import Usd

    from isaaclab.assets import ArticulationCfg


def robot_cfg() -> ArticulationCfg:
    """Return the 29-joint robot ready for the calibrated index-finger grasp."""
    from isaaclab_newton.sim.schemas import (
        MujocoCollisionCfg,
        MujocoJointCfg,
        NewtonArticulationCfg,
        NewtonCollisionCfg,
    )

    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_prepared_robot_path(local_robot_usd())),
            articulation_props=NewtonArticulationCfg(self_collision_enabled=True),
            joint_drive_props={"/Physics/.*": [MujocoJointCfg(actuatorgravcomp=True)]},
            collision_props={
                "(/.*)?": [
                    sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
                    MujocoCollisionCfg(condim=4, solref=(0.004, 1.0)),
                    NewtonCollisionCfg(contact_gap=0.001),
                ]
            },
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=ROBOT_BASE_POS,
            # Leave shoulder travel available throughout the elevated pour and recovery.
            rot=(0.0, 0.0, math.sin(math.pi / 8.0), math.cos(math.pi / 8.0)),
            joint_pos={
                **dict(zip(ARM_JOINT_NAMES, ARM_HOME_POSITIONS, strict=True)),
                **dict(zip(HAND_JOINT_NAMES, HAND_OPEN_POSITIONS, strict=True)),
            },
        ),
        actuators={
            **arm_actuators(),
            "hand": ImplicitActuatorCfg(
                joint_names_expr=["right_.*"],
                joint_effort_limit=3.3,
                joint_velocity_limit=6.0,
                stiffness=24.0,
                damping=1.2,
                # Passive damping remains active when the closing drive reaches its torque limit.
                viscous_friction={joint: 1.0 if joint == "right_index_MCP_AA" else 0.5 for joint in HAND_JOINT_NAMES},
                # The measured contact response stays stable at 800 Hz without changing link inertias.
                armature=2.0e-4,
            ),
        },
        soft_joint_pos_limit_factor=0.95,
    )


def make_solver_cfg(mpm_cfg: MPMSolverCfg) -> CouplerProxyCfg:
    """Resolve rigid hand--pot contacts and synchronize the measured fluid boundary."""
    return CouplerProxyCfg(
        class_type="isaaclab_tasks.contrib.rizon_sharpa_teapot.coupling:TeapotSolver",
        entries=[
            CouplerEntryCfg(
                name="robot",
                solver_cfg=MJWarpSolverCfg(
                    integrator="implicitfast",
                    cone="elliptic",
                    impratio=20.0,
                    iterations=16,
                    ls_iterations=32,
                    # Distributed mesh contacts support the handle against the finger pads.
                    enable_multiccd=True,
                    # Keep headroom for hand self-contacts and the pot without
                    # oversized dense constraint buffers.
                    njmax=128,
                    nconmax=256,
                ),
                bodies=[r"/World/envs/env_.*/Robot", r"/World/envs/env_.*/PourContainer"],
            ),
            CouplerEntryCfg(
                name="fluid",
                solver_cfg=mpm_cfg,
                # The authored bowl has a fixed body, rather than a body-free static shape.
                bodies=[r"/World/envs/env_.*/CatchBowl"],
                all_particles=True,
                include_static_shapes=True,
                in_place=True,
            ),
        ],
    )


@configclass
class GraspedTeapotCfg(sim_utils.MeshFileCfg):
    """Spawn the exact fluid shell and rigid grasp contacts on the same body."""

    func = "{DIR}.scene:_spawn_grasped_teapot"
    contact_source: str = MISSING
    """Teapot USD containing the contact geometry and authored inertia."""

    rigid_props = sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=False)
    mass_props = sim_utils.MassCfg(mass=0.45)


@sim_utils.clone
def _spawn_grasped_teapot(
    prim_path: str,
    cfg: GraspedTeapotCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs: Any,
) -> Usd.Prim:
    """Compose both collider representations before cloning the body."""
    from isaaclab_newton.sim.schemas import MujocoCollisionCfg, NewtonCollisionCfg

    from pxr import Gf, Usd, UsdPhysics

    pot = sim_utils.spawn_from_mesh(prim_path, cfg, translation, orientation, **kwargs)
    contact_cfg = sim_utils.UsdFileCfg(
        usd_path=str(_prepared_contact_path(Path(cfg.contact_source))),
        collision_props=[
            sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
            MujocoCollisionCfg(condim=4, solref=(0.0025, 1.0)),
            NewtonCollisionCfg(contact_gap=0.001),
        ],
        physics_material=UsdPhysicsRigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        rigid_props={"(/.*)?": [sim_utils.UsdPhysicsRigidBodyCfg(rigid_body_enabled=False)]},
        visible=False,
    )
    # Retain the source root's frame relative to the fluid shell.
    contact_root = sim_utils.spawn_from_usd(f"{prim_path}/RigidCollider", contact_cfg)
    shell = UsdPhysics.MassAPI(pot)
    configured_mass = shell.GetMassAttr().Get()
    for prim in Usd.PrimRange(contact_root):
        if prim.HasAPI(UsdPhysics.MassAPI):
            authored = UsdPhysics.MassAPI(prim)
            for field in ("Mass", "CenterOfMass", "DiagonalInertia", "PrincipalAxes"):
                value = getattr(authored, f"Get{field}Attr")().Get()
                if value is not None:
                    getattr(shell, f"Create{field}Attr")(value)
            prim.RemoveAPI(UsdPhysics.MassAPI)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
    authored_mass = shell.GetMassAttr().Get()
    inertia = shell.GetDiagonalInertiaAttr().Get()
    shell.CreateMassAttr(configured_mass)
    if configured_mass > 0.0 and authored_mass > 0.0 and inertia is not None:
        # Scaling the body mass preserves its center of mass and principal axes.
        scale = configured_mass / authored_mass
        shell.CreateDiagonalInertiaAttr(Gf.Vec3f(*(value * scale for value in inertia)))
    return pot


def _prepared_contact_path(source: Path) -> Path:
    """Cache compact rigid contacts without changing the visual or MPM shell."""
    import newton

    from pxr import Usd, UsdGeom, UsdPhysics, Vt

    source = source.expanduser().resolve(strict=True)
    original = Usd.Stage.Open(str(source))
    if original is None:
        raise ValueError(f"Cannot open teapot source: {source}")
    root = original.GetDefaultPrim()
    if not root or not root.HasAPI(UsdPhysics.RigidBodyAPI) or not root.HasAPI(UsdPhysics.MassAPI):
        raise ValueError("Teapot source requires rigid-body and mass APIs on its default prim")
    units = UsdGeom.GetStageMetersPerUnit(original)
    if not np.isfinite(units) or units <= 0:
        raise ValueError("Teapot source has invalid length units")
    options = {
        "threshold": 0.05,
        "mcts_nodes": 20,
        "mcts_iterations": 5,
        "mcts_max_depth": 1,
        "merge": False,
        "seed": 0,
    }
    # The composed stage includes referenced geometry; tool/settings changes also invalidate the cache.
    digest = hashlib.sha256(original.Flatten().ExportToString().encode())
    digest.update(str(source).encode())
    digest.update(json.dumps(options, sort_keys=True).encode())
    digest.update(f"{version('newton')}:{version('coacd')}:{newton.Mesh.MAX_HULL_VERTICES}".encode())
    directory = Path.home() / ".cache/isaaclab/rizon-sharpa-teapot/teapot-prepared-v5"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{digest.hexdigest()}.usda"
    with FileLock(str(destination.with_suffix(".lock"))):
        if destination.is_file():
            return destination
        # Mesh inertia helpers allocate Warp arrays. Keep conversion on CPU,
        # even when invoked by a simulation whose current device is CUDA.
        with wp.ScopedDevice("cpu"):
            builder = newton.ModelBuilder()
            builder.add_usd(
                original,
                root_path=str(root.GetPath()),
                skip_mesh_approximation=True,
                load_visual_shapes=False,
                load_static_visual_shapes=False,
                load_sites=False,
            )
            if builder.body_count != 1 or builder.body_label[0] != str(root.GetPath()):
                raise ValueError("Teapot source must contain exactly one default-prim rigid body")
            colliders = [
                index
                for index, shape_type in enumerate(builder.shape_type)
                if shape_type == newton.GeoType.MESH
                and builder.shape_flags[index] & int(newton.ShapeFlags.COLLIDE_SHAPES)
            ]
            if not colliders or any(builder.shape_body[index] != 0 for index in colliders):
                raise ValueError("Teapot source requires enabled triangle-mesh contacts on its default body")
            collider_paths = [builder.shape_label[index] for index in colliders]
            for index, path in zip(colliders, collider_paths):
                imported = builder.shape_source[index]
                mesh = newton.usd.get_mesh(
                    original.GetPrimAtPath(path),
                    load_normals=False,
                    load_uvs=False,
                    load_visual_materials=False,
                    maxhullvert=imported.maxhullvert,
                    compute_inertia=False,
                )
                mesh.is_solid = imported.is_solid
                builder.shape_source[index] = mesh
            previous_count = builder.shape_count
            remeshed = builder.approximate_meshes(
                method="coacd", shape_indices=colliders, raise_on_failure=True, **options
            )
            if remeshed != set(colliders):
                raise RuntimeError("Teapot contact decomposition did not cover every source collider")
            parts = colliders + list(range(previous_count, builder.shape_count))
            if any(builder.shape_type[index] != newton.GeoType.CONVEX_MESH for index in parts):
                raise RuntimeError("Teapot contact decomposition returned a non-convex part")
            contact_meshes = []
            body_points, spout_points = [], []
            for index in parts:
                mesh = builder.shape_source[index]
                vertices = np.asarray(mesh.vertices, dtype=np.float64) * np.asarray(builder.shape_scale[index])
                transform = np.asarray(builder.shape_transform[index], dtype=np.float64)
                points = Rotation.from_quat(transform[3:]).apply(vertices) + transform[:3]
                # Retain the calibrated Utah handle and its junction with the body.
                # Merge the remaining body and spout; MPM uses a separate exact shell.
                if points[:, 0].min() < -0.055:
                    contact_meshes.append((points, np.asarray(mesh.indices, dtype=np.int32).reshape(-1, 3)))
                elif points[:, 0].max() > 0.075:
                    spout_points.append(points)
                else:
                    body_points.append(points)
            for points in (body_points, spout_points):
                if points:
                    hull = newton.Mesh(
                        np.concatenate(points), np.empty((0, 3), dtype=np.int32), compute_inertia=False
                    ).compute_convex_hull()
                    contact_meshes.append((hull.vertices, hull.indices.reshape(-1, 3)))
            with tempfile.TemporaryDirectory(dir=directory) as temporary:
                prepared = Path(temporary) / "contacts.usda"
                stage = Usd.Stage.CreateNew(str(prepared))
                stage.GetRootLayer().subLayerPaths = [str(source)]
                stage.SetDefaultPrim(stage.GetPrimAtPath(root.GetPath()))
                UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(original))
                UsdGeom.SetStageMetersPerUnit(stage, units)
                UsdPhysics.SetStageKilogramsPerUnit(stage, UsdPhysics.GetStageKilogramsPerUnit(original))
                for path in collider_paths:
                    stage.OverridePrim(path).SetActive(False)
                group = root.GetPath().AppendChild("PreparedConvexContacts")
                UsdGeom.Xform.Define(stage, group)
                for ordinal, (points, faces) in enumerate(contact_meshes):
                    # Points are body-local; the referenced root already carries its pose.
                    points = np.asarray(points / units, dtype=np.float32)
                    prim = UsdGeom.Mesh.Define(stage, group.AppendChild(f"part_{ordinal:04d}"))
                    prim.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
                    prim.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32)))
                    prim.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1)))
                    prim.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
                    UsdPhysics.CollisionAPI.Apply(prim.GetPrim()).CreateCollisionEnabledAttr(True)
                    UsdPhysics.MeshCollisionAPI.Apply(prim.GetPrim()).CreateApproximationAttr(
                        UsdPhysics.Tokens.convexHull
                    )
                stage.GetRootLayer().Save()
                prepared.replace(destination)
    return destination


def _prepared_robot_path(source: Path) -> Path:
    """Overlay repaired normals, tiny fingertip inertias, and thumb-housing filtering."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    location = hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:12]
    directory = Path.home() / ".cache/isaaclab/rizon-sharpa-teapot/robot-prepared-v4" / ROBOT_BUNDLE_SHA256
    destination = directory / f"{location}.usda"
    if destination.is_file():
        return destination
    original = Usd.Stage.Open(str(source))
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        prepared = Path(temporary) / "robot.usda"
        stage = Usd.Stage.CreateNew(str(prepared))
        stage.GetRootLayer().subLayerPaths = [str(source.resolve())]
        stage.SetDefaultPrim(stage.GetPrimAtPath(original.GetDefaultPrim().GetPath()))
        root = original.GetDefaultPrim().GetPath()
        palm_housing = stage.OverridePrim(root.AppendChild("right_hand_C_MC"))
        filtered_pairs = UsdPhysics.FilteredPairsAPI.Apply(palm_housing)
        filtered_pairs.CreateFilteredPairsRel().AddTarget(root.AppendChild("right_thumb_MC"))
        for prim in original.Traverse():
            if prim.GetName().endswith(("_fingertip", "_elastomer")) and prim.HasAPI(UsdPhysics.MassAPI):
                inertia = UsdPhysics.MassAPI(prim).GetDiagonalInertiaAttr().Get()
                # Fixed fingertip frames carry tiny positive inertias. Keep them
                # above Newton's 1e-10 floor instead of its 1e-6 correction.
                if inertia is not None and min(inertia) < 2.0e-10:
                    UsdPhysics.MassAPI(stage.OverridePrim(prim.GetPath())).CreateDiagonalInertiaAttr(
                        Gf.Vec3f(*(max(float(value), 2.0e-10) for value in inertia))
                    )
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            normals = mesh.GetNormalsAttr().Get()
            if (
                normals is not None
                and mesh.GetNormalsInterpolation() == UsdGeom.Tokens.faceVarying
                and len(normals) != len(mesh.GetFaceVertexIndicesAttr().Get())
            ):
                UsdGeom.Mesh(stage.OverridePrim(prim.GetPath())).GetNormalsAttr().Block()
        stage.GetRootLayer().Save()
        prepared.replace(destination)
    return destination
