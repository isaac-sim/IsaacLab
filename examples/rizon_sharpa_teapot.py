# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physically grasp the Utah teapot and pour MPM water during a 25 cm rise.

This experimental example uses a fixed index grasp and motion.

.. code-block:: bash

    # Newton GL:
    uv run isaaclab example rizon-sharpa-teapot


The pinned robot USD and textures are downloaded and verified on first use.
Set ``ISAACLAB_FABRICS_SIM_RIZON_SHARPA_ROOT`` to an existing bundle directory
to use it without downloading.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import runpy
import shutil
import sys
import tempfile
from collections.abc import Callable
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import warp as wp
from filelock import FileLock
from isaaclab_newton.physics import MJWarpSolverCfg, MPMSolverCfg
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from isaaclab.utils import configclass

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, ArticulationCfg, RigidObject

TABLETOP_CONTAINER_POS = (-0.355, 0.0, 0.90041)
CONTAINER_BASE_POS = (-0.340, 0.0, 1.020)
BOWL_BASE_POS = (-0.144, 0.0, 0.900408217)
ROBOT_BASE_POS = (-0.930, 0.0, 1.15041)
PICKUP_TIME = 2.0
GRASP_CLOSE_INTERVAL = (0.6, 1.3)
INITIAL_LIFT_TIME = 2.0
POUR_PREFIX_TIME = PICKUP_TIME + 0.35
HOLD_TIME = 0.0
TILT_TIME = 4.0
LIFT_DELAY = 2.5
LIFT_TIME = 8.0
LIFT_HEIGHT = 0.25
POUR_HOLD_TIME = 0.65
RECOVERY_TIME = 3.5
SEQUENCE_DURATION = POUR_PREFIX_TIME + HOLD_TIME + LIFT_DELAY + LIFT_TIME + POUR_HOLD_TIME + RECOVERY_TIME + 1.0
_POUR_ANGLE = math.radians(42.0)
_SPOUT_LOWERING_ANGLE = math.radians(35.0)
_SPOUT_LOCAL_POS = (0.111, 0.0, 0.0805)
_SPOUT_POUR_POS = (-0.169, 0.0, 0.9937642035)


def _blend(raw: float) -> tuple[float, float]:
    """Return quintic progress and its derivative with respect to raw progress."""
    progress = max(0.0, min(1.0, raw))
    return (
        progress**3 * (10.0 + progress * (-15.0 + 6.0 * progress)),
        30.0 * progress**2 * (1.0 - progress) ** 2 if 0.0 < raw < 1.0 else 0.0,
    )


def pose_at_time(sim_time: float) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Return the teapot root position [m], XYZW quaternion and linear/angular twist [m/s, rad/s]."""
    # Begin tilting while the initial lift is still clearing the table.
    progress, derivative = _blend((sim_time - PICKUP_TIME) / INITIAL_LIFT_TIME)
    displacement = tuple(b - a for a, b in zip(TABLETOP_CONTAINER_POS, CONTAINER_BASE_POS, strict=True))
    base_position = tuple(a + progress * delta for a, delta in zip(TABLETOP_CONTAINER_POS, displacement, strict=True))
    base_velocity = tuple(derivative * delta / INITIAL_LIFT_TIME for delta in displacement)

    time = sim_time - POUR_PREFIX_TIME - HOLD_TIME
    tilt, tilt_derivative = _blend(time / TILT_TIME)
    lift, lift_derivative = _blend((time - LIFT_DELAY) / LIFT_TIME)
    recovery_raw = (time - LIFT_DELAY - LIFT_TIME - POUR_HOLD_TIME) / RECOVERY_TIME
    recovery, recovery_derivative = _blend(recovery_raw)
    # Tilt directly to the pouring angle, then hold it until recovery.
    angle = _POUR_ANGLE * (tilt - recovery)
    angular_speed = _POUR_ANGLE * (tilt_derivative / TILT_TIME - recovery_derivative / RECOVERY_TIME)
    lift_speed = LIFT_HEIGHT * lift_derivative / LIFT_TIME

    # Blend the ongoing pickup into the calibrated outlet alignment.
    alignment, alignment_derivative = _blend(angle / _SPOUT_LOWERING_ANGLE)
    alignment_speed = alignment_derivative * angular_speed / _SPOUT_LOWERING_ANGLE
    x, y, z = _SPOUT_LOCAL_POS
    sine, cosine = math.sin(angle), math.cos(angle)
    rotated_spout = (x * cosine + z * sine, y, -x * sine + z * cosine)
    displacement = tuple(
        target - outlet - origin
        for target, outlet, origin in zip(_SPOUT_POUR_POS, rotated_spout, base_position, strict=True)
    )
    position = tuple(a + alignment * delta for a, delta in zip(base_position, displacement, strict=True))
    target_velocity = (angular_speed * (x * sine - z * cosine), 0.0, angular_speed * (x * cosine + z * sine))
    velocity = tuple(
        (1.0 - alignment) * base_speed + alignment_speed * delta + alignment * speed
        for base_speed, delta, speed in zip(base_velocity, displacement, target_velocity, strict=True)
    )

    # Compensate for the longer falling stream, then ease forward for the final drips.
    gap = 0.028
    fall_speed = math.sqrt(0.45**2 + 2.0 * 9.81 * gap)
    raised_fall_speed = math.sqrt(0.45**2 + 2.0 * 9.81 * (gap + LIFT_HEIGHT * lift))
    forward_speed = 0.30 - 0.04 * lift
    extra_fall_time = (raised_fall_speed - fall_speed) / 9.81
    retreat_speed = -0.04 * lift_speed / LIFT_HEIGHT * extra_fall_time + forward_speed * lift_speed / raised_fall_speed
    drip, drip_derivative = _blend(recovery_raw * RECOVERY_TIME / 0.8)
    position = (
        position[0] - forward_speed * extra_fall_time + 0.01 * drip,
        position[1],
        position[2] + LIFT_HEIGHT * lift,
    )
    velocity = (velocity[0] - retreat_speed + 0.01 * drip_derivative / 0.8, velocity[1], velocity[2] + lift_speed)
    return position, (0.0, math.sin(0.5 * angle), 0.0, math.cos(0.5 * angle)), (*velocity, 0.0, angular_speed, 0.0)


def approach_offset_at_time(sim_time: float) -> tuple[float, ...]:
    """Insert the index finger from beside the handle after a brief settling interval [m]."""
    progress, _ = _blend((sim_time - 0.1) / (GRASP_CLOSE_INTERVAL[0] - 0.1))
    return (0.0, -0.035 * (1.0 - progress), 0.0)


def main() -> None:
    """Run the shared teapot scene with the fixed physical index grasp."""
    original_args = sys.argv[:]
    sys.argv[1:1] = [
        "--robot",
        "rizon_sharpa",
        "--fill_level",
        "0.70",
        "--fluid_render_mode",
        "particles",
        "--max_steps",
        str(round(800 * SEQUENCE_DURATION)),
    ]
    try:
        runpy.run_path(str(Path(__file__).parent / "demos" / "teapot_fill.py"), run_name="__main__")
    finally:
        sys.argv[:] = original_args


_ContainerPose = tuple[
    tuple[float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float, float, float],
]

ASSET_REVISION = "d0dbd1ddaefc4996db546949a7dfb37e39afcbeb"
ASSET_BUNDLE_SHA256 = "ae5d22792b44fb6d29a7691d4276bc061a5529132f01e7a0eb5795a482595d63"
ASSET_USD_NAME = "rizon4s_sharpa_no_spheres_generated.usd"
ASSET_ROOT_ENV = "ISAACLAB_FABRICS_SIM_RIZON_SHARPA_ROOT"
_ASSET_RELATIVE_PATHS = (
    ASSET_USD_NAME,
    "textures/t_Rizon4s_EmissiveMask.1001.png",
    "textures/t_Rizon4s_EmissiveMask.1002.png",
    *[f"textures/t_Rizon4s_{channel}.{tile}.png" for channel in ("alb", "nor", "orm") for tile in (1001, 1002, 1003)],
    "textures/t_Sharpa_alb.png",
    "textures/t_Sharpa_nor.png",
    "textures/t_Sharpa_orm.png",
)
PALM_BODY_NAME = "r_palm_ctrl"
"""Canonical palm: +X toward the knuckles, +Z out of the palm."""

ARM_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))
ARM_HOME_POSITIONS = (
    0.457719810068,
    -2.146584624401,
    -0.617383959981,
    1.860143482555,
    0.884768946633,
    2.692224339911,
    -1.200491354255,
)
"""IK seed with elbow and wrist clearance for pickup and the elevated pour [rad]."""
HAND_JOINT_NAMES = (
    "right_thumb_CMC_FE",
    "right_thumb_CMC_AA",
    "right_thumb_MCP_FE",
    "right_thumb_MCP_AA",
    "right_thumb_IP",
    "right_index_MCP_FE",
    "right_index_MCP_AA",
    "right_index_PIP",
    "right_index_DIP",
    "right_middle_MCP_FE",
    "right_middle_MCP_AA",
    "right_middle_PIP",
    "right_middle_DIP",
    "right_ring_MCP_FE",
    "right_ring_MCP_AA",
    "right_ring_PIP",
    "right_ring_DIP",
    "right_pinky_CMC",
    "right_pinky_MCP_FE",
    "right_pinky_MCP_AA",
    "right_pinky_PIP",
    "right_pinky_DIP",
)
_CLEARED_FINGERS = (
    (0.098851, -0.099312, 1.646411, 0.712023)  # Middle
    + (0.768873, -0.120827, 1.278450, 1.117291)  # Ring
    + (0.002, 1.408754, -0.265713, 1.471442, 1.204550)  # Pinky
)
"""Relaxed unused fingers with clearance from one another, the pot, and the table [rad]."""


HAND_OPEN_POSITIONS = (
    (0.560745308772, 0.113282128071, 0.001196655587, -0.031283024683, 0.331758077572)  # Thumb
    + (0.812310475827, -0.039792563766, 0.680000000000, 0.200000000000)  # Index
    + _CLEARED_FINGERS
)
"""Index aligned for insertion, with the thumb and unused fingers clear [rad]."""

HAND_GRASP_POSITIONS = (
    (0.747349262238, 0.289767593145, 0.094652488828, -0.132383197546, 0.297648280859)  # Thumb
    + (0.835736691952, -0.039792563766, 0.616298854351, 0.907486081123)  # Index
    + _CLEARED_FINGERS
)
"""Index hooked through the handle, opposed by the thumb [rad]."""

HAND_HOLD_POSITIONS = (
    # Oppose the handle without running the thumb pad onto the hooked index's middle phalanx.
    (0.769337998127, 0.317099993706, 0.116388337027, -0.138365810236, 0.304292971503)  # Thumb
    + (0.836523531919, -0.321860616915, 0.615938444479, 0.907486536391)  # Index
    + _CLEARED_FINGERS
)
"""Force-limited squeeze targets; contacts determine the actual joint pose [rad]."""

PALM_IN_POT = (
    (-0.206058637337, -0.066785491216, 0.060669620349),
    (0.815358862991, 0.016218134562, 0.024054733648, -0.578228558999),
)
"""Palm pose near the loaded grasp equilibrium, reducing recoil during pickup."""

_THUMB_CONTACT_POINT = np.array((0.004038638, -0.016705590, -0.004974879))
_THUMB_CONTACT_NORMAL = np.array((-0.320969277, 0.308200555, 0.895539581))
"""Calibrated upper-handle contact point and outward normal in the thumb-pad frame."""


def configured_asset_path() -> Path:
    """Download the pinned robot bundle or verify an explicitly configured local copy."""
    configured = os.environ.get(ASSET_ROOT_ENV)
    root = (
        Path(configured).expanduser()
        if configured
        else Path.home() / ".cache/isaaclab/fabrics-sim/rizon4s-sharpa/sha256" / ASSET_BUNDLE_SHA256
    )
    if root.suffix.lower() in (".usd", ".usda", ".usdc"):
        root = root.parent
    if not configured:
        from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, read_file

        source = f"{ISAACLAB_NUCLEUS_DIR}/Environments/RizonSharpaRJ45/Robots/Rizon4sSharpa"
        root.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(f"{root}.lock"):
            missing = [relative for relative in _ASSET_RELATIVE_PATHS if not (root / relative).is_file()]
            if missing:
                with tempfile.TemporaryDirectory(dir=root.parent) as temporary:
                    staging = Path(temporary)
                    for relative in _ASSET_RELATIVE_PATHS:
                        target = staging / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if (root / relative).is_file():
                            shutil.copyfile(root / relative, target)
                        else:
                            target.write_bytes(read_file(f"{source}/{relative}").getbuffer())
                    _verify_asset_bundle(staging.resolve())
                    if not root.exists():
                        staging.rename(root)
                    else:
                        for relative in missing:
                            target = root / relative
                            target.parent.mkdir(parents=True, exist_ok=True)
                            (staging / relative).replace(target)
    path = root / ASSET_USD_NAME
    _verify_asset_bundle(root.resolve())
    return path


def get_robot_cfg(*, base_position: tuple[float, float, float] = (-0.68, 0.0, 0.90041)) -> ArticulationCfg:
    """Return the 29-joint robot ready for the calibrated index-finger grasp."""
    from isaaclab_newton.sim.schemas import (
        MujocoCollisionCfg,
        MujocoJointCfg,
        NewtonArticulationCfg,
        NewtonCollisionCfg,
    )

    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_prepared_robot_path(configured_asset_path())),
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
            pos=base_position,
            # Leave shoulder travel available throughout the elevated pour and recovery.
            rot=(0.0, 0.0, math.sin(math.pi / 8.0), math.cos(math.pi / 8.0)),
            joint_pos={
                **dict(zip(ARM_JOINT_NAMES, ARM_HOME_POSITIONS, strict=True)),
                **dict(zip(HAND_JOINT_NAMES, HAND_OPEN_POSITIONS, strict=True)),
            },
        ),
        actuators={
            "shoulder": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-2]"],
                joint_effort_limit=123.0,
                joint_velocity_limit=2.094,
                stiffness=6000.0,
                damping=108.5,
            ),
            "elbow": ImplicitActuatorCfg(
                joint_names_expr=["joint[3-4]"],
                joint_effort_limit=64.0,
                joint_velocity_limit=2.443,
                stiffness=4200.0,
                damping=90.7,
            ),
            "wrist": ImplicitActuatorCfg(
                joint_names_expr=["joint[5-7]"],
                joint_effort_limit=39.0,
                joint_velocity_limit=4.887,
                stiffness=1500.0,
                damping=54.2,
            ),
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


def prepared_contact_path(source: Path) -> Path:
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


@lru_cache(maxsize=4)
def _prepared_robot_path(source: Path) -> Path:
    """Overlay repaired normals, tiny fingertip inertias, and thumb-housing filtering."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    location = hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:12]
    directory = Path.home() / ".cache/isaaclab/rizon-sharpa-teapot/robot-prepared-v4" / ASSET_BUNDLE_SHA256
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


@lru_cache(maxsize=4)
def _verify_asset_bundle(root: Path) -> None:
    """Verify the pinned robot USD and textures without modifying them."""
    digest = hashlib.sha256()
    for relative_path in _ASSET_RELATIVE_PATHS:
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing Rizon/Sharpa asset: {path}. Set {ASSET_ROOT_ENV} to the licensed bundle directory "
                f"containing {ASSET_USD_NAME} and textures/."
            )
        with path.open("rb") as stream:
            file_digest = hashlib.file_digest(stream, "sha256").hexdigest()
        digest.update(f"{file_digest}  {relative_path}\n".encode())
    if digest.hexdigest() != ASSET_BUNDLE_SHA256:
        raise ValueError(f"Rizon/Sharpa asset bundle at {root} differs from revision {ASSET_REVISION}.")


class TeapotPourMotion:
    """Drive a simulated arm along an IK path with a contact-acquired grasp.

    Startup IK supplies the approach; bounded differential IK uses the measured
    object pose to compensate for contact compliance during pickup and pouring.
    Thumb feedback maintains acquired opposition while permitting pad rotation.
    Finger contacts hold the handle throughout the lift and pour. The teapot
    remains dynamic; no attachment constraint or body pose override is used.
    """

    def __init__(
        self,
        robot: Articulation,
        container: RigidObject,
        pose_at_time: Callable[[float], _ContainerPose],
        *,
        sim_dt: float,
        duration: float,
        grasp_time: float,
        close_interval: tuple[float, float],
        pickup_end_time: float,
        approach_offset_at_time: Callable[[float], tuple[float, float, float]] | None = None,
    ) -> None:
        import mujoco
        from scipy.optimize import least_squares

        from isaaclab.sim import SimulationContext

        if not 0 <= close_interval[0] < close_interval[1] < grasp_time - 0.3 < grasp_time < pickup_end_time < duration:
            raise ValueError("Closure, pickup and sequence timings must be strictly ordered.")
        if not math.isclose(sim_dt, 1.0 / 800.0, rel_tol=1.0e-9):
            raise ValueError("The calibrated contact grasp requires an 800 Hz simulation timestep.")
        self._robot = robot
        self._container = container
        self._pose_at_time = pose_at_time
        self._grasp_time = grasp_time
        self._pickup_end_time = pickup_end_time
        self._gain_start, self._gain_end = close_interval[1] + 0.1, grasp_time - 0.3
        self._gain_settled = False
        self._path_hz = 100.0
        self._physics_hz = 1.0 / sim_dt
        self._control_decimation = 8
        self._palm = robot.body_names.index(PALM_BODY_NAME)
        self._palm_jacobian = self._palm - int(robot.is_fixed_base)
        joint_ids = robot.find_joints(list(ARM_JOINT_NAMES + HAND_JOINT_NAMES), preserve_order=True)[0]
        self._arm_jacobian_dofs = [joint + robot.num_base_dofs for joint in joint_ids[:7]]
        self._thumb = robot.body_names.index("right_thumb_elastomer")
        self._thumb_jacobian = self._thumb - int(robot.is_fixed_base)
        self._thumb_joints = joint_ids[7:12]
        self._thumb_jacobian_dofs = [joint + robot.num_base_dofs for joint in self._thumb_joints]
        # Startup IK needs MJWarp's native model. Feedback reads public asset state.
        mj = SimulationContext.instance().backend_manager.backend.solver.solver("robot").mj_model
        names = [mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, i).split("_Physics_")[-1] for i in range(mj.njnt)]
        joints = [names.index(name) for name in ARM_JOINT_NAMES + HAND_JOINT_NAMES]
        arm_q = mj.jnt_qposadr[joints[:7]]
        self._qmin, self._qmax = mj.jnt_range[joints[:7], 0] + 0.03, mj.jnt_range[joints[:7], 1] - 0.03
        self._thumb_qmin = mj.jnt_range[joints[7:12], 0] + 0.002
        self._thumb_qmax = mj.jnt_range[joints[7:12], 1] - 0.002
        self._speed_limits = robot.data.joint_vel_limits.warp.numpy()[0, joint_ids].astype(np.float64)
        data = mujoco.MjData(mj)
        mj_palm = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, "_World_envs_env_0_Robot_" + PALM_BODY_NAME)
        palm_rotation = Rotation.from_quat(PALM_IN_POT[1])
        previous = np.array(ARM_HOME_POSITIONS)
        times = np.arange(round(grasp_time * self._path_hz) + 1) / self._path_hz
        trajectory = []
        for time in times:
            position, quaternion, _ = pose_at_time(float(time))
            rotation = Rotation.from_quat(quaternion)
            target_position = np.asarray(position) + rotation.apply(PALM_IN_POT[0])
            if approach_offset_at_time is not None:
                target_position += approach_offset_at_time(float(time))
            target_rotation = rotation * palm_rotation

            def residual(q):
                data.qpos[arm_q] = q
                mujoco.mj_kinematics(mj, data)
                error_rotation = target_rotation * Rotation.from_matrix(data.xmat[mj_palm].reshape(3, 3)).inv()
                return np.concatenate(
                    (
                        data.xpos[mj_palm] - target_position,
                        0.18 * error_rotation.as_rotvec(),
                        1.0e-4 * (q - previous),
                    )
                )

            solution = least_squares(
                residual,
                previous,
                bounds=(self._qmin, self._qmax),
                max_nfev=100,
                gtol=1.0e-9,
                ftol=1.0e-9,
                xtol=1.0e-9,
            )
            error = float(np.linalg.norm(residual(solution.x)[:6]))
            if error > 2.0e-4:
                raise RuntimeError(f"Teapot approach is unreachable at {time:.2f}s (IK error {error:.4g}).")
            previous = solution.x
            alpha = self._blend((time - close_interval[0]) / (close_interval[1] - close_interval[0]))
            squeeze = self._blend((time - close_interval[1]) / (self._gain_end - close_interval[1]))
            hand = (
                np.asarray(HAND_OPEN_POSITIONS)
                + alpha * (np.asarray(HAND_GRASP_POSITIONS) - HAND_OPEN_POSITIONS)
                + squeeze * (np.asarray(HAND_HOLD_POSITIONS) - HAND_GRASP_POSITIONS)
            )
            thumb_alpha = self._blend((time - close_interval[0]) / (0.65 * (close_interval[1] - close_interval[0])))
            hand[:5] = (
                np.asarray(HAND_OPEN_POSITIONS[:5])
                + thumb_alpha * (np.asarray(HAND_GRASP_POSITIONS[:5]) - HAND_OPEN_POSITIONS[:5])
                + squeeze * (np.asarray(HAND_HOLD_POSITIONS[:5]) - HAND_GRASP_POSITIONS[:5])
            )
            trajectory.append(np.concatenate((previous, hand)))
        trajectory = np.asarray(trajectory, dtype=np.float32)
        spline = CubicSpline(times, trajectory, bc_type="clamped")
        speeds = np.max(
            np.abs(spline(np.linspace(0.0, grasp_time, round(grasp_time * self._physics_hz) + 1), 1)), axis=0
        )
        if np.any(speeds > self._speed_limits):
            raise RuntimeError(f"Teapot approach exceeds rated joint speeds: {speeds} rad/s.")
        self._trajectory = wp.array(spline.c.astype(np.float32), dtype=float, device=robot.device)
        self._joint_ids = wp.array(joint_ids, dtype=int, device=robot.device)
        self._hand_joint_ids = wp.array(joint_ids[7:], dtype=int, device=robot.device)
        self._arm_position = trajectory[-1, :7].astype(np.float64)
        self._arm_velocity = np.zeros(7)
        self._thumb_position = np.asarray(HAND_HOLD_POSITIONS[:5]).copy()
        self._thumb_velocity = np.zeros(5)
        self._thumb_anchor: tuple[np.ndarray, np.ndarray] | None = None
        self._thumb_position_gpu = wp.array(self._thumb_position[None].astype(np.float32), device=robot.device)
        self._arm_position_gpu, self._arm_velocity_gpu = (
            wp.zeros(7, device=robot.device),
            wp.zeros(7, device=robot.device),
        )
        self._last_feedback_step = None
        self._next_feedback_step = round(grasp_time * self._physics_hz)
        initial = robot.data.default_joint_pos.warp.numpy()
        initial[0, joint_ids] = trajectory[0]
        self.initial_joint_positions = wp.array(initial, dtype=float, device=robot.device)
        robot.actuators.target_command.set_position_index(value=self.initial_joint_positions)

    @staticmethod
    def _blend(progress: float) -> float:
        """Clamp and evaluate a quintic transition with zero endpoint velocity and acceleration."""
        progress = float(np.clip(progress, 0.0, 1.0))
        return progress**3 * (10.0 + progress * (-15.0 + 6.0 * progress))

    def update(self, sim_time: float) -> None:
        """Acquire the contact grasp, then drive the arm from measured object pose."""
        step = round(sim_time * self._physics_hz)
        if step % self._control_decimation == 0 and not self._gain_settled and sim_time >= self._gain_start:
            gain = 24.0 + 76.0 * self._blend((sim_time - self._gain_start) / (self._gain_end - self._gain_start))
            self._robot.write_joint_stiffness_to_sim_index(stiffness=gain, joint_ids=self._hand_joint_ids)
            self._robot.write_joint_damping_to_sim_index(
                damping=1.2 * math.sqrt(gain / 24.0), joint_ids=self._hand_joint_ids
            )
            self._gain_settled = sim_time >= self._gain_end
        feedback_active = sim_time >= self._grasp_time
        if feedback_active and step >= self._next_feedback_step:
            self._update_feedback(sim_time, step)
        # Advance the arm target by one physics step while holding 100 Hz feedback.
        feedback_elapsed = (step - self._last_feedback_step + 1) / self._physics_hz if feedback_active else 0.0
        wp.launch(
            _interpolate_joints,
            dim=len(ARM_JOINT_NAMES) + len(HAND_JOINT_NAMES),
            inputs=[
                self._trajectory,
                self._joint_ids,
                float(sim_time * self._path_hz),
                1.0 / self._path_hz,
                self._arm_position_gpu,
                self._arm_velocity_gpu,
                self._thumb_position_gpu,
                feedback_active,
                feedback_elapsed,
                self._robot.actuators.target_command.position,
                self._robot.actuators.target_command.velocity,
            ],
            device=self._robot.device,
        )

    def _update_feedback(self, sim_time: float, step: int) -> None:
        """Follow the measured pot with bounded arm and thumb position targets.

        Finger contacts supply every force on the pot. Object feedback compensates
        for finger compliance and pivoting in the handle. The velocity solve respects
        rated speeds, an 8 rad/s^2 acceleration envelope, and joint stopping distances.
        """
        if self._last_feedback_step is not None:
            self._arm_position += self._arm_velocity * (step - self._last_feedback_step) / self._physics_hz
        pot = self._container.data.root_link_pose_w.warp.numpy()[0].astype(np.float64)
        position, quaternion, twist = self._pose_at_time(sim_time)
        pot_rotation = Rotation.from_quat(pot[3:])
        error = np.concatenate(
            (
                np.asarray(position) - pot[:3],
                (Rotation.from_quat(quaternion) * pot_rotation.inv()).as_rotvec(),
            )
        )
        # Level gently as the base clears the table; ease the higher pickup gain before pouring.
        pickup_weight = 1.0 - self._blend((sim_time - self._pickup_end_time + 0.5) / 0.5)
        feedback_weight = self._blend(
            (sim_time - self._grasp_time) / (0.5 * (self._pickup_end_time - self._grasp_time))
        )
        rotation_gain = (8.0 + 4.0 * pickup_weight) * feedback_weight
        command = np.asarray(twist) + error * np.array((10.0, 10.0, 10.0, rotation_gain, rotation_gain, rotation_gain))
        command[:3], command[3:] = np.clip(command[:3], -0.45, 0.45), np.clip(command[3:], -1.5, 1.5)
        poses = self._robot.data.body_link_pose_w.warp.numpy()[0]
        jacobians = self._robot.data.body_link_jacobian_w.warp.numpy()[0]
        palm = poses[self._palm, :3]
        jacobian = jacobians[self._palm_jacobian][:, self._arm_jacobian_dofs].astype(np.float64)
        # Evaluate the palm's twist at the measured pot root, preserving the
        # object-centered task without treating the contact grasp as a weld.
        jacobian[:3] += np.cross(jacobian[3:].T, pot[:3] - palm).T
        jacobian[3:] *= 0.18
        command[3:] *= 0.18
        q = self._arm_position
        posture = 0.4 * ((self._qmin + self._qmax) / 2.0 - q) + 0.08 * (
            1.0 / np.maximum(q - self._qmin, 0.1) ** 2 - 1.0 / np.maximum(self._qmax - q, 0.1) ** 2
        )
        # Preserve the joint-limit avoidance direction when bounding posture speed.
        posture *= 2.0 / max(2.0, float(np.max(np.abs(posture))))
        matrix = np.vstack((jacobian, 0.005 * np.eye(7)))
        target = np.concatenate((command, 0.005 * posture))
        self._arm_velocity = self._bounded_velocity(
            q, self._arm_velocity, self._qmin, self._qmax, self._speed_limits[:7], matrix, target
        )
        self._update_thumb_feedback(pot, poses, jacobians)
        self._arm_position_gpu.assign(self._arm_position.astype(np.float32))
        self._arm_velocity_gpu.assign(self._arm_velocity.astype(np.float32))
        self._last_feedback_step = step
        self._next_feedback_step = step + self._control_decimation

    def _update_thumb_feedback(self, pot: np.ndarray, poses: np.ndarray, jacobians: np.ndarray) -> None:
        """Maintain acquired thumb opposition through physical, torque-limited finger motion."""
        pose = poses[self._thumb].astype(np.float64)
        rotation = Rotation.from_quat(pose[3:])
        offset = rotation.apply(_THUMB_CONTACT_POINT)
        point, normal = pose[:3] + offset, rotation.apply(_THUMB_CONTACT_NORMAL)
        jacobian = jacobians[self._thumb_jacobian][:, self._thumb_jacobian_dofs].astype(np.float64)
        jacobian[:3] += np.cross(jacobian[3:].T, offset).T
        actual = self._robot.data.joint_pos.warp.numpy()[0, self._thumb_joints]
        delta = self._thumb_position - actual
        # Track the commanded pad point rather than integrating elastic compression
        # into more squeeze. This preserves the acquired preload without winding up.
        predicted_point = point + jacobian[:3] @ delta
        predicted_normal = normal + np.cross(jacobian[3:] @ delta, normal)
        predicted_normal /= np.linalg.norm(predicted_normal)
        pot_rotation = Rotation.from_quat(pot[3:])
        if self._thumb_anchor is None:
            self._thumb_anchor = (
                pot_rotation.inv().apply(predicted_point - pot[:3]),
                pot_rotation.inv().apply(predicted_normal),
            )
        desired_point = pot[:3] + pot_rotation.apply(self._thumb_anchor[0])
        desired_normal = pot_rotation.apply(self._thumb_anchor[1])
        # Prioritize opposition at the contact point; a soft normal target permits
        # pad rotation instead of forcing a singular five-coordinate pose lock.
        matrix = np.vstack((jacobian[:3], 0.002 * np.cross(jacobian[3:].T, normal).T, 0.005 * np.eye(5)))
        target = np.concatenate(
            (
                10.0 * (desired_point - predicted_point),
                0.016 * (desired_normal - predicted_normal),
                0.002 * (np.asarray(HAND_HOLD_POSITIONS[:5]) - self._thumb_position),
            )
        )
        self._thumb_velocity = self._bounded_velocity(
            self._thumb_position,
            self._thumb_velocity,
            self._thumb_qmin,
            self._thumb_qmax,
            self._speed_limits[7:12],
            matrix,
            target,
        )
        self._thumb_position += self._thumb_velocity / self._path_hz
        self._thumb_position_gpu.assign(self._thumb_position[None].astype(np.float32))

    def _bounded_velocity(
        self,
        q: np.ndarray,
        velocity: np.ndarray,
        qmin: np.ndarray,
        qmax: np.ndarray,
        speed_limits: np.ndarray,
        matrix: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        """Solve joint velocities within rated speeds, acceleration, and stopping distance."""
        from scipy.optimize import lsq_linear

        period, acceleration = 1.0 / self._path_hz, 8.0
        increment = acceleration * period
        distances = np.maximum(np.stack((q - qmin, qmax - q)), 0.0) / (increment * period)
        braking_steps = np.floor((np.sqrt(1.0 + 8.0 * distances) - 1.0) / 2.0)
        # Invert dt * sum(max(v - k * acceleration * dt, 0)). The continuous
        # stopping distance can leave no feasible velocity at the next update.
        braking_speeds = increment * (distances / (braking_steps + 1.0) + braking_steps / 2.0)
        lower = np.maximum(-speed_limits, np.maximum(-braking_speeds[0], velocity - increment))
        upper = np.minimum(speed_limits, np.minimum(braking_speeds[1], velocity + increment))
        if np.any(lower > upper + 1.0e-7):
            raise RuntimeError("Infeasible joint braking envelope.")
        # A joint at its stopping boundary has one feasible velocity. Eliminate
        # those coordinates because scipy's box solver requires strict bounds.
        fixed = upper - lower < 1.0e-8
        free = ~fixed
        result = np.zeros_like(q)
        result[fixed] = 0.5 * (lower[fixed] + upper[fixed])
        if np.any(free):
            solution = lsq_linear(
                matrix[:, free],
                target - matrix[:, fixed] @ result[fixed],
                bounds=(lower[free], upper[free]),
                method="bvls",
                tol=1.0e-9,
                max_iter=50,
            )
            if not solution.success or not np.isfinite(solution.x).all():
                raise RuntimeError(f"Joint velocity solve failed: {solution.message}")
            result[free] = solution.x
        return result


@wp.kernel
def _interpolate_joints(
    path: wp.array3d[float],
    joints: wp.array[int],
    sample: float,
    path_dt: float,
    arm_position: wp.array[float],
    arm_velocity: wp.array[float],
    thumb_position: wp.array2d[float],
    feedback_active: bool,
    feedback_elapsed: float,
    targets: wp.array2d[float],
    velocities: wp.array2d[float],
):
    i = wp.tid()
    sample = wp.clamp(sample, 0.0, float(path.shape[1]))
    start = wp.min(int(sample), path.shape[1] - 1)
    elapsed = (sample - float(start)) * path_dt
    targets[0, joints[i]] = (
        (path[0, start, i] * elapsed + path[1, start, i]) * elapsed + path[2, start, i]
    ) * elapsed + path[3, start, i]
    # Position control with zero velocity target damps finger contact instead
    # of adding a closure-speed impulse after a phalanx meets the handle.
    velocities[0, joints[i]] = 0.0
    if i < 7:
        velocities[0, joints[i]] = (3.0 * path[0, start, i] * elapsed + 2.0 * path[1, start, i]) * elapsed + path[
            2, start, i
        ]
        if feedback_active:
            targets[0, joints[i]] = arm_position[i] + feedback_elapsed * arm_velocity[i]
            velocities[0, joints[i]] = arm_velocity[i]
    elif feedback_active and i < 12:
        targets[0, joints[i]] = thumb_position[0, i - 7]


@configclass
class TeapotSolverCfg(CouplerProxyCfg):
    """Select convex rigid contacts and the exact MPM shell after simulation launch."""

    class_type: type | str = "_rizon_sharpa_teapot_coupler:TeapotSolver"


def make_solver_cfg(mpm_cfg: MPMSolverCfg) -> TeapotSolverCfg:
    """Resolve rigid hand--pot contacts and synchronize the measured fluid boundary."""
    return TeapotSolverCfg(
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
                substeps=1,
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


if __name__ == "__main__":
    main()
