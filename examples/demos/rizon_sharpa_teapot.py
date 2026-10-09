# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rizon4s/Sharpa asset contract for the teapot demonstration.

The USD and textures remain in the user's asset cache because the upstream
Fabrics-Sim asset is NVIDIA proprietary. The bundle is shared with the Rizon–Sharpa coffee and RJ45 demonstrations.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import warp as wp
from filelock import FileLock
from isaaclab_newton.physics import NewtonManager
from newton import BodyFlags, GeoType, StateFlags
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledProxy
from scipy.interpolate import CubicSpline
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from isaaclab_contrib.coupling.coupler import NewtonCouplerManager

if TYPE_CHECKING:
    from newton import ModelBuilder

    from isaaclab.assets import Articulation, ArticulationCfg, RigidObject

logger = logging.getLogger(__name__)

_ContainerPose = tuple[
    tuple[float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float, float, float],
]

ASSET_REVISION = "d0dbd1ddaefc4996db546949a7dfb37e39afcbeb"
ASSET_BUNDLE_SHA256 = "ae5d22792b44fb6d29a7691d4276bc061a5529132f01e7a0eb5795a482595d63"
ASSET_USD_NAME = "rizon4s_sharpa_no_spheres_generated.usd"
ASSET_ROOT_ENV = "ISAACLAB_FABRICS_SIM_RIZON_SHARPA_ROOT"
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
HAND_OPEN_POSITIONS = (
    (0.845973763383, 0.125171802920, -0.017756344590, 0.050021490807, 0.531562269870)  # Thumb
    + (-0.172307, -0.345035, 1.652764, 0.922972)  # Index
    + (1.031823297928, -0.159787612115, 0.525572413672, 0.002000000000)  # Middle
    + (1.251685, -0.111183, 1.420558, 1.394272)  # Ring
    + (0.227997, 1.038287, 0.057138, 1.739520, 1.385767)  # Pinky
)
"""Straight middle finger for insertion; thumb raised and remaining fingers folded clear [rad]."""

HAND_GRASP_POSITIONS = (
    (0.974189127271, 0.211061400850, 0.039555008147, -0.042665442093, 0.518353740463)  # Thumb
    + HAND_OPEN_POSITIONS[5:9]  # Folded index
    + (1.066771632614, -0.194172242080, 0.477623069828, 0.850343590830)  # Middle
    + HAND_OPEN_POSITIONS[13:]  # Folded ring/pinky
)
"""One middle finger through the original handle, with the thumb on its upper connection [rad]."""

HAND_HOLD_POSITIONS = (
    (1.015033396161, 0.305089982727, 0.097067012757, -0.042814197356, 0.542465552736)  # Thumb
    + HAND_OPEN_POSITIONS[5:9]  # Folded index
    + (1.061720879432, -0.247316271906, 0.468109540381, 0.862159956653)  # Middle
    + HAND_OPEN_POSITIONS[13:]  # Folded ring/pinky
)
"""Force-limited squeeze targets after geometric closure; contacts determine the actual joint pose [rad]."""

PALM_IN_POT = (
    (-0.197511821700, -0.073740792821, 0.069355780931),
    (0.730973745537, -0.022873525489, -0.038976477952, -0.680907496900),
)


@dataclass(frozen=True)
class _GraspProfile:
    """Calibrated contact grasp on the unchanged Utah teapot handle."""

    open_positions: tuple[float, ...]
    grasp_positions: tuple[float, ...]
    hold_positions: tuple[float, ...]
    palm_in_pot: tuple[tuple[float, float, float], tuple[float, float, float, float]]
    thumb_close_fraction: float = 1.0


_INDEX_CLEARED_FINGERS = (
    (0.098851, -0.099312, 1.646411, 0.712023)  # Middle
    + (0.768873, -0.120827, 1.278450, 1.117291)  # Ring
    + (0.002, 1.408754, -0.265713, 1.471442, 1.204550)  # Pinky
)
"""Relaxed unused fingers with clearance from one another, the pot, and the table [rad]."""


_GRASP_PROFILES = {
    "middle": _GraspProfile(HAND_OPEN_POSITIONS, HAND_GRASP_POSITIONS, HAND_HOLD_POSITIONS, PALM_IN_POT),
    "index": _GraspProfile(
        (
            (0.560745308772, 0.113282128071, 0.001196655587, -0.031283024683, 0.331758077572)  # Thumb
            + (0.812310475827, -0.039792563766, 0.680000000000, 0.200000000000)  # Index
            + _INDEX_CLEARED_FINGERS  # Folded middle/ring/pinky
        ),
        (
            (0.747349262238, 0.289767593145, 0.094652488828, -0.132383197546, 0.297648280859)  # Thumb
            + (0.835736691952, -0.039792563766, 0.616298854351, 0.907486081123)  # Index
            + _INDEX_CLEARED_FINGERS  # Folded middle/ring/pinky
        ),
        (
            (0.769337998127, 0.347099993706, 0.116388337027, -0.138365810236, 0.304292971503)  # Thumb
            + (0.836523531919, -0.321860616915, 0.615938444479, 0.907486536391)  # Index
            + _INDEX_CLEARED_FINGERS  # Folded middle/ring/pinky
        ),
        (
            # Start near the loaded grasp equilibrium to reduce recoil as the table unloads.
            (-0.206058637337, -0.066785491216, 0.060669620349),
            (0.815358862991, 0.016218134562, 0.024054733648, -0.578228558999),
        ),
        # Oppose the index finger before its curl can push the pot sideways.
        thumb_close_fraction=0.65,
    ),
}


def configured_asset_path(*, verify: bool = True) -> Path:
    """Resolve the existing RJ45 robot USD, optionally checking all pinned textures."""
    configured = os.environ.get(ASSET_ROOT_ENV)
    root = (
        Path(configured).expanduser()
        if configured
        else Path.home() / ".cache/isaaclab/fabrics-sim/rizon4s-sharpa/sha256" / ASSET_BUNDLE_SHA256
    )
    if root.suffix.lower() in (".usd", ".usda", ".usdc"):
        root = root.parent
    path = root / ASSET_USD_NAME
    if verify:
        _verify_asset_bundle(root.resolve())
    return path


def get_robot_cfg(
    *, base_position: tuple[float, float, float] = (-0.68, 0.0, 0.90041), grasp_finger: str = "middle"
) -> ArticulationCfg:
    """Return the 29-joint robot at the chosen mount with a curled hand."""
    from isaaclab_newton.sim.schemas import (
        MujocoCollisionCfg,
        MujocoJointCfg,
        NewtonArticulationCfg,
        NewtonCollisionCfg,
    )

    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    profile = _GRASP_PROFILES[grasp_finger]
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
                **dict(zip(HAND_JOINT_NAMES, profile.open_positions, strict=True)),
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
                viscous_friction={
                    joint: 1.0 if grasp_finger == "index" and joint == "right_index_MCP_AA" else 0.5
                    for joint in HAND_JOINT_NAMES
                },
                # The measured contact response stays stable at 800 Hz without changing link inertias.
                armature=2.0e-4,
            ),
        },
        soft_joint_pos_limit_factor=0.95,
    )


def prepared_contact_path(source: Path) -> Path:
    """Cache authored convex parts while retaining the source body's mass properties.

    The prepared USD overlays the source, deactivates its original collision
    meshes, and supplies convex hulls in the same body's local frame. The demo
    still loads its exact visual and MPM shell from the original USD. CoACD is
    run only when the composed source, conversion settings, or tool versions
    change; subsequent launches import the authored parts without decomposition.
    Collision preparation reads geometry without shading-driven vertex splits.

    Args:
        source: Local stock teapot USD with one rigid body at its default prim.

    Returns:
        Local prepared contact USD in the Isaac Lab user cache.
    """
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
    layers = []
    for layer in original.GetUsedLayers():
        if layer.realPath:
            with Path(layer.realPath).open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            identifier = layer.identifier
        elif layer.empty:
            # Empty session layers have process-specific anonymous identifiers.
            continue
        else:
            digest = hashlib.sha256(layer.ExportToString().encode()).hexdigest()
            identifier = f"anonymous:{layer.GetDisplayName()}" if layer.anonymous else layer.identifier
        layers.append({"identifier": identifier, "sha256": digest})
    layers.sort(key=lambda item: item["identifier"])
    options = {
        "threshold": 0.05,
        "mcts_nodes": 20,
        "mcts_iterations": 5,
        "mcts_max_depth": 1,
        "merge": False,
        "seed": 0,
    }
    provenance = {
        "format": "teapot-prepared-v2",
        "source_path": str(source),
        "source_layers": layers,
        "newton_version": version("newton"),
        "coacd_version": version("coacd"),
        "options": options,
        "default_max_hull_vertices": newton.Mesh.MAX_HULL_VERTICES,
    }
    key = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    directory = Path.home() / ".cache/isaaclab/rizon-sharpa-teapot/teapot-prepared-v2"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{key}.usda"
    # A shared lock prevents two first launches from freezing different CoACD
    # realizations under the same source/configuration key.
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
                for ordinal, index in enumerate(parts):
                    mesh = builder.shape_source[index]
                    vertices = np.asarray(mesh.vertices, dtype=np.float64) * np.asarray(builder.shape_scale[index])
                    transform = np.asarray(builder.shape_transform[index], dtype=np.float64)
                    position, quaternion = transform[:3], transform[3:]
                    # shape_transform is body-local. Never apply body_q here:
                    # the referenced source root already carries its own pose.
                    rotated = vertices + 2.0 * np.cross(
                        quaternion[:3], np.cross(quaternion[:3], vertices) + quaternion[3] * vertices
                    )
                    points = np.asarray((rotated + position) / units, dtype=np.float32)
                    faces = np.asarray(mesh.indices, dtype=np.int32).reshape(-1, 3)
                    prim = UsdGeom.Mesh.Define(stage, group.AppendChild(f"part_{ordinal:04d}"))
                    prim.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
                    prim.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32)))
                    prim.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1)))
                    prim.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
                    UsdPhysics.CollisionAPI.Apply(prim.GetPrim()).CreateCollisionEnabledAttr(True)
                    UsdPhysics.MeshCollisionAPI.Apply(prim.GetPrim()).CreateApproximationAttr(
                        UsdPhysics.Tokens.convexHull
                    )
                stage.GetRootLayer().customLayerData = {
                    "isaaclabPreparedContacts": json.dumps({**provenance, "hull_count": len(parts)}, sort_keys=True)
                }
                stage.GetRootLayer().Save()
                prepared.replace(destination)
    return destination


@lru_cache(maxsize=4)
def _prepared_robot_path(source: Path) -> Path:
    """Reference the licensed robot with repaired normals, inertias, and joint-housing filtering.

    Two meshes in the pinned bundle have face-varying normal arrays that do not
    match their corner counts. Newton's USD reader rejects them. Blocking only
    those normals lets the importer recompute them from the original geometry.
    A 2e-10 kg*m^2 inertia floor on fixed fingertip frames avoids Newton's much
    larger automatic correction. The thumb CMC's two-axis joint has an intermediate
    body without collision geometry; exclude its adjoining palm and metacarpal
    housings while retaining distal thumb and cross-finger collisions. The overlay
    leaves the licensed bundle intact.
    """
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
    """Check the exact bundle used by the cable task without copying licensed assets."""
    texture_names = [
        "t_Rizon4s_EmissiveMask.1001.png",
        "t_Rizon4s_EmissiveMask.1002.png",
        *[f"t_Rizon4s_{channel}.{tile}.png" for channel in ("alb", "nor", "orm") for tile in (1001, 1002, 1003)],
        "t_Sharpa_alb.png",
        "t_Sharpa_nor.png",
        "t_Sharpa_orm.png",
    ]
    digest = hashlib.sha256()
    for relative_path in [ASSET_USD_NAME, *[f"textures/{name}" for name in texture_names]]:
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing Rizon/Sharpa asset: {path}. Set {ASSET_ROOT_ENV} to the licensed bundle directory "
                f"containing {ASSET_USD_NAME} and textures/. See the Rizon--Sharpa prerequisite in "
                "docs/source/setup/demos.rst."
            )
        with path.open("rb") as stream:
            file_digest = hashlib.file_digest(stream, "sha256").hexdigest()
        digest.update(f"{file_digest}  {relative_path}\n".encode())
    if digest.hexdigest() != ASSET_BUNDLE_SHA256:
        raise ValueError(f"Rizon/Sharpa asset bundle at {root} differs from revision {ASSET_REVISION}.")


class _TeapotGeometry:
    """Use convex contact geometry in MJWarp and the hollow shell in MPM."""

    _one_way = False

    def _step_entry(self, entry, control, contacts, dt, **kwargs):
        """Correct particle contact after the single-substep fluid entry.

        Newton's native correction prevents drift through thin triangle shells.
        Applying it before gathering retains the fluid entry's zero-copy state
        and captures the pass in the same CUDA graph as the coupled solve.
        """
        result = super()._step_entry(entry, control, contacts, dt, **kwargs)
        if entry.name == "fluid":
            entry.solver.project_outside(entry.state_1, entry.state_1, dt)
        return result

    def _apply_entry_shape_visibility(self, view, cfg, proxy_body_keep):
        visible = self._entry_visible_shapes(cfg, set(cfg.bodies) | proxy_body_keep)
        if cfg.name == "robot":
            # MJWarp's native CCD requires zero margins and otherwise warns while
            # discarding them. Keep the fluid view's thin-shell margins intact.
            view.shape_margin = wp.zeros_like(view.shape_margin)
        super()._apply_entry_shape_visibility(view, replace(cfg, shapes=sorted(visible)), proxy_body_keep)
        if cfg.name == "fluid" and self._one_way:
            # A view-local flag keeps the physical pot dynamic in MJWarp while
            # treating its measured motion as a prescribed boundary in MPM.
            flags = view.body_flags.numpy()
            pot = self.model.body_label.index("/World/envs/env_0/PourContainer")
            flags[pot] |= int(BodyFlags.KINEMATIC)
            view.body_flags = wp.array(flags, dtype=view.body_flags.dtype, device=self.model.device)

    def _entry_visible_shapes(self, cfg, visible_bodies):
        visible = super()._entry_visible_shapes(cfg, visible_bodies)
        if cfg.name == "robot":
            visible = {i for i in visible if "/PourContainer/geometry/" not in self.model.shape_label[i]}
            # Static geometry can be visible to both solvers without duplicate ownership.
            visible.update(
                i
                for i, label in enumerate(self.model.shape_label)
                if "/TabletopCollider/" in label or label.startswith("/World/Ground/")
            )
        else:
            visible.update(i for i, label in enumerate(self.model.shape_label) if "/PourContainer/geometry/" in label)
            visible = {
                i
                for i in visible
                if "/PourContainer/" not in self.model.shape_label[i]
                or "/PourContainer/geometry/" in self.model.shape_label[i]
            }
        return visible


class _TeapotProxySolver(_TeapotGeometry, SolverCoupledProxy):
    """Exchange reactions through a dynamic MPM proxy."""


class _TeapotColliderSyncSolver(_TeapotGeometry, SolverCoupled):
    """Advance rigid contacts, then pass the measured pot boundary to MPM."""

    _one_way = True

    def __init__(self, model, entries):
        super().__init__(model=model, entries=entries)
        pot = model.body_label.index("/World/envs/env_0/PourContainer")
        self._source_pot = int(self._entries["robot"].body_global_to_local.numpy()[pot])
        self._fluid_pot = int(self._entries["fluid"].body_global_to_local.numpy()[pot])
        rigid = self._entries["robot"]
        self._global_substeps = NewtonManager._num_substeps
        self._control_phase = 0
        self._interpolate_control = rigid.substeps > 1 or self._global_substeps > 1
        if self._interpolate_control:
            self._rigid_step = replace(rigid, substeps=1)
            q_start = rigid.view.joint_target_q_start.numpy()
            qd_start = rigid.view.joint_qd_start.numpy()
            arm = [i for i, label in enumerate(rigid.view.joint_label) if label.rsplit("/", 1)[-1] in ARM_JOINT_NAMES]
            if len(arm) != len(ARM_JOINT_NAMES):
                raise RuntimeError("Cannot resolve the seven arm control targets in the rigid model view.")
            self._arm_targets = wp.array([(q_start[i], qd_start[i]) for i in arm], dtype=wp.vec2i, device=model.device)

    def reset(self, *args, **kwargs):
        self._control_phase = 0
        return super().reset(*args, **kwargs)

    def _step_entry(self, entry, control, contacts, dt, *, filter_contacts=True, control_callback=None):
        if entry.name != "robot" or not self._interpolate_control:
            return super()._step_entry(
                entry, control, contacts, dt, filter_contacts=filter_contacts, control_callback=control_callback
            )
        # Reuse the native entry's scratch buffers and control mapping, preserving
        # state_0 for collider differencing and leaving the final state in state_1.
        entry.state_1.assign(entry.state_0)
        if entry.state_tmp is not None:
            entry.state_tmp.assign(entry.state_0)
        single = self._rigid_step
        state = entry.state_0
        for substep in range(entry.substeps):
            single.state_0 = state
            single.state_1 = entry.state_1 if (entry.substeps - substep) % 2 else entry.state_tmp
            if state is not entry.state_0:
                state.body_f.assign(entry.state_0.body_f)
            offset = dt * (self._control_phase + (substep + 0.5) / entry.substeps)

            def interpolate(local_control):
                if control_callback is not None:
                    control_callback(local_control)
                wp.launch(
                    _interpolate_arm_control,
                    dim=len(ARM_JOINT_NAMES),
                    inputs=[self._arm_targets, local_control.joint_target_q, local_control.joint_target_qd, offset],
                    device=self.model.device,
                )

            contacts = super()._step_entry(
                single,
                control,
                contacts,
                dt / entry.substeps,
                filter_contacts=filter_contacts and substep == 0,
                control_callback=interpolate,
            )
            state = single.state_1
        return contacts

    def _step_coupled(self, state_in, state_out, control, contacts, dt):
        rigid, fluid = self._entries["robot"], self._entries["fluid"]
        self._step_entry(rigid, control, contacts, dt)
        # MPM's forward boundary starts at the measured beginning pose. The
        # interval twist reproduces the actual end pose, including rigid substeps.
        wp.launch(
            _sync_teapot_collider,
            dim=1,
            inputs=[
                rigid.state_0.body_q,
                rigid.state_1.body_q,
                rigid.view.body_com,
                self._source_pot,
                dt,
                fluid.state_0.body_q,
                fluid.state_0.body_qd,
                self._fluid_pot,
            ],
            device=self.model.device,
        )
        self._notify_input_state_update(fluid, StateFlags.BODY_Q | StateFlags.BODY_QD, dt=dt)
        self._step_entry(fluid, control, contacts, dt)
        self._control_phase = (self._control_phase + 1) % self._global_substeps


class NewtonTeapotCouplerManager(NewtonCouplerManager):
    """Couple the dynamic grasped pot to its exact MPM collision shell."""

    @classmethod
    def _prepare_builder_for_finalize(cls, builder: ModelBuilder) -> None:
        """Keep generated rigid hulls above the exact shell's body-local base plane."""
        for body, label in enumerate(builder.body_label):
            if not label.endswith("/PourContainer"):
                continue
            shell = next(i for i, name in enumerate(builder.shape_label) if name.startswith(label + "/geometry/"))
            transform = builder.shape_transform[shell]
            points = Rotation.from_quat(transform.q).apply(
                builder.shape_source[shell].vertices * np.asarray(builder.shape_scale[shell])
            ) + np.asarray(transform.p)
            base_z = float(points[:, 2].min())
            for shape, name in enumerate(builder.shape_label):
                if (
                    builder.shape_body[shape] != body
                    or not name.startswith(label + "/RigidCollider/")
                    or builder.shape_type[shape] != GeoType.CONVEX_MESH
                ):
                    continue
                mesh = builder.shape_source[shape]
                transform = builder.shape_transform[shape]
                rotation = Rotation.from_quat(transform.q)
                scale = np.asarray(builder.shape_scale[shape])
                points = rotation.apply(mesh.vertices * scale) + np.asarray(transform.p)
                below = points[:, 2] < base_z
                if not below.any():
                    continue
                # CoACD's voxel preprocessing can extend hulls below a flat base,
                # producing a first-step impulse and an uneven support footprint.
                points[below, 2] = base_z
                vertices = mesh.vertices.copy()
                vertices[below] = rotation.inv().apply(points[below] - np.asarray(transform.p)) / scale
                hull = ConvexHull(vertices)
                faces = hull.simplices.copy()
                normals = np.cross(
                    vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]]
                )
                reverse = np.einsum("ij,ij->i", normals, hull.equations[:, :3]) < 0.0
                faces[reverse] = faces[reverse][:, (0, 2, 1)]
                # Copying geometry rebuilds mesh caches while preserving authored
                # mass properties; the exact shell and handle vertices are untouched.
                builder.shape_source[shape] = mesh.copy(vertices=vertices, indices=faces)
        # MPM preparation converts convex parts to triangle meshes; repair their
        # generated support geometry before that classification is lost.
        super()._prepare_builder_for_finalize(builder)

    @classmethod
    def _build_proxy_coupled_solver(cls, model, entries, proxy_cfgs, solver_cfg):
        if not proxy_cfgs:
            return _TeapotColliderSyncSolver(model=model, entries=entries)
        coupling = SolverCoupledProxy.Config(
            proxies=[SolverCoupledProxy.Proxy(**vars(cfg)) for cfg in proxy_cfgs], iterations=solver_cfg.iterations
        )
        return _TeapotProxySolver(model=model, entries=entries, coupling=coupling)


class TeapotPourMotion:
    """Drive a simulated arm along an IK path with a contact-acquired grasp.

    Startup IK supplies the approach; bounded differential IK uses the measured
    object pose to compensate for contact compliance during pickup and pouring.
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
        fluid_coupling: str = "one_way",
        grasp_finger: str = "middle",
        controller_hz: float = 100.0,
        interpolate_arm_targets: bool = False,
    ) -> None:
        import mujoco
        from scipy.optimize import least_squares

        if not 0.0 <= close_interval[0] < close_interval[1] < grasp_time - 0.3 < grasp_time < duration:
            raise ValueError("Closure and squeeze must finish before pickup and the end of the sequence.")
        if not math.isfinite(sim_dt) or sim_dt <= 0.0 or not math.isfinite(controller_hz) or controller_hz <= 0.0:
            raise ValueError("Simulation timestep and controller rate must be positive and finite.")
        decimation = 1.0 / (sim_dt * controller_hz)
        if decimation < 1.0 or not math.isclose(decimation, round(decimation), rel_tol=1.0e-9):
            raise ValueError("The controller period must be an integer multiple of the simulation timestep.")
        profile = _GRASP_PROFILES[grasp_finger]
        self._grasp_finger = grasp_finger
        self._robot = robot
        self._container = container
        self._pose_at_time = pose_at_time
        self._grasp_time = grasp_time
        self._pickup_end_time = pickup_end_time
        self._gain_start, self._gain_end = close_interval[1] + 0.1, grasp_time - 0.3
        self._gain_settled = False
        self._grasped = False
        self._path_hz = controller_hz
        self._arm_target_lead = 0.0 if interpolate_arm_targets else 1.0
        self._physics_hz = 1.0 / sim_dt
        self._control_decimation = round(decimation)
        self.metrics = {
            "max_position_error_m": 0.0,
            "max_orientation_error_deg": 0.0,
            "max_pouring_orientation_error_deg": 0.0,
            "max_grasp_palm_travel_m": 0.0,
            "max_handle_entry_drift_m": 0.0,
            "max_pregrasp_pot_displacement_m": 0.0,
            "max_hand_pot_penetration_m": 0.0,
            "max_hand_self_penetration_m": 0.0,
            "grasp_model": f"physical_single_{grasp_finger}_finger_contact",
            "fluid_coupling": fluid_coupling,
            "proxy_mass_scale": 1000.0 if fluid_coupling == "two_way" else None,
            "object_feedback_hz": self._path_hz,
        }
        self._palm = robot.body_names.index(PALM_BODY_NAME)
        self._palm_jacobian = self._palm - int(robot.is_fixed_base)
        joint_ids = robot.find_joints(list(ARM_JOINT_NAMES + HAND_JOINT_NAMES), preserve_order=True)[0]
        self._arm_jacobian_dofs = [joint + robot.num_base_dofs for joint in joint_ids[:7]]
        self._palm_in_pot = profile.palm_in_pot
        # Startup IK and contact diagnostics need MJWarp's native model.
        # Runtime feedback reads the public asset state and Jacobian below.
        self._rigid_solver = NewtonManager._solver.solver("robot")
        mj = self._rigid_solver.mj_model
        self.metrics["attachment_constraints"] = mj.neq
        geom_roles = []
        for body in mj.geom_bodyid:
            name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, int(body)) or ""
            geom_roles.append(2 if name == "_World_envs_env_0_PourContainer" else int("_Robot_right_" in name))
        self._geom_roles = wp.array(geom_roles, dtype=int, device=robot.device)
        self._contact_penetration = wp.zeros(2, dtype=float, device=robot.device)
        self._contact_counts = wp.zeros(3, dtype=int, device=robot.device)
        self._peak_joint_speed = wp.zeros(29, dtype=float, device=robot.device)
        self._peak_joint_effort = wp.zeros(29, dtype=float, device=robot.device)
        names = [mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, i).split("_Physics_")[-1] for i in range(mj.njnt)]
        joints = [names.index(name) for name in ARM_JOINT_NAMES + HAND_JOINT_NAMES]
        arm_q = mj.jnt_qposadr[joints[:7]]
        self._dof_indices = wp.array(mj.jnt_dofadr[joints], dtype=int, device=robot.device)
        self._qmin, self._qmax = mj.jnt_range[joints[:7], 0] + 0.03, mj.jnt_range[joints[:7], 1] - 0.03
        self._speed_limits = robot.data.joint_vel_limits.warp.numpy()[0, joint_ids].astype(np.float64)
        self._effort_limits = robot.data.joint_effort_limits.warp.numpy()[0, joint_ids].astype(np.float64)
        data = mujoco.MjData(mj)
        mj_palm = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, "_World_envs_env_0_Robot_" + PALM_BODY_NAME)
        mj_container = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, "_World_envs_env_0_PourContainer")
        self.metrics.update(
            teapot_mass_kg=float(mj.body_mass[mj_container]),
            teapot_diagonal_inertia_kg_m2=mj.body_inertia[mj_container].tolist(),
        )
        self._finger_bodies = {
            digit: [robot.body_names.index(f"right_{digit}_{link}") for link in ("PP", "MP", "DP", "fingertip")]
            for digit in ("index", "middle", "ring", "pinky")
        }
        palm_rotation = Rotation.from_quat(profile.palm_in_pot[1])
        previous = np.array(ARM_HOME_POSITIONS)
        times = np.arange(round(grasp_time * self._path_hz) + 1) / self._path_hz
        trajectory = []
        max_error = 0.0
        for time in times:
            position, quaternion, _ = pose_at_time(float(time))
            rotation = Rotation.from_quat(quaternion)
            target_position = np.asarray(position) + rotation.apply(profile.palm_in_pot[0])
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
            max_error = max(error, max_error)
            if error > 2.0e-4:
                raise RuntimeError(f"Teapot approach is unreachable at {time:.2f}s (IK error {error:.4g}).")
            previous = solution.x
            alpha = self._blend((time - close_interval[0]) / (close_interval[1] - close_interval[0]))
            squeeze = self._blend((time - close_interval[1]) / (self._gain_end - close_interval[1]))
            hand = (
                np.asarray(profile.open_positions)
                + alpha * (np.asarray(profile.grasp_positions) - profile.open_positions)
                + squeeze * (np.asarray(profile.hold_positions) - profile.grasp_positions)
            )
            thumb_alpha = self._blend(
                (time - close_interval[0]) / (profile.thumb_close_fraction * (close_interval[1] - close_interval[0]))
            )
            hand[:5] = (
                np.asarray(profile.open_positions[:5])
                + thumb_alpha * (np.asarray(profile.grasp_positions[:5]) - profile.open_positions[:5])
                + squeeze * (np.asarray(profile.hold_positions[:5]) - profile.grasp_positions[:5])
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
        self._arm_position_gpu, self._arm_velocity_gpu = (
            wp.zeros(7, device=robot.device),
            wp.zeros(7, device=robot.device),
        )
        self._last_feedback_step = None
        self._next_feedback_step = round(grasp_time * self._physics_hz)
        initial = robot.data.default_joint_pos.warp.numpy()
        initial[0, joint_ids] = trajectory[0]
        self.initial_joint_positions = wp.array(initial, dtype=float, device=robot.device)
        logger.info(
            "Teapot approach IK maximum error %.3g; peak arm speed %.3f rad/s;"
            " peak finger speed %.3f rad/s; contact-only object feedback at %g Hz.",
            max_error,
            speeds[:7].max(),
            speeds[7:].max(),
            self._path_hz,
        )
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
        if not self._grasped and sim_time >= self._grasp_time:
            self._confirm_grasp(sim_time)
        if self._grasped and step >= self._next_feedback_step:
            self._update_feedback(sim_time, step)
        # Substepped arm targets start at the outer boundary; the coupler samples
        # their velocity at every rigid step instead of holding an endpoint target.
        feedback_elapsed = (
            (step - self._last_feedback_step + self._arm_target_lead) / self._physics_hz if self._grasped else 0.0
        )
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
                self._grasped,
                feedback_elapsed,
                self._robot.actuators.target_command.position,
                self._robot.actuators.target_command.velocity,
            ],
            device=self._robot.device,
        )

    def _update_feedback(self, sim_time: float, step: int) -> None:
        """Solve bounded differential IK at the actual pot root, driving only arm joints.

        Finger contacts supply every force on the pot. Object feedback compensates
        for finger compliance and pivoting in the handle. The velocity solve respects
        rated speeds, an 8 rad/s^2 acceleration envelope, and joint stopping distances.
        """
        from scipy.optimize import lsq_linear

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
        feedback_weight = self._blend((sim_time - self._grasp_time) / 2.0)
        rotation_gain = (8.0 + 4.0 * pickup_weight) * feedback_weight
        command = np.asarray(twist) + error * np.array((10.0, 10.0, 10.0, rotation_gain, rotation_gain, rotation_gain))
        command[:3], command[3:] = np.clip(command[:3], -0.45, 0.45), np.clip(command[3:], -1.5, 1.5)
        palm = self._robot.data.body_link_pose_w.warp.numpy()[0, self._palm, :3]
        jacobian = self._robot.data.body_link_jacobian_w.warp.numpy()[0, self._palm_jacobian][
            :, self._arm_jacobian_dofs
        ].astype(np.float64)
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
        period, acceleration = 1.0 / self._path_hz, 8.0
        half_increment = 0.5 * acceleration * period
        brake_lower = half_increment - np.sqrt(2.0 * acceleration * np.maximum(q - self._qmin, 0.0) + half_increment**2)
        brake_upper = np.sqrt(2.0 * acceleration * np.maximum(self._qmax - q, 0.0) + half_increment**2) - half_increment
        lower = np.maximum.reduce((-self._speed_limits[:7], brake_lower, self._arm_velocity - acceleration * period))
        upper = np.minimum.reduce((self._speed_limits[:7], brake_upper, self._arm_velocity + acceleration * period))
        if np.any(lower > upper + 1.0e-7):
            raise RuntimeError(f"Infeasible arm braking envelope at {sim_time:.3f}s.")
        matrix = np.vstack((jacobian, 0.005 * np.eye(7)))
        target = np.concatenate((command, 0.005 * posture))
        # A joint at its stopping boundary has one feasible velocity. Eliminate
        # those coordinates because scipy's box solver requires strict bounds.
        fixed = upper - lower < 1.0e-8
        free = ~fixed
        self._arm_velocity = np.zeros(7)
        self._arm_velocity[fixed] = 0.5 * (lower[fixed] + upper[fixed])
        if np.any(free):
            solution = lsq_linear(
                matrix[:, free],
                target - matrix[:, fixed] @ self._arm_velocity[fixed],
                bounds=(lower[free], upper[free]),
                method="bvls",
                tol=1.0e-9,
                max_iter=50,
            )
            if not solution.success or not np.isfinite(solution.x).all():
                raise RuntimeError(f"Arm velocity solve failed at {sim_time:.3f}s: {solution.message}")
            self._arm_velocity[free] = solution.x
        self._arm_position_gpu.assign(self._arm_position.astype(np.float32))
        self._arm_velocity_gpu.assign(self._arm_velocity.astype(np.float32))
        self._last_feedback_step = step
        self._next_feedback_step = step + self._control_decimation

    def _handle_entries(self) -> dict[str, np.ndarray]:
        """Find finger centerlines crossing the stock handle's opening plane.

        This geometric check complements actual collision forces and penetration:
        contact on the outside of the loop alone does not demonstrate insertion.
        """
        poses = self._robot.data.body_link_pose_w.warp.numpy()[0]
        pot = self._container.data.root_link_pose_w.warp.numpy()[0]
        rotation = Rotation.from_quat(pot[3:])
        result = {}
        for digit, bodies in self._finger_bodies.items():
            points = rotation.inv().apply(poses[bodies, :3].astype(np.float64) - pot[:3])
            for first, second in zip(points[:-1], points[1:], strict=True):
                if first[1] * second[1] <= 0.0 and abs(second[1] - first[1]) > 1.0e-6:
                    crossing = first - first[1] * (second - first) / (second[1] - first[1])
                    if -0.090 < crossing[0] < -0.062 and 0.037 < crossing[2] < 0.070:
                        result[digit] = crossing
        return result

    def _confirm_grasp(self, sim_time: float) -> None:
        """Confirm one inserted finger and opposing physical contact without changing state."""
        palm = self._robot.data.body_link_pose_w.warp.numpy()[0, self._palm]
        pot = self._container.data.root_link_pose_w.warp.numpy()[0]
        pot_rotation, palm_rotation = Rotation.from_quat(pot[3:]), Rotation.from_quat(palm[3:])
        target_position = pot[:3] + pot_rotation.apply(self._palm_in_pot[0])
        target_rotation = pot_rotation * Rotation.from_quat(self._palm_in_pot[1])
        position_error = float(np.linalg.norm(palm[:3] - target_position))
        rotation_error = math.degrees(float((palm_rotation * target_rotation.inv()).magnitude()))
        contacts = self._hand_contacts()
        thumb = any("right_thumb" in name and force > 0.05 for name, _, force in contacts)
        hook = any(f"right_{self._grasp_finger}" in name and force > 0.05 for name, _, force in contacts)
        entries = self._handle_entries()
        if (
            position_error > 0.020
            or rotation_error > 5.0
            or not (thumb and hook)
            or set(entries) != {self._grasp_finger}
        ):
            raise RuntimeError(
                f"Hand is not ready to grasp at {sim_time:.3f}s: alignment {position_error:.4f} m /"
                f" {rotation_error:.2f} deg, thumb/{self._grasp_finger} contacts {thumb}/{hook},"
                f" inserted fingers {list(entries)}."
            )
        self._palm_in_pot = (
            pot_rotation.inv().apply(palm[:3] - pot[:3]),
            (pot_rotation.inv() * palm_rotation).as_quat(),
        )
        self._handle_entry = entries[self._grasp_finger]
        self._grasped = True
        self.metrics.update(
            grasp_time_s=sim_time,
            grasp_position_error_m=position_error,
            grasp_orientation_error_deg=rotation_error,
            handle_fingers=[self._grasp_finger],
            handle_entry_m=self._handle_entry.tolist(),
            grasp_contact_bodies=sorted({name for name, _, _ in contacts}),
            grasp_contacts=[
                {"body": name, "distance_m": distance, "normal_force_n": force} for name, distance, force in contacts
            ],
        )
        logger.info(
            "Single-finger contact grasp at %.2fs; alignment %.2f mm;"
            " %d contacts; the teapot remains dynamic without an attachment.",
            sim_time,
            position_error * 1000,
            len(contacts),
        )

    def _hand_contacts(self) -> list[tuple[str, float, float]]:
        """Read MJWarp's real hand--pot contacts for acquisition and validation."""
        import mujoco

        solver = self._rigid_solver
        mj = solver.mj_model
        data = solver.mjw_data
        count = int(data.nacon.numpy()[0])
        broadphase_count = int(data.ncollision.numpy()[0])
        constraint_count = int(data.nefc.numpy()[0])
        if max(count, broadphase_count) > data.naconmax or constraint_count > data.njmax:
            raise RuntimeError("MJWarp contact/constraint capacity overflow; increase nconmax/njmax.")
        self.metrics["max_broadphase_pairs"] = max(self.metrics.get("max_broadphase_pairs", 0), broadphase_count)
        self.metrics["max_contact_points"] = max(self.metrics.get("max_contact_points", 0), count)
        self.metrics["max_constraint_rows"] = max(self.metrics.get("max_constraint_rows", 0), constraint_count)
        pairs = data.contact.geom.numpy()[:count]
        distances = data.contact.dist.numpy()[:count]
        addresses = data.contact.efc_address.numpy()[:count, 0]
        forces = data.efc.force.numpy()[0]
        result = []
        for pair, distance, address in zip(pairs, distances, addresses, strict=True):
            if min(pair) < 0 or distance > 0.0005:
                continue
            bodies = mj.geom_bodyid[pair]
            names = [mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, int(body)) for body in bodies]
            if "_World_envs_env_0_PourContainer" not in names:
                continue
            for name in names:
                if "_Robot_right_" in name:
                    result.append((name, float(distance), float(forces[address]) if address >= 0 else 0.0))
        return result

    def observe_contacts(self) -> None:
        """Accumulate contact penetration and buffer usage at each outer-step boundary.

        Called only during validation; benchmark playback avoids the observer.
        Intermediate rigid substeps overwrite these native buffers, so their
        peaks are not included when coupled or rigid substeps exceed one.
        """
        data = self._rigid_solver.mjw_data
        wp.launch(
            _record_contacts,
            dim=data.naconmax,
            inputs=[
                data.nacon,
                data.ncollision,
                data.nefc,
                data.contact.geom,
                data.contact.dist,
                self._geom_roles,
                self._contact_penetration,
                self._contact_counts,
                data.qvel,
                data.qfrc_actuator,
                self._dof_indices,
                self._peak_joint_speed,
                self._peak_joint_effort,
            ],
            device=self._robot.device,
        )

    def check(self, sim_time: float, pose_at_time: Callable[[float], _ContainerPose]) -> None:
        """Validate tracking, a single hooked finger, opposing contact, and actuator limits."""
        poses = self._robot.data.body_link_pose_w.warp.numpy()[0]
        actual = self._container.data.root_link_pose_w.warp.numpy()[0]
        if (
            not np.isfinite(poses).all()
            or not np.isfinite(actual).all()
            or not np.isfinite(self._robot.data.joint_vel.warp.numpy()).all()
            or not np.isfinite(self._container.data.root_link_vel_w.warp.numpy()).all()
        ):
            raise RuntimeError(f"Nonfinite robot state at {sim_time:.3f}s.")
        expected_position, expected_quat, _ = pose_at_time(sim_time)
        position_error = float(np.linalg.norm(actual[:3] - expected_position))
        orientation_error = math.degrees(
            float((Rotation.from_quat(actual[3:]) * Rotation.from_quat(expected_quat).inv()).magnitude())
        )
        palm = poses[self._palm]
        expected_palm = actual[:3] + Rotation.from_quat(actual[3:]).apply(self._palm_in_pot[0])
        palm_travel = float(np.linalg.norm(palm[:3] - expected_palm)) if self._grasped else 0.0
        if not self._grasped:
            self.metrics["max_pregrasp_pot_displacement_m"] = max(
                self.metrics["max_pregrasp_pot_displacement_m"], position_error
            )
            if position_error > 0.008:
                raise RuntimeError(f"The teapot moved before the grasp at {sim_time:.3f}s.")
        else:
            entries = self._handle_entries()
            contacts = self._hand_contacts()
            opposing = all(
                any(f"right_{digit}" in name and force > 0.05 for name, _, force in contacts)
                for digit in (self._grasp_finger, "thumb")
            )
            if set(entries) != {self._grasp_finger} or not opposing:
                raise RuntimeError(f"Lost the single-finger contact grasp at {sim_time:.3f}s: {list(entries)}.")
            drift = float(np.linalg.norm(entries[self._grasp_finger] - self._handle_entry))
            self.metrics["max_handle_entry_drift_m"] = max(self.metrics["max_handle_entry_drift_m"], drift)
            if drift > 0.015:
                raise RuntimeError(f"Finger moved {drift * 1000:.2f} mm in the handle at {sim_time:.3f}s.")
        penetration, self_penetration = self._contact_penetration.numpy()
        penetration, self_penetration = float(penetration), float(self_penetration)
        self.metrics["max_hand_pot_penetration_m"] = penetration
        self.metrics["max_hand_self_penetration_m"] = self_penetration
        counts = self._contact_counts.numpy()
        data = self._rigid_solver.mjw_data
        if max(counts[:2]) > data.naconmax or counts[2] > data.njmax:
            raise RuntimeError("MJWarp contact/constraint capacity overflow during the sequence.")
        self.metrics.update(
            max_contact_points=int(counts[0]),
            max_broadphase_pairs=int(counts[1]),
            max_constraint_rows=int(counts[2]),
            contact_observation_hz=self._physics_hz,
            contact_observation_phase="outer_step_end",
        )
        if penetration > 0.002:
            raise RuntimeError(f"Hand--pot penetration {penetration * 1000:.2f} mm at {sim_time:.3f}s.")
        if self_penetration > 0.002:
            raise RuntimeError(f"Hand self-penetration {self_penetration * 1000:.2f} mm at {sim_time:.3f}s.")
        speeds, efforts = self._peak_joint_speed.numpy(), self._peak_joint_effort.numpy()
        self.metrics.update(peak_joint_speed_rad_s=speeds.tolist(), peak_joint_effort_nm=efforts.tolist())
        if np.any(efforts > self._effort_limits + 1.0e-3) or np.any(speeds > 1.1 * self._speed_limits):
            raise RuntimeError(f"Joint effort or speed envelope exceeded at {sim_time:.3f}s: {efforts}, {speeds}.")
        for key, value in zip(
            ("max_position_error_m", "max_orientation_error_deg", "max_grasp_palm_travel_m"),
            (position_error, orientation_error, palm_travel),
            strict=True,
        ):
            self.metrics[key] = max(self.metrics[key], value)
        if sim_time >= self._pickup_end_time:
            self.metrics["max_pouring_orientation_error_deg"] = max(
                self.metrics["max_pouring_orientation_error_deg"], orientation_error
            )
        # The palm can pivot about the hooked finger as the table unloads;
        # validate insertion/contact rather than imposing a rigid palm--pot transform.
        orientation_limit = 10.0 if sim_time < self._pickup_end_time else 5.0
        if position_error > 0.015 or orientation_error > orientation_limit:
            raise RuntimeError(
                f"Teapot tracking failed at {sim_time:.3f}s: position {position_error:.4f} m,"
                f" orientation {orientation_error:.2f} deg."
            )


@wp.kernel
def _interpolate_arm_control(
    indices: wp.array[wp.vec2i],
    position: wp.array[float],
    velocity: wp.array[float],
    offset: float,
):
    index = indices[wp.tid()]
    position[index[0]] += offset * velocity[index[1]]


@wp.kernel
def _sync_teapot_collider(
    source_start: wp.array[wp.transform],
    source_end: wp.array[wp.transform],
    source_com: wp.array[wp.vec3],
    source: int,
    dt: float,
    fluid_pose: wp.array[wp.transform],
    fluid_velocity: wp.array[wp.spatial_vector],
    destination: int,
):
    start = source_start[source]
    end = source_end[source]
    linear = (wp.transform_point(end, source_com[source]) - wp.transform_point(start, source_com[source])) / dt
    rotation = wp.normalize(wp.transform_get_rotation(end) * wp.quat_inverse(wp.transform_get_rotation(start)))
    axis, angle = wp.quat_to_axis_angle(rotation)
    fluid_pose[destination] = start
    fluid_velocity[destination] = wp.spatial_vector(linear, axis * (angle / dt))


@wp.kernel
def _record_contacts(
    nacon: wp.array[int],
    ncollision: wp.array[int],
    nefc: wp.array[int],
    pairs: wp.array[wp.vec2i],
    distances: wp.array[float],
    roles: wp.array[int],
    penetration: wp.array[float],
    counts: wp.array[int],
    joint_velocities: wp.array2d[float],
    joint_efforts: wp.array2d[float],
    joint_dofs: wp.array[int],
    peak_speed: wp.array[float],
    peak_effort: wp.array[float],
):
    i = wp.tid()
    if i < joint_dofs.shape[0]:
        dof = joint_dofs[i]
        wp.atomic_max(peak_speed, i, wp.abs(joint_velocities[0, dof]))
        wp.atomic_max(peak_effort, i, wp.abs(joint_efforts[0, dof]))
    if i == 0:
        wp.atomic_max(counts, 0, nacon[0])
        wp.atomic_max(counts, 1, ncollision[0])
        wp.atomic_max(counts, 2, nefc[0])
    if i < nacon[0]:
        pair = pairs[i]
        if pair[0] >= 0 and pair[1] >= 0:
            a, b = roles[pair[0]], roles[pair[1]]
            if (a == 1 and b == 2) or (a == 2 and b == 1):
                wp.atomic_max(penetration, 0, wp.max(0.0, -distances[i]))
            elif a == 1 and b == 1:
                wp.atomic_max(penetration, 1, wp.max(0.0, -distances[i]))


@wp.kernel
def _interpolate_joints(
    path: wp.array3d[float],
    joints: wp.array[int],
    sample: float,
    path_dt: float,
    arm_position: wp.array[float],
    arm_velocity: wp.array[float],
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
