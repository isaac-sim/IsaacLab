# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fill and pour water from the Utah teapot with Newton implicit MPM.

The teapot is a hollow, double-walled shell, so the fluid is seeded in its enclosed air cavity with
:func:`~isaaclab.utils.warp.sample_particles_in_cavity` instead of using a winding-number volume fill.

.. code-block:: bash

    # Fast Newton visualizer (the default):
    uvx isaaclab demo teapot-fill --device cuda:0 --visualizer newton_gl
    # Display both the raw particles and reconstructed surface:
    uvx isaaclab demo teapot-fill --visualizer newton_gl --fluid_render_mode both
    # Fuller / coarser (faster) fill:
    uvx isaaclab demo teapot-fill --fill_level 1.0 --fill_spacing 0.003
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable
from dataclasses import MISSING
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from isaaclab_visualizers.newton import NewtonGLBackendCfg

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR, retrieve_file_path
from isaaclab.visualizers import VisualizerCfg

if TYPE_CHECKING:
    from isaaclab_visualizers.newton import ParticleSurfaceRenderer

DEFAULT_VOXEL_SIZE = 0.003
DEFAULT_PARTICLES_PER_VOXEL_AXIS = 2.0
DEFAULT_FILL_LEVEL = 0.70
DEFAULT_MIN_RAY_HITS = 5
DEFAULT_ISLAND_USD = (
    f"{ISAAC_NUCLEUS_DIR}/SimReady/Residential/Kitchen/Counters/Island_A01/sm_fixture_island_a01_01.usd"
)
DEFAULT_BOWL_USD = f"{ISAAC_NUCLEUS_DIR}/SimReady/Residential/Kitchen/Dishware/Bowl_G01/sm_kitchenware_bowl_g01_01.usd"


def _positive_finite_float(value: str) -> float:
    """Parse a positive finite floating-point argument."""
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise argparse.ArgumentTypeError(f"expected a positive finite value, got {value!r}")
    return resolved


def _unit_interval_float(value: str) -> float:
    """Parse a finite floating-point argument in the closed unit interval."""
    resolved = float(value)
    if not math.isfinite(resolved) or not 0.0 <= resolved <= 1.0:
        raise argparse.ArgumentTypeError(f"expected a value in [0, 1], got {value!r}")
    return resolved


parser = argparse.ArgumentParser(description="Newton implicit MPM teapot-fill demo.", conflict_handler="resolve")
parser.add_argument("--robot", choices=["none", "rizon_sharpa"], default="none", help="Use the fixed physical grasp.")
parser.add_argument(
    "--max_steps",
    type=int,
    default=6000,
    help="Stop after this many simulation steps; negative runs forever.",
)
parser.add_argument(
    "--voxel_size",
    type=_positive_finite_float,
    default=DEFAULT_VOXEL_SIZE,
    help=f"MPM grid voxel size in meters. Defaults to {DEFAULT_VOXEL_SIZE:g}.",
)
parser.add_argument(
    "--grid_type",
    type=str,
    default="sparse",
    choices=["fixed", "sparse"],
    help="MPM grid topology. Sparse uses a bounded rebuildable grid over active voxels and is "
    "CUDA-graph-capturable (fastest); fixed pre-allocates a frozen padded grid.",
)
parser.add_argument("--disable_cuda_graph", action="store_true", help="Disable Newton CUDA graph capture.")
parser.add_argument(
    "--fill_spacing",
    type=_positive_finite_float,
    default=None,
    help=(
        "Particle spacing used to sample the cavity volume [m]."
        f" Defaults to voxel_size / {DEFAULT_PARTICLES_PER_VOXEL_AXIS:g}."
    ),
)
parser.add_argument(
    "--fill_level",
    type=_unit_interval_float,
    default=DEFAULT_FILL_LEVEL,
    help=(
        "Water line as a fraction [0, 1] of the teapot height to fill the cavity up to. Use 1.0 to fill to the brim."
    ),
)
parser.add_argument(
    "--min_ray_hits",
    type=int,
    default=DEFAULT_MIN_RAY_HITS,
    choices=range(1, 7),
    help=(
        "Enclosure strictness for cavity detection: how many of the 6 axis rays must hit the shell"
        f" (1-6). Higher drops thin features like the spout/handle. Defaults to {DEFAULT_MIN_RAY_HITS}."
    ),
)
parser.add_argument(
    "--fluid_render_mode",
    type=str,
    default="surface",
    choices=["particles", "surface", "both"],
    help="Fluid visualization: raw MPM particles, reconstructed surface mesh, or both.",
)
parser.add_argument(
    "--container_usd",
    type=str,
    default=f"{ISAAC_NUCLEUS_DIR}/Props/Teapot/utah_teapot.usdc",
    help="USD asset used as the pouring container (rigid collider).",
)
parser.add_argument(
    "--island_usd",
    type=str,
    default=DEFAULT_ISLAND_USD,
    help="SimReady kitchen-island USD used by every visualizer.",
)
parser.add_argument(
    "--bowl_usd",
    type=str,
    default=DEFAULT_BOWL_USD,
    help="SimReady catch-bowl USD used for both visualization and fluid collisions.",
)
add_launcher_args(parser)
parser.add_argument("--visualizer", "--viz", choices=["newton_gl"], default="newton_gl", help="Visualizer backend.")
args_cli = parser.parse_args()
ROBOT_ENABLED = args_cli.robot == "rizon_sharpa"


# Use one 1.25 ms solver step per outer loop iteration and retain the original 400 Hz render cadence.
SIMULATION_HZ = 800
RENDER_INTERVAL = 2
VOXEL_SIZE = args_cli.voxel_size
FILL_SPACING = (
    args_cli.fill_spacing if args_cli.fill_spacing is not None else VOXEL_SIZE / DEFAULT_PARTICLES_PER_VOXEL_AXIS
)
FILL_LEVEL = args_cli.fill_level
MIN_RAY_HITS = args_cli.min_ray_hits
SHOW_FLUID_PARTICLES = args_cli.fluid_render_mode in ("particles", "both")
SHOW_FLUID_SURFACE = args_cli.fluid_render_mode in ("surface", "both")

# Sparse grids reserve capture-stable storage; fixed grids need padding for the full pour trajectory.
GRID_TYPE = args_cli.grid_type
GRID_PADDING = 0 if GRID_TYPE == "sparse" else 64
MAX_ACTIVE_CELL_COUNT = (1 << 16) if GRID_TYPE == "sparse" else (1 << 18)
MPM_SUBSTEPS = 1

PARTICLE_DENSITY = 1000.0
FILL_JITTER = 0.2
FILL_SEED = 7
PARTICLE_RADIUS = 0.5 * FILL_SPACING
PARTICLE_MASS = FILL_SPACING**3 * PARTICLE_DENSITY

COLLIDER_MARGIN = 0.5 * VOXEL_SIZE
PARTICLE_SURFACE_CLEARANCE = COLLIDER_MARGIN + PARTICLE_RADIUS
CONTAINER_FRICTION = 0.0
BOWL_FRICTION = 0.20
TABLE_FRICTION = 0.5

HOLD_TIME = 0.55
TILT_TIME = 2.0
POUR_ANGLE = math.radians(65.0)
CONTAINER_LIFT_HEIGHT = 0.24
CONTAINER_LIFT_TIME = 3.0

TABLE_TOP_Z = 0.90041
TABLE_HALF_EXTENTS = (0.31565, 0.62045, 0.009)
TABLE_ORIENTATION = (0.0, 0.0, -math.sin(0.25 * math.pi), math.cos(0.25 * math.pi))
BOWL_SCALE = 1.0
# Local bounds of the default SimReady Bowl_G01 asset [m].
BOWL_LOCAL_Z_MIN = 0.000001783
BOWL_LOCAL_Z_MAX = 0.043571252
BOWL_BASE_POS = (0.106, 0.0, TABLE_TOP_Z - BOWL_SCALE * BOWL_LOCAL_Z_MIN)
BOWL_WORLD_TOP_Z = BOWL_BASE_POS[2] + BOWL_SCALE * BOWL_LOCAL_Z_MAX
CONTAINER_BASE_POS = (-0.105, 0.0, BOWL_WORLD_TOP_Z + 0.115)

CONTAINER_COLOR = (0.70, 0.35, 0.16)
WATER_COLOR = (0.12, 0.35, 0.78)
WATER_OPACITY = 0.65

CAMERA_EYE = (0.0, -0.72, 1.18)
CAMERA_TARGET = (0.0, 0.0, TABLE_TOP_Z + 0.10)

if ROBOT_ENABLED:
    # The optional robot example lives outside the curated demos.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rizon_sharpa_teapot import BOWL_BASE_POS, HOLD_TIME, ROBOT_BASE_POS, pose_at_time

    BOWL_WORLD_TOP_Z = BOWL_BASE_POS[2] + BOWL_SCALE * BOWL_LOCAL_Z_MAX

    CAMERA_EYE = (0.55, -1.5, 1.72)
    CAMERA_TARGET = (-0.28, 0.0, 1.22)

# Use tighter kernels than the open-tank dam-break preset so the reconstructed
# surface stays inside narrow features such as the teapot spout.
SURFACE_VOXEL_SIZE = 0.5 * VOXEL_SIZE
SURFACE_KERNEL_RADIUS = max(3.0 * FILL_SPACING, 1.5 * SURFACE_VOXEL_SIZE)
SURFACE_MAX_GRID_CELLS = 4_000_000
SURFACE_PATH = "/fluid_surface"


def create_surface_renderer(sim) -> ParticleSurfaceRenderer:
    """Configure teapot-specific surface extraction after Newton initializes."""
    from isaaclab_visualizers.newton import ParticleSurfaceRenderer
    from newton.geometry import ParticleSurface

    model = sim.backend_manager.backend.model
    surface = ParticleSurface(
        voxel_size=SURFACE_VOXEL_SIZE,
        max_grid_cells=SURFACE_MAX_GRID_CELLS,
        world_count=max(model.world_count, 1),
        kernel_radius=SURFACE_KERNEL_RADIUS,
        threshold=0.4,
        smooth_lambda=0.0,
        anisotropic=True,
        kernel_scale=0.5,
        anisotropy_ratio=16.0,
        anisotropy_scale=1.0,
        anisotropy_min_neighbors=4,
        anisotropy_binning=True,
        anisotropy_strength=0.95,
        field_smooth_iterations=0,
        mesh_smooth_iterations=1,
        device=model.device,
    )
    return ParticleSurfaceRenderer(
        sim.visualizers,
        surface,
        path=SURFACE_PATH,
        color=WATER_COLOR,
        opacity=WATER_OPACITY,
        roughness=0.1,
        use_cuda_graph=not args_cli.disable_cuda_graph,
    )


def quat_y(angle_rad: float) -> tuple[float, float, float, float]:
    """Return an XYZW quaternion for a rotation about +Y."""
    half = 0.5 * angle_rad
    return (0.0, math.sin(half), 0.0, math.cos(half))


def container_pose_at_time(
    sim_time: float,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float, float, float],
]:
    """Return teapot ``(position, orientation, twist)`` for the scripted pour."""
    if ROBOT_ENABLED:
        return pose_at_time(sim_time)
    raw = (sim_time - HOLD_TIME) / TILT_TIME
    clamped = max(0.0, min(1.0, raw))
    alpha = clamped * clamped * (3.0 - 2.0 * clamped)
    alpha_dot = (6.0 * clamped * (1.0 - clamped)) / TILT_TIME if 0.0 < raw < 1.0 else 0.0

    lift_raw = (sim_time - HOLD_TIME - TILT_TIME) / CONTAINER_LIFT_TIME
    lift_alpha = max(0.0, min(1.0, lift_raw))
    lift_speed = CONTAINER_LIFT_HEIGHT / CONTAINER_LIFT_TIME if 0.0 < lift_raw < 1.0 else 0.0

    angle = POUR_ANGLE * alpha
    angular_speed = POUR_ANGLE * alpha_dot
    pos = (
        CONTAINER_BASE_POS[0],
        CONTAINER_BASE_POS[1],
        CONTAINER_BASE_POS[2] + CONTAINER_LIFT_HEIGHT * lift_alpha,
    )
    # Newton spatial vectors are (linear, angular).
    twist = (0.0, 0.0, lift_speed, 0.0, angular_speed, 0.0)
    return pos, quat_y(angle), twist


def spawn_demo_mesh(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn an exact triangle mesh with standard Isaac Lab rigid/collision schemas."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    from isaaclab.sim import schemas
    from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg, spawn_physics_material
    from isaaclab.sim.utils import bind_physics_material, bind_visual_material, create_prim, get_current_stage

    stage = get_current_stage()
    vertices = np.asarray(cfg.vertices, dtype=np.float32)
    faces = np.asarray(cfg.faces, dtype=np.int32)

    create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation, stage=stage)
    geom_prim_path = f"{prim_path}/geometry"
    mesh_prim_path = f"{geom_prim_path}/mesh"
    create_prim(geom_prim_path, prim_type="Xform", stage=stage)
    create_prim(
        mesh_prim_path,
        prim_type="Mesh",
        attributes={
            "points": vertices,
            "faceVertexIndices": faces.reshape(-1),
            "faceVertexCounts": np.full(faces.shape[0], 3, dtype=np.int32),
            "subdivisionScheme": "bilinear",
        },
        stage=stage,
    )
    if not cfg.visible:
        UsdGeom.Imageable(stage.GetPrimAtPath(mesh_prim_path)).MakeInvisible()

    def as_fragments(value) -> list:
        # the schema slots accept a bare fragment or a list of fragments
        return list(value) if isinstance(value, (list, tuple)) else [value]

    if cfg.collision_props is not None:
        schemas.apply_collision_properties(
            mesh_prim_path, as_fragments(cfg.collision_props), create_if_missing=True, stage=stage
        )
    if cfg.mesh_collision_props is not None:
        schemas.apply_mesh_collision_properties(mesh_prim_path, as_fragments(cfg.mesh_collision_props), stage=stage)
    if cfg.rigid_props is not None:
        schemas.apply_rigid_body_properties(
            prim_path, as_fragments(cfg.rigid_props), create_if_missing=True, stage=stage
        )
    if cfg.rigid_contact_usd is not None:
        from isaaclab_newton.sim.schemas import MujocoCollisionCfg, NewtonCollisionCfg

        import isaaclab.sim as sim_utils

        # Keep the authored convex decomposition, including the handle opening,
        # on the same body as the exact fluid shell. Only the shell is rendered.
        contact_cfg = sim_utils.UsdFileCfg(
            usd_path=cfg.rigid_contact_usd,
            rigid_props={"(/.*)?": [sim_utils.UsdPhysicsRigidBodyCfg(rigid_body_enabled=False)]},
            collision_props=[
                sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
                MujocoCollisionCfg(condim=4, solref=(0.0025, 1.0)),
                NewtonCollisionCfg(contact_gap=0.001),
            ],
            physics_material=UsdPhysicsRigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            visible=False,
        )
        # The public spawner applies visible=False to the referenced hierarchy;
        # its low-level file helper would leave a duplicate opaque teapot visible.
        contact_root = sim_utils.spawn_from_usd(f"{prim_path}/RigidCollider", contact_cfg)
        mass = UsdPhysics.MassAPI(stage.GetPrimAtPath(prim_path))
        for prim in Usd.PrimRange(contact_root):
            if prim.HasAPI(UsdPhysics.MassAPI):
                original_mass = UsdPhysics.MassAPI(prim)
                for field in ("Mass", "CenterOfMass", "DiagonalInertia", "PrincipalAxes"):
                    value = getattr(original_mass, f"Get{field}Attr")().Get()
                    if value is not None:
                        getattr(mass, f"Create{field}Attr")(value)
                prim.RemoveAPI(UsdPhysics.MassAPI)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)

    if cfg.mass_props is not None:
        mass = UsdPhysics.MassAPI(stage.GetPrimAtPath(prim_path))
        authored_mass = mass.GetMassAttr().Get()
        authored_inertia = mass.GetDiagonalInertiaAttr().Get()
        schemas.apply_mass_properties(prim_path, as_fragments(cfg.mass_props), create_if_missing=True, stage=stage)
        configured_mass = mass.GetMassAttr().Get()
        if (
            authored_mass is not None
            and authored_mass > 0.0
            and authored_inertia is not None
            and configured_mass is not None
            and configured_mass > 0.0
        ):
            # Keep the stock COM and principal axes; a uniform mass change scales
            # its authored inertia rather than leaving mismatched mechanics.
            scale = configured_mass / authored_mass
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*(value * scale for value in authored_inertia)))

    if cfg.visual_material is not None:
        material_path = cfg.visual_material_path
        if not material_path.startswith("/"):
            material_path = f"{geom_prim_path}/{material_path}"
        cfg.visual_material.func(material_path, cfg.visual_material)
        bind_visual_material(mesh_prim_path, material_path, stage=stage)

    if cfg.physics_material is not None:
        material_path = cfg.physics_material_path
        if not material_path.startswith("/"):
            material_path = f"{geom_prim_path}/{material_path}"
        spawn_physics_material(material_path, cfg.physics_material, stage=stage)
        bind_physics_material(mesh_prim_path, material_path, stage=stage)

    return stage.GetPrimAtPath(prim_path)


def load_asset_mesh(usd_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read a USD asset and return its triangulated geometry in the asset's local frame.

    Meshes, including instance proxies, are concatenated, transformed into the asset frame, and
    fan-triangulated so visual assets and their exact MPM colliders align.
    """
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"Could not open USD asset: {usd_path}")

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    vertices_list: list[np.ndarray] = []
    triangles_list: list[np.ndarray] = []
    vertex_offset = 0
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
        face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not points or not face_vertex_counts or not face_vertex_indices:
            continue

        points = np.asarray(points, dtype=np.float64)
        face_vertex_counts = np.asarray(face_vertex_counts, dtype=np.int64)
        face_vertex_indices = np.asarray(face_vertex_indices, dtype=np.int64)
        # Bake the local-to-root transform (USD uses row-vector convention: v' = v * M).
        matrix = np.asarray(xform_cache.GetLocalToWorldTransform(prim), dtype=np.float64).reshape(4, 4)
        homogeneous = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
        transformed_points = (homogeneous @ matrix)[:, :3]

        triangles: list[tuple[int, int, int]] = []
        cursor = 0
        for count in face_vertex_counts:
            for k in range(1, count - 1):
                triangles.append(
                    (
                        face_vertex_indices[cursor],
                        face_vertex_indices[cursor + k],
                        face_vertex_indices[cursor + k + 1],
                    )
                )
            cursor += count
        if not triangles:
            continue
        vertices_list.append(transformed_points.astype(np.float32))
        triangles_list.append(np.asarray(triangles, dtype=np.int64) + vertex_offset)
        vertex_offset += points.shape[0]

    if not vertices_list:
        raise RuntimeError(f"No meshes found in USD asset: {usd_path}")
    return np.concatenate(vertices_list), np.concatenate(triangles_list).astype(np.int32)


def create_fluid_particles(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Sample local-space MPM particles in the teapot's enclosed cavity."""
    from isaaclab.utils.warp import sample_particles_in_cavity

    water_level = float(vertices[:, 2].min() + FILL_LEVEL * (vertices[:, 2].max() - vertices[:, 2].min()))
    points = sample_particles_in_cavity(
        vertices,
        faces,
        spacing=FILL_SPACING,
        device=args_cli.device,
        jitter=FILL_JITTER,
        seed=FILL_SEED,
        surface_margin=PARTICLE_SURFACE_CLEARANCE,
        min_ray_hits=MIN_RAY_HITS,
        water_level=water_level,
    )
    if points.shape[0] == 0:
        raise RuntimeError("Teapot cavity sampling produced no particles; reduce --fill_spacing or --min_ray_hits.")

    return points.astype(np.float32, copy=False), PARTICLE_RADIUS, PARTICLE_MASS


def create_sim_cfg():
    """Create the Isaac Lab simulation config using the MPM manager."""
    from isaaclab_newton.physics import MPMSolverCfg, NewtonCfg

    import isaaclab.sim as sim_utils

    cfg = sim_utils.SimulationCfg(
        dt=1.0 / SIMULATION_HZ,
        device=args_cli.device,
        gravity=(0.0, 0.0, -9.81),
        visualizer_cfgs=VisualizerCfg(
            eye=CAMERA_EYE,
            lookat=CAMERA_TARGET,
            backend=NewtonGLBackendCfg(show_particles=SHOW_FLUID_PARTICLES, particle_color=WATER_COLOR),
        ),
        physics=NewtonCfg(
            solver_cfg=MPMSolverCfg(
                voxel_size=VOXEL_SIZE,
                grid_type=GRID_TYPE,
                grid_padding=GRID_PADDING,
                max_active_cell_count=MAX_ACTIVE_CELL_COUNT,
                max_iterations=100,
                tolerance=1.0e-4,
                collider_basis="S2",
                strain_basis="P0",
                transfer_scheme="apic",
                integration_scheme="pic",
                air_drag=1.0e-3,
                collider_velocity_mode="forward",
                # Grid contact avoids impulses from projecting particles out of colliders.
                project_outside_colliders=False,
            ),
            # The outer 800 Hz step is already the desired MPM collision interval.
            num_substeps=MPM_SUBSTEPS,
            use_cuda_graph=not args_cli.disable_cuda_graph,
        ),
    )
    if ROBOT_ENABLED:
        from rizon_sharpa_teapot import make_solver_cfg

        solver_cfg = cfg.physics.solver_cfg
        solver_cfg.air_drag = 1.0
        solver_cfg.strain_basis = "P1d"
        solver_cfg.transfer_scheme = "pic"
        # Quadratic velocity interpolation reduces grid-aligned flow in the bowl.
        solver_cfg.velocity_basis = "B2"
        solver_cfg.max_iterations = 25
        solver_cfg.max_active_cell_count = 1 << 15
        # Cover dense cells during bowl impact; pic27 can omit wall contacts.
        solver_cfg.collider_basis = "pic512"
        cfg.physics = NewtonCfg(
            solver_cfg=make_solver_cfg(solver_cfg),
            num_substeps=MPM_SUBSTEPS,
            use_cuda_graph=not args_cli.disable_cuda_graph,
        )
    return cfg


def create_scene_cfg(container_usd: str, island_usd: str, bowl_usd: str):
    """Create the teapot-fill scene using declarative Isaac Lab assets."""
    from isaaclab_newton.assets import MPMObjectCfg
    from isaaclab_newton.sim.schemas import NewtonCollisionCfg
    from isaaclab_newton.sim.spawners.mpm import MPMParticleMaterialCfg, MPMPointsCfg
    from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    container_pos, container_rot, _ = container_pose_at_time(0.0)
    container_vertices, container_faces = load_asset_mesh(container_usd)
    fluid_points, particle_radius, particle_mass = create_fluid_particles(container_vertices, container_faces)

    @configclass
    class DemoMeshCfg(sim_utils.MeshCfg):
        """Demo-local exact triangle-mesh asset config."""

        func: Callable | str = clone(spawn_demo_mesh)
        vertices: list[list[float]] = MISSING
        faces: list[list[int]] = MISSING
        mesh_collision_props: sim_utils.UsdPhysicsMeshCollisionCfg | None = None
        rigid_contact_usd: str | None = None

    # Explicit mappings modify authored physics without adding a body to the asset root.
    disable_asset_rigid_bodies = {"(/.*)?": [sim_utils.UsdPhysicsRigidBodyCfg(rigid_body_enabled=False)]}
    disable_asset_colliders = {"(/.*)?": [sim_utils.UsdPhysicsCollisionCfg(collision_enabled=False)]}

    @configclass
    class TeapotFillSceneCfg(InteractiveSceneCfg):
        """Scene containing MPM colliders and one MPM fluid object sampled inside the teapot."""

        island = AssetBaseCfg(
            prim_path="/World/Island",
            spawn=sim_utils.UsdFileCfg(
                usd_path=island_usd,
                make_uninstanceable=True,
                rigid_props=disable_asset_rigid_bodies,
                collision_props=disable_asset_colliders,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(rot=TABLE_ORIENTATION),
        )

        tabletop_collider = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/TabletopCollider",
            spawn=sim_utils.CuboidCfg(
                size=(2.0 * TABLE_HALF_EXTENTS[0], 2.0 * TABLE_HALF_EXTENTS[1], 2.0 * TABLE_HALF_EXTENTS[2]),
                collision_props=[
                    sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
                    NewtonCollisionCfg(contact_margin=COLLIDER_MARGIN),
                ],
                physics_material=UsdPhysicsRigidBodyMaterialCfg(
                    static_friction=TABLE_FRICTION,
                    dynamic_friction=TABLE_FRICTION,
                ),
                physics_material_path="physicsMaterial",
                visible=False,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.0, 0.0, TABLE_TOP_Z - TABLE_HALF_EXTENTS[2]),
                rot=TABLE_ORIENTATION,
            ),
        )

        catch_bowl = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/CatchBowl",
            spawn=sim_utils.UsdFileCfg(
                usd_path=bowl_usd,
                scale=(BOWL_SCALE,) * 3,
                variants={"Newton": "Enabled", "PhysX": "Disabled", "MuJoCo": "Disabled"},
                make_uninstanceable=True,
                rigid_props={
                    "(/.*)?": [sim_utils.UsdPhysicsRigidBodyCfg(rigid_body_enabled=False, kinematic_enabled=True)]
                },
                collision_props=[
                    sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
                    NewtonCollisionCfg(contact_margin=COLLIDER_MARGIN),
                ],
                mesh_collision_props=sim_utils.UsdPhysicsMeshCollisionCfg(mesh_approximation_name="none"),
                physics_material=UsdPhysicsRigidBodyMaterialCfg(
                    static_friction=BOWL_FRICTION,
                    dynamic_friction=BOWL_FRICTION,
                ),
                physics_material_path="physicsMaterial",
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=BOWL_BASE_POS),
        )

        # Utah Teapot: free for any use; credit as the (Modified) Utah Teapot
        # (Univ. of Utah); provided "as is", no warranty.
        container = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/PourContainer",
            # Re-spawn the source geometry as one exact triangle mesh. The USD
            # asset's authored convex decomposition is unsuitable for a hollow
            # MPM collider and SolverImplicitMPM does not accept convex meshes.
            spawn=DemoMeshCfg(
                vertices=container_vertices.tolist(),
                faces=container_faces.tolist(),
                rigid_props=[
                    sim_utils.UsdPhysicsRigidBodyCfg(rigid_body_enabled=True, kinematic_enabled=True),
                    PhysxRigidBodyCfg(disable_gravity=True),
                ],
                collision_props=[
                    sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
                    NewtonCollisionCfg(contact_margin=COLLIDER_MARGIN),
                ],
                mesh_collision_props=sim_utils.UsdPhysicsMeshCollisionCfg(mesh_approximation_name="none"),
                physics_material=UsdPhysicsRigidBodyMaterialCfg(
                    static_friction=CONTAINER_FRICTION,
                    dynamic_friction=CONTAINER_FRICTION,
                ),
                physics_material_path="physicsMaterial",
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=CONTAINER_COLOR),
                visual_material_path="visualMaterial",
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=container_pos, rot=container_rot),
        )

        fluid = MPMObjectCfg(
            prim_path="{ENV_REGEX_NS}/Fluid",
            spawn=MPMPointsCfg(
                visible=SHOW_FLUID_PARTICLES,
                positions=fluid_points.tolist(),
                mass=particle_mass,
                radius=particle_radius,
                material=MPMParticleMaterialCfg(
                    viscosity=1.0e-3,
                    friction=0.0,
                    damping=1.0e-3,
                    yield_pressure=1.0e15,
                    tensile_yield_ratio=1.0,
                ),
                visual_color=WATER_COLOR,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=WATER_COLOR,
                    roughness=0.1,
                    opacity=0.7,
                ),
            ),
            init_state=MPMObjectCfg.InitialStateCfg(pos=container_pos),
        )

        ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())

        dome_light = AssetBaseCfg(
            prim_path="/World/DomeLight",
            spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.78, 0.78, 0.78)),
        )

    cfg = TeapotFillSceneCfg(num_envs=1, env_spacing=0.0)
    if ROBOT_ENABLED:
        from isaaclab_newton.sim.schemas import MujocoCollisionCfg
        from rizon_sharpa_teapot import get_robot_cfg, prepared_contact_path

        cfg.robot = get_robot_cfg(base_position=ROBOT_BASE_POS)
        cfg.tabletop_collider.spawn.collision_props[1].contact_gap = 0.001
        cfg.tabletop_collider.spawn.collision_props.append(MujocoCollisionCfg(condim=4, solref=(0.0025, 1.0)))
        cfg.container.spawn.rigid_props = sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=False)
        cfg.container.spawn.mass_props = sim_utils.MassCfg(mass=0.45)
        # Keep the exact fluid passage open; the initial particles retain their larger clearance.
        cfg.container.spawn.collision_props[1].contact_margin = 0.0005
        cfg.container.spawn.rigid_contact_usd = str(prepared_contact_path(Path(container_usd)))
        cfg.container.spawn.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0))
        cfg.robot_pedestal = AssetBaseCfg(
            prim_path="/World/RobotPedestal",
            spawn=sim_utils.CylinderCfg(
                radius=0.10,
                height=ROBOT_BASE_POS[2],
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.10, 0.12, 0.15),
                    metallic=0.65,
                    roughness=0.3,
                ),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(*ROBOT_BASE_POS[:2], 0.5 * ROBOT_BASE_POS[2])),
        )
        if {"kit", "newton_gl", "newton_rtx"}.intersection(args_cli.visualizer or []):
            cfg.room = AssetBaseCfg(
                prim_path="/World/Room",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=(
                        f"{ISAACLAB_NUCLEUS_DIR}/Environments/RizonSharpaTeapot/Environment/enclosed_simple_room_v2.usda"
                    ),
                    make_uninstanceable=True,
                    rigid_props=disable_asset_rigid_bodies,
                    collision_props=disable_asset_colliders,
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -0.65, 0.7685705931503188), rot=(0.0, 0.0, 1.0, 0.0)),
            )
            cfg.ground.spawn.visible = False
    return cfg


def particle_count(scene) -> int:
    """Return the number of MPM particles in the scene."""
    fluid = scene["fluid"]
    return fluid.num_instances * fluid.particles_per_object


def keep_running(sim, count: int) -> bool:
    """Return whether the demo loop should continue this frame."""
    if args_cli.max_steps >= 0 and count >= args_cli.max_steps:
        return False
    return sim.is_running()


def write_container_state(container, sim_time: float) -> None:
    """Write the scripted container pose and velocity through the rigid-object API."""
    pos, quat, twist = container_pose_at_time(sim_time)
    pose = torch.tensor([pos + quat], dtype=torch.float32, device=container.device)
    velocity = torch.tensor([twist], dtype=torch.float32, device=container.device)
    container.write_root_link_pose_to_sim_index(root_pose=pose)
    container.write_root_link_velocity_to_sim_index(root_velocity=velocity)


def run_simulator(sim, scene, surface_renderer: ParticleSurfaceRenderer | None, robot_motion=None) -> None:
    """Run the scripted teapot-fill MPM loop."""
    sim_dt = sim.get_physics_dt()
    container = scene["container"]
    count = 0
    while keep_running(sim, count):
        if robot_motion is None:
            write_container_state(container, count / SIMULATION_HZ)
        else:
            robot_motion.update(count / SIMULATION_HZ)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(sim_dt)
        if sim.is_rendering and count % RENDER_INTERVAL == 0:
            if surface_renderer is not None:
                surface_renderer.update()
            sim.render()
        count += 1


def main() -> None:
    """Set up and run the Isaac Lab Newton MPM teapot-fill demo."""
    sim_cfg = create_sim_cfg()
    with launch_simulation(sim_cfg, args_cli):
        if ROBOT_ENABLED:
            from rizon_sharpa_teapot import configured_asset_path

            configured_asset_path()
        container_usd = retrieve_file_path(args_cli.container_usd)
        island_usd = retrieve_file_path(args_cli.island_usd)
        bowl_usd = retrieve_file_path(args_cli.bowl_usd)

        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene

        sim = sim_utils.SimulationContext(sim_cfg)
        scene = InteractiveScene(create_scene_cfg(container_usd, island_usd, bowl_usd))
        sim.reset()
        robot_motion = None
        if ROBOT_ENABLED:
            from rizon_sharpa_teapot import (
                GRASP_CLOSE_INTERVAL,
                INITIAL_LIFT_TIME,
                PICKUP_TIME,
                POUR_PREFIX_TIME,
                SEQUENCE_DURATION,
                TeapotPourMotion,
                approach_offset_at_time,
            )

            robot = scene["robot"]
            robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.warp)
            robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.warp)
            sim.forward()
            robot_motion = TeapotPourMotion(
                robot,
                scene["container"],
                container_pose_at_time,
                sim_dt=sim.get_physics_dt(),
                duration=SEQUENCE_DURATION,
                approach_offset_at_time=approach_offset_at_time,
                grasp_time=PICKUP_TIME,
                pickup_end_time=PICKUP_TIME + INITIAL_LIFT_TIME,
                close_interval=GRASP_CLOSE_INTERVAL,
            )
            robot.write_joint_position_to_sim_index(position=robot_motion.initial_joint_positions)
            sim.forward()
        surface_renderer = create_surface_renderer(sim) if SHOW_FLUID_SURFACE else None
        surface_triangle_count = surface_renderer.update() if surface_renderer is not None else 0

        print(
            "[INFO]: Isaac Lab Newton teapot-fill MPM demo ready."
            f" Sampled {particle_count(scene)} MPM particles inside the teapot;"
            f" extracted {surface_triangle_count} surface triangles;"
            f" rendering {args_cli.fluid_render_mode};"
            f" fill spacing {FILL_SPACING:.4g} m;"
            f" the teapot will tilt after {POUR_PREFIX_TIME + HOLD_TIME if ROBOT_ENABLED else HOLD_TIME:.2f}s.",
            flush=True,
        )
        run_simulator(sim, scene, surface_renderer, robot_motion)


if __name__ == "__main__":
    main()
