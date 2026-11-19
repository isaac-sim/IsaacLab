# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared scene and fluid configuration for the Newton teapot examples."""

from __future__ import annotations

import argparse
import math
from dataclasses import MISSING
from typing import TYPE_CHECKING

import numpy as np
from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.physics import MPMSolverCfg, NewtonCfg
from isaaclab_newton.sim.schemas import NewtonCollisionCfg
from isaaclab_newton.sim.spawners.mpm import MPMParticleMaterialCfg, MPMPointsCfg
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg
from isaaclab_visualizers.newton import NewtonGLBackendCfg

import isaaclab.sim as sim_utils
from isaaclab.app import add_launcher_args
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, retrieve_file_path
from isaaclab.visualizers import VisualizerCfg

if TYPE_CHECKING:
    from isaaclab_visualizers.newton import ParticleSurfaceRenderer

DEFAULT_VOXEL_SIZE = 0.003
DEFAULT_PARTICLES_PER_VOXEL_AXIS = 2.0
DEFAULT_FILL_LEVEL = 0.70
DEFAULT_CONTAINER_USD = f"{ISAAC_NUCLEUS_DIR}/Props/Teapot/utah_teapot.usdc"
DEFAULT_MIN_RAY_HITS = 5
DEFAULT_ISLAND_USD = (
    f"{ISAAC_NUCLEUS_DIR}/SimReady/Residential/Kitchen/Counters/Island_A01/sm_fixture_island_a01_01.usd"
)
DEFAULT_BOWL_USD = f"{ISAAC_NUCLEUS_DIR}/SimReady/Residential/Kitchen/Dishware/Bowl_G01/sm_kitchenware_bowl_g01_01.usd"


# Use one 1.25 ms solver step per outer loop iteration and retain the original 400 Hz render cadence.
SIMULATION_HZ = 800
RENDER_INTERVAL = 2
PARTICLE_DENSITY = 1000.0
FILL_JITTER = 0.2
FILL_SEED = 7
CONTAINER_FRICTION = 0.0
BOWL_FRICTION = 0.20
TABLE_FRICTION = 0.5

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

SURFACE_PATH = "/fluid_surface"

_DISABLED_RIGID_BODIES = {"(/.*)?": [sim_utils.UsdPhysicsRigidBodyCfg(rigid_body_enabled=False)]}
_DISABLED_COLLIDERS = {"(/.*)?": [sim_utils.UsdPhysicsCollisionCfg(collision_enabled=False)]}


def add_teapot_args(parser: argparse.ArgumentParser, *, max_steps: int, fluid_render_mode: str) -> None:
    """Add the options shared by the two standalone teapot examples."""
    parser.add_argument(
        "--max_steps",
        type=int,
        default=max_steps,
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
        help="Water line as a fraction [0, 1] of the teapot height. Use 1.0 to fill to the brim.",
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
        default=fluid_render_mode,
        choices=["particles", "surface", "both"],
        help="Fluid visualization: raw MPM particles, reconstructed surface mesh, or both.",
    )
    parser.add_argument(
        "--container_usd",
        type=str,
        default=DEFAULT_CONTAINER_USD,
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


def create_sim_cfg(args: argparse.Namespace) -> sim_utils.SimulationCfg:
    """Configure Newton MPM and its GL view."""
    sparse = args.grid_type == "sparse"
    solver = MPMSolverCfg(
        voxel_size=args.voxel_size,
        grid_type=args.grid_type,
        grid_padding=0 if sparse else 64,
        max_active_cell_count=(1 << 16) if sparse else (1 << 18),
        max_iterations=100,
        tolerance=1.0e-4,
        collider_basis="S2",
        strain_basis="P0",
        transfer_scheme="apic",
        integration_scheme="pic",
        air_drag=1.0e-3,
        collider_velocity_mode="forward",
        project_outside_colliders=False,
    )
    return sim_utils.SimulationCfg(
        dt=1.0 / SIMULATION_HZ,
        device=args.device,
        gravity=(0.0, 0.0, -9.81),
        visualizer_cfgs=VisualizerCfg(
            eye=CAMERA_EYE,
            lookat=CAMERA_TARGET,
            backend=NewtonGLBackendCfg(show_particles=args.fluid_render_mode != "surface", particle_color=WATER_COLOR),
        ),
        physics=NewtonCfg(solver_cfg=solver, num_substeps=1, use_cuda_graph=not args.disable_cuda_graph),
    )


def configure_scene(
    cfg: TeapotFillSceneCfg,
    args: argparse.Namespace,
    container_usd: str,
    fill_spacing: float,
    *,
    shell_margin: float | None = None,
) -> None:
    """Apply CLI settings and sample fluid from the same exact shell used for collisions."""
    vertices, faces = _load_asset_mesh(container_usd)
    cfg.island.spawn.usd_path = retrieve_file_path(args.island_usd)
    cfg.catch_bowl.spawn.usd_path = retrieve_file_path(args.bowl_usd)
    margin = 0.5 * args.voxel_size
    cfg.catch_bowl.spawn.collision_props[1].contact_margin = margin
    cfg.tabletop_collider.spawn.collision_props[1].contact_margin = margin
    cfg.container.spawn.collision_props[1].contact_margin = margin if shell_margin is None else shell_margin
    cfg.container.spawn.mesh = sim_utils.MeshFileCfg.TriangleMeshCfg(vertices=vertices, faces=faces)
    cfg.fluid.spawn.visible = args.fluid_render_mode != "surface"
    cfg.fluid.spawn.positions = _sample_fluid_particles(vertices, faces, args, fill_spacing)
    cfg.fluid.spawn.mass = fill_spacing**3 * PARTICLE_DENSITY
    cfg.fluid.spawn.radius = 0.5 * fill_spacing


def particle_spacing(args: argparse.Namespace) -> float:
    """Return the explicit spacing or derive it from the grid voxel size."""
    return args.voxel_size / DEFAULT_PARTICLES_PER_VOXEL_AXIS if args.fill_spacing is None else args.fill_spacing


def create_surface_renderer(
    sim: sim_utils.SimulationContext, args: argparse.Namespace, fill_spacing: float
) -> ParticleSurfaceRenderer | None:
    """Configure teapot-specific surface extraction after Newton initializes."""
    from isaaclab_visualizers.newton import ParticleSurfaceRenderer
    from newton.geometry import ParticleSurface

    if args.fluid_render_mode == "particles":
        return None
    model = sim.backend_manager.backend.model
    surface = ParticleSurface(
        voxel_size=0.5 * args.voxel_size,
        max_grid_cells=4_000_000,
        world_count=max(model.world_count, 1),
        kernel_radius=max(3.0 * fill_spacing, 0.75 * args.voxel_size),
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
        backend_manager=sim.backend_manager,
        path=SURFACE_PATH,
        color=WATER_COLOR,
        opacity=WATER_OPACITY,
        roughness=0.1,
        use_cuda_graph=not args.disable_cuda_graph,
    )


@configclass
class TeapotFillSceneCfg(InteractiveSceneCfg):
    """Scene containing MPM colliders and one MPM fluid object sampled inside the teapot."""

    island: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Island",
        spawn=sim_utils.UsdFileCfg(
            usd_path=DEFAULT_ISLAND_USD,
            make_uninstanceable=True,
            rigid_props=_DISABLED_RIGID_BODIES,
            collision_props=_DISABLED_COLLIDERS,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(rot=TABLE_ORIENTATION),
    )

    tabletop_collider: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TabletopCollider",
        spawn=sim_utils.CuboidCfg(
            size=(2.0 * TABLE_HALF_EXTENTS[0], 2.0 * TABLE_HALF_EXTENTS[1], 2.0 * TABLE_HALF_EXTENTS[2]),
            collision_props=[
                sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
                NewtonCollisionCfg(contact_margin=0.5 * DEFAULT_VOXEL_SIZE),
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

    catch_bowl: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CatchBowl",
        spawn=sim_utils.UsdFileCfg(
            usd_path=DEFAULT_BOWL_USD,
            scale=(BOWL_SCALE,) * 3,
            variants={"Newton": "Enabled", "PhysX": "Disabled", "MuJoCo": "Disabled"},
            make_uninstanceable=True,
            rigid_props={
                "(/.*)?": [sim_utils.UsdPhysicsRigidBodyCfg(rigid_body_enabled=False, kinematic_enabled=True)]
            },
            collision_props=[
                sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
                NewtonCollisionCfg(contact_margin=0.5 * DEFAULT_VOXEL_SIZE),
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
    container: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PourContainer",
        spawn=sim_utils.MeshFileCfg(
            mesh=MISSING,
            rigid_props=[
                sim_utils.UsdPhysicsRigidBodyCfg(rigid_body_enabled=True, kinematic_enabled=True),
                PhysxRigidBodyCfg(disable_gravity=True),
            ],
            collision_props=[
                sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
                NewtonCollisionCfg(contact_margin=0.5 * DEFAULT_VOXEL_SIZE),
            ],
            mesh_collision_props=sim_utils.UsdPhysicsMeshCollisionCfg(mesh_approximation_name="none"),
            physics_material=UsdPhysicsRigidBodyMaterialCfg(
                static_friction=CONTAINER_FRICTION, dynamic_friction=CONTAINER_FRICTION
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=CONTAINER_COLOR),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=CONTAINER_BASE_POS),
    )

    fluid: MPMObjectCfg = MPMObjectCfg(
        prim_path="{ENV_REGEX_NS}/Fluid",
        spawn=MPMPointsCfg(
            positions=MISSING,
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
        init_state=MPMObjectCfg.InitialStateCfg(pos=CONTAINER_BASE_POS),
    )

    ground: AssetBaseCfg = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())

    dome_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.78, 0.78, 0.78)),
    )


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


def _load_asset_mesh(usd_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read a USD asset and return its triangulated geometry in the asset's local frame.

    Meshes, including instance proxies, are concatenated, transformed into the asset frame, and
    fan-triangulated so visual assets and their exact MPM colliders align.
    """
    from pxr import Usd, UsdGeom

    from isaaclab.utils.mesh import convert_faces_to_triangles

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

        triangles = convert_faces_to_triangles(face_vertex_indices, face_vertex_counts)
        if not len(triangles):
            continue
        vertices_list.append(transformed_points.astype(np.float32))
        triangles_list.append(triangles + vertex_offset)
        vertex_offset += points.shape[0]

    if not vertices_list:
        raise RuntimeError(f"No meshes found in USD asset: {usd_path}")
    return np.concatenate(vertices_list), np.concatenate(triangles_list).astype(np.int32)


def _sample_fluid_particles(
    vertices: np.ndarray, faces: np.ndarray, args: argparse.Namespace, spacing: float
) -> np.ndarray:
    """Sample local-space MPM particles in the teapot's enclosed cavity."""
    from isaaclab.utils.warp import sample_particles_in_cavity

    water_level = float(vertices[:, 2].min() + args.fill_level * (vertices[:, 2].max() - vertices[:, 2].min()))
    points = sample_particles_in_cavity(
        vertices,
        faces,
        spacing=spacing,
        device=args.device,
        jitter=FILL_JITTER,
        seed=FILL_SEED,
        surface_margin=0.5 * (args.voxel_size + spacing),
        min_ray_hits=args.min_ray_hits,
        water_level=water_level,
    )
    if points.shape[0] == 0:
        raise RuntimeError("Teapot cavity sampling produced no particles; reduce --fill_spacing or --min_ray_hits.")

    return points.astype(np.float32, copy=False)
