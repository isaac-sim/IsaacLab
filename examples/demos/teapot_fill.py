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
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from isaaclab_visualizers.newton import NewtonGLBackendCfg

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, retrieve_file_path
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
# Use one 1.25 ms solver step per outer loop iteration and retain the original 400 Hz render cadence.
SIMULATION_HZ = 800
RENDER_INTERVAL = 2
PARTICLE_DENSITY = 1000.0
FILL_JITTER = 0.2
FILL_SEED = 7
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

SURFACE_PATH = "/fluid_surface"


def create_surface_renderer(sim, args: argparse.Namespace, fill_spacing: float) -> ParticleSurfaceRenderer:
    """Configure teapot-specific surface extraction after Newton initializes."""
    from isaaclab_visualizers.newton import ParticleSurfaceRenderer
    from newton.geometry import ParticleSurface

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


def load_asset_mesh(usd_path: str) -> tuple[np.ndarray, np.ndarray]:
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


def create_fluid_particles(
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


def create_sim_cfg(args: argparse.Namespace | None = None):
    """Create the Newton simulation for the selected teapot motion."""
    from isaaclab_newton.physics import MPMSolverCfg, NewtonCfg

    import isaaclab.sim as sim_utils

    if args is None:
        args = parser.parse_args()
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
    eye, lookat = CAMERA_EYE, CAMERA_TARGET
    if args.robot == "rizon_sharpa":
        from isaaclab_tasks.contrib.rizon_sharpa_teapot.scene import make_solver_cfg

        # Quadratic velocity interpolation and dense collider sampling keep the pouring stream inside the bowl.
        solver = make_solver_cfg(
            solver.replace(
                air_drag=1.0,
                strain_basis="P1d",
                transfer_scheme="pic",
                velocity_basis="B2",
                max_iterations=25,
                max_active_cell_count=1 << 15,
                collider_basis="pic512",
            )
        )
        eye, lookat = (0.55, -1.5, 1.72), (-0.28, 0.0, 1.22)
    return sim_utils.SimulationCfg(
        dt=1.0 / SIMULATION_HZ,
        device=args.device,
        gravity=(0.0, 0.0, -9.81),
        visualizer_cfgs=VisualizerCfg(
            eye=eye,
            lookat=lookat,
            backend=NewtonGLBackendCfg(show_particles=args.fluid_render_mode != "surface", particle_color=WATER_COLOR),
        ),
        physics=NewtonCfg(solver_cfg=solver, num_substeps=1, use_cuda_graph=not args.disable_cuda_graph),
    )


def create_scene_cfg(args: argparse.Namespace, fill_spacing: float):
    """Create the teapot-fill scene using declarative Isaac Lab assets."""
    from isaaclab_newton.assets import MPMObjectCfg
    from isaaclab_newton.sim.schemas import NewtonCollisionCfg
    from isaaclab_newton.sim.spawners.mpm import MPMParticleMaterialCfg, MPMPointsCfg
    from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg
    from isaaclab.utils import configclass

    container_usd = retrieve_file_path(args.container_usd)
    bowl_pos = BOWL_BASE_POS
    container_pos, container_rot, _ = container_pose_at_time(0.0)
    if args.robot == "rizon_sharpa":
        from isaaclab_tasks.contrib.rizon_sharpa_teapot import motion

        bowl_pos = motion.BOWL_BASE_POS
        container_pos, container_rot, _ = motion.pose_at_time(0.0)
    vertices, faces = load_asset_mesh(container_usd)
    fluid_points = create_fluid_particles(vertices, faces, args, fill_spacing)
    collider_margin = 0.5 * args.voxel_size

    # Explicit mappings modify authored physics without adding a body to the asset root.
    disable_asset_rigid_bodies = {"(/.*)?": [sim_utils.UsdPhysicsRigidBodyCfg(rigid_body_enabled=False)]}
    disable_asset_colliders = {"(/.*)?": [sim_utils.UsdPhysicsCollisionCfg(collision_enabled=False)]}

    @configclass
    class TeapotFillSceneCfg(InteractiveSceneCfg):
        """Scene containing MPM colliders and one MPM fluid object sampled inside the teapot."""

        island = AssetBaseCfg(
            prim_path="/World/Island",
            spawn=sim_utils.UsdFileCfg(
                usd_path=retrieve_file_path(args.island_usd),
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
                    NewtonCollisionCfg(contact_margin=collider_margin),
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
                usd_path=retrieve_file_path(args.bowl_usd),
                scale=(BOWL_SCALE,) * 3,
                variants={"Newton": "Enabled", "PhysX": "Disabled", "MuJoCo": "Disabled"},
                make_uninstanceable=True,
                rigid_props={
                    "(/.*)?": [sim_utils.UsdPhysicsRigidBodyCfg(rigid_body_enabled=False, kinematic_enabled=True)]
                },
                collision_props=[
                    sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
                    NewtonCollisionCfg(contact_margin=collider_margin),
                ],
                mesh_collision_props=sim_utils.UsdPhysicsMeshCollisionCfg(mesh_approximation_name="none"),
                physics_material=UsdPhysicsRigidBodyMaterialCfg(
                    static_friction=BOWL_FRICTION,
                    dynamic_friction=BOWL_FRICTION,
                ),
                physics_material_path="physicsMaterial",
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=bowl_pos),
        )

        # Utah Teapot: free for any use; credit as the (Modified) Utah Teapot
        # (Univ. of Utah); provided "as is", no warranty.
        container = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/PourContainer",
            # Re-spawn the source geometry as one exact triangle mesh. The USD
            # asset's authored convex decomposition is unsuitable for a hollow
            # MPM collider and SolverImplicitMPM does not accept convex meshes.
            spawn=sim_utils.MeshFileCfg(
                mesh=sim_utils.MeshFileCfg.TriangleMeshCfg(vertices=vertices, faces=faces),
                rigid_props=[
                    sim_utils.UsdPhysicsRigidBodyCfg(rigid_body_enabled=True, kinematic_enabled=True),
                    PhysxRigidBodyCfg(disable_gravity=True),
                ],
                collision_props=[
                    sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
                    NewtonCollisionCfg(contact_margin=collider_margin),
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
                visible=args.fluid_render_mode != "surface",
                positions=fluid_points,
                mass=fill_spacing**3 * PARTICLE_DENSITY,
                radius=0.5 * fill_spacing,
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
    if args.robot == "rizon_sharpa":
        from isaaclab_newton.sim.schemas import MujocoCollisionCfg

        from isaaclab_tasks.contrib.rizon_sharpa_teapot.scene import (
            prepared_contact_path,
            robot_cfg,
            spawn_grasp_colliders,
        )

        cfg.robot = robot_cfg()
        cfg.tabletop_collider.spawn.collision_props[1].contact_gap = 0.001
        cfg.tabletop_collider.spawn.collision_props.append(MujocoCollisionCfg(condim=4, solref=(0.0025, 1.0)))
        cfg.container.spawn.rigid_props = sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=False)
        cfg.container.spawn.mass_props = sim_utils.MassCfg(mass=0.45)
        # Keep the exact fluid passage open; the initial particles retain their larger clearance.
        cfg.container.spawn.collision_props[1].contact_margin = 0.0005
        cfg.grasp_colliders = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/PourContainer/RigidCollider",
            spawn=sim_utils.UsdFileCfg(
                func=spawn_grasp_colliders,
                usd_path=str(prepared_contact_path(Path(container_usd))),
                collision_props=[
                    sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
                    MujocoCollisionCfg(condim=4, solref=(0.0025, 1.0)),
                    NewtonCollisionCfg(contact_gap=0.001),
                ],
                physics_material=UsdPhysicsRigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
                rigid_props=disable_asset_rigid_bodies,
                visible=False,
            ),
        )
        cfg.container.spawn.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0))
        cfg.robot_pedestal = AssetBaseCfg(
            prim_path="/World/RobotPedestal",
            spawn=sim_utils.CylinderCfg(
                radius=0.10,
                height=motion.ROBOT_BASE_POS[2],
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.10, 0.12, 0.15),
                    metallic=0.65,
                    roughness=0.3,
                ),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(*motion.ROBOT_BASE_POS[:2], 0.5 * motion.ROBOT_BASE_POS[2])),
        )
    return cfg


def write_container_state(container, sim_time: float) -> None:
    """Write the scripted container pose and velocity through the rigid-object API."""
    pos, quat, twist = container_pose_at_time(sim_time)
    pose = torch.tensor([pos + quat], dtype=torch.float32, device=container.device)
    velocity = torch.tensor([twist], dtype=torch.float32, device=container.device)
    container.write_root_link_pose_to_sim_index(root_pose=pose)
    container.write_root_link_velocity_to_sim_index(root_velocity=velocity)


def run_simulator(
    sim, scene, surface_renderer: ParticleSurfaceRenderer | None, max_steps: int, robot_motion=None
) -> None:
    """Run the scripted teapot-fill MPM loop."""
    sim_dt = sim.get_physics_dt()
    container = scene["container"]
    count = 0
    while (max_steps < 0 or count < max_steps) and sim.is_running():
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


def main(defaults: argparse.Namespace | None = None) -> None:
    """Set up and run the teapot demo with explicit command-line options taking precedence."""
    args = parser.parse_args(namespace=defaults)
    fill_spacing = (
        args.voxel_size / DEFAULT_PARTICLES_PER_VOXEL_AXIS if args.fill_spacing is None else args.fill_spacing
    )
    sim_cfg = create_sim_cfg(args)
    with launch_simulation(sim_cfg, args):
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene

        sim = sim_utils.SimulationContext(sim_cfg)
        scene = InteractiveScene(create_scene_cfg(args, fill_spacing))
        sim.reset()
        robot_motion = None
        tilt_start = HOLD_TIME
        if args.robot == "rizon_sharpa":
            from isaaclab_tasks.contrib.rizon_sharpa_teapot.motion import POUR_PREFIX_TIME, TeapotPourMotion

            tilt_start = POUR_PREFIX_TIME
            robot = scene["robot"]
            robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.warp)
            robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.warp)
            sim.forward()
            robot_motion = TeapotPourMotion(
                robot,
                scene["container"],
                sim.backend_manager.backend.solver.solver("robot").mj_model,
                sim_dt=sim.get_physics_dt(),
            )
            robot.write_joint_position_to_sim_index(position=robot_motion.initial_joint_positions)
            sim.forward()
        surface_renderer = (
            create_surface_renderer(sim, args, fill_spacing) if args.fluid_render_mode != "particles" else None
        )
        surface_triangle_count = surface_renderer.update() if surface_renderer is not None else 0

        fluid = scene["fluid"]
        print(
            "[INFO]: Isaac Lab Newton teapot-fill MPM demo ready."
            f" Sampled {fluid.num_instances * fluid.particles_per_object} MPM particles inside the teapot;"
            f" extracted {surface_triangle_count} surface triangles;"
            f" rendering {args.fluid_render_mode};"
            f" fill spacing {fill_spacing:.4g} m;"
            f" the teapot will tilt after {tilt_start:.2f}s.",
            flush=True,
        )
        run_simulator(sim, scene, surface_renderer, args.max_steps, robot_motion)


if __name__ == "__main__":
    main()
