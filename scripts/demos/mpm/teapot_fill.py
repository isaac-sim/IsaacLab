# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fill and pour water from the Utah teapot with Newton implicit MPM.

The teapot is a hollow, double-walled shell, so the fluid is seeded in its enclosed air cavity with
:func:`~isaaclab.utils.warp.sample_particles_in_cavity` instead of using a winding-number volume fill.

.. code-block:: bash

    # Fast Newton visualizer (the default):
    uv run python scripts/demos/mpm/teapot_fill.py --device cuda:0 --visualizer newton
    # The translucent water material needs the RTX/Kit visualizer:
    uv run python scripts/demos/mpm/teapot_fill.py --device cuda:0 --visualizer kit
    # Fuller / coarser (faster) fill:
    uv run python scripts/demos/mpm/teapot_fill.py --fill_level 1.0 --fill_spacing 0.003
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from dataclasses import MISSING

import numpy as np
import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, retrieve_file_path

DEFAULT_VOXEL_SIZE = 0.003
DEFAULT_PARTICLES_PER_CELL = 1.2
DEFAULT_FILL_LEVEL = 0.70
DEFAULT_MIN_RAY_HITS = 5


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


parser = argparse.ArgumentParser(description="Newton implicit MPM teapot-fill demo.")
parser.add_argument(
    "--max_steps",
    type=int,
    default=3000,
    help="Stop after this many frames; negative runs forever.",
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
        f" Defaults to voxel_size / {DEFAULT_PARTICLES_PER_CELL:g}."
    ),
)
parser.add_argument(
    "--fill_level",
    type=_unit_interval_float,
    default=DEFAULT_FILL_LEVEL,
    help=(
        "Water line as a fraction [0, 1] of the teapot height to fill the cavity up to."
        f" Defaults to {DEFAULT_FILL_LEVEL:g}; use 1.0 to fill to the brim."
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
    "--container_usd",
    type=str,
    default=f"{ISAAC_NUCLEUS_DIR}/Props/Teapot/utah_teapot.usdc",
    help="USD asset used as the pouring container (rigid collider).",
)
add_launcher_args(parser)
parser.set_defaults(visualizer=["newton_gl"])
args_cli = parser.parse_args()


# Update the kinematic spout at 400 Hz; its per-step displacement must remain inside the MPM contact band.
FPS = 400
VOXEL_SIZE = args_cli.voxel_size
FILL_SPACING = args_cli.fill_spacing if args_cli.fill_spacing is not None else VOXEL_SIZE / DEFAULT_PARTICLES_PER_CELL
FILL_LEVEL = args_cli.fill_level
MIN_RAY_HITS = args_cli.min_ray_hits

# Sparse grids reserve capture-stable storage; fixed grids need padding for the full pour trajectory.
GRID_TYPE = args_cli.grid_type
GRID_PADDING = 0 if GRID_TYPE == "sparse" else 64
MAX_ACTIVE_CELL_COUNT = (1 << 16) if GRID_TYPE == "sparse" else (1 << 18)
MPM_SUBSTEPS = 2

PARTICLE_DENSITY = 1000.0
FILL_JITTER = 0.2
FILL_SEED = 7
PARTICLE_RADIUS = 0.5 * FILL_SPACING
PARTICLE_MASS = FILL_SPACING**3 * PARTICLE_DENSITY

COLLIDER_MARGIN = 0.5 * VOXEL_SIZE
CONTAINER_MARGIN = 0.00125
PARTICLE_SURFACE_CLEARANCE = CONTAINER_MARGIN + 0.5 * FILL_SPACING
CONTAINER_FRICTION = 0.0
BOWL_FRICTION = 0.20
TABLE_FRICTION = 0.5

HOLD_TIME = 0.55
TILT_TIME = 2.0
POUR_ANGLE = math.radians(65.0)
CONTAINER_LIFT_HEIGHT = 0.24
CONTAINER_LIFT_TIME = 3.0

TABLE_TOP_Z = 0.255
TABLE_HALF_EXTENTS = (0.255, 0.165, 0.009)
BOWL_BASE_POS = (0.066, 0.0, TABLE_TOP_Z + 0.006)
BOWL_HEIGHT = 0.039
CONTAINER_BASE_POS = (-0.105, 0.0, BOWL_BASE_POS[2] + BOWL_HEIGHT + 0.115)

BOWL_INNER_BOTTOM_RADIUS = 0.0135
BOWL_INNER_TOP_RADIUS = 0.057
BOWL_WALL_THICKNESS = 0.0075
BOWL_BOTTOM_THICKNESS = 0.0075
BOWL_COLOR = (1.0, 1.0, 1.0)
CONTAINER_COLOR = (0.70, 0.35, 0.16)
TABLE_COLOR = (0.48, 0.38, 0.26)
WATER_COLOR = (0.12, 0.35, 0.78)

CAMERA_EYE = (0.0, -0.54, 0.64)
CAMERA_TARGET = (-0.01, 0.0, 0.30)


def create_visualizer_cfgs():
    """Create demo-specific visualizer configs for requested backends."""
    if not any(v in (args_cli.visualizer or []) for v in ("newton", "newton_gl", "newton_rtx")):
        return []

    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg

    cfg_type = NewtonRTXVisualizerCfg if args_cli.visualizer == ["newton_rtx"] else NewtonGLVisualizerCfg
    return [
        cfg_type(
            show_particles=True,
            particle_color=WATER_COLOR,
        )
    ]


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


def create_demo_bowl_mesh(num_segments: int = 96) -> tuple[np.ndarray, np.ndarray]:
    """Build one local-space open bowl mesh used by the catch-bowl collider."""
    theta = np.linspace(0.0, 2.0 * math.pi, num_segments, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    outer_bottom_radius = BOWL_INNER_BOTTOM_RADIUS + BOWL_WALL_THICKNESS
    outer_top_radius = BOWL_INNER_TOP_RADIUS + BOWL_WALL_THICKNESS

    def ring(radius: float, z: float) -> np.ndarray:
        return np.column_stack([radius * cos_t, radius * sin_t, np.full(num_segments, z)])

    vertices = np.vstack(
        [
            ring(BOWL_INNER_BOTTOM_RADIUS, BOWL_BOTTOM_THICKNESS),
            ring(BOWL_INNER_TOP_RADIUS, BOWL_HEIGHT),
            ring(outer_top_radius, BOWL_HEIGHT),
            ring(outer_bottom_radius, 0.0),
            np.array([[0.0, 0.0, BOWL_BOTTOM_THICKNESS], [0.0, 0.0, 0.0]], dtype=np.float32),
        ]
    ).astype(np.float32)

    inner_center_id = 4 * num_segments
    outer_center_id = inner_center_id + 1
    indices: list[int] = []
    for i in range(num_segments):
        j = (i + 1) % num_segments
        ib_i, ib_j = i, j
        it_i, it_j = i + num_segments, j + num_segments
        ot_i, ot_j = i + 2 * num_segments, j + 2 * num_segments
        ob_i, ob_j = i + 3 * num_segments, j + 3 * num_segments
        indices.extend([ib_i, it_i, ib_j, ib_j, it_i, it_j])
        indices.extend([ob_i, ob_j, ot_i, ot_i, ob_j, ot_j])
        indices.extend([it_i, ot_i, it_j, it_j, ot_i, ot_j])
        indices.extend([inner_center_id, ib_i, ib_j, outer_center_id, ob_j, ob_i])

    return vertices, np.asarray(indices, dtype=np.int32).reshape(-1, 3)


def spawn_demo_mesh(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn an exact triangle mesh with standard Isaac Lab rigid/collision schemas."""
    from isaaclab.sim import schemas
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

    if cfg.collision_props is not None:
        schemas.define_collision_properties(mesh_prim_path, cfg.collision_props, stage=stage)
    if cfg.mesh_collision_props is not None:
        schemas.define_mesh_collision_properties(mesh_prim_path, cfg.mesh_collision_props, stage=stage)
    if cfg.rigid_props is not None:
        schemas.define_rigid_body_properties(prim_path, cfg.rigid_props, stage=stage)

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
        cfg.physics_material.func(material_path, cfg.physics_material)
        bind_physics_material(mesh_prim_path, material_path, stage=stage)

    return stage.GetPrimAtPath(prim_path)


def load_container_mesh(usd_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read the container USD and return its triangulated geometry in the asset's local frame.

    Meshes are concatenated, transformed into the asset frame, and fan-triangulated so sampled
    particles align with the rigid teapot collider.
    """
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"Could not open container USD for cavity sampling: {usd_path}")

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    vertices_list: list[np.ndarray] = []
    triangles_list: list[np.ndarray] = []
    vertex_offset = 0
    for prim in stage.Traverse():
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
        raise RuntimeError(f"No meshes found in container USD for cavity sampling: {usd_path}")
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

    return sim_utils.SimulationCfg(
        dt=1.0 / FPS,
        device=args_cli.device,
        gravity=(0.0, 0.0, -9.81),
        visualizer_cfgs=create_visualizer_cfgs(),
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
                air_drag=0.2,
                collider_velocity_mode="forward",
                # Grid contact avoids impulses from projecting particles out of colliders.
                project_outside_colliders=False,
            ),
            # Resolve the material at 800 Hz while refreshing the kinematic collider at 400 Hz.
            num_substeps=MPM_SUBSTEPS,
            use_cuda_graph=not args_cli.disable_cuda_graph,
        ),
    )


def create_scene_cfg():
    """Create the teapot-fill scene using declarative Isaac Lab assets."""
    from isaaclab_newton.assets import MPMObjectCfg
    from isaaclab_newton.sim.spawners.mpm import MPMParticleMaterialCfg, MPMPointsCfg

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils.configclass import configclass

    container_pos, container_rot, _ = container_pose_at_time(0.0)
    container_vertices, container_faces = load_container_mesh(args_cli.container_usd)
    fluid_points, particle_radius, particle_mass = create_fluid_particles(container_vertices, container_faces)
    bowl_vertices, bowl_faces = create_demo_bowl_mesh()

    @configclass
    class DemoMeshCfg(sim_utils.MeshCfg):
        """Demo-local exact triangle-mesh asset config."""

        func: Callable | str = clone(spawn_demo_mesh)
        vertices: list[list[float]] = MISSING
        faces: list[list[int]] = MISSING
        mesh_collision_props: sim_utils.NewtonMeshCollisionPropertiesCfg | None = None

    @configclass
    class TeapotFillSceneCfg(InteractiveSceneCfg):
        """Scene containing MPM colliders and one MPM fluid object sampled inside the teapot."""

        table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            spawn=sim_utils.CuboidCfg(
                size=(2.0 * TABLE_HALF_EXTENTS[0], 2.0 * TABLE_HALF_EXTENTS[1], 2.0 * TABLE_HALF_EXTENTS[2]),
                collision_props=sim_utils.NewtonCollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_margin=COLLIDER_MARGIN,
                ),
                physics_material=sim_utils.NewtonMaterialPropertiesCfg(
                    static_friction=TABLE_FRICTION,
                    dynamic_friction=TABLE_FRICTION,
                ),
                physics_material_path="physicsMaterial",
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=TABLE_COLOR),
                visual_material_path="visualMaterial",
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.0, 0.0, TABLE_TOP_Z - TABLE_HALF_EXTENTS[2]),
            ),
        )

        catch_bowl = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/CatchBowl",
            spawn=DemoMeshCfg(
                vertices=bowl_vertices.tolist(),
                faces=bowl_faces.tolist(),
                collision_props=sim_utils.NewtonCollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_margin=COLLIDER_MARGIN,
                ),
                mesh_collision_props=sim_utils.NewtonMeshCollisionPropertiesCfg(mesh_approximation_name="none"),
                physics_material=sim_utils.NewtonMaterialPropertiesCfg(
                    static_friction=BOWL_FRICTION,
                    dynamic_friction=BOWL_FRICTION,
                ),
                physics_material_path="physicsMaterial",
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=BOWL_COLOR),
                visual_material_path="visualMaterial",
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
                rigid_props=sim_utils.NewtonRigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
                collision_props=sim_utils.NewtonCollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_margin=CONTAINER_MARGIN,
                ),
                mesh_collision_props=sim_utils.NewtonMeshCollisionPropertiesCfg(mesh_approximation_name="none"),
                physics_material=sim_utils.NewtonMaterialPropertiesCfg(
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
                positions=fluid_points.tolist(),
                mass=particle_mass,
                radius=particle_radius,
                material=MPMParticleMaterialCfg(
                    viscosity=0.1,
                    friction=0.0,
                    damping=0.02,
                    yield_pressure=1.0e15,
                    tensile_yield_ratio=5.0,
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

        ground = AssetBaseCfg(
            prim_path="/World/Ground",
            spawn=sim_utils.GroundPlaneCfg(size=(12.0, 12.0), color=(0.30, 0.30, 0.30)),
        )

        dome_light = AssetBaseCfg(
            prim_path="/World/DomeLight",
            spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.78, 0.78, 0.78)),
        )

    return TeapotFillSceneCfg(num_envs=1, env_spacing=0.0)


def particle_count(scene) -> int:
    """Return the number of MPM particles in the scene."""
    fluid = scene["fluid"]
    return fluid.num_instances * fluid.particles_per_object


def keep_running(sim, count: int) -> bool:
    """Return whether the demo loop should continue this frame."""
    if args_cli.max_steps >= 0 and count >= args_cli.max_steps:
        return False
    return sim.is_headless_or_exist_active_visualizer()


def write_container_state(container, sim_time: float) -> None:
    """Write the scripted container pose and velocity through the rigid-object API."""
    pos, quat, twist = container_pose_at_time(sim_time)
    pose = torch.tensor([pos + quat], dtype=torch.float32, device=container.device)
    velocity = torch.tensor([twist], dtype=torch.float32, device=container.device)
    container.write_root_link_pose_to_sim_index(root_pose=pose)
    container.write_root_link_velocity_to_sim_index(root_velocity=velocity)


def run_simulator(sim, scene) -> None:
    """Run the scripted teapot-fill MPM loop."""
    sim_dt = sim.get_physics_dt()
    container = scene["container"]
    count = 0
    while keep_running(sim, count):
        write_container_state(container, count / FPS)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(sim_dt)
        if sim.is_rendering:
            sim.render()
        count += 1


def main() -> None:
    """Set up and run the Isaac Lab Newton MPM teapot-fill demo."""
    sim_cfg = create_sim_cfg()
    with launch_simulation(sim_cfg, args_cli):
        if "kit" in (args_cli.visualizer or []):
            from isaaclab_physx.renderers import IsaacRtxRendererGlobalSettingsCfg
            from isaaclab_physx.renderers.isaac_rtx_renderer_utils import apply_isaac_rtx_global_settings

            apply_isaac_rtx_global_settings(
                IsaacRtxRendererGlobalSettingsCfg(enable_translucency=True),
            )

        # Resolve after launching so Kit runs never import USD modules before
        # AppLauncher; Newton-only runs still use standalone omni.client.
        args_cli.container_usd = retrieve_file_path(args_cli.container_usd)

        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene

        sim = sim_utils.SimulationContext(sim_cfg)
        scene = InteractiveScene(create_scene_cfg())
        sim.reset()
        sim.set_camera_view(eye=CAMERA_EYE, target=CAMERA_TARGET)

        print(
            "[INFO]: Isaac Lab Newton teapot-fill MPM demo ready."
            f" Sampled {particle_count(scene)} MPM particles inside the teapot;"
            f" fill spacing {FILL_SPACING:.4g} m;"
            f" the teapot will tilt after {HOLD_TIME:.2f}s.",
            flush=True,
        )
        run_simulator(sim, scene)


if __name__ == "__main__":
    main()
