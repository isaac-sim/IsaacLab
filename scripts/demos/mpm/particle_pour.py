# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton implicit MPM particle-pour demo.

This demo shows the Isaac Lab MPM scene path:

* configure :class:`~isaaclab_newton.physics.MPMSolverCfg`;
* add fluid as an :class:`~isaaclab_newton.assets.mpm_object.MPMObject`;
* use standard Isaac Lab USD and mesh assets as MPM colliders;
* drive the pouring container through the standard rigid-object API.

.. code-block:: bash

    ./isaaclab.sh -p scripts/demos/mpm/particle_pour.py --device cuda:0 --visualizer newton
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from dataclasses import MISSING

import numpy as np
import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

DEFAULT_VOXEL_SIZE = 0.003

parser = argparse.ArgumentParser(description="Newton implicit MPM particle-pour demo.")
parser.add_argument("--max-steps", type=int, default=3000, help="Stop after this many frames; negative runs forever.")
parser.add_argument(
    "--voxel-size",
    type=float,
    default=DEFAULT_VOXEL_SIZE,
    help=f"MPM grid voxel size in meters. Defaults to {DEFAULT_VOXEL_SIZE:g}.",
)
parser.add_argument(
    "--container-usd",
    type=str,
    default=f"{ISAAC_NUCLEUS_DIR}/Props/Teapot/utah_teapot.usdc",
    help="USD asset used as the pouring container (kinematic mesh collider).",
)
add_launcher_args(parser)
parser.set_defaults(visualizer=["newton"])
args_cli = parser.parse_args()


FPS = 200
VOXEL_SIZE = float(args_cli.voxel_size)
NEWTON_VISUAL_UPDATE_FREQUENCY = 1
KIT_PARTICLE_VISUAL_UPDATE_FREQUENCY = 4

GRID_TYPE = "fixed"
GRID_PADDING = 64
MAX_ACTIVE_CELL_COUNT = 1 << 17
# With a fixed grid the solver loop is captured in a CUDA graph, so the rheology
# solve always runs exactly ``max_iterations`` (the convergence tolerance cannot
# trigger an early exit inside the graph). 100 iterations measures ~1.6x faster
# than the 250 default with an end state identical to within noise.
MPM_MAX_ITERATIONS = 100

PARTICLES_PER_CELL = 2.0
PARTICLE_DENSITY = 1000.0

PIPE_EMITTER_CENTER_XY = (0.0, 0.0)
PIPE_EMITTER_RADIUS = 0.035
PIPE_EMITTER_Z_RANGE = (0.018, 0.064)
PIPE_EMITTER_LO = (
    PIPE_EMITTER_CENTER_XY[0] - PIPE_EMITTER_RADIUS,
    PIPE_EMITTER_CENTER_XY[1] - PIPE_EMITTER_RADIUS,
    PIPE_EMITTER_Z_RANGE[0],
)
PIPE_EMITTER_HI = (
    PIPE_EMITTER_CENTER_XY[0] + PIPE_EMITTER_RADIUS,
    PIPE_EMITTER_CENTER_XY[1] + PIPE_EMITTER_RADIUS,
    PIPE_EMITTER_Z_RANGE[1],
)

COLLIDER_MARGIN = 0.0003
CONTAINER_MARGIN = 0.001
CONTAINER_FRICTION = 0.0
BOWL_MARGIN = 0.0025
BOWL_FRICTION = 0.05
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
BOWL_RIM_Z = BOWL_BASE_POS[2] + BOWL_HEIGHT
CONTAINER_BASE_POS = (-0.105, 0.0, BOWL_RIM_Z + 0.115)

BOWL_INNER_BOTTOM_RADIUS = 0.0135
BOWL_INNER_TOP_RADIUS = 0.057
BOWL_WALL_THICKNESS = 0.0075
BOWL_BOTTOM_THICKNESS = 0.0075
BOWL_COLOR = (1.0, 1.0, 1.0)
TABLE_COLOR = (0.48, 0.38, 0.26)
PARTICLE_COLOR = (0.12, 0.35, 0.78)

CAMERA_EYE = (0.0, -0.36, 0.46)
CAMERA_TARGET = (-0.01, 0.0, 0.38)


def create_visualizer_cfgs():
    """Create demo-specific visualizer configs for requested backends."""
    if "newton" not in args_cli.visualizer:
        return []

    from isaaclab_visualizers.newton import NewtonVisualizerCfg

    return [
        NewtonVisualizerCfg(
            show_particles=True,
            particle_color=PARTICLE_COLOR,
            update_frequency=NEWTON_VISUAL_UPDATE_FREQUENCY,
        )
    ]


def quat_y(angle_rad: float) -> tuple[float, float, float, float]:
    """Return an XYZW quaternion for a rotation about +Y."""
    half = 0.5 * angle_rad
    return (0.0, math.sin(half), 0.0, math.cos(half))


def smoothstep(value: float) -> float:
    """Smoothly remap a 0..1 value for the scripted tilt."""
    t = max(0.0, min(1.0, value))
    return t * t * (3.0 - 2.0 * t)


def container_pose_at_time(sim_time: float):
    """Return teapot ``(position, orientation, twist)`` for the scripted pour."""
    raw = (sim_time - HOLD_TIME) / TILT_TIME
    clamped = max(0.0, min(1.0, raw))
    alpha = smoothstep(clamped)
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


def launch_omniverse_asset_resolver():
    """Start Kit for Newton-only runs that need to resolve remote USD layers.

    The default teapot container is served from Nucleus over HTTPS, which plain
    ``pxr`` cannot resolve; Kit ships the asset resolver that can. Kit-visualizer
    runs boot Kit anyway, so this only applies to Newton-only runs.
    """
    if "kit" in args_cli.visualizer:
        return None

    from isaaclab.utils import has_kit

    if has_kit():
        return None

    from isaaclab.app import AppLauncher

    return AppLauncher(args_cli)


def create_demo_bowl_mesh(num_segments: int = 96):
    """Build one local-space open bowl mesh used by the catch-bowl collider."""
    theta = np.linspace(0.0, 2.0 * math.pi, num_segments, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    outer_bottom_radius = BOWL_INNER_BOTTOM_RADIUS + BOWL_WALL_THICKNESS
    outer_top_radius = BOWL_INNER_TOP_RADIUS + BOWL_WALL_THICKNESS

    def ring(radius: float, z: float):
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

    return vertices, np.asarray(indices, dtype=np.int32).reshape((-1, 3))


def spawn_demo_bowl_mesh(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn the demo bowl as a standard Isaac Lab mesh asset."""
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


def create_fluid_particles():
    """Return local-space MPM particle points seeded inside the teapot."""
    center_xy = np.array(PIPE_EMITTER_CENTER_XY, dtype=np.float32)
    particle_lo = np.asarray(PIPE_EMITTER_LO, dtype=np.float32)
    particle_hi = np.asarray(PIPE_EMITTER_HI, dtype=np.float32)
    resolution = np.maximum(np.ceil(PARTICLES_PER_CELL * (particle_hi - particle_lo) / VOXEL_SIZE), 1).astype(int)
    cell_size = (particle_hi - particle_lo) / resolution
    cell_volume = float(np.prod(cell_size))
    radius = float(np.max(cell_size) * 0.45)
    mass = float(cell_volume * PARTICLE_DENSITY)

    px = np.arange(int(resolution[0]) + 1) * cell_size[0]
    py = np.arange(int(resolution[1]) + 1) * cell_size[1]
    pz = np.arange(int(resolution[2]) + 1) * cell_size[2]
    points = np.stack(np.meshgrid(px, py, pz, indexing="ij")).reshape(3, -1).T

    rng = np.random.default_rng(7)
    points += (rng.random(points.shape) - 0.5) * (0.10 * np.max(cell_size))
    points += particle_lo

    normalized_xy = (points[:, :2] - center_xy) / PIPE_EMITTER_RADIUS
    points = points[np.sum(normalized_xy * normalized_xy, axis=1) < 1.0]
    if points.shape[0] == 0:
        raise RuntimeError("Particle initialization produced no particles; reduce --voxel-size.")

    return points.astype(np.float32, copy=False), radius, mass


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
                max_iterations=MPM_MAX_ITERATIONS,
                air_drag=0.2,
                collider_velocity_mode="backward",
                project_outside_colliders=True,
            ),
            use_cuda_graph=True,
            simplify_meshes=False,
        ),
    )


def preview_material(color):
    """Return a preview-surface material for Kit runs; Kit-less runs spawn no USD materials."""
    if "kit" not in args_cli.visualizer:
        return None

    import isaaclab.sim as sim_utils

    return sim_utils.PreviewSurfaceCfg(diffuse_color=color)


def create_scene_cfg():
    """Create the particle-pour scene using declarative Isaac Lab assets."""
    from isaaclab_newton.assets.mpm_object import MPMObjectCfg
    from isaaclab_newton.sim.spawners.mpm import MPMParticleMaterialCfg, MPMPointsCfg

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils.configclass import configclass

    container_pos, container_rot, _ = container_pose_at_time(0.0)
    fluid_points, particle_radius, particle_mass = create_fluid_particles()
    bowl_vertices, bowl_faces = create_demo_bowl_mesh()

    @configclass
    class DemoBowlMeshCfg(sim_utils.MeshCfg):
        """Demo-local arbitrary mesh asset config for the catch bowl."""

        func: Callable | str = clone(spawn_demo_bowl_mesh)
        vertices: list[list[float]] = MISSING
        faces: list[list[int]] = MISSING
        mesh_collision_props: sim_utils.NewtonMeshCollisionPropertiesCfg | None = None

    @configclass
    class PourSceneCfg(InteractiveSceneCfg):
        """Scene containing MPM colliders and one MPM fluid object."""

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
                visual_material=preview_material(TABLE_COLOR),
                visual_material_path="visualMaterial",
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.0, 0.0, TABLE_TOP_Z - TABLE_HALF_EXTENTS[2]),
            ),
        )

        catch_bowl = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/CatchBowl",
            spawn=DemoBowlMeshCfg(
                vertices=bowl_vertices.tolist(),
                faces=bowl_faces.tolist(),
                collision_props=sim_utils.NewtonCollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_margin=BOWL_MARGIN,
                ),
                mesh_collision_props=sim_utils.NewtonMeshCollisionPropertiesCfg(mesh_approximation_name="none"),
                physics_material=sim_utils.NewtonMaterialPropertiesCfg(
                    static_friction=BOWL_FRICTION,
                    dynamic_friction=BOWL_FRICTION,
                ),
                physics_material_path="physicsMaterial",
                visual_material=preview_material(BOWL_COLOR),
                visual_material_path="visualMaterial",
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=BOWL_BASE_POS),
        )

        # Utah Teapot: free for any use; credit as the (Modified) Utah Teapot
        # (Univ. of Utah); provided "as is", no warranty.
        container = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/PourContainer",
            spawn=sim_utils.UsdFileCfg(
                usd_path=args_cli.container_usd,
                rigid_props=sim_utils.NewtonRigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
                collision_props=sim_utils.NewtonCollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_margin=CONTAINER_MARGIN,
                ),
                physics_material=sim_utils.NewtonMaterialPropertiesCfg(
                    static_friction=CONTAINER_FRICTION,
                    dynamic_friction=CONTAINER_FRICTION,
                ),
                physics_material_path="physicsMaterial",
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
                visual_color=PARTICLE_COLOR,
                visual_update_frequency=KIT_PARTICLE_VISUAL_UPDATE_FREQUENCY,
            ),
            init_state=MPMObjectCfg.InitialStateCfg(pos=container_pos),
        )

        ground = AssetBaseCfg(
            prim_path="/World/Ground",
            spawn=sim_utils.GroundPlaneCfg(size=(1.0, 1.0), color=(0.30, 0.30, 0.30)),
        )

        dome_light = AssetBaseCfg(
            prim_path="/World/DomeLight",
            spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.78, 0.78, 0.78)),
        )

    return PourSceneCfg(num_envs=1, env_spacing=0.0)


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
    pos, quat, qd = container_pose_at_time(sim_time)
    pose = torch.tensor([tuple(pos) + tuple(quat)], dtype=torch.float32, device=container.device)
    velocity = torch.tensor([qd], dtype=torch.float32, device=container.device)
    container.write_root_link_pose_to_sim_index(root_pose=pose)
    container.write_root_link_velocity_to_sim_index(root_velocity=velocity)


def run_simulator(sim, scene) -> None:
    """Run the scripted particle-pour MPM loop."""
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
    """Set up and run the Isaac Lab Newton MPM particle-pour demo."""
    app_launcher = launch_omniverse_asset_resolver()
    try:
        sim_cfg = create_sim_cfg()
        with launch_simulation(sim_cfg, args_cli):
            import isaaclab.sim as sim_utils
            from isaaclab.scene import InteractiveScene

            sim = sim_utils.SimulationContext(sim_cfg)
            scene = InteractiveScene(create_scene_cfg())
            sim.reset()
            sim.set_camera_view(eye=CAMERA_EYE, target=CAMERA_TARGET)

            print(
                "[INFO]: Isaac Lab Newton particle-pour MPM demo ready."
                f" Spawned {particle_count(scene)} MPM particles;"
                f" voxel size {VOXEL_SIZE:.4g} m;"
                f" the teapot will tilt after {HOLD_TIME:.2f}s.",
                flush=True,
            )
            run_simulator(sim, scene)
    finally:
        if app_launcher is not None:
            app_launcher.app.close()


if __name__ == "__main__":
    main()
