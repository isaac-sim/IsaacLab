# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare MJWarp rigid bodies with shape-matched, nearly rigid MPM objects.

This demo extends the rigid-ball experiment from Figure 17 of Daviet's
mixed-MPM paper. A sphere, cube, and capsule descend matched inclined lanes:
MJWarp rigid bodies are on the left and MPM particle discretizations are on the
right. Corresponding objects use the same authored color and discretized mass.

.. code-block:: bash

    uv run python scripts/demos/mpm/tuning/rigid_body_equivalence.py \
      --visualizer kit
"""

from __future__ import annotations

import argparse
import math
from typing import NamedTuple

import numpy as np

from isaaclab.app import add_launcher_args, launch_simulation


class ShapeSpec(NamedTuple):
    """Geometry and presentation values for one matched object pair."""

    name: str
    color: tuple[float, float, float]
    lane_offset_x: float
    friction: float
    sphere_radius: float = 0.0
    cuboid_size: tuple[float, float, float] = (0.0, 0.0, 0.0)
    capsule_radius: float = 0.0
    capsule_height: float = 0.0


SHAPES = (
    ShapeSpec("sphere", (0.10, 0.42, 0.95), -1.30, 0.68, sphere_radius=0.50),
    ShapeSpec("cube", (0.96, 0.42, 0.08), 0.0, 0.12, cuboid_size=(0.85, 0.85, 0.85)),
    ShapeSpec("capsule", (0.12, 0.72, 0.38), 1.30, 0.68, capsule_radius=0.30, capsule_height=1.30),
)

SIMULATION_HZ = 240
DEFAULT_VOXEL_SIZE = 0.03
PARTICLES_PER_VOXEL_AXIS = 3.0
DENSITY = 1600.0
STRAIN_BASIS = "P0"
VELOCITY_BASIS = "Q1"
RIGID_GROUP_X = -2.85
MPM_GROUP_X = 2.85
RAMP_CENTER_Y = 0.0
RAMP_CENTER_Z = 2.55
RAMP_SIZE = (5.20, 6.40, 0.16)
RAMP_ANGLE_RAD = math.radians(38.0)
RAMP_ROTATION = (math.sin(0.5 * RAMP_ANGLE_RAD), 0.0, 0.0, math.cos(0.5 * RAMP_ANGLE_RAD))
START_Y = 1.70
INITIAL_CLEARANCE = 0.08
INITIAL_SPEED = 0.0
RIGID_BODY_PATTERN = r"/World/envs/env_.*/Rigid_.*"
CAMERA_EYE = (0.0, -26.0, 6.55)
CAMERA_TARGET = (0.0, -2.5, 2.15)
CAMERA_FOCAL_LENGTH = 18.0

parser = argparse.ArgumentParser(description="MJWarp rigid bodies versus nearly rigid Newton MPM objects.")
parser.add_argument("--max_steps", type=int, default=-1, help="Stop after this many steps; negative runs forever.")
parser.add_argument(
    "--voxel_size",
    type=float,
    default=DEFAULT_VOXEL_SIZE,
    help=f"MPM grid voxel size [m]. Defaults to {DEFAULT_VOXEL_SIZE:g}.",
)
parser.add_argument(
    "--particle_jitter_fraction",
    type=float,
    default=0.12,
    help="Maximum per-axis particle jitter as a fraction of particle spacing in [0, 0.5].",
)
parser.add_argument("--particle_seed", type=int, default=42, help="Seed for deterministic particle jitter.")
parser.add_argument(
    "--mpm_young_modulus",
    type=float,
    default=1.0e20,
    help="Finite approximation of the paper's infinite MPM stiffness [Pa].",
)
parser.add_argument(
    "--mpm_yield_pressure",
    type=float,
    default=1.0e20,
    help="Pressure yield threshold for the nearly rigid MPM material [Pa].",
)
parser.add_argument(
    "--solver_iterations",
    type=int,
    default=240,
    help="Maximum implicit MPM solver iterations per physics step. Defaults to the paper's rigid-ball range.",
)
parser.add_argument("--mpm_substeps", type=int, default=8, help="Implicit MPM substeps per simulation step.")
parser.add_argument("--rigid_substeps", type=int, default=4, help="MJWarp substeps per simulation step.")
parser.add_argument("--disable_cuda_graph", action="store_true", help="Disable Newton CUDA graphs for debugging.")
add_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli = parser.parse_args()

if not np.isfinite(args_cli.voxel_size) or args_cli.voxel_size <= 0.0:
    parser.error("--voxel_size must be finite and positive.")
if not np.isfinite(args_cli.particle_jitter_fraction) or not 0.0 <= args_cli.particle_jitter_fraction <= 0.5:
    parser.error("--particle_jitter_fraction must be finite and in [0, 0.5].")
if not np.isfinite(args_cli.mpm_young_modulus) or args_cli.mpm_young_modulus <= 0.0:
    parser.error("--mpm_young_modulus must be finite and positive.")
if not np.isfinite(args_cli.mpm_yield_pressure) or args_cli.mpm_yield_pressure <= 0.0:
    parser.error("--mpm_yield_pressure must be finite and positive.")
if args_cli.solver_iterations <= 0:
    parser.error("--solver_iterations must be positive.")
if args_cli.mpm_substeps <= 0:
    parser.error("--mpm_substeps must be positive.")
if args_cli.rigid_substeps <= 0:
    parser.error("--rigid_substeps must be positive.")

PARTICLE_SPACING = args_cli.voxel_size / PARTICLES_PER_VOXEL_AXIS
PARTICLE_RADIUS = 0.5 * PARTICLE_SPACING
PARTICLE_MASS = PARTICLE_SPACING**3 * DENSITY


def create_shape_points(shape: ShapeSpec) -> np.ndarray:
    """Create deterministic cell-centered particles inside one analytic shape."""
    spacing = PARTICLE_SPACING
    if shape.sphere_radius > 0.0:
        radius = shape.sphere_radius
        axis = np.arange(-radius + 0.5 * spacing, radius, spacing)
        points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
        points = points[np.linalg.norm(points, axis=1) <= radius - 0.20 * spacing]
    elif any(shape.cuboid_size):
        axes = tuple(np.arange(-0.5 * extent + 0.5 * spacing, 0.5 * extent, spacing) for extent in shape.cuboid_size)
        points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    elif shape.capsule_radius > 0.0:
        radius = shape.capsule_radius
        half_segment = 0.5 * shape.capsule_height - radius
        x_axis = np.arange(-0.5 * shape.capsule_height + 0.5 * spacing, 0.5 * shape.capsule_height, spacing)
        yz_axis = np.arange(-radius + 0.5 * spacing, radius, spacing)
        points = np.stack(np.meshgrid(x_axis, yz_axis, yz_axis, indexing="ij"), axis=-1).reshape(-1, 3)
        axial_distance = np.maximum(np.abs(points[:, 0]) - half_segment, 0.0)
        radial_distance = np.sqrt(points[:, 1] ** 2 + points[:, 2] ** 2)
        points = points[np.sqrt(axial_distance**2 + radial_distance**2) <= radius - 0.20 * spacing]
    else:
        raise ValueError(f"Shape {shape.name!r} has no geometry.")

    if points.shape[0] == 0:
        raise RuntimeError(f"Particle generation for {shape.name!r} produced no points.")
    jitter = args_cli.particle_jitter_fraction * spacing
    seed_offset = sum(ord(character) for character in shape.name)
    points += np.random.default_rng(args_cli.particle_seed + seed_offset).uniform(-jitter, jitter, points.shape)
    if shape.sphere_radius > 0.0:
        target_half_extents = np.full(3, shape.sphere_radius, dtype=np.float32)
    elif any(shape.cuboid_size):
        target_half_extents = 0.5 * np.asarray(shape.cuboid_size, dtype=np.float32)
    else:
        target_half_extents = np.asarray(
            (0.5 * shape.capsule_height, shape.capsule_radius, shape.capsule_radius), dtype=np.float32
        )
    # Account for the rendered and colliding particle radius so the outside
    # bounds match the corresponding analytic rigid primitive exactly.
    target_center_extents = target_half_extents - PARTICLE_RADIUS
    points *= target_center_extents / np.abs(points).max(axis=0)
    return points.astype(np.float32)


SHAPE_POINTS = {shape.name: create_shape_points(shape) for shape in SHAPES}


def shape_half_height(shape: ShapeSpec) -> float:
    """Return one shape's local vertical half extent [m]."""
    if shape.sphere_radius > 0.0:
        return shape.sphere_radius
    if any(shape.cuboid_size):
        return 0.5 * shape.cuboid_size[2]
    return shape.capsule_radius


def ramp_surface_height(y_position: float) -> float:
    """Return the upper ramp surface height at one world-space y coordinate [m]."""
    return (
        RAMP_CENTER_Z
        + math.tan(RAMP_ANGLE_RAD) * (y_position - RAMP_CENTER_Y)
        + 0.5 * RAMP_SIZE[2] / math.cos(RAMP_ANGLE_RAD)
    )


def initial_center_height(shape: ShapeSpec) -> float:
    """Place a shape fully above the incline with a small visible gap [m]."""
    if shape.name == "cube":
        half_y = 0.5 * shape.cuboid_size[1]
        return ramp_surface_height(START_Y + half_y) + shape_half_height(shape) + INITIAL_CLEARANCE
    return ramp_surface_height(START_Y) + shape_half_height(shape) / math.cos(RAMP_ANGLE_RAD) + INITIAL_CLEARANCE


def create_initial_velocities(shape: ShapeSpec, points: np.ndarray) -> np.ndarray:
    """Create matched forward and rolling particle velocities [m/s]."""
    velocities = np.zeros_like(points)
    velocities[:, 1] = -INITIAL_SPEED * math.cos(RAMP_ANGLE_RAD)
    velocities[:, 2] = -INITIAL_SPEED * math.sin(RAMP_ANGLE_RAD)
    if shape.name in {"sphere", "capsule"}:
        radius = shape.sphere_radius or shape.capsule_radius
        angular_velocity = np.array((INITIAL_SPEED / radius, 0.0, 0.0), dtype=np.float32)
        velocities += np.cross(angular_velocity, points)
    elif shape.name == "cube":
        velocities += np.cross(np.array((INITIAL_SPEED, 0.0, 0.0), dtype=np.float32), points)
    return velocities


def create_visualizer_cfgs():
    """Create the requested Kit or Newton visualizer configuration."""
    requested = args_cli.visualizer or []
    cfgs = []
    if "kit" in requested:
        from isaaclab_visualizers.kit import KitVisualizerCfg

        cfgs.append(KitVisualizerCfg(eye=CAMERA_EYE, lookat=CAMERA_TARGET, focal_length=CAMERA_FOCAL_LENGTH))
    if not {"newton", "newton_gl", "newton_rtx"}.intersection(requested):
        return cfgs

    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg

    cfg_type = NewtonRTXVisualizerCfg if requested == ["newton_rtx"] else NewtonGLVisualizerCfg
    visualizer_kwargs = {}
    if cfg_type is NewtonRTXVisualizerCfg:
        visualizer_kwargs = {"rtx_environment": "studio"}
    cfgs.append(
        cfg_type(
            eye=CAMERA_EYE,
            lookat=CAMERA_TARGET,
            streaming_view=False,
            show_particles=True,
            update_frequency=1,
            **visualizer_kwargs,
        )
    )
    return cfgs


def create_sim_cfg():
    """Create independent MJWarp and implicit-MPM entries in one Newton model."""
    from isaaclab_newton.physics import MJWarpSolverCfg, MPMSolverCfg, NewtonCfg

    import isaaclab.sim as sim_utils

    from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg

    solver_cfg = CouplerProxyCfg(
        entries=[
            CouplerEntryCfg(
                name="rigid",
                solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=False, njmax=128),
                bodies=[RIGID_BODY_PATTERN],
                include_static_shapes=True,
                substeps=args_cli.rigid_substeps,
            ),
            CouplerEntryCfg(
                name="mpm",
                solver_cfg=MPMSolverCfg(
                    voxel_size=args_cli.voxel_size,
                    grid_type="sparse",
                    grid_padding=4,
                    max_iterations=args_cli.solver_iterations,
                    tolerance=1.0e-4,
                    solver="auto",
                    warmstart_mode="auto",
                    transfer_scheme="apic",
                    integration_scheme="pic",
                    strain_basis=STRAIN_BASIS,
                    velocity_basis=VELOCITY_BASIS,
                    collider_basis="S2",
                ),
                all_particles=True,
                in_place=True,
                substeps=args_cli.mpm_substeps,
            ),
        ],
        # The lanes do not overlap. This mapping lets both entries share the
        # outer contact pipeline without introducing visible cross-interaction.
        proxies=[
            CouplerProxyMappingCfg(
                source="rigid",
                destination="mpm",
                bodies=[RIGID_BODY_PATTERN],
                mode="lagged",
                collision_pipeline=None,
            )
        ],
        iterations=1,
    )
    return sim_utils.SimulationCfg(
        dt=1.0 / SIMULATION_HZ,
        device=args_cli.device,
        gravity=(0.0, 0.0, -9.81),
        visualizer_cfgs=create_visualizer_cfgs(),
        physics=NewtonCfg(solver_cfg=solver_cfg, use_cuda_graph=not args_cli.disable_cuda_graph),
    )


def create_scene_cfg():
    """Create matched rigid and MPM lanes from procedural primitives."""
    from isaaclab_newton.assets import MPMObjectCfg
    from isaaclab_newton.sim.spawners.mpm import MPMParticleMaterialCfg, MPMPointsCfg

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.utils.configclass import configclass

    def rigid_spawn(shape: ShapeSpec):
        common = {
            "rigid_props": sim_utils.RigidBodyPropertiesCfg(),
            "mass_props": sim_utils.MassPropertiesCfg(mass=len(SHAPE_POINTS[shape.name]) * PARTICLE_MASS),
            "collision_props": sim_utils.NewtonCollisionPropertiesCfg(contact_margin=0.5 * PARTICLE_SPACING),
            "physics_material": sim_utils.NewtonMaterialPropertiesCfg(
                static_friction=shape.friction,
                dynamic_friction=0.8 * shape.friction,
                rolling_friction=0.005,
            ),
            "visual_material": sim_utils.PreviewSurfaceCfg(
                diffuse_color=shape.color,
                roughness=0.24,
                metallic=0.08,
            ),
        }
        if shape.name == "sphere":
            return sim_utils.SphereCfg(radius=shape.sphere_radius, **common)
        if shape.name == "cube":
            return sim_utils.CuboidCfg(size=shape.cuboid_size, **common)
        # USD capsule height is the cylinder spine length, whereas ShapeSpec
        # stores the desired overall end-to-end size.
        spine_height = shape.capsule_height - 2.0 * shape.capsule_radius
        return sim_utils.CapsuleCfg(radius=shape.capsule_radius, height=spine_height, axis="X", **common)

    def ramp_cfg(group_x: float) -> AssetBaseCfg:
        return AssetBaseCfg(
            prim_path=f"/World/Ramp_{'Left' if group_x < 0.0 else 'Right'}",
            spawn=sim_utils.CuboidCfg(
                size=RAMP_SIZE,
                collision_props=sim_utils.NewtonCollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_margin=0.5 * PARTICLE_SPACING,
                ),
                physics_material=sim_utils.NewtonMaterialPropertiesCfg(
                    static_friction=0.52,
                    dynamic_friction=0.42,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.16, 0.18, 0.22),
                    roughness=0.32,
                    metallic=0.12,
                ),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(group_x, RAMP_CENTER_Y, RAMP_CENTER_Z), rot=RAMP_ROTATION),
        )

    rigid_object_cfgs = {}
    mpm_objects = {}
    for shape in SHAPES:
        position = (
            RIGID_GROUP_X + shape.lane_offset_x,
            START_Y,
            initial_center_height(shape),
        )
        rigid_object_cfgs[shape.name] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Rigid_{shape.name.title()}",
            spawn=rigid_spawn(shape),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=position,
                lin_vel=(0.0, -INITIAL_SPEED, 0.0),
                ang_vel=(
                    INITIAL_SPEED / (shape.sphere_radius or shape.capsule_radius)
                    if shape.name in {"sphere", "capsule"}
                    else 0.0,
                    0.0,
                    0.0,
                ),
            ),
        )

        points = SHAPE_POINTS[shape.name]
        mpm_objects[shape.name] = MPMObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/MPM_{shape.name.title()}",
            spawn=MPMPointsCfg(
                positions=points.tolist(),
                velocities=create_initial_velocities(shape, points).tolist(),
                mass=PARTICLE_MASS,
                radius=PARTICLE_RADIUS,
                material=MPMParticleMaterialCfg(
                    density=DENSITY,
                    young_modulus=args_cli.mpm_young_modulus,
                    poisson_ratio=0.3,
                    friction=shape.friction,
                    yield_pressure=args_cli.mpm_yield_pressure,
                    tensile_yield_ratio=1.0,
                    yield_stress=0.0,
                ),
                visual_color=shape.color,
            ),
            init_state=MPMObjectCfg.InitialStateCfg(
                pos=(
                    MPM_GROUP_X + shape.lane_offset_x,
                    START_Y,
                    initial_center_height(shape),
                )
            ),
        )

    @configclass
    class RigidMPMComparisonSceneCfg(InteractiveSceneCfg):
        """Scene containing matched MJWarp and nearly rigid MPM objects."""

        ground = AssetBaseCfg(
            prim_path="/World/Ground",
            spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
        )
        dome_light = AssetBaseCfg(
            prim_path="/World/DomeLight",
            spawn=sim_utils.DomeLightCfg(intensity=2200.0, color=(0.82, 0.86, 0.92)),
        )
        left_ramp = ramp_cfg(RIGID_GROUP_X)
        right_ramp = ramp_cfg(MPM_GROUP_X)
        divider = AssetBaseCfg(
            prim_path="/World/Divider",
            spawn=sim_utils.CuboidCfg(
                size=(0.08, 6.8, 0.18),
                collision_props=sim_utils.NewtonCollisionPropertiesCfg(collision_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.68, 0.72, 0.78), roughness=0.35, metallic=0.35
                ),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -0.05, 0.09)),
        )

        rigid_objects = RigidObjectCollectionCfg(rigid_objects=rigid_object_cfgs)
        mpm_sphere = mpm_objects["sphere"]
        mpm_cube = mpm_objects["cube"]
        mpm_capsule = mpm_objects["capsule"]

    return RigidMPMComparisonSceneCfg(num_envs=1, env_spacing=0.0)


def run_simulator(sim, scene) -> None:
    """Run until the viewer closes or the step limit is reached."""
    sim_dt = sim.get_physics_dt()
    step_count = 0
    while sim.is_headless_or_exist_active_visualizer() and (args_cli.max_steps < 0 or step_count < args_cli.max_steps):
        sim.step(render=False)
        scene.update(sim_dt)
        if sim.is_rendering:
            sim.render()
        step_count += 1


def main() -> None:
    """Launch the rigid-body versus nearly rigid MPM comparison."""
    sim_cfg = create_sim_cfg()
    with launch_simulation(sim_cfg, args_cli):
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene

        sim = sim_utils.SimulationContext(sim_cfg)
        scene = InteractiveScene(create_scene_cfg())
        sim.reset()
        sim.set_camera_view(eye=CAMERA_EYE, target=CAMERA_TARGET)
        particle_count = sum(len(points) for points in SHAPE_POINTS.values())
        print(
            f"[INFO]: Rigid-versus-MPM comparison ready: {particle_count} MPM particles across {len(SHAPES)} pairs.",
            flush=True,
        )
        print("[INFO]: Left lane = MJWarp rigid bodies; right lane = nearly rigid MPM.", flush=True)
        print(
            f"[INFO]: MPM settings: voxel={1000 * args_cli.voxel_size:g} mm, "
            f"{PARTICLES_PER_VOXEL_AXIS:g} particles/voxel axis, E={args_cli.mpm_young_modulus:.0e} Pa, "
            f"yield pressure={args_cli.mpm_yield_pressure:.0e} Pa.",
            flush=True,
        )
        run_simulator(sim, scene)


if __name__ == "__main__":
    main()
