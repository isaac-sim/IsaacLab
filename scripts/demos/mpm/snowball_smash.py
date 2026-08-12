# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smash a stack of rigid crates with three implicit-MPM snowballs.

Newton's elastic-plastic snow exchanges impulses with MuJoCo-Warp rigid bodies
through proxy coupling. The default bounded sparse grid supports CUDA graph
capture.

.. code-block:: bash

    uv run python scripts/demos/mpm/snowball_smash.py --device cuda:0 --visualizer newton

Use ``--grid_type fixed`` to compare against the larger fixed-grid fallback.
"""

from __future__ import annotations

import argparse
from typing import NamedTuple

import numpy as np

from isaaclab.app import add_launcher_args, launch_simulation


class SnowballSpec(NamedTuple):
    """Initial state and geometry for one snowball."""

    name: str
    asset: str
    radius: float
    center: tuple[float, float, float]
    velocity: tuple[float, float, float]
    seed: int


DEFAULT_VOXEL_SIZE = 0.05

parser = argparse.ArgumentParser(description="Newton implicit-MPM snowball-smash demo.")
parser.add_argument("--max_steps", type=int, default=900, help="Stop after this many frames; negative runs forever.")
parser.add_argument(
    "--voxel_size",
    type=float,
    default=DEFAULT_VOXEL_SIZE,
    help=f"MPM grid voxel size [m]. Defaults to {DEFAULT_VOXEL_SIZE:g}.",
)
parser.add_argument(
    "--grid_type",
    choices=("sparse", "fixed"),
    default="sparse",
    help="Use the graph-capturable bounded sparse grid or the fixed-grid fallback.",
)
parser.add_argument("--throw_speed", type=float, default=6.5, help="Horizontal snowball speed [m/s].")
parser.add_argument("--crate_mass", type=float, default=20.0, help="Mass of each crate [kg].")
parser.add_argument("--mpm_iterations", type=int, default=100, help="Maximum MPM rheology iterations.")
parser.add_argument("--rigid_substeps", type=int, default=4, help="MuJoCo-Warp substeps per coupled step.")
parser.add_argument("--disable_cuda_graph", action="store_true", help="Disable CUDA graph capture for debugging.")
add_launcher_args(parser)
parser.set_defaults(visualizer=["newton"])
args_cli = parser.parse_args()


FPS = 60.0
VOXEL_SIZE = args_cli.voxel_size
SNOW_SPACING = 0.5 * VOXEL_SIZE

# Bounded sparse topology keeps allocation stable during CUDA graph capture.
GRID_TYPE = args_cli.grid_type
MAX_ACTIVE_CELL_COUNT = 1 << 17

RIGID_ENTRY = "crates"
SNOW_ENTRY = "snow"
CRATE_BODY_PATTERN = r"/World/envs/env_.*/Crate_.*"
MPM_GROUND_BODY_PATTERN = r"/World/envs/env_.*/MPMGround"

# Soft elastic-plastic snow with pressure yielding, hardening, and dilatancy.
SNOW_DENSITY = 400.0
SNOW_COLOR = (0.88, 0.92, 0.98)

CRATE_SIZE = 0.6
CRATE_FRICTION = 0.6
CRATE_STACK = (
    ((0.0, -0.62, 0.30), (0.85, 0.18, 0.18)),
    ((0.0, 0.0, 0.30), (0.95, 0.55, 0.10)),
    ((0.0, 0.62, 0.30), (0.93, 0.80, 0.15)),
    ((0.0, -0.31, 0.92), (0.20, 0.65, 0.30)),
    ((0.0, 0.31, 0.92), (0.15, 0.40, 0.80)),
    ((0.0, 0.0, 1.54), (0.55, 0.25, 0.70)),
)

# Name, scene key, radius, center, initial velocity, and RNG seed.
SNOWBALLS = (
    SnowballSpec("SnowballFirst", "snowball_first", 0.25, (-2.8, 0.0, 1.2), (args_cli.throw_speed, 0.0, 1.8), 11),
    SnowballSpec("SnowballSecond", "snowball_second", 0.22, (-3.8, -0.5, 1.8), (args_cli.throw_speed, 0.9, 1.8), 12),
    SnowballSpec("SnowballThird", "snowball_third", 0.20, (-4.8, 0.5, 2.0), (args_cli.throw_speed, -0.8, 2.2), 13),
)

CAMERA_EYE = (-4.2, -3.2, 2.2)
CAMERA_TARGET = (0.0, 0.0, 0.9)


def create_visualizer_cfgs():
    """Create demo-specific visualizer configs for the requested backends."""
    if not any(v in (args_cli.visualizer or []) for v in ("newton", "newton_gl", "newton_rtx")):
        return []

    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg

    cfg_type = NewtonRTXVisualizerCfg if args_cli.visualizer == ["newton_rtx"] else NewtonGLVisualizerCfg
    return [
        cfg_type(
            show_particles=True,
            particle_color=SNOW_COLOR,
        )
    ]


def create_snowball_points(radius: float, seed: int) -> np.ndarray:
    """Create jittered local-space sphere points."""
    dim = int(2.0 * radius / SNOW_SPACING) + 1
    axis = (np.arange(dim) - 0.5 * (dim - 1)) * SNOW_SPACING
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij")).reshape(3, -1).T
    points = points[np.linalg.norm(points, axis=1) <= radius]
    if points.shape[0] == 0:
        raise RuntimeError("Snowball generation produced no particles; reduce --voxel_size.")

    rng = np.random.default_rng(seed)
    points += (rng.random(points.shape) - 0.5) * SNOW_SPACING

    return points.astype(np.float32)


def create_sim_cfg():
    """Create the proxy-coupled Newton simulation configuration."""
    from isaaclab_newton.physics import MJWarpSolverCfg, MPMSolverCfg, NewtonCfg

    import isaaclab.sim as sim_utils

    from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg

    if not str(args_cli.device).startswith("cuda"):
        raise RuntimeError("The graph-captured implicit-MPM coupling demo requires a CUDA device.")

    is_sparse_grid = GRID_TYPE == "sparse"
    solver_cfg = CouplerProxyCfg(
        entries=[
            CouplerEntryCfg(
                name=RIGID_ENTRY,
                # MuJoCo handles crate-crate and crate-ground contact. The
                # coupled proxy below exposes only the crates to MPM.
                solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=True, njmax=500, nconmax=400),
                bodies=[CRATE_BODY_PATTERN],
                include_static_shapes=True,
                substeps=args_cli.rigid_substeps,
            ),
            CouplerEntryCfg(
                name=SNOW_ENTRY,
                solver_cfg=MPMSolverCfg(
                    voxel_size=VOXEL_SIZE,
                    grid_type=GRID_TYPE,
                    grid_padding=0 if is_sparse_grid else 96,
                    max_active_cell_count=MAX_ACTIVE_CELL_COUNT,
                    max_leaf_node_count=(1 << 14) if is_sparse_grid else -1,
                    max_lower_node_count=(1 << 10) if is_sparse_grid else -1,
                    max_upper_node_count=(1 << 8) if is_sparse_grid else -1,
                    strain_basis="P0",
                    velocity_basis="Q1",
                    collider_basis="S2",
                    transfer_scheme="apic",
                    max_iterations=args_cli.mpm_iterations,
                    tolerance=1.0e-4,
                    solver="jacobi",
                    warmstart_mode="none",
                    air_drag=0.2,
                    collider_velocity_mode="forward",
                    project_outside_colliders=False,
                ),
                bodies=[MPM_GROUND_BODY_PATTERN],
                all_particles=True,
                include_static_shapes=False,
                include_child_joints=False,
                in_place=True,
            ),
        ],
        proxies=[
            CouplerProxyMappingCfg(
                source=RIGID_ENTRY,
                destination=SNOW_ENTRY,
                bodies=[CRATE_BODY_PATTERN],
                mode="lagged",
                mass_scale=1.0,
                # Implicit MPM resolves proxy collision internally.
                collision_pipeline=None,
            )
        ],
        iterations=1,
    )

    return sim_utils.SimulationCfg(
        dt=1.0 / FPS,
        device=args_cli.device,
        gravity=(0.0, 0.0, -9.81),
        visualizer_cfgs=create_visualizer_cfgs(),
        physics=NewtonCfg(
            solver_cfg=solver_cfg,
            num_substeps=1,
            use_cuda_graph=not args_cli.disable_cuda_graph,
        ),
    )


def create_scene_cfg():
    """Create the declarative snowball, crate, and ground scene."""
    from isaaclab_newton.assets import MPMObjectCfg
    from isaaclab_newton.sim.spawners.mpm import MPMParticleMaterialCfg, MPMPointsCfg

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.utils.configclass import configclass

    particle_mass = SNOW_SPACING**3 * SNOW_DENSITY
    particle_radius = 0.5 * SNOW_SPACING

    def crate_cfg(index: int) -> RigidObjectCfg:
        center, color = CRATE_STACK[index]
        return RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Crate_{index}",
            spawn=sim_utils.CuboidCfg(
                size=(CRATE_SIZE, CRATE_SIZE, CRATE_SIZE),
                rigid_props=sim_utils.NewtonRigidBodyPropertiesCfg(rigid_body_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=args_cli.crate_mass),
                collision_props=sim_utils.NewtonCollisionPropertiesCfg(collision_enabled=True),
                physics_material=sim_utils.NewtonMaterialPropertiesCfg(
                    static_friction=CRATE_FRICTION,
                    dynamic_friction=CRATE_FRICTION,
                ),
                physics_material_path="physicsMaterial",
                visual_material=(
                    sim_utils.PreviewSurfaceCfg(diffuse_color=color) if "kit" in (args_cli.visualizer or []) else None
                ),
                visual_material_path="visualMaterial",
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=center),
        )

    def snowball_cfg(ball: SnowballSpec) -> MPMObjectCfg:
        points = create_snowball_points(ball.radius, ball.seed)
        return MPMObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{ball.name}",
            spawn=MPMPointsCfg(
                positions=points.tolist(),
                velocities=[list(ball.velocity)] * points.shape[0],
                mass=particle_mass,
                radius=particle_radius,
                material=MPMParticleMaterialCfg(
                    density=SNOW_DENSITY,
                    young_modulus=1.4e6,
                    poisson_ratio=0.3,
                    friction=0.5,
                    damping=0.01,
                    yield_pressure=1.4e6,
                    tensile_yield_ratio=0.2,
                    hardening=5.0,
                    dilatancy=1.0,
                ),
                visual_color=SNOW_COLOR,
            ),
            init_state=MPMObjectCfg.InitialStateCfg(pos=ball.center),
        )

    @configclass
    class SnowballSmashSceneCfg(InteractiveSceneCfg):
        """Scene with rigid crates and independently owned rigid/MPM ground collision."""

        # This visible static plane belongs to the rigid entry, so MuJoCo can
        # resolve crate-ground contact without duplicating the ground shape in
        # the MPM model view.
        ground = AssetBaseCfg(
            prim_path="/World/Ground",
            spawn=sim_utils.GroundPlaneCfg(size=(12.0, 12.0), color=(0.32, 0.34, 0.38)),
        )

        # The co-located hidden kinematic slab belongs only to the MPM entry.
        # This mirrors the shared geometry without assigning one Newton shape
        # to two coupled solver entries.
        mpm_ground = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/MPMGround",
            spawn=sim_utils.CuboidCfg(
                size=(12.0, 12.0, 0.10),
                rigid_props=sim_utils.NewtonRigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=True,
                ),
                collision_props=sim_utils.NewtonCollisionPropertiesCfg(collision_enabled=True),
                physics_material=sim_utils.NewtonMaterialPropertiesCfg(
                    static_friction=CRATE_FRICTION,
                    dynamic_friction=CRATE_FRICTION,
                ),
                visible=False,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
        )

        dome_light = AssetBaseCfg(
            prim_path="/World/DomeLight",
            spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.78, 0.78, 0.78)),
        )

        crate_0 = crate_cfg(0)
        crate_1 = crate_cfg(1)
        crate_2 = crate_cfg(2)
        crate_3 = crate_cfg(3)
        crate_4 = crate_cfg(4)
        crate_5 = crate_cfg(5)

        snowball_first = snowball_cfg(SNOWBALLS[0])
        snowball_second = snowball_cfg(SNOWBALLS[1])
        snowball_third = snowball_cfg(SNOWBALLS[2])

    return SnowballSmashSceneCfg(num_envs=1, env_spacing=0.0)


def particle_count(scene) -> int:
    """Return the total number of snow particles."""
    return sum(scene[ball.asset].num_instances * scene[ball.asset].particles_per_object for ball in SNOWBALLS)


def keep_running(sim, count: int) -> bool:
    """Return whether the demo loop should continue."""
    if args_cli.max_steps >= 0 and count >= args_cli.max_steps:
        return False
    return sim.is_headless_or_exist_active_visualizer()


def run_simulator(sim, scene) -> None:
    """Run the coupled snowball-smash loop."""
    sim_dt = sim.get_physics_dt()
    count = 0
    while keep_running(sim, count):
        sim.step(render=False)
        scene.update(sim_dt)
        if sim.is_rendering:
            sim.render()
        count += 1


def main() -> None:
    """Set up and run the Newton MPM snowball-smash demo."""
    sim_cfg = create_sim_cfg()
    with launch_simulation(sim_cfg, args_cli):
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene

        sim = sim_utils.SimulationContext(sim_cfg)
        scene = InteractiveScene(create_scene_cfg())
        sim.reset()
        sim.set_camera_view(eye=CAMERA_EYE, target=CAMERA_TARGET)

        print(
            "[INFO]: Isaac Lab Newton snowball-smash demo ready."
            f" Spawned {particle_count(scene)} particles across {len(SNOWBALLS)} snowballs;"
            f" grid={GRID_TYPE} ({MAX_ACTIVE_CELL_COUNT} active-cell capacity);"
            f" CUDA graph={'enabled' if sim_cfg.physics.use_cuda_graph else 'disabled'}.",
            flush=True,
        )
        run_simulator(sim, scene)


if __name__ == "__main__":
    main()
