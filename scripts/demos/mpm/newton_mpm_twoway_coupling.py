# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Drag three rigid spheres in a bath of Newton implicit-MPM sand.

This Isaac Lab port of Newton's ``mpm_twoway_coupling`` example uses a proxy
coupler to expose dynamic rigid spheres as MPM colliders and feed the resulting
impulses back into the rigid-body solver.

.. code-block:: bash

    uv run python scripts/demos/mpm/newton_mpm_twoway_coupling.py

The spheres roll through three V-shaped chutes into the bath. Right-click and
drag any sphere to apply an interactive force.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from functools import partial

from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg

from pxr import Gf, Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Newton rigid-sphere and MPM-sand two-way coupling demo.")
parser.add_argument("--max_steps", type=int, default=-1, help="Stop after this many frames; negative runs forever.")
parser.add_argument("--voxel_size", type=float, default=0.08, help="MPM grid voxel size [m].")
parser.add_argument("--rigid_substeps", type=int, default=4, help="Rigid-solver substeps per coupled step.")
add_launcher_args(parser)
parser.set_defaults(visualizer=["newton_gl"])
args_cli = parser.parse_args()


FPS = 100.0
GRAVITY = (0.0, 0.0, -9.81)
PARTICLES_PER_VOXEL_AXIS = 2.0
PARTICLE_SPACING = args_cli.voxel_size / PARTICLES_PER_VOXEL_AXIS
COLLIDER_MARGIN = 0.5 * args_cli.voxel_size
PARTICLE_COLOR = (0.7, 0.6, 0.4)
SPHERE_BODY_PATTERN = r"/World/envs/env_.*/Sphere_[0-9]+"
SPHERE_RADIUS = 0.30
SPHERE_MASS = 450.0
SPHERE_COLORS = ((0.20, 0.45, 0.85), (0.85, 0.25, 0.20), (0.25, 0.70, 0.30))
SPHERE_POSITIONS = ((-3.392, 0.0, 2.167), (0.0, 2.872, 2.167), (3.392, 0.0, 2.167))

BATH_INTERIOR_SIZE = (3.6, 2.6)
BATH_WALL_HEIGHT = 1.2
BATH_WALL_THICKNESS = 0.15
SAND_LOWER = (-1.65, -1.15, 0.05)
SAND_UPPER = (1.65, 1.15, 0.72)

CHUTE_PANEL_SIZE = (2.4, 0.62, 0.08)
CHUTE_COLOR = (0.25, 0.28, 0.32)
# Paired poses form the left, back, and right V-shaped chutes.
CHUTE_PANEL_POSES = (
    ((-2.933, -0.274, 1.771), (-0.23957, 0.13504, 0.03367, 0.96085)),
    ((-2.933, 0.274, 1.771), (0.23957, 0.13504, -0.03367, 0.96085)),
    ((-0.274, 2.413, 1.771), (-0.07391, 0.26489, -0.65562, 0.70323)),
    ((0.274, 2.413, 1.771), (0.26489, -0.07391, -0.70323, 0.65562)),
    ((2.933, 0.274, 1.771), (0.13504, 0.23957, -0.96085, 0.03367)),
    ((2.933, -0.274, 1.771), (-0.13504, 0.23957, 0.96085, 0.03367)),
)


@sim_utils.clone
def _spawn_colored_shape(
    prim_path: str,
    cfg: sim_utils.SpawnerCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    *,
    spawn_func: Callable[..., Usd.Prim],
    color: tuple[float, float, float],
) -> Usd.Prim:
    """Spawn a shape with a display color understood by the Newton viewer."""
    prim = spawn_func(prim_path, cfg, translation, orientation)
    mesh = UsdGeom.Gprim(prim.GetStage().GetPrimAtPath(f"{prim_path}/geometry/mesh"))
    mesh.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
    return prim


def create_visualizer_cfgs():
    """Create the demo-specific Newton visualizer configuration."""
    requested = args_cli.visualizer or []
    if not {"newton", "newton_gl", "newton_rtx"}.intersection(requested):
        return []

    cfg_type = NewtonRTXVisualizerCfg if requested == ["newton_rtx"] else NewtonGLVisualizerCfg
    return [
        cfg_type(
            streaming_view=False,
            show_particles=True,
            particle_color=PARTICLE_COLOR,
            update_frequency=1,
        )
    ]


def create_sim_cfg():
    """Create the proxy-coupled MJWarp and MPM simulation configuration."""
    from isaaclab_newton.physics import MJWarpSolverCfg, MPMSolverCfg, NewtonCfg

    from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg

    solver_cfg = CouplerProxyCfg(
        entries=[
            CouplerEntryCfg(
                name="rigid",
                solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=False, njmax=128),
                bodies=[SPHERE_BODY_PATTERN],
                include_static_shapes=True,
                substeps=args_cli.rigid_substeps,
            ),
            CouplerEntryCfg(
                name="mpm",
                solver_cfg=MPMSolverCfg(
                    voxel_size=args_cli.voxel_size,
                    grid_type="fixed",
                    grid_padding=50,
                    max_active_cell_count=1 << 16,
                    strain_basis="P0",
                    max_iterations=50,
                    critical_fraction=0.0,
                ),
                all_particles=True,
                in_place=True,
            ),
        ],
        proxies=[
            CouplerProxyMappingCfg(
                source="rigid",
                destination="mpm",
                bodies=[SPHERE_BODY_PATTERN],
                mode="lagged",
                collision_pipeline=None,
            )
        ],
        iterations=1,
    )
    return sim_utils.SimulationCfg(
        dt=1.0 / FPS,
        device=args_cli.device,
        gravity=GRAVITY,
        visualizer_cfgs=create_visualizer_cfgs(),
        physics=NewtonCfg(solver_cfg=solver_cfg),
    )


def create_scene_cfg():
    """Create the declarative rigid-sphere and granular-bath scene."""
    from isaaclab_newton.assets.mpm_object import MPMObjectCfg
    from isaaclab_newton.sim.spawners.mpm import MPMGridCfg, MPMParticleMaterialCfg

    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.utils.configclass import configclass

    def bath_collider(
        prim_path: str,
        size: tuple[float, float, float],
        position: tuple[float, float, float],
        orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        color: tuple[float, float, float] | None = None,
    ) -> AssetBaseCfg:
        return AssetBaseCfg(
            prim_path=prim_path,
            spawn=sim_utils.CuboidCfg(
                func=(
                    partial(_spawn_colored_shape, spawn_func=sim_utils.spawn_cuboid, color=color)
                    if color is not None
                    else sim_utils.spawn_cuboid
                ),
                size=size,
                collision_props=sim_utils.NewtonCollisionPropertiesCfg(contact_margin=COLLIDER_MARGIN),
                physics_material=sim_utils.NewtonMaterialPropertiesCfg(
                    static_friction=0.6,
                    dynamic_friction=0.6,
                ),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=position, rot=orientation),
        )

    rigid_objects = {}
    for index, position in enumerate(SPHERE_POSITIONS):
        rigid_objects[f"sphere_{index}"] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Sphere_{index}",
            spawn=sim_utils.SphereCfg(
                func=partial(
                    _spawn_colored_shape,
                    spawn_func=sim_utils.spawn_sphere,
                    color=SPHERE_COLORS[index],
                ),
                radius=SPHERE_RADIUS,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=SPHERE_MASS),
                collision_props=sim_utils.NewtonCollisionPropertiesCfg(),
                physics_material=sim_utils.NewtonMaterialPropertiesCfg(
                    static_friction=0.5,
                    dynamic_friction=0.5,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=position),
        )

    bath_x, bath_y = BATH_INTERIOR_SIZE
    wall_t = BATH_WALL_THICKNESS
    wall_z = 0.5 * BATH_WALL_HEIGHT
    chute_panels = tuple(
        bath_collider(
            f"/World/Chutes/Panel_{index}",
            CHUTE_PANEL_SIZE,
            position,
            orientation,
            CHUTE_COLOR,
        )
        for index, (position, orientation) in enumerate(CHUTE_PANEL_POSES)
    )

    @configclass
    class CoupledSceneCfg(InteractiveSceneCfg):
        """Scene containing a static bath, three rigid spheres, and MPM sand."""

        ground = AssetBaseCfg(
            prim_path="/World/Ground",
            spawn=sim_utils.GroundPlaneCfg(size=(12.0, 12.0), color=(0.30, 0.30, 0.30)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -wall_t)),
        )
        dome_light = AssetBaseCfg(
            prim_path="/World/DomeLight",
            spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.78, 0.78, 0.78)),
        )

        bath_floor = bath_collider(
            "/World/Bath/Floor",
            (bath_x + 2.0 * wall_t, bath_y + 2.0 * wall_t, wall_t),
            (0.0, 0.0, -0.5 * wall_t),
        )
        bath_left = bath_collider(
            "/World/Bath/LeftWall",
            (wall_t, bath_y, BATH_WALL_HEIGHT),
            (-0.5 * (bath_x + wall_t), 0.0, wall_z),
        )
        bath_right = bath_collider(
            "/World/Bath/RightWall",
            (wall_t, bath_y, BATH_WALL_HEIGHT),
            (0.5 * (bath_x + wall_t), 0.0, wall_z),
        )
        bath_front = bath_collider(
            "/World/Bath/FrontWall",
            (bath_x + 2.0 * wall_t, wall_t, BATH_WALL_HEIGHT),
            (0.0, -0.5 * (bath_y + wall_t), wall_z),
        )
        bath_back = bath_collider(
            "/World/Bath/BackWall",
            (bath_x + 2.0 * wall_t, wall_t, BATH_WALL_HEIGHT),
            (0.0, 0.5 * (bath_y + wall_t), wall_z),
        )

        chute_left_a, chute_left_b, chute_back_a, chute_back_b, chute_right_a, chute_right_b = chute_panels

        spheres = RigidObjectCollectionCfg(rigid_objects=rigid_objects)

        sand = MPMObjectCfg(
            prim_path="{ENV_REGEX_NS}/Sand",
            spawn=MPMGridCfg(
                lower=SAND_LOWER,
                upper=SAND_UPPER,
                voxel_size=args_cli.voxel_size,
                particles_per_cell=PARTICLES_PER_VOXEL_AXIS,
                particle_placement="cell_center",
                jitter=PARTICLE_SPACING,
                material=MPMParticleMaterialCfg(density=2500.0, friction=0.5, yield_pressure=1.0e5),
                visual_color=PARTICLE_COLOR,
            ),
        )

    return CoupledSceneCfg(num_envs=1, env_spacing=0.0)


def run_simulator(sim, scene) -> None:
    """Run until the viewer closes or the optional step limit is reached."""
    sim_dt = sim.get_physics_dt()
    step_count = 0
    while sim.is_headless_or_exist_active_visualizer() and (args_cli.max_steps < 0 or step_count < args_cli.max_steps):
        sim.step(render=False)
        scene.update(sim_dt)
        if sim.is_rendering:
            sim.render()
        step_count += 1


def main() -> None:
    """Launch the two-way rigid-MPM coupling demo."""
    sim_cfg = create_sim_cfg()
    with launch_simulation(sim_cfg, args_cli):
        from isaaclab.scene import InteractiveScene

        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view(eye=(6.0, -7.0, 5.0), target=(0.0, 0.4, 1.3))
        scene = InteractiveScene(create_scene_cfg())
        sim.reset()
        sand = scene["sand"]
        particle_count = sand.num_instances * sand.particles_per_object
        print(
            f"[INFO]: Isaac Lab Newton two-way MPM demo ready. Spawned {particle_count} particles.",
            flush=True,
        )
        print("[INFO]: Right-click and drag any sphere in the Newton viewer.", flush=True)
        run_simulator(sim, scene)


if __name__ == "__main__":
    main()
