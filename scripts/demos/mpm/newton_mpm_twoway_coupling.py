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

Right-click and drag a sphere to apply an interactive force. Use ``Space`` to
pause or resume the simulation and ``.`` to advance one step while paused.
"""

from __future__ import annotations

import argparse

from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Newton rigid-sphere and MPM-sand two-way coupling demo.")
parser.add_argument("--max_steps", type=int, default=-1, help="Stop after this many frames; negative runs forever.")
parser.add_argument("--voxel_size", type=float, default=0.08, help="MPM grid voxel size [m].")
parser.add_argument("--rigid_substeps", type=int, default=4, help="Rigid-solver substeps per coupled step.")
add_launcher_args(parser)
parser.set_defaults(visualizer=["newton"])
args_cli = parser.parse_args()


FPS = 100.0
GRAVITY = (0.0, 0.0, -9.81)
PARTICLES_PER_CELL = 2.0
PARTICLE_COLOR = (0.7, 0.6, 0.4)
SPHERE_BODY_PATTERN = r"/World/envs/env_.*/Sphere_[0-9]+"
SPHERE_RADIUS = 0.30
SPHERE_MASS = 450.0
SPHERE_POSITIONS = ((-0.90, 0.25, 1.10), (0.0, -0.30, 1.10), (0.90, 0.30, 1.10))

BATH_INTERIOR_SIZE = (3.6, 2.6)
BATH_WALL_HEIGHT = 1.5
BATH_WALL_THICKNESS = 0.15
SAND_LOWER = (-1.65, -1.15, 0.05)
SAND_UPPER = (1.65, 1.15, 0.72)


def create_visualizer_cfgs():
    """Create the demo-specific Newton visualizer configuration."""
    if not {"newton", "newton_gl"}.intersection(args_cli.visualizer or []):
        return []

    return [
        NewtonGLVisualizerCfg(
            streaming_view=False,
            show_particles=True,
            particle_color=PARTICLE_COLOR,
            update_frequency=1,
        )
    ]


def create_sim_cfg():
    """Create the proxy-coupled MJWarp and MPM simulation configuration."""
    from isaaclab_newton.physics import MJWarpSolverCfg, MPMSolverCfg, NewtonCfg

    import isaaclab.sim as sim_utils

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

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.utils.configclass import configclass

    def bath_collider(
        prim_path: str, size: tuple[float, float, float], position: tuple[float, float, float]
    ) -> AssetBaseCfg:
        return AssetBaseCfg(
            prim_path=prim_path,
            spawn=sim_utils.CuboidCfg(
                size=size,
                collision_props=sim_utils.NewtonCollisionPropertiesCfg(contact_margin=0.04),
                physics_material=sim_utils.NewtonMaterialPropertiesCfg(
                    static_friction=0.6,
                    dynamic_friction=0.6,
                ),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=position),
        )

    rigid_objects = {}
    for index, position in enumerate(SPHERE_POSITIONS):
        rigid_objects[f"sphere_{index}"] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Sphere_{index}",
            spawn=sim_utils.SphereCfg(
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

    @configclass
    class CoupledSceneCfg(InteractiveSceneCfg):
        """Scene containing a static bath, three rigid spheres, and MPM sand."""

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

        spheres = RigidObjectCollectionCfg(rigid_objects=rigid_objects)

        sand = MPMObjectCfg(
            prim_path="{ENV_REGEX_NS}/Sand",
            spawn=MPMGridCfg(
                lower=SAND_LOWER,
                upper=SAND_UPPER,
                voxel_size=args_cli.voxel_size,
                particles_per_cell=PARTICLES_PER_CELL,
                jitter=args_cli.voxel_size / PARTICLES_PER_CELL,
                material=MPMParticleMaterialCfg(density=2500.0, friction=0.75, yield_pressure=1.0e15),
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
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene

        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view(eye=(4.5, -5.5, 3.5), target=(0.0, 0.0, 0.65))
        scene = InteractiveScene(create_scene_cfg())
        sim.reset()
        sand = scene["sand"]
        particle_count = sand.num_instances * sand.particles_per_object
        print(
            f"[INFO]: Isaac Lab Newton two-way MPM demo ready. Spawned {particle_count} particles.",
            flush=True,
        )
        print("[INFO]: Right-click and drag a sphere in the Newton viewer.", flush=True)
        run_simulator(sim, scene)


if __name__ == "__main__":
    main()
