# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script shows how to use the terrain generator from the Isaac Lab framework.

The terrains are generated using the :class:`TerrainGenerator` class and imported using the :class:`TerrainImporter`
class. The terrains can be imported from a file or generated procedurally.

Example usage:

.. code-block:: bash

    # generate terrain
    # -- use physics sphere mesh
    ./isaaclab.sh -p source/isaaclab/test/terrains/check_terrain_importer.py --terrain_type generator
    # -- usd usd sphere geom
    ./isaaclab.sh -p source/isaaclab/test/terrains/check_terrain_importer.py --terrain_type generator --geom_sphere

    # usd terrain
    ./isaaclab.sh -p source/isaaclab/test/terrains/check_terrain_importer.py --terrain_type usd

    # plane terrain
    ./isaaclab.sh -p source/isaaclab/test/terrains/check_terrain_importer.py --terrain_type plane
"""

"""Launch Isaac Sim Simulator first."""

import argparse

# isaaclab
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script shows how to use the terrain importer.")
parser.add_argument("--geom_sphere", action="store_true", default=False, help="Whether to use sphere mesh or shape.")
parser.add_argument(
    "--terrain_type",
    type=str,
    choices=["generator", "usd", "plane"],
    default="generator",
    help="Type of terrain to import. Can be 'generator' or 'usd' or 'plane'.",
)
parser.add_argument(
    "--color_scheme",
    type=str,
    default="height",
    choices=["height", "random", "none"],
    help="The color scheme to use for the generated terrain.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import torch

from isaacsim.core.cloner import GridCloner

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.terrains.terrain_importer import TerrainImporter
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


def main():
    """Generates a terrain from isaaclab."""

    # Load kit helper
    sim = SimulationContext(SimulationCfg())
    # Set main camera
    sim.set_camera_view(eye=(0.0, 30.0, 25.0), target=(0.0, 0.0, -2.5))

    # Parameters
    num_balls = 2048

    # Create interface to clone the scene
    cloner = GridCloner(spacing=2.0, stage=sim.stage)
    cloner.define_base_env("/World/envs")
    # Everything under the namespace "/World/envs/env_0" will be cloned
    sim_utils.define_prim("/World/envs/env_0")

    # Handler for terrains importing
    terrain_importer_cfg = terrain_gen.TerrainImporterCfg(
        num_envs=2048,
        env_spacing=3.0,
        prim_path="/World/ground",
        max_init_terrain_level=None,
        terrain_type=args_cli.terrain_type,
        terrain_generator=ROUGH_TERRAINS_CFG.replace(curriculum=True, color_scheme=args_cli.color_scheme),
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd",
    )
    terrain_importer = TerrainImporter(terrain_importer_cfg)

    # Define the scene
    # -- Light
    cfg = sim_utils.DistantLightCfg(intensity=1000.0)
    cfg.func("/World/Light", cfg)

    # -- Ball with physics properties using Isaac Lab spawners
    ball_prim_path = "/World/envs/env_0/ball"

    # Create physics material
    physics_material_cfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=0.2,
        dynamic_friction=1.0,
        restitution=0.0,
    )

    # Create visual material
    visual_material_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0))

    if args_cli.geom_sphere:
        # Spawn a geom sphere with rigid body properties
        sphere_cfg = sim_utils.SphereCfg(
            radius=0.25,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=visual_material_cfg,
            physics_material=physics_material_cfg,
        )
        sphere_cfg.func(ball_prim_path, sphere_cfg, translation=(0.0, 0.0, 5.0))
    else:
        # Spawn a mesh sphere with rigid body properties
        mesh_sphere_cfg = sim_utils.MeshSphereCfg(
            radius=0.25,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=visual_material_cfg,
            physics_material=physics_material_cfg,
        )
        mesh_sphere_cfg.func(ball_prim_path, mesh_sphere_cfg, translation=(0.0, 0.0, 0.5))

    # Clone the scene
    cloner.define_base_env("/World/envs")
    envs_prim_paths = cloner.generate_paths("/World/envs/env", num_paths=num_balls)
    cloner.clone(source_prim_path="/World/envs/env_0", prim_paths=envs_prim_paths, replicate_physics=True)
    physics_scene_path = sim.cfg.physics.physics_prim_path
    cloner.filter_collisions(
        physics_scene_path, "/World/collisions", prim_paths=envs_prim_paths, global_paths=["/World/ground"]
    )

    # Set ball positions over terrain origins using XformPrimView (before simulation starts)
    xform_view = sim_utils.XformPrimView("/World/envs/env_.*/ball")
    # cache initial state of the balls
    ball_initial_positions = terrain_importer.env_origins.clone()
    ball_initial_positions[:, 2] += 5.0
    # set initial poses (writes to USD before simulation)
    xform_view.set_world_poses(positions=ball_initial_positions)

    # Play simulator
    sim.reset()

    # Create a PhysX rigid body view for physics simulation
    physics_sim_view = sim.physics_manager.get_physics_sim_view()
    ball_view = physics_sim_view.create_rigid_body_view("/World/envs/env_*/ball")

    # Cache initial velocities (all zeros)
    ball_initial_velocities = ball_view.get_velocities()

    # Build initial transforms tensor for reset: (N, 7) = [pos(3), quat_xyzw(4)]
    num_balls_actual = ball_initial_positions.shape[0]
    ball_initial_transforms = torch.zeros(num_balls_actual, 7, device=ball_initial_positions.device)
    ball_initial_transforms[:, :3] = ball_initial_positions
    ball_initial_transforms[:, 6] = 1.0  # w=1 for identity quaternion (xyzw format)

    # Create indices for all balls (required by PhysX view API)
    all_indices = torch.arange(num_balls_actual, dtype=torch.int32, device=ball_initial_positions.device)

    # Create a counter for resetting the scene
    step_count = 0
    # Simulate physics
    while simulation_app.is_running():
        # If simulation is stopped, then exit.
        if sim.is_stopped():
            break
        # If simulation is paused, then skip.
        if not sim.is_playing():
            sim.step()
            continue
        # Reset the scene
        if step_count % 500 == 0:
            # reset the balls using PhysX tensor API
            ball_view.set_transforms(ball_initial_transforms, all_indices)
            ball_view.set_velocities(ball_initial_velocities, all_indices)
            # reset the counter
            step_count = 0
        # Step simulation
        sim.step()
        # Update counter
        step_count += 1


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
