# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script shows how to use the ray caster from the Isaac Lab framework.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/isaaclab/test/sensors/test_ray_caster.py --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Ray Caster Test Script")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments to clone.")
parser.add_argument(
    "--terrain_type",
    type=str,
    default="generator",
    help="Type of terrain to import. Can be 'generator' or 'usd' or 'plane'.",
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
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors.ray_caster import RayCaster, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.terrains.terrain_importer import TerrainImporter
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.timer import Timer


def design_scene(sim: SimulationContext, num_envs: int = 2048):
    """Design the scene."""
    # Create interface to clone the scene
    cloner = GridCloner(spacing=2.0, stage=sim.stage)
    cloner.define_base_env("/World/envs")
    # Everything under the namespace "/World/envs/env_0" will be cloned
    sim.stage.DefinePrim("/World/envs/env_0", "Xform")
    # Define the scene
    # -- Light
    cfg = sim_utils.DistantLightCfg(intensity=2000)
    cfg.func("/World/light", cfg)
    # -- Balls
    cfg = sim_utils.SphereCfg(
        radius=0.25,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
    )
    cfg.func("/World/envs/env_0/ball", cfg, translation=(0.0, 0.0, 5.0))
    # Clone the scene
    cloner.define_base_env("/World/envs")
    envs_prim_paths = cloner.generate_paths("/World/envs/env", num_paths=num_envs)
    cloner.clone(source_prim_path="/World/envs/env_0", prim_paths=envs_prim_paths, replicate_physics=True)
    physics_scene_path = sim.get_physics_context().prim_path
    cloner.filter_collisions(
        physics_scene_path, "/World/collisions", prim_paths=envs_prim_paths, global_paths=["/World/ground"]
    )


def main():
    """Main function."""

    sim = SimulationContext(SimulationCfg())
    # Set main camera
    sim.set_camera_view([0.0, 30.0, 25.0], [0.0, 0.0, -2.5])

    # Parameters
    num_envs = args_cli.num_envs
    # Design the scene
    design_scene(sim=sim, num_envs=num_envs)
    # Handler for terrains importing
    terrain_importer_cfg = terrain_gen.TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type=args_cli.terrain_type,
        terrain_generator=ROUGH_TERRAINS_CFG,
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd",
        max_init_terrain_level=None,
        num_envs=1,
    )
    _ = TerrainImporter(terrain_importer_cfg)

    # Create a ray-caster sensor
    ray_caster_cfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/ball",
        mesh_prim_paths=["/World/ground"],
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        ray_alignment="yaw",
        debug_vis=not args_cli.headless,
    )
    ray_caster = RayCaster(cfg=ray_caster_cfg)
    # Create a view over all the balls
    balls_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ball",
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 5.0)),
    )
    balls = RigidObject(cfg=balls_cfg)

    # Play simulator
    sim.reset()

    # Initialize the views
    # -- balls
    print(balls)
    # Print the sensor information
    print(ray_caster)

    # Get the initial positions of the balls
    ball_initial_poses = balls.data.root_pose_w.clone()
    ball_initial_velocities = balls.data.root_vel_w.clone()

    # Create a counter for resetting the scene
    step_count = 0
    # Simulate physics
    while simulation_app.is_running():
        # If simulation is stopped, then exit.
        if sim.is_stopped():
            break
        # If simulation is paused, then skip.
        if not sim.is_playing():
            sim.step(render=False)
            continue
        # Reset the scene
        if step_count % 500 == 0:
            # sample random indices to reset
            reset_indices = torch.randint(0, num_envs, (num_envs // 2,), device=sim.device)
            # reset the balls
            balls.write_root_pose_to_sim(ball_initial_poses[reset_indices], env_ids=reset_indices)
            balls.write_root_velocity_to_sim(ball_initial_velocities[reset_indices], env_ids=reset_indices)
            balls.reset(reset_indices)
            # reset the sensor
            ray_caster.reset(reset_indices)
            # reset the counter
            step_count = 0
        # Step simulation
        sim.step()
        # Update the ray-caster
        with Timer(f"Ray-caster update with {num_envs} x {ray_caster.num_rays} rays"):
            ray_caster.update(dt=sim.get_physics_dt(), force_recompute=True)
        # Update counter
        step_count += 1


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
