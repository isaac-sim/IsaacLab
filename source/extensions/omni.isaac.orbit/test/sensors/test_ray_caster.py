# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
This script shows how to use the ray caster from the Orbit framework.

.. code-block:: bash

    # Usage
    ./orbit.sh -p source/extensions/omni.isaac.orbit/test/sensors/test_ray_caster.py --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse

# omni-isaac-orbit
from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments to clone.")
parser.add_argument(
    "--terrain_type",
    type=str,
    default="generated",
    help="Type of terrain to import. Can be 'generated' or 'usd' or 'plane'.",
)
args_cli = parser.parse_args()

# launch omniverse app
config = {"headless": args_cli.headless}
simulation_app = SimulationApp(config)


"""Rest everything follows."""


import numpy as np

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.cloner import GridCloner
from omni.isaac.core.objects import DynamicSphere
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.viewports import set_camera_view

import omni.isaac.orbit.terrains as terrain_gen
import omni.isaac.orbit.utils.kit as kit_utils
from omni.isaac.orbit.sensors.ray_caster import GridPatternCfg, RayCaster, RayCasterCfg
from omni.isaac.orbit.terrains.config.rough import ASSORTED_TERRAINS_CFG
from omni.isaac.orbit.terrains.terrain_importer import TerrainImporter
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.orbit.utils.timer import Timer


def design_scene(sim: SimulationContext, num_envs: int = 2048):
    """Design the scene."""
    # Create interface to clone the scene
    cloner = GridCloner(spacing=2.0)
    cloner.define_base_env("/World/envs")
    # Everything under the namespace "/World/envs/env_0" will be cloned
    prim_utils.define_prim("/World/envs/env_0")
    # Define the scene
    # -- Light
    prim_utils.create_prim(
        "/World/sphereLight",
        "SphereLight",
        translation=(0.0, 0.0, 500.0),
        attributes={"radius": 100.0, "intensity": 50000.0, "color": (0.75, 0.75, 0.75)},
    )
    # -- Balls
    # -- Ball physics
    DynamicSphere(
        prim_path="/World/envs/env_0/ball",
        translation=np.array([0.0, 0.0, 5.0]),
        mass=0.5,
        radius=0.25,
        color=np.asarray((0.0, 0.0, 1.0)),
    )
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

    # Load kit helper
    sim_params = {
        "use_gpu": True,
        "use_gpu_pipeline": True,
        "use_flatcache": True,
        "enable_scene_query_support": True,
    }
    sim = SimulationContext(
        physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0, sim_params=sim_params, backend="torch", device="cuda:0"
    )
    # Set main camera
    set_camera_view([0.0, 30.0, 25.0], [0.0, 0.0, -2.5])

    # Parameters
    num_envs = args_cli.num_envs
    # Design the scene
    design_scene(sim=sim, num_envs=num_envs)
    # Handler for terrains importing
    if args_cli.terrain_type == "generated":
        terrain_importer_cfg = terrain_gen.TerrainImporterCfg(prim_path="/World/ground", max_init_terrain_level=None)
        terrain_importer = TerrainImporter(terrain_importer_cfg, device=sim.device)

        terrain_generator = terrain_gen.TerrainGenerator(cfg=ASSORTED_TERRAINS_CFG, curriculum=True)
        terrain_importer.import_mesh(terrain_generator.terrain_mesh, key="rough")
    elif args_cli.terrain_type == "usd":
        prim_utils.create_prim("/World/ground", usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd")
    elif args_cli.terrain_type == "plane":
        kit_utils.create_ground_plane("/World/ground")
    else:
        raise NotImplementedError(f"Terrain type {args_cli.terrain_type} not supported!")

    # Create a ray-caster sensor
    pattern_cfg = GridPatternCfg(resolution=0.1, size=(1.6, 1.0))
    ray_caster_cfg = RayCasterCfg(
        mesh_prim_paths=["/World/ground"], pattern_cfg=pattern_cfg, attach_yaw_only=True, debug_vis=True
    )
    ray_caster = RayCaster(cfg=ray_caster_cfg)
    # Create a view over all the balls
    ball_view = RigidPrimView("/World/envs/env_.*/ball", reset_xform_properties=False)

    # Play simulator
    sim.reset()

    # Initialize the views
    # -- balls
    ball_view.initialize()
    # -- sensors
    ray_caster.initialize("/World/envs/env_.*/ball")
    # Print the sensor information
    print(ray_caster)

    # Get the initial positions of the balls
    ball_initial_poses = ball_view.get_world_poses()
    ball_initial_velocities = ball_view.get_velocities()

    # Create a counter for resetting the scene
    step_count = 0
    # Simulate physics
    while simulation_app.is_running():
        # If simulation is stopped, then exit.
        if sim.is_stopped():
            break
        # If simulation is paused, then skip.
        if not sim.is_playing():
            sim.step(render=not args_cli.headless)
            continue
        # Reset the scene
        if step_count % 500 == 0:
            # reset the balls
            ball_view.set_world_poses(*ball_initial_poses)
            ball_view.set_velocities(ball_initial_velocities)
            # reset the sensor
            ray_caster.reset_buffers()
            # reset the counter
            step_count = 0
        # Step simulation
        sim.step()
        # Update the ray-caster
        with Timer(f"Ray-caster update with {ray_caster.count} x {ray_caster.num_rays} rays"):
            ray_caster.update_buffers(dt=sim.get_physics_dt())
        # Visualize the ray-caster
        if not args_cli.headless:
            with Timer(f"Ray-caster debug visualization\t\t"):
                ray_caster.debug_vis()
        # Update counter
        step_count += 1


if __name__ == "__main__":
    # Runs the main function
    main()
    # Close the simulator
    simulation_app.close()
