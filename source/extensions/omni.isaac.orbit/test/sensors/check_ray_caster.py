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

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse

# omni-isaac-orbit
from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser(description="Ray Caster Test Script")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments to clone.")
parser.add_argument(
    "--terrain_type",
    type=str,
    default="generator",
    help="Type of terrain to import. Can be 'generator' or 'usd' or 'plane'.",
)
args_cli = parser.parse_args()

# launch omniverse app
config = {"headless": args_cli.headless}
simulation_app = SimulationApp(config)


"""Rest everything follows."""


import numpy as np
import torch

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.cloner import GridCloner
from omni.isaac.core.objects import DynamicSphere
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.viewports import set_camera_view

import omni.isaac.orbit.terrains as terrain_gen
from omni.isaac.orbit.sensors.ray_caster import RayCaster, RayCasterCfg, patterns
from omni.isaac.orbit.terrains.config.rough import ROUGH_TERRAINS_CFG
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
    terrain_importer_cfg = terrain_gen.TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type=args_cli.terrain_type,
        terrain_generator=ROUGH_TERRAINS_CFG,
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd",
        max_init_terrain_level=None,
        num_envs=1,
    )
    terrain_importer = TerrainImporter(terrain_importer_cfg)

    # Create a ray-caster sensor
    ray_caster_cfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/ball",
        mesh_prim_paths=["/World/ground"],
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        attach_yaw_only=True,
        debug_vis=False if args_cli.headless else True,
    )
    ray_caster = RayCaster(cfg=ray_caster_cfg)
    # Create a view over all the balls
    ball_view = RigidPrimView("/World/envs/env_.*/ball", reset_xform_properties=False)

    # Play simulator
    sim.reset()

    # Initialize the views
    # -- balls
    ball_view.initialize()
    # Print the sensor information
    print(ray_caster)

    # Get the initial positions of the balls
    ball_initial_positions, ball_initial_orientations = ball_view.get_world_poses()
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
            # sample random indices to reset
            reset_indices = torch.randint(0, num_envs, (num_envs // 2,))
            # reset the balls
            ball_view.set_world_poses(
                ball_initial_positions[reset_indices], ball_initial_orientations[reset_indices], indices=reset_indices
            )
            ball_view.set_velocities(ball_initial_velocities[reset_indices], indices=reset_indices)
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
    # Runs the main function
    main()
    # Close the simulator
    simulation_app.close()
