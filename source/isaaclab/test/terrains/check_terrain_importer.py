# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
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


import numpy as np

import isaacsim.core.utils.prims as prim_utils
import omni.kit
import omni.kit.commands
from isaacsim.core.api.simulation_context import SimulationContext
from isaacsim.core.cloner import GridCloner
from isaacsim.core.materials import PhysicsMaterial, PreviewSurface
from isaacsim.core.objects import DynamicSphere
from isaacsim.core.prims import RigidPrim, SingleGeometryPrim, SingleRigidPrim
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.viewports import set_camera_view

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.terrains.terrain_importer import TerrainImporter
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

enable_extension("omni.kit.primitive.mesh")


def main():
    """Generates a terrain from isaaclab."""

    # Load kit helper
    sim_params = {
        "use_gpu": True,
        "use_gpu_pipeline": True,
        "use_flatcache": True,
        "use_fabric": True,
        "enable_scene_query_support": True,
    }
    sim = SimulationContext(
        physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0, sim_params=sim_params, backend="torch", device="cuda:0"
    )
    # Set main camera
    set_camera_view([0.0, 30.0, 25.0], [0.0, 0.0, -2.5])

    # Parameters
    num_balls = 2048

    # Create interface to clone the scene
    cloner = GridCloner(spacing=2.0)
    cloner.define_base_env("/World/envs")
    # Everything under the namespace "/World/envs/env_0" will be cloned
    prim_utils.define_prim("/World/envs/env_0")

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
    # -- Ball
    if args_cli.geom_sphere:
        # -- Ball physics
        _ = DynamicSphere(
            prim_path="/World/envs/env_0/ball", translation=np.array([0.0, 0.0, 5.0]), mass=0.5, radius=0.25
        )
    else:
        # -- Ball geometry
        cube_prim_path = omni.kit.commands.execute("CreateMeshPrimCommand", prim_type="Sphere")[1]
        prim_utils.move_prim(cube_prim_path, "/World/envs/env_0/ball")
        # -- Ball physics
        SingleRigidPrim(
            prim_path="/World/envs/env_0/ball", mass=0.5, scale=(0.5, 0.5, 0.5), translation=(0.0, 0.0, 0.5)
        )
        SingleGeometryPrim(prim_path="/World/envs/env_0/ball", collision=True)
    # -- Ball material
    sphere_geom = SingleGeometryPrim(prim_path="/World/envs/env_0/ball", collision=True)
    visual_material = PreviewSurface(prim_path="/World/Looks/ballColorMaterial", color=np.asarray([0.0, 0.0, 1.0]))
    physics_material = PhysicsMaterial(
        prim_path="/World/Looks/ballPhysicsMaterial",
        dynamic_friction=1.0,
        static_friction=0.2,
        restitution=0.0,
    )
    sphere_geom.set_collision_approximation("convexHull")
    sphere_geom.apply_visual_material(visual_material)
    sphere_geom.apply_physics_material(physics_material)

    # Clone the scene
    cloner.define_base_env("/World/envs")
    envs_prim_paths = cloner.generate_paths("/World/envs/env", num_paths=num_balls)
    cloner.clone(source_prim_path="/World/envs/env_0", prim_paths=envs_prim_paths, replicate_physics=True)
    physics_scene_path = sim.get_physics_context().prim_path
    cloner.filter_collisions(
        physics_scene_path, "/World/collisions", prim_paths=envs_prim_paths, global_paths=["/World/ground"]
    )

    # Set ball positions over terrain origins
    # Create a view over all the balls
    ball_view = RigidPrim("/World/envs/env_.*/ball", reset_xform_properties=False)
    # cache initial state of the balls
    ball_initial_positions = terrain_importer.env_origins
    ball_initial_positions[:, 2] += 5.0
    # set initial poses
    # note: setting here writes to USD :)
    ball_view.set_world_poses(positions=ball_initial_positions)

    # Play simulator
    sim.reset()
    # Initialize the ball views for physics simulation
    ball_view.initialize()
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
            sim.step()
            continue
        # Reset the scene
        if step_count % 500 == 0:
            # reset the balls
            ball_view.set_world_poses(positions=ball_initial_positions)
            ball_view.set_velocities(ball_initial_velocities)
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
