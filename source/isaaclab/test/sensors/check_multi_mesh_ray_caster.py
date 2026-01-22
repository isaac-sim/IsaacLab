# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""
This script shows how to use the multi-mesh ray caster from the Isaac Lab framework.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/isaaclab/test/sensors/check_multi_mesh_ray_caster.py --headless

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Ray Caster Test Script")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to clone.")
parser.add_argument("--num_objects", type=int, default=0, help="Number of additional objects to clone.")
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

import random

import torch

from isaacsim.core.api.simulation_context import SimulationContext
from isaacsim.core.cloner import GridCloner
from isaacsim.core.prims import RigidPrim
from isaacsim.core.utils.viewports import set_camera_view

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.sensors.ray_caster import MultiMeshRayCaster, MultiMeshRayCasterCfg, patterns
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.terrains.terrain_importer import TerrainImporter
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab.utils.timer import Timer


def design_scene(sim: SimulationContext, num_envs: int = 2048):
    """Design the scene."""
    # Create interface to clone the scene
    cloner = GridCloner(spacing=10.0)
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

    for i in range(args_cli.num_objects):
        object = sim_utils.CuboidCfg(
            size=(0.5 + random.random() * 0.5, 0.5 + random.random() * 0.5, 0.1 + random.random() * 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0 + i / args_cli.num_objects, 0.0, 1.0 - i / args_cli.num_objects)
            ),
        )
        object.func(
            f"/World/envs/env_0/object_{i}",
            object,
            translation=(0.0 + random.random(), 0.0 + random.random(), 1.0),
            orientation=quat_from_euler_xyz(torch.Tensor(0), torch.Tensor(0), torch.rand(1) * torch.pi).numpy(),
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
        "use_flatcache": True,  # deprecated from Isaac Sim 2023.1 onwards
        "use_fabric": True,  # used from Isaac Sim 2023.1 onwards
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
        max_init_terrain_level=0,
        num_envs=1,
    )
    _ = TerrainImporter(terrain_importer_cfg)

    mesh_targets: list[MultiMeshRayCasterCfg.RaycastTargetCfg] = [
        MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="/World/ground", track_mesh_transforms=False),
    ]
    if args_cli.num_objects != 0:
        mesh_targets.append(
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="/World/envs/env_.*/object_.*", track_mesh_transforms=True)
        )
    # Create a ray-caster sensor
    ray_caster_cfg = MultiMeshRayCasterCfg(
        prim_path="/World/envs/env_.*/ball",
        mesh_prim_paths=mesh_targets,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        attach_yaw_only=True,
        debug_vis=not args_cli.headless,
    )
    ray_caster = MultiMeshRayCaster(cfg=ray_caster_cfg)
    # Create a view over all the balls
    ball_view = RigidPrim("/World/envs/env_.*/ball", reset_xform_properties=False)

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
            sim.step(render=False)
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
    # run the main function
    main()
    # close sim app
    simulation_app.close()
