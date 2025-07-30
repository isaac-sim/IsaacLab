# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Visual test script for the imu sensor from the Orbit framework.
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse

from isaacsim import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser(description="Imu Test Script")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments to clone.")
parser.add_argument(
    "--terrain_type",
    type=str,
    default="generator",
    choices=["generator", "usd", "plane"],
    help="Type of terrain to import. Can be 'generator' or 'usd' or 'plane'.",
)
args_cli = parser.parse_args()

# launch omniverse app
config = {"headless": args_cli.headless}
simulation_app = SimulationApp(config)


"""Rest everything follows."""

import torch
import traceback

import carb
import omni
from isaacsim.core.api.simulation_context import SimulationContext
from isaacsim.core.cloner import GridCloner
from isaacsim.core.utils.viewports import set_camera_view
from pxr import PhysxSchema

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors.imu import Imu, ImuCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.terrains.terrain_importer import TerrainImporter
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.timer import Timer


def design_scene(sim: SimulationContext, num_envs: int = 2048) -> RigidObject:
    """Design the scene."""
    # Handler for terrains importing
    terrain_importer_cfg = terrain_gen.TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd",
        max_init_terrain_level=None,
        num_envs=1,
    )
    _ = TerrainImporter(terrain_importer_cfg)
    # obtain the current stage
    stage = omni.usd.get_context().get_stage()
    # Create interface to clone the scene
    cloner = GridCloner(spacing=2.0)
    cloner.define_base_env("/World/envs")
    envs_prim_paths = cloner.generate_paths("/World/envs/env", num_paths=num_envs)
    # create source prim
    stage.DefinePrim(envs_prim_paths[0], "Xform")
    # clone the env xform
    cloner.clone(source_prim_path="/World/envs/env_0", prim_paths=envs_prim_paths, replicate_physics=True)
    # Define the scene
    # -- Light
    cfg = sim_utils.DistantLightCfg(intensity=2000)
    cfg.func("/World/light", cfg)
    # -- Balls
    cfg = RigidObjectCfg(
        spawn=sim_utils.SphereCfg(
            radius=0.25,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
        prim_path="/World/envs/env_.*/ball",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 5.0)),
    )
    balls = RigidObject(cfg)
    # Clone the scene
    # obtain the current physics scene
    physics_scene_prim_path = None
    for prim in stage.Traverse():
        if prim.HasAPI(PhysxSchema.PhysxSceneAPI):
            physics_scene_prim_path = prim.GetPrimPath()
            carb.log_info(f"Physics scene prim path: {physics_scene_prim_path}")
            break
    # filter collisions within each environment instance
    cloner.filter_collisions(
        physics_scene_prim_path,
        "/World/collisions",
        envs_prim_paths,
    )
    return balls


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
    balls = design_scene(sim=sim, num_envs=num_envs)

    # Create a ray-caster sensor
    imu_cfg = ImuCfg(
        prim_path="/World/envs/env_.*/ball",
        debug_vis=not args_cli.headless,
    )
    # increase scale of the arrows for better visualization
    imu_cfg.visualizer_cfg.markers["arrow"].scale = (1.0, 0.2, 0.2)
    imu = Imu(cfg=imu_cfg)

    # Play simulator and init the Imu
    sim.reset()

    # Print the sensor information
    print(imu)

    # Get the ball initial positions
    sim.step(render=not args_cli.headless)
    balls.update(sim.get_physics_dt())
    ball_initial_positions = balls.data.root_pos_w.clone()
    ball_initial_orientations = balls.data.root_quat_w.clone()

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
            # reset ball positions
            balls.write_root_pose_to_sim(torch.cat([ball_initial_positions, ball_initial_orientations], dim=-1))
            balls.reset()
            # reset the sensor
            imu.reset()
            # reset the counter
            step_count = 0
        # Step simulation
        sim.step()
        # Update the imu sensor
        with Timer(f"Imu sensor update with {num_envs}"):
            imu.update(dt=sim.get_physics_dt(), force_recompute=True)
        # Update counter
        step_count += 1


if __name__ == "__main__":
    try:
        # Run the main function
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()
