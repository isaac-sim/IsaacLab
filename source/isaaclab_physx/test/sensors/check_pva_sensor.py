# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Visual test script for the pva sensor from the Orbit framework.
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse

from isaacsim import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser(description="Pva Test Script")
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

import logging
import traceback

import torch

from isaacsim.core.rendering_manager import ViewportManager

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab import cloner as lab_cloner
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors.pva import Pva, PvaCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.terrains.terrain_importer import TerrainImporter
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.timer import Timer

# import logger
logger = logging.getLogger(__name__)


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
    stage = sim_utils.get_current_stage()
    # Create interface to clone the scene
    # Create environment clones using Lab's cloner utilities
    env_fmt = "/World/envs/env_{}"
    env_ids = torch.arange(num_envs, dtype=torch.long, device=sim.device)
    env_origins, _ = lab_cloner.grid_transforms(num_envs, spacing=2.0, device=sim.device)
    envs_prim_paths = [f"/World/envs/env_{i}" for i in range(num_envs)]
    # create source prim
    stage.DefinePrim(envs_prim_paths[0], "Xform")
    # clone the env xform
    lab_cloner.usd_replicate(stage, [env_fmt.format(0)], [env_fmt], env_ids, positions=env_origins)
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
        if "PhysxSceneAPI" in prim.GetAppliedSchemas():
            physics_scene_prim_path = prim.GetPrimPath()
            logging.info(f"Physics scene prim path: {physics_scene_prim_path}")
            break
    # filter collisions within each environment instance
    lab_cloner.filter_collisions(
        stage,
        physics_scene_prim_path,
        "/World/collisions",
        envs_prim_paths,
    )
    return balls


def main():
    """Main function."""

    # Load kit helper
    sim = SimulationContext(SimulationCfg())
    # Set main camera
    ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=[0.0, 30.0, 25.0], target=[0.0, 0.0, -2.5])

    # Parameters
    num_envs = args_cli.num_envs
    # Design the scene
    balls = design_scene(sim=sim, num_envs=num_envs)

    # Create a pva sensor
    pva_cfg = PvaCfg(
        prim_path="/World/envs/env_.*/ball",
        debug_vis=not args_cli.headless,
    )
    # increase scale of the arrows for better visualization
    pva_cfg.visualizer_cfg.markers["arrow"].scale = (1.0, 0.2, 0.2)
    pva = Pva(cfg=pva_cfg)

    # Play simulator and init the Pva
    sim.reset()

    # Print the sensor information
    print(pva)

    # Get the ball initial positions
    sim.step(render=not args_cli.headless)
    balls.update(sim.get_physics_dt())
    ball_initial_positions = balls.data.root_pos_w.torch.clone()
    ball_initial_orientations = balls.data.root_quat_w.torch.clone()

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
            pva.reset()
            # reset the counter
            step_count = 0
        # Step simulation
        sim.step()
        # Update the pva sensor
        with Timer(f"Pva sensor update with {num_envs}"):
            pva.update(dt=sim.get_physics_dt(), force_recompute=True)
        # Update counter
        step_count += 1


if __name__ == "__main__":
    try:
        # Run the main function
        main()
    except Exception as err:
        logger.error(err)
        logger.error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()
