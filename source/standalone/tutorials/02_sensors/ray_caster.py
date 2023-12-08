# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstartes how to use the ray-caster sensor.

.. code-block:: bash

    # Usage
    ./orbit.sh -p source/standalone/how_to_guides/02_sensors/ray_caster.py

"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser(description="Ray Caster Test Script")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
args_cli = parser.parse_args()

# launch omniverse app
config = {"headless": args_cli.headless}
simulation_app = SimulationApp(config)


"""Rest everything follows."""


import torch
import traceback

import carb
from omni.isaac.core.utils.viewports import set_camera_view

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import RigidObject, RigidObjectCfg
from omni.isaac.orbit.sensors.ray_caster import RayCaster, RayCasterCfg, patterns
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.orbit.utils.timer import Timer


def design_scene():
    """Design the scene."""
    # Populate scene
    # -- Rough terrain
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd")
    cfg.func("/World/ground", cfg)
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
    cfg.func("/World/envs/env_0/ball", cfg)
    cfg.func("/World/envs/env_1/ball", cfg)
    cfg.func("/World/envs/env_2/ball", cfg)
    cfg.func("/World/envs/env_3/ball", cfg)

    # Setup rigid object
    cfg = RigidObjectCfg(prim_path="/World/envs/env_.*/ball")
    # Create rigid object handler
    balls = RigidObject(cfg)

    # Create a ray-caster sensor
    ray_caster = add_sensor()

    return ray_caster, balls


def add_sensor():
    # Create a ray-caster sensor
    ray_caster_cfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/ball",
        mesh_prim_paths=["/World/ground"],
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        attach_yaw_only=True,
        debug_vis=not args_cli.headless,
    )
    ray_caster = RayCaster(cfg=ray_caster_cfg)

    return ray_caster


def run_simulator(sim: sim_utils.SimulationContext, ray_caster: RayCaster, balls: RigidObject):
    """Run the simulator."""

    # define an initial position of the sensor
    ball_default_state = balls.data.default_root_state.clone()
    ball_default_state[:, :3] = torch.rand_like(ball_default_state[:, :3]) * 10

    # Create a counter for resetting the scene
    step_count = 0
    # Simulate physics
    while simulation_app.is_running():
        # Reset the scene
        if step_count % 250 == 0:
            # reset the balls
            balls.write_root_state_to_sim(ball_default_state)
            # reset the sensor
            ray_caster.reset()
            # reset the counter
            step_count = 0
        # Step simulation
        sim.step()
        # Update the ray-caster
        with Timer(
            f"Ray-caster update with {4} x {ray_caster.num_rays} rays with max height of"
            f" {torch.max(ray_caster.data.pos_w).item():.2f}"
        ):
            ray_caster.update(dt=sim.get_physics_dt(), force_recompute=True)
        # Update counter
        step_count += 1


def main():
    """Main function."""
    # Load simulation context
    sim_cfg = sim_utils.SimulationCfg()
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    set_camera_view([0.0, 15.0, 15.0], [0.0, 0.0, -2.5])
    # Design the scene
    ray_caster, balls = design_scene()
    # Play simulator
    sim.reset()
    # Print the sensor information
    print(ray_caster)
    # Run simulator
    run_simulator(sim=sim, ray_caster=ray_caster, balls=balls)


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
