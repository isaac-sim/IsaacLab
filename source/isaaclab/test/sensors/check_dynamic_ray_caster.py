# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script showcases ray-caster with multiple and dynamic meshes.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/isaaclab/test/sensors/check_dynamic_ray_caster.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Dynamic Ray Caster Test Script")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors.ray_caster import RayCaster, RayCasterCfg, patterns
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.timer import Timer


def define_sensor() -> RayCaster:
    """Defines the ray-caster sensor to add to the scene."""
    # Create a ray-caster sensor
    ray_caster_cfg = RayCasterCfg(
        prim_path="/World/Origin.*/ball",
        mesh_prim_paths=["/World/ground", "/World/Origin.*/MovingCuboid"],
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(2.0, 2.0)),
        attach_yaw_only=True,
        debug_vis=not args_cli.headless,
    )
    ray_caster = RayCaster(cfg=ray_caster_cfg)

    return ray_caster


def design_scene() -> dict:
    """Design the scene."""
    # Populate scene
    # -- Rough terrain
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd")
    cfg.func("/World/ground", cfg)
    # -- Light
    cfg = sim_utils.DistantLightCfg(intensity=2000)
    cfg.func("/World/light", cfg)

    # Create separate groups called "Origin0", "Origin1", "Origin2", "Origin3"
    origins = [[0.25, 0.25, 0.0], [-0.25, 0.25, 0.0], [0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]]
    for i, origin in enumerate(origins):
        prim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)
    # -- Balls
    cfg = RigidObjectCfg(
        prim_path="/World/Origin.*/ball",
        spawn=sim_utils.SphereCfg(
            radius=0.25,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
    )
    balls = RigidObject(cfg)
    # -- Moving Cuboid
    cfg = RigidObjectCfg(
        prim_path="/World/Origin.*/MovingCuboid",
        spawn=sim_utils.MeshCuboidCfg(
            size=(1.5, 1.5, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                rigid_body_enabled=True,
                kinematic_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
    )
    moving_cuboid = RigidObject(cfg)
    # -- Sensors
    ray_caster = define_sensor()

    # return the scene information
    scene_entities = {"balls": balls, "moving_cuboid": moving_cuboid, "ray_caster": ray_caster}
    return scene_entities


def run_simulator(sim: sim_utils.SimulationContext, scene_entities: dict):
    """Run the simulator."""
    ray_caster: RayCaster = scene_entities["ray_caster"]
    balls: RigidObject = scene_entities["balls"]
    moving_cuboid: RigidObject = scene_entities["moving_cuboid"]

    dt = sim.get_physics_dt()

    # Get the default state of the balls and randomize their positions (x,y).
    ball_default_state = balls.data.default_root_state.clone()
    ball_default_state[:, 2] = 3
    ball_default_state[:, :2] = torch.rand_like(ball_default_state[:, :2]) * 10

    # Compute the initial cuboid state based on the ball state.
    cuboid_default_state = ball_default_state.clone()
    cuboid_default_state[:, 2] = 1
    # Write the initial cuboid pose
    moving_cuboid.write_root_pose_to_sim(cuboid_default_state[:, :7])

    # Parameters for sine wave motion along x.
    amplitude = 0.5
    frequency = 1.0

    # Simulation step counter.
    step_count = 0

    while simulation_app.is_running():
        time_val = step_count * dt
        sin_offset = amplitude * math.sin(frequency * time_val)

        # Update cuboid positions: add the sine offset to the x-coordinate.
        new_cuboid_state = cuboid_default_state.clone()
        new_cuboid_state[:, 0] += sin_offset

        # Write updated pose
        moving_cuboid.write_root_pose_to_sim(new_cuboid_state[:, :7])

        # Reset the scene every 250 steps.
        if step_count % 250 == 0:
            # Reset balls.
            balls.write_root_pose_to_sim(ball_default_state[:, :7])
            balls.write_root_velocity_to_sim(ball_default_state[:, 7:])
            # Reset the sensor.
            ray_caster.reset()
            # Recompute cuboid default state from the reset ball state.
            cuboid_default_state = ball_default_state.clone()
            cuboid_default_state[:, 2] = 1
            moving_cuboid.write_root_pose_to_sim(cuboid_default_state[:, :7])
            step_count = 0

        # Step the simulation.
        sim.step()

        # Update the ray-caster.
        with Timer(
            f"Ray-caster update with {4} x {ray_caster.num_rays} rays with max height of "
            f"{torch.max(ray_caster.data.pos_w).item():.2f}"
        ):
            ray_caster.update(dt=dt, force_recompute=True)

        step_count += 1


def main():
    """Main function."""
    # Load simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([0.0, 15.0, 15.0], [0.0, 0.0, -2.5])
    # Design the scene
    scene_entities = design_scene()
    # Play simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run simulator
    run_simulator(sim=sim, scene_entities=scene_entities)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
