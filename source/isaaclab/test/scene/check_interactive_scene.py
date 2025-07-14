# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the scene interface to quickly setup a scene with multiple
articulated robots and sensors.
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to use the scene interface.")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns
from isaaclab.sim import SimulationContext
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.timer import Timer

##
# Pre-defined configs
##
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Example scene configuration."""

    # terrain - flat terrain plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
    )

    # articulation - robot 1
    robot_1 = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_1")
    # articulation - robot 2
    robot_2 = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_2")
    robot_2.init_state.pos = (0.0, 1.0, 0.6)

    # sensor - ray caster attached to the base of robot 1 that scans the ground
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot_1/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )

    # extras - light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 500.0)),
    )


def main():
    """Main function."""

    # Load kit helper
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005))
    # Set main camera
    sim.set_camera_view(eye=[5, 5, 5], target=[0.0, 0.0, 0.0])

    # Spawn things into stage
    with Timer("Setup scene"):
        scene = InteractiveScene(MySceneCfg(num_envs=args_cli.num_envs, env_spacing=5.0, lazy_sensor_update=False))

    # Check that parsing happened as expected
    assert len(scene.env_prim_paths) == args_cli.num_envs, "Number of environments does not match."
    assert scene.terrain is not None, "Terrain not found."
    assert len(scene.articulations) == 2, "Number of robots does not match."
    assert len(scene.sensors) == 1, "Number of sensors does not match."
    assert len(scene.extras) == 1, "Number of extras does not match."

    # Play the simulator
    with Timer("Time taken to play the simulator"):
        sim.reset()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # default joint targets
    robot_1_actions = scene.articulations["robot_1"].data.default_joint_pos.clone()
    robot_2_actions = scene.articulations["robot_2"].data.default_joint_pos.clone()
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    # Simulate physics
    while simulation_app.is_running():
        # If simulation is stopped, then exit.
        if sim.is_stopped():
            break
        # If simulation is paused, then skip.
        if not sim.is_playing():
            sim.step()
            continue
        # reset
        if count % 50 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # reset root state
            root_state = scene.articulations["robot_1"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            joint_pos = scene.articulations["robot_1"].data.default_joint_pos
            joint_vel = scene.articulations["robot_1"].data.default_joint_vel
            # -- set root state
            # -- robot 1
            scene.articulations["robot_1"].write_root_pose_to_sim(root_state[:, :7])
            scene.articulations["robot_1"].write_root_velocity_to_sim(root_state[:, 7:])
            scene.articulations["robot_1"].write_joint_state_to_sim(joint_pos, joint_vel)
            # -- robot 2
            root_state[:, 1] += 1.0
            scene.articulations["robot_2"].write_root_pose_to_sim(root_state[:, :7])
            scene.articulations["robot_2"].write_root_velocity_to_sim(root_state[:, 7:])
            scene.articulations["robot_2"].write_joint_state_to_sim(joint_pos, joint_vel)
            # reset buffers
            scene.reset()
            print(">>>>>>>> Reset!")
        # perform this loop at policy control freq (50 Hz)
        for _ in range(4):
            # set joint targets
            scene.articulations["robot_1"].set_joint_position_target(robot_1_actions)
            scene.articulations["robot_2"].set_joint_position_target(robot_2_actions)
            # write data to sim
            scene.write_data_to_sim()
            # perform step
            sim.step()
            # read data from sim
            scene.update(sim_dt)
        # update sim-time
        sim_time += sim_dt * 4
        count += 1


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
