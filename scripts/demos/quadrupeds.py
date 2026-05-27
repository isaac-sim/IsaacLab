# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates different legged robots.

.. code-block:: bash

    # Usage with the default PhysX backend (launches Isaac Sim Kit by default).
    ./isaaclab.sh -p scripts/demos/quadrupeds.py

    # Usage with the kit-less Newton (MJWarp) backend (launches Isaac Sim Kit by default).
    ./isaaclab.sh -p scripts/demos/quadrupeds.py --physics newton_mjwarp

    # Usage with the Newton (MJWarp) backend without Kit (launches Newton visualizer).
    ./isaaclab.sh -p scripts/demos/quadrupeds.py --physics newton_mjwarp --visualizer newton

    # Usage with the PhysX backend without Kit (launches Newton visualizer).
    # TODO(yizew@nvidia.com): Not supported yet. Investigation needed.
    ./isaaclab.sh -p scripts/demos/quadrupeds.py --physics physx --visualizer newton

"""

"""Parse CLI first so we can decide whether to launch Isaac Sim Kit."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates different legged robots.", conflict_handler="resolve")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--physics", default="physx", choices=["physx", "newton_mjwarp"], help="Physics backend.")
parser.add_argument("--visualizer", default="kit", choices=["kit", "newton"], help="Visualizer backend.")
# parse the arguments
args_cli = parser.parse_args()

import numpy as np
import torch

from demo_helper import has_no_alive_visualizer_window, resolve_backend_and_visualizer
from isaaclab_assets.robots.anymal import ANYMAL_B_CFG, ANYMAL_C_CFG, ANYMAL_D_CFG  # isort:skip
from isaaclab_assets.robots.spot import SPOT_CFG  # isort:skip
from isaaclab_assets.robots.unitree import UNITREE_A1_CFG, UNITREE_GO1_CFG, UNITREE_GO2_CFG  # isort:skip


def define_origins(num_origins: int, spacing: float) -> list[list[float]]:
    """Defines the origins of the scene."""
    # create tensor based on number of environments
    env_origins = torch.zeros(num_origins, 3)
    # create a grid of origins
    num_cols = np.floor(np.sqrt(num_origins))
    num_rows = np.ceil(num_origins / num_cols)
    xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="xy")
    env_origins[:, 0] = spacing * xx.flatten()[:num_origins] - spacing * (num_rows - 1) / 2
    env_origins[:, 1] = spacing * yy.flatten()[:num_origins] - spacing * (num_cols - 1) / 2
    env_origins[:, 2] = 0.0
    # return the origins
    return env_origins.tolist()


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation

    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Create separate groups called "Origin1", "Origin2", "Origin3"
    # Each group will have a mount and a robot on top of it
    origins = define_origins(num_origins=7, spacing=1.25)

    # Origin 1 with Anymal B
    sim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    # -- Robot
    anymal_b = Articulation(ANYMAL_B_CFG.replace(prim_path="/World/Origin1/Robot"))
    # Origin 2 with Anymal C
    sim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])
    # -- Robot
    anymal_c = Articulation(ANYMAL_C_CFG.replace(prim_path="/World/Origin2/Robot"))

    # Origin 3 with Anymal D
    sim_utils.create_prim("/World/Origin3", "Xform", translation=origins[2])
    # -- Robot
    anymal_d = Articulation(ANYMAL_D_CFG.replace(prim_path="/World/Origin3/Robot"))

    # Origin 4 with Unitree A1
    sim_utils.create_prim("/World/Origin4", "Xform", translation=origins[3])
    # -- Robot
    unitree_a1 = Articulation(UNITREE_A1_CFG.replace(prim_path="/World/Origin4/Robot"))

    # Origin 5 with Unitree Go1
    sim_utils.create_prim("/World/Origin5", "Xform", translation=origins[4])
    # -- Robot
    unitree_go1 = Articulation(UNITREE_GO1_CFG.replace(prim_path="/World/Origin5/Robot"))

    # Origin 6 with Unitree Go2
    sim_utils.create_prim("/World/Origin6", "Xform", translation=origins[5])
    # -- Robot
    unitree_go2 = Articulation(UNITREE_GO2_CFG.replace(prim_path="/World/Origin6/Robot"))

    # Origin 7 with Boston Dynamics Spot
    sim_utils.create_prim("/World/Origin7", "Xform", translation=origins[6])
    # -- Robot
    spot = Articulation(SPOT_CFG.replace(prim_path="/World/Origin7/Robot"))

    # return the scene information
    scene_entities = {
        "anymal_b": anymal_b,
        "anymal_c": anymal_c,
        "anymal_d": anymal_d,
        "unitree_a1": unitree_a1,
        "unitree_go1": unitree_go1,
        "unitree_go2": unitree_go2,
        "spot": spot,
    }
    return scene_entities, origins


def run_simulator(sim, entities: dict, origins):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    # Simulate physics
    # Exit when every visualizer window has been closed (works for kit and newton)
    while not has_no_alive_visualizer_window(sim):
        # reset
        if count % 200 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # reset robots
            for index, robot in enumerate(entities.values()):
                # root state
                root_pose = robot.data.default_root_pose.torch.clone()
                root_pose[:, :3] += origins[index]
                robot.write_root_pose_to_sim_index(root_pose=root_pose)
                root_vel = robot.data.default_root_vel.torch.clone()
                robot.write_root_velocity_to_sim_index(root_velocity=root_vel)
                # joint state
                joint_pos, joint_vel = (
                    robot.data.default_joint_pos.torch.clone(),
                    robot.data.default_joint_vel.torch.clone(),
                )
                robot.write_joint_position_to_sim_index(position=joint_pos)
                robot.write_joint_velocity_to_sim_index(velocity=joint_vel)
                # reset the internal state
                robot.reset()
            print("[INFO]: Resetting robots state...")
        # apply default actions to the quadrupedal robots
        for robot in entities.values():
            # generate random joint positions
            joint_pos_target = robot.data.default_joint_pos.torch + torch.randn_like(robot.data.joint_pos.torch) * 0.1
            # apply action to the robot
            robot.set_joint_position_target_index(target=joint_pos_target)
            # write data to sim
            robot.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        for robot in entities.values():
            robot.update(sim_dt)


def main():
    """Main function."""
    with resolve_backend_and_visualizer(args_cli) as (physics_cfg, visualizer_cfg):
        import isaaclab.sim as sim_utils

        # define simulation time step
        dt = 1 / 200
        # define simulation configuration
        sim_cfg = sim_utils.SimulationCfg(dt=dt, device=args_cli.device, physics=physics_cfg, visualizer_cfgs=[visualizer_cfg])
        # Initialize the simulation context
        sim = sim_utils.SimulationContext(sim_cfg)
        # Set main camera
        sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])
        # design scene
        scene_entities, scene_origins = design_scene()
        scene_origins = torch.tensor(scene_origins, device=sim.device)
        # Play the simulator
        sim.reset()
        # Now we are ready!
        print("[INFO]: Setup complete...")
        # Run the simulator
        run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    main()
