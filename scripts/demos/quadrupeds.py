# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates different legged robots.

.. code-block:: bash

    # Usage with default PhysX physics and default kit visualizer.
    uv run python scripts/demos/quadrupeds.py

    # Usage with Newton visualizer and default PhysX physics.
    uv run python scripts/demos/quadrupeds.py --visualizer newton

    # Usage with Newton (MJWarp) physics and default kit visualizer.
    uv run python scripts/demos/quadrupeds.py --physics newton_mjwarp

    # Usage with Newton visualizer and Newton (MJWarp) physics.
    uv run python scripts/demos/quadrupeds.py --visualizer newton --physics newton_mjwarp

"""

"""Parse CLI first so we can decide whether to launch Isaac Sim Kit."""

import argparse
from typing import TYPE_CHECKING

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(
    description="This script demonstrates different legged robots.",
    conflict_handler="resolve",
)
parser.add_argument(
    "--physics", default="isaacsim_physx", choices=["isaacsim_physx", "newton_mjwarp"], help="Physics backend."
)
add_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli = parser.parse_args()

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg

##
# Pre-defined configs
##
from isaaclab.physics import PhysicsCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.anymal import ANYMAL_B_CFG, ANYMAL_C_CFG, ANYMAL_D_CFG  # isort:skip
from isaaclab_assets.robots.spot import SPOT_CFG  # isort:skip
from isaaclab_assets.robots.unitree import UNITREE_A1_CFG, UNITREE_GO1_CFG, UNITREE_GO2_CFG  # isort:skip

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


@configclass
class QuadrupedsSceneCfg(InteractiveSceneCfg):
    """Seven quadrupeds arranged on a 1.25 m grid."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    )
    anymal_b = ANYMAL_B_CFG.replace(prim_path="/World/Origin1/Robot")
    anymal_b.init_state.pos = (-1.875, -0.625, 0.6)
    anymal_c = ANYMAL_C_CFG.replace(prim_path="/World/Origin2/Robot")
    anymal_c.init_state.pos = (-0.625, -0.625, 0.6)
    anymal_d = ANYMAL_D_CFG.replace(prim_path="/World/Origin3/Robot")
    anymal_d.init_state.pos = (0.625, -0.625, 0.6)
    unitree_a1 = UNITREE_A1_CFG.replace(prim_path="/World/Origin4/Robot")
    unitree_a1.init_state.pos = (1.875, -0.625, 0.42)
    unitree_go1 = UNITREE_GO1_CFG.replace(prim_path="/World/Origin5/Robot")
    unitree_go1.init_state.pos = (-1.875, 0.625, 0.4)
    unitree_go2 = UNITREE_GO2_CFG.replace(prim_path="/World/Origin6/Robot")
    unitree_go2.init_state.pos = (-0.625, 0.625, 0.4)
    spot = SPOT_CFG.replace(prim_path="/World/Origin7/Robot")
    spot.init_state.pos = (0.625, 0.625, 0.5)


def run_simulator(sim: "sim_utils.SimulationContext", entities: dict[str, "Articulation"]):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Step while a visualizer window is still open (or none exist, e.g. headless); works for kit and newton.
    while sim.is_headless_or_exist_active_visualizer():
        # Reset robots every 200 steps.
        if count % 200 == 0:
            # reset counters
            count = 0
            # reset robots
            for robot in entities.values():
                # root state
                root_pose = robot.data.default_root_pose.torch.clone()
                robot.write_root_pose_to_sim_index(root_pose=root_pose)
                root_vel = robot.data.default_root_vel.torch.clone()
                robot.write_root_velocity_to_sim_index(root_velocity=root_vel)
                # joint state
                joint_pos = robot.data.default_joint_pos.torch.clone()
                robot.write_joint_position_to_sim_index(position=joint_pos)
                joint_vel = robot.data.default_joint_vel.torch.clone()
                robot.write_joint_velocity_to_sim_index(velocity=joint_vel)
                # reset the internal state
                robot.reset()
            print("[INFO]: Reset robots' state...")
        # Apply default actions to the quadrupedal robots.
        for robot in entities.values():
            # generate random joint positions
            joint_pos_target = robot.data.default_joint_pos.torch + torch.randn_like(robot.data.joint_pos.torch) * 0.1
            # apply action to the robot
            robot.set_joint_position_target_index(target=joint_pos_target)
            # write data to sim
            robot.write_data_to_sim()
        # perform step
        sim.step()
        # update counter
        count += 1
        # update buffers
        for robot in entities.values():
            robot.update(sim_dt)


def main():
    """Main function."""
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        dt = 1 / 200
        sim_cfg: sim_utils.SimulationCfg = sim_utils.SimulationCfg(dt=dt, device=args_cli.device, physics=physics_cfg)
        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])
        scene_cfg = QuadrupedsSceneCfg(num_envs=1, env_spacing=0.0, filter_collisions=False)
        scene = scene_cfg.class_type(scene_cfg)
        sim.reset()
        print("[INFO]: Setup complete...")
        run_simulator(sim, scene.articulations)


if __name__ == "__main__":
    main()
