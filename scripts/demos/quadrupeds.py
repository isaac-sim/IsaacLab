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

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg

##
# Pre-defined configs
##
from isaaclab.physics import PhysicsCfg
from isaaclab.scene import InteractiveSceneCfg

from isaaclab_assets.robots.anymal import ANYMAL_B_CFG, ANYMAL_C_CFG, ANYMAL_D_CFG  # isort:skip
from isaaclab_assets.robots.spot import SPOT_CFG  # isort:skip
from isaaclab_assets.robots.unitree import UNITREE_A1_CFG, UNITREE_GO1_CFG, UNITREE_GO2_CFG  # isort:skip

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.scene import InteractiveScene


def define_origins(num_origins: int, spacing: float) -> torch.Tensor:
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
    return env_origins


def design_scene() -> "InteractiveScene":
    """Designs the scene."""
    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=0.0, filter_collisions=False)
    scene_cfg.ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    scene_cfg.light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    )
    origins = define_origins(num_origins=7, spacing=1.25)
    robots = {
        "anymal_b": ANYMAL_B_CFG,
        "anymal_c": ANYMAL_C_CFG,
        "anymal_d": ANYMAL_D_CFG,
        "unitree_a1": UNITREE_A1_CFG,
        "unitree_go1": UNITREE_GO1_CFG,
        "unitree_go2": UNITREE_GO2_CFG,
        "spot": SPOT_CFG,
    }
    for index, (name, cfg) in enumerate(robots.items()):
        cfg = cfg.replace(prim_path=f"/World/Origin{index + 1}/Robot")
        cfg.init_state.pos = tuple(p + float(o) for p, o in zip(cfg.init_state.pos, origins[index], strict=True))
        setattr(scene_cfg, name, cfg)
    return scene_cfg.class_type(scene_cfg)


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
        scene = design_scene()
        sim.reset()
        print("[INFO]: Setup complete...")
        run_simulator(sim, scene.articulations)


if __name__ == "__main__":
    main()
