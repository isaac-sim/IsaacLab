# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates different single-arm manipulators.

.. code-block:: bash

    # Usage with default PhysX physics and default kit visualizer.
    uv run python scripts/demos/arms.py

    # Usage with Newton visualizer and default PhysX physics.
    uv run python scripts/demos/arms.py --visualizer newton

    # Usage with Newton (MJWarp) physics and default kit visualizer.
    uv run python scripts/demos/arms.py --physics newton_mjwarp

    # Usage with Newton visualizer and Newton (MJWarp) physics.
    uv run python scripts/demos/arms.py --visualizer newton --physics newton_mjwarp

"""

"""Parse CLI first so we can decide whether to launch Isaac Sim Kit."""

import argparse
from typing import TYPE_CHECKING

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(
    description="This script demonstrates different single-arm manipulators.",
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
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg  # isort:skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort:skip
from isaaclab_assets.robots.kinova import KINOVA_GEN3_N7_CFG, KINOVA_JACO2_N6S300_CFG, KINOVA_JACO2_N7S300_CFG  # isort:skip
from isaaclab_assets.robots.sawyer import SAWYER_CFG  # isort:skip
from isaaclab_assets.robots.universal_robots import UR10_CFG  # isort:skip

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


def define_origins(num_origins: int, spacing: float) -> list[list[float]]:
    """Defines the origins of the scene."""
    # create tensor based on number of environments
    env_origins = torch.zeros(num_origins, 3)
    # create a grid of origins
    num_rows = np.floor(np.sqrt(num_origins))
    num_cols = np.ceil(num_origins / num_rows)
    xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="xy")
    env_origins[:, 0] = spacing * xx.flatten()[:num_origins] - spacing * (num_rows - 1) / 2
    env_origins[:, 1] = spacing * yy.flatten()[:num_origins] - spacing * (num_cols - 1) / 2
    env_origins[:, 2] = 0.0
    # return the origins
    return env_origins.tolist()


def design_scene() -> InteractiveScene:
    """Designs the scene."""
    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=0.0, filter_collisions=False)
    scene_cfg.ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    scene_cfg.light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    )
    origins = define_origins(num_origins=6, spacing=2.0)
    franka_arm_cfg = FRANKA_PANDA_CFG.copy()
    franka_arm_cfg.spawn.usd_path = f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    seattle_rotation = (0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5))  # Authored 90-degree rotation around Z.
    mounts = (
        ("franka_panda", franka_arm_cfg, "SeattleLabTable", 1.05, 0.55, None, seattle_rotation),
        ("ur10", UR10_CFG, "Stand", 1.03, 0.0, (2.0, 2.0, 2.0), (0.0, 0.0, 0.0, 1.0)),
        ("kinova_j2n7s300", KINOVA_JACO2_N7S300_CFG, "ThorlabsTable", 0.8, 0.0, None, (0.0, 0.0, 0.0, 1.0)),
        ("kinova_j2n6s300", KINOVA_JACO2_N6S300_CFG, "ThorlabsTable", 0.8, 0.0, None, (0.0, 0.0, 0.0, 1.0)),
        ("kinova_gen3n7", KINOVA_GEN3_N7_CFG, "SeattleLabTable", 1.05, 0.55, None, seattle_rotation),
        ("sawyer", SAWYER_CFG, "Stand", 1.03, 0.0, (2.0, 2.0, 2.0), (0.0, 0.0, 0.0, 1.0)),
    )
    for index, (name, robot_cfg, mount, height, table_x, scale, rotation) in enumerate(mounts):
        x, y, _ = origins[index]
        root = f"/World/Origin{index + 1}"
        table_file = "stand_instanceable.usd" if mount == "Stand" else "table_instanceable.usd"
        setattr(
            scene_cfg,
            f"{name}_table",
            AssetBaseCfg(
                prim_path=f"{root}/Table",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/{mount}/{table_file}", scale=scale
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(x + table_x, y, height), rot=rotation),
            ),
        )
        robot_cfg = robot_cfg.replace(prim_path=f"{root}/Robot")
        robot_cfg.init_state.pos = (x, y, height)
        setattr(scene_cfg, name, robot_cfg)
    return InteractiveScene(scene_cfg)


def run_simulator(sim: "sim_utils.SimulationContext", entities: dict[str, "Articulation"]):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    # Step while a visualizer window is still open (or none exist, e.g. headless); works for kit and newton.
    while sim.is_headless_or_exist_active_visualizer():
        # reset
        if count % 200 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # reset the scene entities
            for robot in entities.values():
                # root state
                root_pose = robot.data.default_root_pose.torch.clone()
                robot.write_root_pose_to_sim_index(root_pose=root_pose)
                root_vel = robot.data.default_root_vel.torch.clone()
                robot.write_root_velocity_to_sim_index(root_velocity=root_vel)
                # set joint positions
                joint_pos, joint_vel = (
                    robot.data.default_joint_pos.torch.clone(),
                    robot.data.default_joint_vel.torch.clone(),
                )
                robot.write_joint_position_to_sim_index(position=joint_pos)
                robot.write_joint_velocity_to_sim_index(velocity=joint_vel)
                # clear internal buffers
                robot.reset()
            print("[INFO]: Resetting robots state...")
        # apply random actions to the robots
        for robot in entities.values():
            # generate random joint positions
            joint_pos_target = robot.data.default_joint_pos.torch + torch.randn_like(robot.data.joint_pos.torch) * 0.1
            soft_limits = robot.data.soft_joint_pos_limits.torch
            joint_pos_target = joint_pos_target.clamp_(soft_limits[..., 0], soft_limits[..., 1])
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
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        # The default newton mjwarp solver configuration needs to be tuned for these arms.
        if isinstance(physics_cfg, NewtonCfg) and isinstance(physics_cfg.solver_cfg, MJWarpSolverCfg):
            physics_cfg.solver_cfg.njmax = 70
            physics_cfg.solver_cfg.nconmax = 70
            physics_cfg.solver_cfg.ls_iterations = 40
            physics_cfg.solver_cfg.cone = "elliptic"
            physics_cfg.solver_cfg.impratio = 100
            physics_cfg.solver_cfg.ls_parallel = False
            physics_cfg.solver_cfg.integrator = "implicitfast"
            physics_cfg.num_substeps = 2

        # Initialize the simulation context
        sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, physics=physics_cfg)
        sim = sim_utils.SimulationContext(sim_cfg)
        # Set main camera
        sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
        # design scene
        scene = design_scene()
        # Play the simulator
        sim.reset()
        # Now we are ready!
        print("[INFO]: Setup complete...")
        # Run the simulator
        run_simulator(sim, scene.articulations)


if __name__ == "__main__":
    # run the main function
    main()
