# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to view ARL Robot 1.

.. code-block:: bash

    # Usage with default Newton physics and Newton GL visualizer.
    uvx isaaclab example arl-robot-1

    # Previous PhysX/Kit path.
    uvx --from 'isaaclab[isaacsim]' isaaclab example arl-robot-1 --physics isaacsim_physx --viz kit

"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils import instantiate, replace

parser = argparse.ArgumentParser(
    description="View ARL Robot 1 with Lee Position Controller.",
    conflict_handler="resolve",
)
parser.add_argument(
    "--physics", default="newton_mjwarp", choices=["isaacsim_physx", "newton_mjwarp"], help="Physics backend."
)
parser.add_argument("--max_steps", type=int, default=-1, help="Stop after this many steps; negative runs forever.")
add_launcher_args(parser)
parser.set_defaults(visualizer=["newton_gl"])
args_cli = parser.parse_args()
if args_cli.max_steps == 0 or args_cli.max_steps < -1:
    parser.error("--max_steps must be positive or -1.")

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.physics import PhysicsCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from isaaclab_contrib.actuators import Thruster
from isaaclab_contrib.controllers.lee_position_control import LeePosController
from isaaclab_contrib.controllers.lee_position_control_cfg import LeePosControllerCfg
from isaaclab_contrib.utils.types import MultiRotorActions

from isaaclab_assets.robots.arl_robot_1 import ARL_ROBOT_1_CFG


@configclass
class ARLSceneCfg(InteractiveSceneCfg):
    """ARL robot and its shared world assets."""

    robot: ArticulationCfg = replace(ARL_ROBOT_1_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    ground: AssetBaseCfg = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/DomeLight", spawn=sim_utils.DomeLightCfg(intensity=1000.0, color=(0.53, 0.81, 0.92))
    )


def main():
    """Main function to spawn arl_robot_1."""
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        # Create simulation context
        sim_cfg = sim_utils.SimulationCfg(
            dt=0.01,
            device=args_cli.device,
            physics=physics_cfg,
            use_newton_actuators=False,
        )
        sim = sim_utils.SimulationContext(sim_cfg)

        scene_cfg = ARLSceneCfg(num_envs=1, env_spacing=2.0)
        robot_cfg = scene_cfg.robot
        robot_cfg.actuators["thrusters"].dt = sim_cfg.dt
        if args_cli.physics == "newton_mjwarp":
            thruster_cfg = robot_cfg.actuators["thrusters"]
            init_state = robot_cfg.init_state
            articulation_cfg = ArticulationCfg(
                prim_path=robot_cfg.prim_path,
                spawn=robot_cfg.spawn,
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=init_state.pos,
                    rot=init_state.rot,
                    lin_vel=init_state.lin_vel,
                    ang_vel=init_state.ang_vel,
                    joint_pos={},
                    joint_vel={},
                ),
                actuators={},
            )
            scene_cfg.robot = articulation_cfg
        scene = instantiate(scene_cfg)
        robot = scene["robot"]
        sim.reset()

        # Create Lee position controller
        controller_cfg = LeePosControllerCfg(
            K_pos_range=((2.5, 2.5, 1.5), (3.5, 3.5, 2.0)),
            K_vel_range=((2.5, 2.5, 1.5), (3.5, 3.5, 2.0)),
            K_rot_range=((1.6, 1.6, 0.25), (1.85, 1.85, 0.4)),
            K_angvel_range=((0.4, 0.4, 0.075), (0.5, 0.5, 0.09)),
            max_inclination_angle_rad=1.0471975511965976,
            max_yaw_rate=1.0471975511965976,
        )
        controller = LeePosController(controller_cfg, robot, num_envs=1, device=str(sim.device))

        # Get allocation matrix and compute pseudoinverse
        allocation_matrix = torch.tensor(robot_cfg.allocation_matrix, device=sim.device, dtype=torch.float32)
        # allocation_matrix is (6, num_thrusters), we need pseudoinverse for wrench -> thrust
        alloc_pinv = torch.linalg.pinv(allocation_matrix)  # Shape: (num_thrusters, 6)
        if args_cli.physics == "newton_mjwarp":
            thruster_names = thruster_cfg.thruster_names_expr
            initial_rps = torch.tensor(
                [[robot_cfg.init_state.rps[name] for name in thruster_names]], device=sim.device, dtype=torch.float32
            )
            thruster: Thruster = instantiate(
                thruster_cfg,
                thruster_names=thruster_names,
                thruster_ids=slice(None),
                num_envs=1,
                device=str(sim.device),
                init_thruster_rps=initial_rps,
            )

        # Position command: hover in place (zero position, zero yaw)
        pos_command = torch.zeros((1, 4), device=sim.device)  # [x, y, z, yaw]
        pos_command[0, 2] = 1.0  # Hover at 1 meter height

        # Simulation loop
        print("[INFO] Starting example with Lee Position Controller. Press Ctrl+C to stop.")

        step_count = 0
        # Step while a visualizer window is still open (or none exist, e.g. headless); works for kit and newton.
        while sim.is_running() and (args_cli.max_steps < 0 or step_count < args_cli.max_steps):
            # Compute wrench from position controller
            wrench = controller.compute(pos_command)  # Shape: (1, 6)

            # Allocate wrench to thrusters: thrust = pinv(A) @ wrench
            thrust_cmd = torch.matmul(wrench, alloc_pinv.T)  # Shape: (1, num_thrusters)
            thrust_cmd = thrust_cmd.clamp(min=0.0)  # Ensure non-negative thrust

            # Apply thrust
            if args_cli.physics == "newton_mjwarp":
                thrust = thruster.compute(MultiRotorActions(thrusts=thrust_cmd)).thrusts
                wrench_b = thrust @ allocation_matrix.T
                robot.permanent_wrench_composer.set_forces_and_torques_index(
                    forces=wrench_b[:, None, :3], torques=wrench_b[:, None, 3:], body_ids=[0]
                )
            else:
                robot.set_thrust_target(thrust_cmd)

            # Step simulation
            robot.write_data_to_sim()
            sim.step()
            step_count += 1

            # Update robot
            robot.update(sim_cfg.dt)


if __name__ == "__main__":
    main()
