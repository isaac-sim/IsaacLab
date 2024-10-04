# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the operational space controller (OSC) with the simulator.

The OSC controller can be configured in different modes. It uses the dynamical quantities such as Jacobians and 
mass matricescomputed by PhysX. 

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/05_controllers/run_osc.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, AssetBaseCfg
from omni.isaac.lab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.sensors import ContactSensorCfg
from omni.isaac.lab.utils.math import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_inv,
    quat_rotate_inverse,
    subtract_frame_transforms,
)

##
# Pre-defined configs
##
from omni.isaac.lab_assets import FRANKA_PANDA_HIGH_PD_CFG  # isort:skip


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for a simple scene with a tilted wall."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # Tilted wall
    tilted_wall = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TiltedWall",
        spawn=sim_utils.CuboidCfg(
            size=(2.0, 1.5, 0.01),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), opacity=0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            activate_contact_sensors=True,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.6 + 0.085, 0.0, 0.3), rot=(0.9238795325, 0.0, -0.3826834324, 0.0)
        ),
    )

    contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/TiltedWall",
        update_period=0.0,
        history_length=2,
        debug_vis=False,
    )

    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.actuators["panda_shoulder"].stiffness = 0.0
    robot.actuators["panda_shoulder"].damping = 0.0
    robot.actuators["panda_forearm"].stiffness = 0.0
    robot.actuators["panda_forearm"].damping = 0.0
    robot.spawn.rigid_props.disable_gravity = True


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""

    # Extract scene entities for readability.
    robot = scene["robot"]
    contact_forces = scene["contact_forces"]

    # Obtain indices for the end-effector and arm joints
    ee_frame_name = "panda_leftfinger"
    arm_joint_names = ["panda_joint.*"]
    ee_frame_idx = robot.find_bodies(ee_frame_name)[0][0]
    arm_joint_ids = robot.find_joints(arm_joint_names)[0]

    # Create the OSC
    opc_cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs", "wrench_abs"],
        impedance_mode="variable_kp",
        inertial_compensation=True,
        decoupled_motion_calculations=False,
        gravity_compensation=False,
        motion_damping_ratio_task=1.0,
        contact_wrench_stiffness_task=[0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
        motion_control_axes_task=[1, 1, 0, 1, 1, 1],
        contact_wrench_control_axes_task=[0, 0, 1, 0, 0, 0],
    )
    opc = OperationalSpaceController(opc_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Define targets for the arm
    ee_goal_pose_set_tilted_b = torch.tensor([
        [0.6, 0.15, 0.3, 0.0, 0.92387953, 0.0, 0.38268343],
        [0.6, -0.3, 0.3, 0.0, 0.92387953, 0.0, 0.38268343],
        [0.8, 0.0, 0.5, 0.0, 0.92387953, 0.0, 0.38268343],
    ], device=sim.device)
    ee_goal_wrench_set_tilted_task = torch.tensor([
        [0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
    ], device=sim.device)
    kp_set_task = torch.tensor(
        [
            [360.0, 360.0, 360.0, 360.0, 360.0, 360.0],
            [420.0, 420.0, 420.0, 420.0, 420.0, 420.0],
            [320.0, 320.0, 320.0, 320.0, 320.0, 320.0],
        ],
        device=sim.device,
    )
    ee_target_set = torch.cat([ee_goal_pose_set_tilted_b, ee_goal_wrench_set_tilted_task, kp_set_task], dim=-1)

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    # Update existing buffers
    # Note: We need to update buffers before the first step for the controller.
    robot.update(dt=sim_dt)

    # get the updated states
    jacobian_b, mass_matrix, gravity, ee_pose_b, ee_vel_b, root_pose_w, ee_pose_w, ee_force_b = update_states(
        sim, scene, robot, ee_frame_idx, arm_joint_ids, contact_forces
    )

    # Track the given target command
    current_goal_idx = 0  # Current goal index for the arm
    command = torch.zeros(
        scene.num_envs, opc.action_dim, device=sim.device
    )  # Generic target command, which can be pose, position, force, etc.
    ee_target_pose_b = torch.zeros(scene.num_envs, 7, device=sim.device)  # Target pose in the body frame
    ee_target_pose_w = torch.zeros(
        scene.num_envs, 7, device=sim.device
    )  # Target pose in the world frame (for marker)

    # Set joint efforts to zero
    zero_joint_efforts = torch.zeros(scene.num_envs, robot.num_joints, device=sim.device)
    joint_efforts = torch.zeros(scene.num_envs, len(arm_joint_ids), device=sim.device)

    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # reset every 500 steps
        if count % 500 == 0:
            # reset joint state to default
            default_joint_pos = robot.data.default_joint_pos.clone()
            default_joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
            robot.set_joint_effort_target(zero_joint_efforts)  # Set zero torques in the initial step
            robot.write_data_to_sim()
            robot.reset()
            # reset contact sensor
            contact_forces.reset()
            # reset target pose
            robot.update(sim_dt)
            _, _, _, ee_pose_b, _, _, _, _ = update_states(
                sim, scene, robot, ee_frame_idx, arm_joint_ids, contact_forces
            )  # at reset, the jacobians are not updated to the latest state
            command, ee_target_pose_b, ee_target_pose_w, current_goal_idx = update_target(
                sim, scene, opc, root_pose_w, ee_pose_b, ee_target_set, current_goal_idx
            )
            # set the opc command
            opc.reset()
            command, task_frame_pose_b = convert_to_task_frame(
                opc, command=command, ee_target_pose_b=ee_target_pose_b
            )
            opc.set_command(
                command=command, current_ee_pose_b=ee_pose_b, current_task_frame_pose_b=task_frame_pose_b
            )
        else:
            # get the updated states
            jacobian_b, mass_matrix, gravity, ee_pose_b, ee_vel_b, root_pose_w, ee_pose_w, ee_force_b = (
                update_states(sim, scene, robot, ee_frame_idx, arm_joint_ids, contact_forces)
            )
            # compute the joint commands
            joint_efforts = opc.compute(
                jacobian_b=jacobian_b,
                current_ee_pose_b=ee_pose_b,
                current_ee_vel_b=ee_vel_b,
                current_ee_force_b=ee_force_b,
                mass_matrix=mass_matrix,
                gravity=gravity,
            )
            robot.set_joint_effort_target(joint_efforts, joint_ids=arm_joint_ids)
            robot.write_data_to_sim()

        # update marker positions
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(ee_target_pose_w[:, 0:3], ee_target_pose_w[:, 3:7])

        # perform step
        sim.step(render=True)
        # update robot buffers
        robot.update(sim_dt)
        # update buffers
        scene.update(sim_dt)
        # update sim-time
        count += 1


def update_states(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    ee_frame_idx: int,
    arm_joint_ids: list[int],
    contact_forces,
):
    # obtain dynamics related quantities from simulation
    ee_jacobi_idx = ee_frame_idx - 1
    jacobian_w = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_joint_ids]
    mass_matrix = robot.root_physx_view.get_mass_matrices()[:, arm_joint_ids, :][:, :, arm_joint_ids]
    gravity = robot.root_physx_view.get_generalized_gravity_forces()[:, arm_joint_ids]
    # Convert the Jacobian from world to root frame
    jacobian_b = jacobian_w.clone()
    root_rot_matrix = matrix_from_quat(quat_inv(robot.data.root_state_w[:, 3:7]))
    jacobian_b[:, :3, :] = torch.bmm(root_rot_matrix, jacobian_b[:, :3, :])
    jacobian_b[:, 3:, :] = torch.bmm(root_rot_matrix, jacobian_b[:, 3:, :])

    # Compute current pose of the end-effector
    root_pose_w = robot.data.root_state_w[:, 0:7]
    ee_pose_w = robot.data.body_state_w[:, ee_frame_idx, 0:7]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    ee_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)

    # Compute the current velocity of the end-effector
    ee_vel_w = robot.data.body_vel_w[:, ee_frame_idx, :]  # Extract end-effector velocity in the world frame
    root_vel_w = robot.data.root_vel_w  # Extract root velocity in the world frame
    relative_vel_w = ee_vel_w - root_vel_w  # Compute the relative velocity in the world frame
    ee_lin_vel_b = quat_rotate_inverse(robot.data.root_quat_w, relative_vel_w[:, 0:3])  # From world to root frame
    ee_ang_vel_b = quat_rotate_inverse(robot.data.root_quat_w, relative_vel_w[:, 3:6])
    ee_vel_b = torch.cat([ee_lin_vel_b, ee_ang_vel_b], dim=-1)

    # Calculate the contact force
    ee_force_w = torch.zeros(scene.num_envs, 3, device=sim.device)
    sim_dt = sim.get_physics_dt()
    contact_forces.update(sim_dt)  # update contact sensor
    # Calculate the contact force by averaging over last four time steps (i.e., to smoothen) and
    # taking the max of three surfaces as only one should be the contact of interest
    ee_force_w, _ = torch.max(torch.mean(contact_forces.data.net_forces_w_history, dim=1), dim=1)

    # This is a simplification, only for the sake of testing.
    ee_force_b = ee_force_w

    return jacobian_b, mass_matrix, gravity, ee_pose_b, ee_vel_b, root_pose_w, ee_pose_w, ee_force_b


def update_target(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    opc: OperationalSpaceController,
    root_pose_w: torch.tensor,
    ee_pose_b: torch.tensor,
    ee_target_set: torch.tensor,
    current_goal_idx: int,
):
    # update the ee desired command
    command = torch.zeros(scene.num_envs, opc.action_dim, device=sim.device)
    command[:] = ee_target_set[current_goal_idx]

    # update the ee desired pose
    ee_target_pose_b = torch.zeros(scene.num_envs, 7, device=sim.device)
    for target_type in opc.cfg.target_types:
        if target_type == "pose_abs":
            ee_target_pose_b[:] = command[:, :7]
        elif target_type == "wrench_abs":
            pass  # ee_target_pose_b could stay at the root frame for force control, what matters is ee_target_b
        else:
            raise ValueError("Undefined target_type within update_target().")

    # update the target desired pose in world frame (for marker)
    ee_target_pos_w, ee_target_quat_w = combine_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_target_pose_b[:, 0:3], ee_target_pose_b[:, 3:7]
    )
    ee_target_pose_w = torch.cat([ee_target_pos_w, ee_target_quat_w], dim=-1)

    next_goal_idx = (current_goal_idx + 1) % len(ee_target_set)

    return command, ee_target_pose_b, ee_target_pose_w, next_goal_idx


def convert_to_task_frame(
    opc: OperationalSpaceController, command: torch.tensor, ee_target_pose_b: torch.tensor
):
    command = command.clone()
    task_frame_pose_b = ee_target_pose_b.clone()

    cmd_idx = 0
    for target_type in opc.cfg.target_types:
        if target_type == "pose_abs":
            command[:, :3], command[:, 3:7] = subtract_frame_transforms(
                task_frame_pose_b[:, :3], task_frame_pose_b[:, 3:], command[:, :3], command[:, 3:7]
            )
            cmd_idx += 7
        elif target_type == "wrench_abs":
            # These are already defined in target frame for ee_goal_wrench_set_tilted_task (since it is
            # easier), so not transforming
            cmd_idx += 6
        else:
            raise ValueError("Undefined target_type within _convert_to_task_frame().")

    return command, task_frame_pose_b


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
