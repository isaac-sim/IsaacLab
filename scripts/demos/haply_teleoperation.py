# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Demonstration of Haply device teleoperation with a robotic arm.

This script demonstrates how to use a Haply device (Inverse3 + VerseGrip) to
teleoperate a robotic arm in Isaac Lab. The Haply provides:
- Position tracking from the Inverse3 device
- Orientation and button inputs from the VerseGrip device
- Force feedback

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/haply_teleoperation.py

    # With custom WebSocket URI
    ./isaaclab.sh -p scripts/demos/haply_teleoperation.py --websocket_uri ws://localhost:10001

    # With sensitivity adjustment
    ./isaaclab.sh -p scripts/demos/haply_teleoperation.py --pos_sensitivity 2.0

Prerequisites:
    1. Install websockets package: pip install websockets
    2. Have Haply SDK running and accessible via WebSocket
    3. Connect Inverse3 and VerseGrip devices
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Demonstration of Haply device teleoperation with Isaac Lab.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument(
    "--websocket_uri",
    type=str,
    default="ws://localhost:10001",
    help="WebSocket URI for Haply SDK connection.",
)
parser.add_argument(
    "--pos_sensitivity",
    type=float,
    default=1.0,
    help="Position sensitivity scaling factor.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.devices import HaplyDevice, HaplyDeviceCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip

# Workspace mapping constants
HAPLY_Z_OFFSET = 0.35
WORKSPACE_LIMITS = {
    "x": (0.1, 0.9),
    "y": (-0.50, 0.50),
    "z": (1.05, 1.85),
}


def apply_haply_to_robot_mapping(
    haply_pos: np.ndarray | torch.Tensor,
    haply_initial_pos: np.ndarray | list,
    robot_initial_pos: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Apply coordinate mapping from Haply workspace to Franka Panda end-effector.

    Uses absolute position control: robot position = robot_initial_pos + haply_pos (transformed)

    Args:
        haply_pos: Current Haply absolute position [x, y, z] in meters
        haply_initial_pos: Haply's zero reference position [x, y, z]
        robot_initial_pos: Base offset for robot end-effector

    Returns:
        robot_pos: Target position for robot EE in world frame [x, y, z]

    """
    # Convert to numpy
    if isinstance(haply_pos, torch.Tensor):
        haply_pos = haply_pos.cpu().numpy()
    if isinstance(robot_initial_pos, torch.Tensor):
        robot_initial_pos = robot_initial_pos.cpu().numpy()

    haply_delta = haply_pos - haply_initial_pos

    # Coordinate system mapping: Haply (X, Y, Z) -> Robot (-Y, X, Z-offset)
    robot_offset = np.array([-haply_delta[1], haply_delta[0], haply_delta[2] - HAPLY_Z_OFFSET])
    robot_pos = robot_initial_pos + robot_offset

    # Apply workspace limits for safety
    robot_pos[0] = np.clip(robot_pos[0], WORKSPACE_LIMITS["x"][0], WORKSPACE_LIMITS["x"][1])
    robot_pos[1] = np.clip(robot_pos[1], WORKSPACE_LIMITS["y"][0], WORKSPACE_LIMITS["y"][1])
    robot_pos[2] = np.clip(robot_pos[2], WORKSPACE_LIMITS["z"][0], WORKSPACE_LIMITS["z"][1])

    return robot_pos


@configclass
class FrankaHaplySceneCfg(InteractiveSceneCfg):
    """Configuration for Franka scene with Haply teleoperation and contact sensors."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.50, 0.0, 1.05), rot=(0.707, 0, 0, 0.707)),
    )

    robot: Articulation = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = (-0.02, 0.0, 1.05)
    robot.spawn.activate_contact_sensors = True

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.06),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.2), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.60, 0.00, 1.15)),
    )

    left_finger_contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
        update_period=0.0,
        history_length=3,
        debug_vis=True,
        track_pose=True,
    )

    right_finger_contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
        update_period=0.0,
        history_length=3,
        debug_vis=True,
        track_pose=True,
    )


def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    haply_device: HaplyDevice,
):
    """Runs the simulation loop with Haply teleoperation."""
    sim_dt = sim.get_physics_dt()
    count = 1

    robot: Articulation = scene["robot"]
    cube: RigidObject = scene["cube"]
    left_finger_sensor: ContactSensor = scene["left_finger_contact_sensor"]
    right_finger_sensor: ContactSensor = scene["right_finger_contact_sensor"]

    ee_body_name = "panda_hand"
    ee_body_idx = robot.body_names.index(ee_body_name)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_pos[0, :7] = torch.tensor([0.0, -0.569, 0.0, -2.81, 0.0, 3.037, 0.741], device=robot.device)
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    for _ in range(10):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    # Initialize the position of franka
    robot_initial_pos = robot.data.body_pos_w[0, ee_body_idx].cpu().numpy()
    haply_initial_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    ik_controller_cfg = DifferentialIKControllerCfg(
        command_type="position",
        use_relative_mode=False,
        ik_method="dls",
        ik_params={"lambda_val": 0.05},
    )

    # IK joints control arms, buttons control ee rotation and gripper open/close
    arm_joint_names = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
    ]
    arm_joint_indices = [robot.joint_names.index(name) for name in arm_joint_names]

    # Initialize IK controller
    ik_controller = DifferentialIKController(cfg=ik_controller_cfg, num_envs=scene.num_envs, device=sim.device)
    initial_ee_quat = robot.data.body_quat_w[:, ee_body_idx]
    ik_controller.set_command(command=torch.zeros(scene.num_envs, 3, device=sim.device), ee_quat=initial_ee_quat)

    prev_button_a = False
    prev_button_b = False
    prev_button_c = False
    gripper_target = 0.04

    # Initialize the rotation of franka end-effector
    ee_rotation_angle = robot.data.joint_pos[0, 6].item()
    rotation_step = np.pi / 3

    print("\n[INFO] Teleoperation ready!")
    print("  Move handler: Control pose of the end-effector")
    print("  Button A: Open | Button B: Close | Button C: Rotate EE (60°)\n")

    while simulation_app.is_running():
        if count % 10000 == 0:
            count = 1
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])

            joint_pos = robot.data.default_joint_pos.clone()
            joint_pos[0, :7] = torch.tensor([0.0, -0.569, 0.0, -2.81, 0.0, 3.037, 0.741], device=robot.device)
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)

            cube_state = cube.data.default_root_state.clone()
            cube_state[:, :3] += scene.env_origins
            cube.write_root_pose_to_sim(cube_state[:, :7])
            cube.write_root_velocity_to_sim(cube_state[:, 7:])

            scene.reset()
            haply_device.reset()
            ik_controller.reset()
            print("[INFO]: Resetting robot state...")

        # Get the data from Haply device
        haply_data = haply_device.advance()

        haply_pos = haply_data[:3]
        button_a = haply_data[7].item() > 0.5
        button_b = haply_data[8].item() > 0.5
        button_c = haply_data[9].item() > 0.5

        if button_a and not prev_button_a:
            gripper_target = 0.04  # Open gripper
        if button_b and not prev_button_b:
            gripper_target = 0.0  # Close gripper
        if button_c and not prev_button_c:
            joint_7_limit = 3.0
            ee_rotation_angle += rotation_step

            if ee_rotation_angle > joint_7_limit:
                ee_rotation_angle = -joint_7_limit + (ee_rotation_angle - joint_7_limit)
            elif ee_rotation_angle < -joint_7_limit:
                ee_rotation_angle = joint_7_limit + (ee_rotation_angle + joint_7_limit)

        prev_button_a = button_a
        prev_button_b = button_b
        prev_button_c = button_c

        # Compute IK
        target_pos = apply_haply_to_robot_mapping(
            haply_pos,
            haply_initial_pos,
            robot_initial_pos,
        )

        target_pos_tensor = torch.tensor(target_pos, dtype=torch.float32, device=sim.device).unsqueeze(0)

        current_joint_pos = robot.data.joint_pos[:, arm_joint_indices]
        ee_pos_w = robot.data.body_pos_w[:, ee_body_idx]
        ee_quat_w = robot.data.body_quat_w[:, ee_body_idx]

        # get jacobian to IK controller
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_body_idx, :, arm_joint_indices]
        ik_controller.set_command(command=target_pos_tensor, ee_quat=ee_quat_w)
        joint_pos_des = ik_controller.compute(ee_pos_w, ee_quat_w, jacobian, current_joint_pos)

        joint_pos_target = robot.data.joint_pos[0].clone()

        # Update joints: 6 from IK + 1 from button control (correct by design)
        joint_pos_target[arm_joint_indices] = joint_pos_des[0]  # panda_joint1-6 from IK
        joint_pos_target[6] = ee_rotation_angle  # panda_joint7 - end-effector rotation (button C)
        joint_pos_target[[-2, -1]] = gripper_target  # gripper

        robot.set_joint_position_target(joint_pos_target.unsqueeze(0))

        for _ in range(5):
            scene.write_data_to_sim()
            sim.step()

        scene.update(sim_dt)
        count += 1

        # get contact forces and apply force feedback
        left_finger_forces = left_finger_sensor.data.net_forces_w[0, 0]
        right_finger_forces = right_finger_sensor.data.net_forces_w[0, 0]
        total_contact_force = (left_finger_forces + right_finger_forces) * 0.5
        haply_device.push_force(forces=total_contact_force.unsqueeze(0), position=torch.tensor([0]))


def main():
    """Main function to set up and run the Haply teleoperation demo."""
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=1 / 200)
    sim = sim_utils.SimulationContext(sim_cfg)

    # set the simulation view
    sim.set_camera_view([1.6, 1.0, 1.70], [0.4, 0.0, 1.0])

    scene_cfg = FrankaHaplySceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # Create Haply device
    haply_cfg = HaplyDeviceCfg(
        websocket_uri=args_cli.websocket_uri,
        pos_sensitivity=args_cli.pos_sensitivity,
        sim_device=args_cli.device,
        limit_force=2.0,
    )
    haply_device = HaplyDevice(cfg=haply_cfg)
    print(f"[INFO] Haply connected: {args_cli.websocket_uri}")

    sim.reset()

    run_simulator(sim, scene, haply_device)


if __name__ == "__main__":
    main()
    simulation_app.close()
