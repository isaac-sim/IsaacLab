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
- Bidirectional force feedback

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/haply_teleoperation.py

    # With custom WebSocket URI
    ./isaaclab.sh -p scripts/demos/haply_teleoperation.py --websocket_uri ws://localhost:10001

    # With sensitivity adjustment
    ./isaaclab.sh -p scripts/demos/haply_teleoperation.py --sensitivity 2.0

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
parser.add_argument(
    "--enable_force_feedback",
    action="store_true",
    default=False,
    help="Enable force feedback (experimental).",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.devices import HaplyDevice, HaplyDeviceCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_mul, subtract_frame_transforms

##
# Pre-defined configs
##
from isaaclab_assets import FRANKA_PANDA_CFG, FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


def apply_haply_to_robot_mapping(haply_pos, haply_quat, haply_initial_pos, robot_initial_pos, debug=False):
    """Apply coordinate mapping from Haply workspace to Franka Panda end-effector.
    
    Uses absolute position control: robot position = robot_initial_pos + haply_pos (transformed)
    
    Args:
        haply_pos: Current Haply absolute position [x, y, z] in meters
        haply_quat: Current Haply quaternion [qx, qy, qz, qw] from VerseGrip
        haply_initial_pos: Haply's zero reference position [x, y, z]
        robot_initial_pos: Base offset for robot end-effector
        debug: If True, print detailed calculation steps
    
    Returns:
        robot_pos: Target position for robot EE in world frame [x, y, z]
        robot_quat: Target quaternion in Isaac format [qw, qx, qy, qz]
    """
    # Convert to numpy
    if isinstance(haply_pos, torch.Tensor):
        haply_pos = haply_pos.cpu().numpy()
    if isinstance(haply_quat, torch.Tensor):
        haply_quat = haply_quat.cpu().numpy()
    if isinstance(robot_initial_pos, torch.Tensor):
        robot_initial_pos = robot_initial_pos.cpu().numpy()
    
    # Calculate Haply delta from its zero position
    haply_delta = haply_pos - haply_initial_pos
    
    # Coordinate system mapping: 
    # Haply Y (front-back, push-pull) → Robot -X (forward-backward, inverted)
    # Haply X (left-right) → Robot Y (left-right)
    # Haply Z (up-down) → Robot Z (up-down)
    robot_offset = np.array([-haply_delta[1], haply_delta[0], haply_delta[2]])
    
    # Apply scaling (smaller = more precise control, easier for IK)
    position_scale = 0.5  # Conservative scaling for stable control
    robot_offset_scaled = robot_offset * position_scale
    
    # Final position = initial position + scaled offset
    robot_pos = robot_initial_pos + robot_offset_scaled
    
    # Apply workspace limits (safety: keep robot above table)
    # Table is at Z=1.05, so minimum safe Z is 1.20 (15cm above table)
    robot_pos_before_limit = robot_pos.copy()
    robot_pos[0] = np.clip(robot_pos[0], 0.30, 0.70)  # X limits: front-back range
    robot_pos[1] = np.clip(robot_pos[1], -0.50, 0.50)  # Y limits: left-right range
    robot_pos[2] = np.clip(robot_pos[2], 1.20, 1.70)  # Z limits: height above table
    if debug:
        print(f"[MAPPING] haply_pos: {haply_pos}")
        print(f"[MAPPING] haply_initial_pos: {haply_initial_pos}")
        print(f"[MAPPING] haply_delta: {haply_delta}")
        print(f"[MAPPING] robot_offset (after coord transform): {robot_offset}")
        print(f"[MAPPING] robot_offset_scaled (x{position_scale}): {robot_offset_scaled}")
        print(f"[MAPPING] robot_initial_pos: {robot_initial_pos}")
        print(f"[MAPPING] robot_pos (before Z limit): {robot_pos_before_limit}")
        print(f"[MAPPING] robot_pos (final): {robot_pos}")
        if abs(robot_pos[2] - robot_pos_before_limit[2]) > 0.001:
            print(f"[WARNING] ⚠️  Z clamped: {robot_pos_before_limit[2]:.3f} → {robot_pos[2]:.3f}")
    
    # Quaternion format conversion: Haply [qx,qy,qz,qw] -> Isaac [qw,qx,qy,qz]
    robot_quat = np.array([haply_quat[3], haply_quat[0], haply_quat[1], haply_quat[2]])
    
    return robot_pos, robot_quat


@configclass
class FrankaHaplySceneCfg(InteractiveSceneCfg):
    """Configuration for Franka scene with Haply teleoperation and contact sensors."""

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # Lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.50, 0.0, 1.05)),
    )

    # Franka Panda robot with high PD gains for responsive teleoperation
    # HIGH_PD_CFG provides: stiffness=400, damping=80, gravity=disabled
    robot: Articulation = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = (0.0, 0.0, 1.05)  # Base at table height
    # Enable contact sensors for force feedback
    robot.spawn.activate_contact_sensors = True

    # Test object - cube for interaction
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.2), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 1.15)),
    )

    # Contact sensor for left finger
    left_finger_contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
        update_period=0.0,  # Update every step
        history_length=3,  # Keep 3 timesteps of history
        debug_vis=True,  # Visualize contact forces
        track_pose=True,  # Track sensor pose
    )

    # Contact sensor for right finger
    right_finger_contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
        update_period=0.0,  # Update every step
        history_length=3,  # Keep 3 timesteps of history
        debug_vis=True,  # Visualize contact forces
        track_pose=True,  # Track sensor pose
    )


def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    haply_device: HaplyDevice,
):
    """Runs the simulation loop with Haply teleoperation and force feedback."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 1
    
    # Get the robot, cube, and sensors
    robot: Articulation = scene["robot"]
    cube: RigidObject = scene["cube"]
    left_finger_sensor: ContactSensor = scene["left_finger_contact_sensor"]
    right_finger_sensor: ContactSensor = scene["right_finger_contact_sensor"]
    
    # Get end-effector body name for Franka
    ee_body_name = "panda_hand"
    ee_body_idx = robot.body_names.index(ee_body_name)
    
    # Set robot to a good initial pose explicitly
    # This configuration puts EE at approximately [0.5, 0.0, 1.3]
    print(f"[INFO] Setting robot to initial working pose...")
    joint_pos = robot.data.default_joint_pos.clone()
    joint_pos[0, :7] = torch.tensor(
        [0.0, -0.569, 0.0, -2.81, 0.0, 3.037, 0.741], 
        device=robot.device
    )
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    
    # Update scene multiple times to ensure position settles
    for i in range(10):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
    
    # Now read the actual EE position
    robot_initial_pos = robot.data.body_pos_w[0, ee_body_idx].cpu().numpy()
    print(f"[INFO] Robot EE position after initialization: {robot_initial_pos}")

    # Read Haply's ACTUAL current position as the zero reference
    print(f"[INFO] Reading Haply initial position...")
    initial_haply_data = haply_device.advance()
    haply_initial_pos = initial_haply_data[:3].cpu().numpy()
    print(f"[INFO] Haply initial position (read from device): {haply_initial_pos}")
    
    print(f"\n[INFO] ========== Coordinate Mapping Configuration ==========")
    print(f"[INFO] Robot EE initial position: {robot_initial_pos}")
    print(f"[INFO] Haply initial position: {haply_initial_pos}")
    print(f"[INFO] Position scale: 0.5x (moderate scaling for responsive control)")
    print(f"[INFO] Formula: robot_pos = robot_initial + (haply_pos - haply_zero) * 0.5")
    print(f"[INFO] When Haply is at physical start (0,0,0.2), robot stays at base")
    print(f"[INFO] Table Z: 1.05m")
    print(f"[INFO] ===============================================")
    print(f"[INFO] Setup complete! Move the Haply device.")
    
    # Create Differential IK Controller for Franka
    ik_controller_cfg = DifferentialIKControllerCfg(
        command_type="pose",  # Control both position and orientation
        use_relative_mode=False,  # Use absolute target pose
        ik_method="dls",  # Damped Least Squares
        ik_params={
            "lambda_val": 0.05,  # Damping factor: balance between responsiveness and stability
            # 0.01 = very responsive but unstable for large movements
            # 0.05 = balanced (recommended)
            # 0.1+ = too conservative
        },
    )
    
    print(f"[INFO] IK Controller config: {ik_controller_cfg}")
    
    # Get arm joint indices (excluding gripper joints)
    arm_joint_names = [
        "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
        "panda_joint5", "panda_joint6", "panda_joint7"
    ]
    arm_joint_indices = [robot.joint_names.index(name) for name in arm_joint_names]
    
    # Initialize IK controller
    ik_controller = DifferentialIKController(
        cfg=ik_controller_cfg,
        num_envs=scene.num_envs,
        device=sim.device
    )
    
    # Set IK controller parameters
    ik_controller.set_command(command=torch.zeros(scene.num_envs, 7, device=sim.device))
    
    # Button state tracking
    prev_button_a = False
    prev_button_b = False
    prev_button_c = False
    
    # Control mode
    teleoperation_enabled = True
    gripper_target = 0.04  # Open gripper initially (Franka gripper fully open)
    
    print("\n" + "="*60)
    print("HAPLY IK TELEOPERATION WITH FORCE FEEDBACK")
    print("="*60)
    print("Controls:")
    print("  - Move Inverse3: Control end-effector position")
    print("  - Rotate VerseGrip: Control end-effector orientation")
    print("  - Button A: Open gripper")
    print("  - Button B: Close gripper")
    print("  - Button C: Exit program")
    print("\nTask:")
    print("  - Try to grab the green cube!")
    print("  - Feel the contact forces through Haply device")
    print("\nIK Method: Differential IK with Damped Least Squares")
    print("="*60 + "\n")

    # Simulate physics
    while simulation_app.is_running():
        # Reset periodically (set to a very large number to effectively disable auto-reset)
        # Change this value if you want automatic reset: 500 = ~8 seconds, 3000 = ~50 seconds
        if count % 10000 == 0:
            count = 1
            # Reset the scene entities
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            
            # Set joint positions - use a pose that puts EE near the target region
            # This configuration puts EE roughly at [0.5, 0.0, 1.3] - close to our workspace center
            joint_pos = robot.data.default_joint_pos.clone()
            joint_pos[0, :7] = torch.tensor(
                [0.0, -0.569, 0.0, -2.81, 0.0, 3.037, 0.741], 
                device=robot.device
            )
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            
            # Reset cube position
            cube_state = cube.data.default_root_state.clone()
            cube_state[:, :3] += scene.env_origins
            cube.write_root_pose_to_sim(cube_state[:, :7])
            cube.write_root_velocity_to_sim(cube_state[:, 7:])
            
            # Reset scene
            scene.reset()
            
            # Reset Haply device
            haply_device.reset()
            
            # Reset IK controller
            ik_controller.reset()
            
            # In absolute mode, robot_initial_pos stays constant (no need to recalculate)
            print("[INFO]: Resetting robot state...")

        # Get Haply device data
        haply_data = haply_device.advance()
        
        # Extract data: [x, y, z, qx, qy, qz, qw, button_a, button_b, button_c]
        haply_pos = haply_data[:3]
        haply_quat = haply_data[3:7]  # [qx, qy, qz, qw]
        button_a = haply_data[7].item() > 0.5
        button_b = haply_data[8].item() > 0.5
        button_c = haply_data[9].item() > 0.5
        
        # Handle button presses (detect rising edge)
        if button_a and not prev_button_a:
            gripper_target = 0.04  # Open gripper
            print("[Button A] Opening gripper")
            
        if button_b and not prev_button_b:
            gripper_target = 0.0  # Close gripper
            print("[Button B] Closing gripper")
            
        if button_c and not prev_button_c:
            print("[Button C] Exiting program...")
            break
        
        prev_button_a = button_a
        prev_button_b = button_b
        prev_button_c = button_c
        
        # Compute IK if teleoperation is enabled
        if teleoperation_enabled:
            # Map Haply coordinates to robot workspace (using relative position)
            # Debug every 50 frames
            debug_mapping = (count % 50 == 0)
            target_pos, target_quat = apply_haply_to_robot_mapping(
                haply_pos, haply_quat, haply_initial_pos, robot_initial_pos, debug=debug_mapping
            )
            
            # Debug: Print every 50 frames
            if debug_mapping:
                print(f"[DEBUG] Current EE pos: {robot.data.body_pos_w[0, ee_body_idx].cpu().numpy()}")

            # Convert to tensors
            target_pos_tensor = torch.tensor(target_pos, dtype=torch.float32, device=sim.device).unsqueeze(0)
            target_quat_tensor = torch.tensor(target_quat, dtype=torch.float32, device=sim.device).unsqueeze(0)
            
            # Get current robot state
            current_joint_pos = robot.data.joint_pos[:, arm_joint_indices]
            ee_pos_w = robot.data.body_pos_w[:, ee_body_idx]
            ee_quat_w = robot.data.body_quat_w[:, ee_body_idx]
            
            # Get Jacobian for arm joints
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_body_idx, :, arm_joint_indices]
            
            # Compute IK
            joint_pos_des = ik_controller.compute(
                target_pos_tensor,
                target_quat_tensor,
                jacobian,
                current_joint_pos
            )
            
            # Debug: Check IK output
            if count % 50 == 0:
                joint_delta = joint_pos_des[0] - current_joint_pos[0]
                print(f"[DEBUG IK] Current joints: {current_joint_pos[0].cpu().numpy()}")
                print(f"[DEBUG IK] Desired joints: {joint_pos_des[0].cpu().numpy()}")
                print(f"[DEBUG IK] Joint delta: {joint_delta.cpu().numpy()}")
                print(f"[DEBUG IK] Max delta: {torch.max(torch.abs(joint_delta)).item():.3f}")
            
            # Create full joint position command (arm + gripper)
            joint_pos_target = robot.data.joint_pos[0].clone()
            joint_pos_target[arm_joint_indices] = joint_pos_des[0]
        else:
            # Use current joint positions when teleoperation is disabled
            joint_pos_target = robot.data.joint_pos[0].clone()
        
        # Set gripper target (last two joints for Franka)
        gripper_joint_indices = [-2, -1]
        joint_pos_target[gripper_joint_indices] = gripper_target
        
        # Apply joint targets to robot
        robot.set_joint_position_target(joint_pos_target.unsqueeze(0))
        
        # Write data to sim and step multiple times for better stability
        for _ in range(5):
            scene.write_data_to_sim()
            sim.step()
        
        # Update scene once after all physics steps (this updates robot and sensors)
        scene.update(sim_dt)
        count += 1
        
        # Get contact forces and apply force feedback
        if args_cli.enable_force_feedback:
            # Get net contact forces from finger sensors
            # net_forces_w shape: [num_envs, num_bodies, 3]
            left_finger_forces = left_finger_sensor.data.net_forces_w[0, 0]  # [3]
            right_finger_forces = right_finger_sensor.data.net_forces_w[0, 0]  # [3]
            
            # Sum contact forces from both fingers
            total_contact_force = left_finger_forces + right_finger_forces
            
            # Scale force for haptic feedback (Haply has limited force output)
            force_scale = 0.5  # Adjust this to tune feedback strength
            feedback_force = total_contact_force * force_scale
            
            # Clamp forces to safe range for Haply device
            max_force = 3.0  
            feedback_force = torch.clamp(feedback_force, -max_force, max_force)
            
            # Send force feedback to Haply (negate to create resistance)
            haply_device.set_force_feedback(
                -feedback_force[0].item(),
                -feedback_force[1].item(),
                -feedback_force[2].item()
            )



def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=1/200)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 1.0])
    
    # Design scene using InteractiveScene
    scene_cfg = FrankaHaplySceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # Create Haply device
    print(f"[INFO] Connecting to Haply device at {args_cli.websocket_uri}...")
    haply_cfg = HaplyDeviceCfg(
        websocket_uri=args_cli.websocket_uri,
        pos_sensitivity=args_cli.pos_sensitivity,
        sim_device=args_cli.device,
    )
    
    try:
        haply_device = HaplyDevice(cfg=haply_cfg)
        print(f"[INFO] Haply device connected successfully!")
        print(haply_device)
    except Exception as e:
        print(f"[ERROR] Failed to connect to Haply device: {e}")
        simulation_app.close()
        return
    
    # Play the simulator
    sim.reset()
    
    print("[INFO]: Setup complete...")
    print("[INFO]: Starting teleoperation...")
    
    # Run the simulator
    run_simulator(sim, scene, haply_device)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

