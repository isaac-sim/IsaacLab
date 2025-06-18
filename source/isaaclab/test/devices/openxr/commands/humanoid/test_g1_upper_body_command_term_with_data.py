# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""
import json
import os
import sys
import numpy as np
from pathlib import Path

# Import pinocchio in the main script to force the use of the dependencies installed by IsaacLab and not the one installed by Isaac Sim
# pinocchio is required by the Pink IK controller
if sys.platform != "win32":
    import pinocchio  # noqa: F401

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=False).app

"""Rest everything follows."""

import contextlib
import gymnasium as gym
import numpy as np
import pytest
import torch
import time
from typing import Dict, List

from pink.configuration import Configuration

from isaaclab.utils.math import axis_angle_from_quat, matrix_from_quat, quat_from_matrix, quat_inv
from isaaclab.devices.hand_tracking_device import Hand, HandTrackingDevice
from isaaclab.devices.openxr.commands.humanoid.g1_upper_body_command_term import (
    G1UpperBodyCommandTerm,
    G1UpperBodyCommandTermCfg,
)

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def create_test_env(num_envs):
    """Create a test environment with the Pink IK controller."""
    env_name = "Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0"
    device = "cuda:0"

    try:
        env_cfg = parse_env_cfg(env_name, device=device, num_envs=num_envs)
        # Modify scene config to not spawn the packing table to avoid collision with the robot
        del env_cfg.terminations.time_out
        return gym.make(env_name, cfg=env_cfg).unwrapped, env_cfg
    except Exception as e:
        print(f"Failed to create environment: {str(e)}")
        raise


@pytest.fixture(scope="module")
def env_and_cfg():
    """Create environment and configuration for tests."""
    env, env_cfg = create_test_env(num_envs=1)
    
    # Set up camera view
    env.sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 1.0])
    
    return env, env_cfg


@pytest.fixture
def test_setup(env_and_cfg):
    """Set up test case - runs before each test."""
    env, env_cfg = env_and_cfg
    
    num_joints_in_robot_hands = env_cfg.actions.pink_ik.controller.num_hand_joints

    # Get Action Term and IK controller
    action_term = env.action_manager.get_term(name="pink_ik")
    pink_controllers = action_term._ik_controllers
    articulation = action_term._asset

    # Initialize Pink Configuration for forward kinematics
    kinematics_model = Configuration(
        pink_controllers[0].robot_wrapper.model,
        pink_controllers[0].robot_wrapper.data,
        pink_controllers[0].robot_wrapper.q0,
    )

    return {
        "env": env,
        "env_cfg": env_cfg,
        "num_joints_in_robot_hands": num_joints_in_robot_hands,
        "action_term": action_term,
        "pink_controllers": pink_controllers,
        "articulation": articulation,
        "kinematics_model": kinematics_model,
    }


@pytest.fixture
def test_setup_g1_command(env_and_cfg):
    """Set up test case for G1 command term tests - runs before each test."""
    env, env_cfg = env_and_cfg
    
    # Get hand joint names from the environment configuration
    hand_joint_names = env_cfg.actions.upper_body_ik.hand_joint_names

    return {
        "env": env,
        "env_cfg": env_cfg,
        "hand_joint_names": hand_joint_names,
    }


@pytest.fixture
def mock_hand_tracking_device():
    """Create a mock hand tracking device for testing that cycles through multiple frames."""
    
    class MockHandTrackingDevice(HandTrackingDevice):
        def __init__(self, height_offset=-0.3, frame_rate=10.0, trim_frames=200):
            self.frame_counter = 0
            self.height_offset = height_offset
            self.frame_rate = frame_rate
            
            # Load the hand pose data
            data_path = os.path.join(os.path.dirname(__file__), "test_data_hand_pose_recording.npz")
            hand_data = np.load(data_path, allow_pickle=True)
            
            # Extract the data
            left_hand_poses = hand_data['left_hand_poses']  # [718, 26, 7]
            right_hand_poses = hand_data['right_hand_poses']  # [718, 26, 7]
            left_joint_names = hand_data['left_joint_names']  # [26]
            right_joint_names = hand_data['right_joint_names']  # [26]

            # Trim the beginning of the data by specified number of frames
            if trim_frames > 0:
                left_hand_poses = left_hand_poses[trim_frames:]
                right_hand_poses = right_hand_poses[trim_frames:]

            # Apply height offset to the z-coordinate (index 2) of position data
            left_hand_poses[:, :, 2] += height_offset
            right_hand_poses[:, :, 2] += height_offset

            # Store the data for use in get_hand_poses method
            self.left_hand_poses = left_hand_poses
            self.right_hand_poses = right_hand_poses
            self.left_joint_names = left_joint_names
            self.right_joint_names = right_joint_names
            self.num_timesteps = left_hand_poses.shape[0]
            
            # Create a list of frames with different hand poses
            self.hand_poses_frames = []
            
            for frame_idx in range(self.num_timesteps):
                
                # Create frame data for left hand
                left_frame_data = {}
                for i, joint_name in enumerate(self.left_joint_names):
                    left_frame_data[joint_name] = self.left_hand_poses[frame_idx, i].tolist()
                
                # Create frame data for right hand  
                right_frame_data = {}
                for i, joint_name in enumerate(self.right_joint_names):
                    right_frame_data[joint_name] = self.right_hand_poses[frame_idx, i].tolist()
                
                # Store both hands' data for this frame
                self.hand_poses_frames.append({
                    Hand.LEFT: left_frame_data,
                    Hand.RIGHT: right_frame_data
                })
        
        def get_hand_poses(self, hand):
            """Get hand poses for the specified hand, cycling through different frames."""
            # Get the current frame data
            current_frame = self.frame_counter % self.num_timesteps
            frame_data = self.hand_poses_frames[current_frame]
            
            # Return the data for the specific hand requested
            return frame_data[hand].copy()
        
        def advance_frame(self):
            """Advance to the next frame based on the configured frame rate."""
            # Calculate how many frames to advance based on time elapsed
            current_time = time.time()
            if not hasattr(self, '_last_advance_time'):
                self._last_advance_time = current_time
            
            time_elapsed = current_time - self._last_advance_time
            frames_to_advance = int(time_elapsed * self.frame_rate)
            
            # Only advance if we have enough time elapsed
            if frames_to_advance > 0:
                self.frame_counter += frames_to_advance
                self._last_advance_time = current_time
        
        def reset_frame_counter(self):
            """Reset the frame counter to start from the beginning."""
            self.frame_counter = 0
    
    return MockHandTrackingDevice()


class TestG1UpperBodyCommandTermWithData:
    """Test G1 upper body command term with data-driven hand tracking."""

    def test_initialization(self, test_setup_g1_command, mock_hand_tracking_device):
        """Test initialization of the command term with real environment."""
        env = test_setup_g1_command["env"]
        hand_joint_names = test_setup_g1_command["hand_joint_names"]
        
        # Set the teleop device in the environment
        env.handtracking = mock_hand_tracking_device
        env._device_name = "handtracking"
        
        cfg = G1UpperBodyCommandTermCfg(
            device_name="teleop_device",
            hand_joint_names=hand_joint_names,
            resampling_time_range=(0.1, 0.2),
        )
        cmd_term = G1UpperBodyCommandTerm(cfg, env)

        assert cmd_term.cfg == cfg
        assert cmd_term.num_envs == env.num_envs
        assert cmd_term.device == env.device
        assert cmd_term._device_name == "handtracking"
        assert cmd_term._get_hand_tracking_device() is not None
        assert cmd_term._hands_controller is not None
        assert cmd_term._hand_joint_names == hand_joint_names

    def test_resample_command(self, test_setup_g1_command, mock_hand_tracking_device):
        """Test resampling of the command."""
        env = test_setup_g1_command["env"]
        hand_joint_names = test_setup_g1_command["hand_joint_names"]
        
        # Set the teleop device in the environment
        env.teleop_device = mock_hand_tracking_device
        
        cfg = G1UpperBodyCommandTermCfg(
            device_name="teleop_device",
            hand_joint_names=hand_joint_names,
            resampling_time_range=(0.1, 0.2),
        )
        cmd_term = G1UpperBodyCommandTerm(cfg, env)
        
        # Test resampling with mock data
        cmd_term._resample_command(env_ids=torch.arange(env.num_envs))
        
        # Check that the command is not None
        assert cmd_term.command is not None
        
        # Check that the command has the correct shape
        assert cmd_term.command.shape == (env.num_envs, 28)
        
        # Check that the command is not all zeros
        assert not torch.all(cmd_term.command == 0.0)

    def test_mock_device_frame_cycling(self, mock_hand_tracking_device):
        """Test that the mock device cycles through different frames correctly."""
        # Reset the frame counter to start from the beginning
        mock_hand_tracking_device.reset_frame_counter()
        
        # Get poses for multiple frames and verify they cycle
        frames_data = []
        for i in range(6):  # Test more than the number of frames to verify cycling
            mock_hand_tracking_device.advance_frame()  # Advance to next frame
            left_poses = mock_hand_tracking_device.get_hand_poses(Hand.LEFT)
            right_poses = mock_hand_tracking_device.get_hand_poses(Hand.RIGHT)
            frames_data.append((left_poses, right_poses))
        
        # Verify that we have different frames (the mock currently returns same data but cycles through frame indices)
        # In a real implementation, each frame would have different pose data
        assert len(frames_data) == 6
        
        # Verify that the frame counter has been incremented correctly
        # After 12 calls (6 left + 6 right), the counter should be at 12
        assert mock_hand_tracking_device.frame_counter == 12
        
        # Test that cycling works (frame 4 should be same as frame 0 due to modulo)
        mock_hand_tracking_device.reset_frame_counter()
        frame_0_data = mock_hand_tracking_device.get_hand_poses(Hand.LEFT)
        
        # Advance to frame 3 (which should be same as frame 0 due to modulo 4)
        for _ in range(3):
            mock_hand_tracking_device.advance_frame()
            mock_hand_tracking_device.get_hand_poses(Hand.LEFT)
        
        mock_hand_tracking_device.advance_frame()
        frame_4_data = mock_hand_tracking_device.get_hand_poses(Hand.LEFT)
        
        # The data should be the same since we're cycling through the same frames
        assert frame_0_data == frame_4_data
        
        # Test that different frames have different data
        mock_hand_tracking_device.reset_frame_counter()
        
        # Get data from different frames
        frame_0_wrist = mock_hand_tracking_device.get_hand_poses(Hand.LEFT)["wrist"]
        mock_hand_tracking_device.advance_frame()
        frame_1_wrist = mock_hand_tracking_device.get_hand_poses(Hand.LEFT)["wrist"]
        mock_hand_tracking_device.advance_frame()
        frame_2_wrist = mock_hand_tracking_device.get_hand_poses(Hand.LEFT)["wrist"]
        mock_hand_tracking_device.advance_frame()
        frame_3_wrist = mock_hand_tracking_device.get_hand_poses(Hand.LEFT)["wrist"]
        
        # Verify that different frames have different wrist positions (if we have enough data)
        # Note: If the loaded data doesn't have enough variety, some frames might be the same
        # This is acceptable for testing purposes
        frames_are_different = (
            frame_0_wrist != frame_1_wrist or 
            frame_1_wrist != frame_2_wrist or 
            frame_2_wrist != frame_3_wrist
        )
        assert frames_are_different, "At least some frames should have different wrist positions"
        
        # Verify that frame 4 (which cycles back to frame 0) has the same data as frame 0
        mock_hand_tracking_device.advance_frame()
        frame_4_wrist = mock_hand_tracking_device.get_hand_poses(Hand.LEFT)["wrist"]
        assert frame_0_wrist == frame_4_wrist, "Frame 0 and Frame 4 should have the same wrist position (cycling)"
        
    def test_step_environment_with_mock_data(self, test_setup_g1_command, mock_hand_tracking_device):
        """Test stepping the environment with commands from the mock device."""
        env = test_setup_g1_command["env"]
        hand_joint_names = test_setup_g1_command["hand_joint_names"]

        # Set the teleop device in the environment
        env.handtracking = mock_hand_tracking_device
        env._device_name = "handtracking"

        obs, _ = env.reset()
        mock_hand_tracking_device.reset_frame_counter()

        action_term = env.action_manager.get_term("upper_body_ik")
        articulation = action_term._asset
        initial_dof_pos = articulation.data.joint_pos.clone()

        # Step for a few iterations
        for i in range(5000):
            env.handtracking.advance_frame()
            # Get command from command manager
            command = env.command_manager.get_command("upper_body_command")
            
            # # Replace indexes 14 to 28 with sinusoidal signal for rotation angles from -90 to 90 degrees
            # # Convert degrees to radians and create sinusoidal signal
            # angle_rad = np.radians(90.0)  # 90 degrees in radians
            # frequency = 1.0  # Adjust frequency as needed
            # time_step = i * 0.01  # Assuming 0.01 second per step

            # # Generate sinusoidal signal that oscillates between -angle_rad and +angle_rad
            # sinusoidal_angle = angle_rad * np.sin(2 * np.pi * frequency * time_step)
            
            # # Apply the sinusoidal signal to command indexes 14-28
            # command[:, 14:] = sinusoidal_angle
            
            # The environment has a single action term `upper_body_ik`, so we can pass the command tensor directly.
            obs, _, _, _, _ = env.step(command)

        final_dof_pos = articulation.data.joint_pos.clone()
        assert not torch.allclose(initial_dof_pos, final_dof_pos), "Robot should have moved."
        