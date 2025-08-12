# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test G1 upper body retargeter with data-driven hand tracking."""

import os
import sys
import time

# Import pinocchio in the main script to force the use of the dependencies installed by IsaacLab and not the one installed by Isaac Sim
# pinocchio is required by the Pink IK controller
if sys.platform != "win32":
    import pinocchio  # noqa: F401

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=False).app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch

import pytest

from isaaclab.devices import DeviceBase
from isaaclab.devices.openxr.openxr_device import OpenXRDevice
from isaaclab.devices.openxr.retargeters.humanoid.unitree.g1_upper_body_retargeter import (
    G1UpperBodyRetargeter,
    G1UpperBodyRetargeterCfg,
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
def mock_hand_tracking_device(env_and_cfg):
    """Create a mock OpenXR-like hand tracking device for testing that cycles through multiple frames.

    The output format matches OpenXRDevice._get_raw_data:
        {
            TrackingTarget.HAND_LEFT: {joint_name: np.ndarray shape (7,) ...},
            TrackingTarget.HAND_RIGHT: {joint_name: np.ndarray shape (7,) ...},
            TrackingTarget.HEAD: np.ndarray shape (7,)
        }
    """

    class MockHandTrackingDevice(DeviceBase):
        TrackingTarget = OpenXRDevice.TrackingTarget

        def __init__(self, height_offset=-0.3, interval_per_frame=0.1, trim_frames=200, retargeters=None):
            super().__init__(retargeters=retargeters)
            self.frame_counter = 0
            self.height_offset = height_offset
            self.interval_per_frame = interval_per_frame

            # Load the hand pose data
            data_path = os.path.join(os.path.dirname(__file__), "test_data_hand_pose_recording.npz")
            hand_data = np.load(data_path, allow_pickle=True)

            # Extract the data
            left_hand_poses = hand_data["left_hand_poses"]  # [N, 26, 7]
            right_hand_poses = hand_data["right_hand_poses"]  # [N, 26, 7]
            left_joint_names = [str(j) for j in hand_data["left_joint_names"]]  # [26]
            right_joint_names = [str(j) for j in hand_data["right_joint_names"]]  # [26]

            # Trim the beginning of the data by specified number of frames
            if trim_frames > 0:
                left_hand_poses = left_hand_poses[trim_frames:]
                right_hand_poses = right_hand_poses[trim_frames:]

            # Apply height offset to the z-coordinate (index 2) of position data
            left_hand_poses[:, :, 2] += height_offset
            right_hand_poses[:, :, 2] += height_offset

            self.left_hand_poses = left_hand_poses
            self.right_hand_poses = right_hand_poses
            self.left_joint_names = left_joint_names
            self.right_joint_names = right_joint_names
            self.num_timesteps = left_hand_poses.shape[0]

            # For callback support
            self._callbacks = {}

        def _get_hand_dict(self, hand: str, frame_idx: int):
            """Return a dict of joint_name: np.ndarray (7,) for the given hand and frame."""
            if hand == "left":
                joint_names = self.left_joint_names
                hand_poses = self.left_hand_poses
            elif hand == "right":
                joint_names = self.right_joint_names
                hand_poses = self.right_hand_poses
            else:
                raise ValueError(f"Unknown hand: {hand}")

            pose_dict = {}
            for i, joint_name in enumerate(joint_names):
                pose_dict[joint_name] = np.array(hand_poses[frame_idx, i], dtype=np.float32)
            return pose_dict

        def _get_raw_data(self):
            """Mimic OpenXRDevice._get_raw_data output."""
            current_frame = self.frame_counter % self.num_timesteps
            return {
                self.TrackingTarget.HAND_LEFT: self._get_hand_dict("left", current_frame),
                self.TrackingTarget.HAND_RIGHT: self._get_hand_dict("right", current_frame),
            }

        def advance(self):
            """Advance the device state and return the raw data (or retargeted if configured)."""
            self._advance_frame()
            # For this mock, just return the raw data (retargeting handled by base if needed)
            return super().advance()

        def add_callback(self, key, func):
            """Add a callback function for a given key."""
            self._callbacks[key] = func

        def _advance_frame(self):
            """Advance to the next frame based on the configured frame rate."""
            current_time = time.time()
            if not hasattr(self, "_last_advance_time"):
                self._last_advance_time = current_time

            time_elapsed = current_time - self._last_advance_time

            if time_elapsed > self.interval_per_frame:
                self.frame_counter += 1
                self._last_advance_time = current_time

        def reset(self):
            """Reset the frame counter to start from the beginning."""
            self.frame_counter = 0
            self._last_advance_time = time.time()

    _, env_cfg = env_and_cfg

    retargeter_cfg = G1UpperBodyRetargeterCfg(
        enable_visualization=True,
        # OpenXR hand tracking has 26 joints per hand
        num_open_xr_hand_joints=2 * 26,
        sim_device="cuda:0",
        hand_joint_names=env_cfg.actions.upper_body_ik.hand_joint_names,
    )
    return MockHandTrackingDevice(retargeters=[G1UpperBodyRetargeter(retargeter_cfg)])


@pytest.fixture
def test_setup_g1_env(env_and_cfg):
    """Set up test case for G1 command term tests - runs before each test."""
    env, env_cfg = env_and_cfg

    # Get hand joint names from the environment configuration
    hand_joint_names = env_cfg.actions.upper_body_ik.hand_joint_names

    return {
        "env": env,
        "env_cfg": env_cfg,
        "hand_joint_names": hand_joint_names,
    }


class TestG1UpperBodyCommandTermWithData:
    """Test G1 upper body command term with data-driven hand tracking."""

    def test_initialization(self, test_setup_g1_env, mock_hand_tracking_device):
        """Test initialization of the command term with real environment."""
        env = test_setup_g1_env["env"]

        # Set the teleop device in the environment
        env.teleop_device = mock_hand_tracking_device
        env._device_name = "handtracking"

        assert env.teleop_device == mock_hand_tracking_device
        assert env._device_name == "handtracking"

    def test_get_raw_data(self, test_setup_g1_env, mock_hand_tracking_device):
        """Test resampling of the command."""
        env = test_setup_g1_env["env"]

        # Set the teleop device in the environment
        env.teleop_device = mock_hand_tracking_device
        env._device_name = "handtracking"

        data = env.teleop_device.advance()
        data = data.repeat(env.num_envs, 1)

        # Check that the command is not None
        assert data is not None

        # Check that the command has the correct shape
        assert data.shape == (env.num_envs, 28)

        # Check that the command is not all zeros
        assert not torch.all(data == 0.0)

    def test_mock_device_frame_rate(self, mock_hand_tracking_device):
        """Test that the mock device cycles through different frames correctly."""
        # Reset the frame counter to start from the beginning
        mock_hand_tracking_device.reset()

        # Get poses for multiple frames and verify they cycle
        frames_data = []
        for i in range(6):  # Test more than the number of frames to verify cycling
            time.sleep(0.15)
            device_data = mock_hand_tracking_device.advance()  # Advance to next frame
            frames_data.append(device_data)

        # Verify that we have different frames (the mock currently returns same data but cycles through frame indices)
        # In a real implementation, each frame would have different pose data
        assert len(frames_data) == 6

        # Verify that the frame counter has been incremented correctly
        # After 6 calls, the counter should be at 6
        assert mock_hand_tracking_device.frame_counter == 6

    def test_step_environment_with_mock_data(self, test_setup_g1_env, mock_hand_tracking_device):
        """Test stepping the environment with commands from the mock device."""
        env = test_setup_g1_env["env"]

        # Set the teleop device in the environment
        env.teleop_device = mock_hand_tracking_device
        env._device_name = "handtracking"

        obs, _ = env.reset()
        mock_hand_tracking_device.reset()

        action_term = env.action_manager.get_term("upper_body_ik")
        articulation = action_term._asset
        initial_dof_pos = articulation.data.joint_pos.clone()

        # Step for a few iterations
        for i in range(100):
            # Get command from command manager
            action = env.teleop_device.advance().repeat(env.num_envs, 1)

            # The environment has a single action term `upper_body_ik`, so we can pass the command tensor directly.
            obs, _, _, _, _ = env.step(action)

        final_dof_pos = articulation.data.joint_pos.clone()
        assert not torch.allclose(initial_dof_pos, final_dof_pos), "Robot should have moved."
