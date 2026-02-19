# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Unit tests for retargeters.
"""

from isaaclab.app import AppLauncher

# Can set this to False to see the GUI for debugging.
HEADLESS = True

# Launch omniverse app.
app_launcher = AppLauncher(headless=HEADLESS)
simulation_app = app_launcher.app

import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

# Mock dependencies that might require a running simulation or specific hardware
sys.modules["isaaclab.markers"] = MagicMock()
sys.modules["isaaclab.markers.config"] = MagicMock()
sys.modules["isaaclab.sim"] = MagicMock()
sys.modules["isaaclab.sim.SimulationContext"] = MagicMock()

# Mock SimulationContext instance
mock_sim_context = MagicMock()
mock_sim_context.get_rendering_dt.return_value = 0.016  # 60Hz
sys.modules["isaaclab.sim"].SimulationContext.instance.return_value = mock_sim_context


# Import after mocking
from isaaclab_teleop.deprecated.openxr.retargeters.humanoid.unitree.g1_lower_body_standing import (
    G1LowerBodyStandingRetargeter,
    G1LowerBodyStandingRetargeterCfg,
)
from isaaclab_teleop.deprecated.openxr.retargeters.humanoid.unitree.g1_motion_controller_locomotion import (
    G1LowerBodyStandingMotionControllerRetargeter,
    G1LowerBodyStandingMotionControllerRetargeterCfg,
)
from isaaclab_teleop.deprecated.openxr.retargeters.manipulator.gripper_retargeter import (
    GripperRetargeter,
    GripperRetargeterCfg,
)
from isaaclab_teleop.deprecated.openxr.retargeters.manipulator.se3_abs_retargeter import (
    Se3AbsRetargeter,
    Se3AbsRetargeterCfg,
)
from isaaclab_teleop.deprecated.openxr.retargeters.manipulator.se3_rel_retargeter import (
    Se3RelRetargeter,
    Se3RelRetargeterCfg,
)

from isaaclab.devices.device_base import DeviceBase

# Mock dex retargeting utils
with patch.dict(
    sys.modules,
    {
        "isaaclab_teleop.deprecated.openxr.retargeters.humanoid.unitree.inspire.g1_dex_retargeting_utils": MagicMock(),
        "isaaclab_teleop.deprecated.openxr.retargeters.humanoid.fourier.gr1_t2_dex_retargeting_utils": MagicMock(),
        "isaaclab_teleop.deprecated.openxr.retargeters.humanoid.unitree.trihand.g1_dex_retargeting_utils": MagicMock(),
    },
):
    from isaaclab_teleop.deprecated.openxr.retargeters.humanoid.fourier.gr1t2_retargeter import (
        GR1T2Retargeter,
        GR1T2RetargeterCfg,
    )
    from isaaclab_teleop.deprecated.openxr.retargeters.humanoid.unitree.inspire.g1_upper_body_retargeter import (
        UnitreeG1Retargeter,
        UnitreeG1RetargeterCfg,
    )
    from isaaclab_teleop.deprecated.openxr.retargeters.humanoid.unitree.trihand.g1_upper_body_motion_ctrl_gripper import (  # noqa: E501
        G1TriHandUpperBodyMotionControllerGripperRetargeter,
        G1TriHandUpperBodyMotionControllerGripperRetargeterCfg,
    )
    from isaaclab_teleop.deprecated.openxr.retargeters.humanoid.unitree.trihand.g1_upper_body_motion_ctrl_retargeter import (  # noqa: E501
        G1TriHandUpperBodyMotionControllerRetargeter,
        G1TriHandUpperBodyMotionControllerRetargeterCfg,
    )
    from isaaclab_teleop.deprecated.openxr.retargeters.humanoid.unitree.trihand.g1_upper_body_retargeter import (
        G1TriHandUpperBodyRetargeter,
        G1TriHandUpperBodyRetargeterCfg,
    )


class TestSe3AbsRetargeter(unittest.TestCase):
    def setUp(self):
        self.cfg = Se3AbsRetargeterCfg(
            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT, enable_visualization=False, sim_device="cpu"
        )
        self.retargeter = Se3AbsRetargeter(self.cfg)

    def test_retarget_defaults(self):
        # Mock input data
        wrist_pose = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0])
        thumb_tip_pose = np.array([0.15, 0.25, 0.35, 0.0, 0.0, 0.0, 1.0])
        index_tip_pose = np.array([0.15, 0.20, 0.35, 0.0, 0.0, 0.0, 1.0])

        data = {
            DeviceBase.TrackingTarget.HAND_RIGHT: {
                "wrist": wrist_pose,
                "thumb_tip": thumb_tip_pose,
                "index_tip": index_tip_pose,
            }
        }

        result = self.retargeter.retarget(data)

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (7,))
        np.testing.assert_allclose(result[:3].numpy(), wrist_pose[:3], rtol=1e-5)
        self.assertAlmostEqual(torch.linalg.norm(result[3:]).item(), 1.0, places=4)

    def test_pinch_position(self):
        self.cfg.use_wrist_position = False
        retargeter = Se3AbsRetargeter(self.cfg)

        wrist_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        thumb_tip_pose = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        index_tip_pose = np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

        data = {
            DeviceBase.TrackingTarget.HAND_RIGHT: {
                "wrist": wrist_pose,
                "thumb_tip": thumb_tip_pose,
                "index_tip": index_tip_pose,
            }
        }

        result = retargeter.retarget(data)
        expected_pos = np.array([2.0, 0.0, 0.0])
        np.testing.assert_allclose(result[:3].numpy(), expected_pos, rtol=1e-5)


class TestSe3RelRetargeter(unittest.TestCase):
    def setUp(self):
        self.cfg = Se3RelRetargeterCfg(
            bound_hand=DeviceBase.TrackingTarget.HAND_LEFT,
            enable_visualization=False,
            sim_device="cpu",
            delta_pos_scale_factor=1.0,
            delta_rot_scale_factor=1.0,
            alpha_pos=1.0,
            alpha_rot=1.0,
        )
        self.retargeter = Se3RelRetargeter(self.cfg)

    def test_retarget_movement(self):
        wrist_pose_1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        thumb_tip_pose_1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        index_tip_pose_1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

        data_1 = {
            DeviceBase.TrackingTarget.HAND_LEFT: {
                "wrist": wrist_pose_1,
                "thumb_tip": thumb_tip_pose_1,
                "index_tip": index_tip_pose_1,
            }
        }

        _ = self.retargeter.retarget(data_1)

        wrist_pose_2 = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        thumb_tip_pose_2 = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        index_tip_pose_2 = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

        data_2 = {
            DeviceBase.TrackingTarget.HAND_LEFT: {
                "wrist": wrist_pose_2,
                "thumb_tip": thumb_tip_pose_2,
                "index_tip": index_tip_pose_2,
            }
        }

        result = self.retargeter.retarget(data_2)
        self.assertEqual(result.shape, (6,))
        np.testing.assert_allclose(result[:3].numpy(), [0.1, 0.0, 0.0], rtol=1e-4)


class TestGripperRetargeter(unittest.TestCase):
    def setUp(self):
        self.cfg = GripperRetargeterCfg(bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT, sim_device="cpu")
        self.retargeter = GripperRetargeter(self.cfg)

    def test_gripper_logic(self):
        data_open = {
            DeviceBase.TrackingTarget.HAND_RIGHT: {
                "thumb_tip": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
                "index_tip": np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            }
        }
        result = self.retargeter.retarget(data_open)
        self.assertEqual(result.item(), 1.0)

        data_close = {
            DeviceBase.TrackingTarget.HAND_RIGHT: {
                "thumb_tip": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
                "index_tip": np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            }
        }
        result = self.retargeter.retarget(data_close)
        self.assertEqual(result.item(), -1.0)


class TestG1LowerBodyStandingRetargeter(unittest.TestCase):
    def test_retarget(self):
        cfg = G1LowerBodyStandingRetargeterCfg(hip_height=0.8, sim_device="cpu")
        retargeter = G1LowerBodyStandingRetargeter(cfg)
        result = retargeter.retarget({})
        self.assertTrue(torch.equal(result, torch.tensor([0.0, 0.0, 0.0, 0.8])))


class TestUnitreeG1Retargeter(unittest.TestCase):
    @patch(
        "isaaclab_teleop.deprecated.openxr.retargeters.humanoid.unitree.inspire.g1_upper_body_retargeter.UnitreeG1DexRetargeting"
    )
    def test_retarget(self, mock_dex_retargeting_cls):
        mock_dex_retargeting = mock_dex_retargeting_cls.return_value
        mock_dex_retargeting.get_joint_names.return_value = ["joint1", "joint2"]
        mock_dex_retargeting.get_left_joint_names.return_value = ["joint1"]
        mock_dex_retargeting.get_right_joint_names.return_value = ["joint2"]
        mock_dex_retargeting.compute_left.return_value = np.array([0.1])
        mock_dex_retargeting.compute_right.return_value = np.array([0.2])

        cfg = UnitreeG1RetargeterCfg(
            enable_visualization=False, sim_device="cpu", hand_joint_names=["joint1", "joint2"]
        )
        retargeter = UnitreeG1Retargeter(cfg)

        wrist_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        data = {
            DeviceBase.TrackingTarget.HAND_LEFT: {"wrist": wrist_pose},
            DeviceBase.TrackingTarget.HAND_RIGHT: {"wrist": wrist_pose},
        }

        result = retargeter.retarget(data)
        self.assertEqual(result.shape, (16,))


class TestGR1T2Retargeter(unittest.TestCase):
    @patch("isaaclab_teleop.deprecated.openxr.retargeters.humanoid.fourier.gr1t2_retargeter.GR1TR2DexRetargeting")
    def test_retarget(self, mock_dex_retargeting_cls):
        mock_dex_retargeting = mock_dex_retargeting_cls.return_value
        mock_dex_retargeting.get_joint_names.return_value = ["joint1", "joint2"]
        mock_dex_retargeting.get_left_joint_names.return_value = ["joint1"]
        mock_dex_retargeting.get_right_joint_names.return_value = ["joint2"]
        mock_dex_retargeting.compute_left.return_value = np.array([0.1])
        mock_dex_retargeting.compute_right.return_value = np.array([0.2])

        cfg = GR1T2RetargeterCfg(enable_visualization=False, sim_device="cpu", hand_joint_names=["joint1", "joint2"])
        retargeter = GR1T2Retargeter(cfg)

        wrist_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        data = {
            DeviceBase.TrackingTarget.HAND_LEFT: {"wrist": wrist_pose},
            DeviceBase.TrackingTarget.HAND_RIGHT: {"wrist": wrist_pose},
        }

        result = retargeter.retarget(data)
        self.assertEqual(result.shape, (16,))


class TestG1LowerBodyStandingMotionControllerRetargeter(unittest.TestCase):
    def test_retarget(self):
        cfg = G1LowerBodyStandingMotionControllerRetargeterCfg(
            hip_height=0.8, movement_scale=1.0, rotation_scale=1.0, sim_device="cpu"
        )
        retargeter = G1LowerBodyStandingMotionControllerRetargeter(cfg)

        # Mock input data
        # Inputs array structure: [thumbstick_x, thumbstick_y, trigger, squeeze, button_0, button_1, padding]
        left_inputs = np.zeros(7)
        left_inputs[0] = 0.5  # thumbstick x
        left_inputs[1] = 0.5  # thumbstick y

        right_inputs = np.zeros(7)
        right_inputs[0] = -0.5  # thumbstick x
        right_inputs[1] = -0.5  # thumbstick y

        data = {
            DeviceBase.TrackingTarget.CONTROLLER_LEFT: [np.zeros(7), left_inputs],
            DeviceBase.TrackingTarget.CONTROLLER_RIGHT: [np.zeros(7), right_inputs],
        }

        result = retargeter.retarget(data)
        # Output: [-left_thumbstick_y, -left_thumbstick_x, -right_thumbstick_x, hip_height]
        # hip_height modified by right_thumbstick_y

        self.assertEqual(result.shape, (4,))
        self.assertAlmostEqual(result[0].item(), -0.5)  # -left y
        self.assertAlmostEqual(result[1].item(), -0.5)  # -left x
        self.assertAlmostEqual(result[2].item(), 0.5)  # -right x
        # Check hip height modification logic if needed, but basic execution is key here


class TestG1TriHandUpperBodyMotionControllerGripperRetargeter(unittest.TestCase):
    def test_retarget(self):
        cfg = G1TriHandUpperBodyMotionControllerGripperRetargeterCfg(
            threshold_high=0.6, threshold_low=0.4, sim_device="cpu"
        )
        retargeter = G1TriHandUpperBodyMotionControllerGripperRetargeter(cfg)

        pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        inputs_trigger_high = np.zeros(7)
        inputs_trigger_high[2] = 0.8  # Trigger

        inputs_trigger_low = np.zeros(7)
        inputs_trigger_low[2] = 0.2  # Trigger

        data = {
            DeviceBase.TrackingTarget.CONTROLLER_LEFT: [pose, inputs_trigger_high],
            DeviceBase.TrackingTarget.CONTROLLER_RIGHT: [pose, inputs_trigger_low],
        }

        result = retargeter.retarget(data)
        # Output: [left_state, right_state, left_wrist(7), right_wrist(7)]
        self.assertEqual(result.shape, (16,))
        self.assertEqual(result[0].item(), 1.0)  # Left closed
        self.assertEqual(result[1].item(), 0.0)  # Right open


class TestG1TriHandUpperBodyMotionControllerRetargeter(unittest.TestCase):
    def test_retarget(self):
        cfg = G1TriHandUpperBodyMotionControllerRetargeterCfg(
            hand_joint_names=["dummy"] * 14,  # Not really used in logic, just passed to config
            sim_device="cpu",
            enable_visualization=False,
        )
        retargeter = G1TriHandUpperBodyMotionControllerRetargeter(cfg)

        pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        inputs = np.zeros(7)

        data = {
            DeviceBase.TrackingTarget.CONTROLLER_LEFT: [pose, inputs],
            DeviceBase.TrackingTarget.CONTROLLER_RIGHT: [pose, inputs],
        }

        result = retargeter.retarget(data)
        # Output: [left_wrist(7), right_wrist(7), hand_joints(14)]
        self.assertEqual(result.shape, (28,))


class TestG1TriHandUpperBodyRetargeter(unittest.TestCase):
    @patch(
        "isaaclab_teleop.deprecated.openxr.retargeters.humanoid.unitree.trihand.g1_upper_body_retargeter.G1TriHandDexRetargeting"
    )
    def test_retarget(self, mock_dex_retargeting_cls):
        mock_dex_retargeting = mock_dex_retargeting_cls.return_value
        mock_dex_retargeting.get_joint_names.return_value = ["joint1", "joint2"]
        mock_dex_retargeting.get_left_joint_names.return_value = ["joint1"]
        mock_dex_retargeting.get_right_joint_names.return_value = ["joint2"]
        mock_dex_retargeting.compute_left.return_value = np.array([0.1])
        mock_dex_retargeting.compute_right.return_value = np.array([0.2])

        cfg = G1TriHandUpperBodyRetargeterCfg(
            enable_visualization=False, sim_device="cpu", hand_joint_names=["joint1", "joint2"]
        )
        retargeter = G1TriHandUpperBodyRetargeter(cfg)

        wrist_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        data = {
            DeviceBase.TrackingTarget.HAND_LEFT: {"wrist": wrist_pose},
            DeviceBase.TrackingTarget.HAND_RIGHT: {"wrist": wrist_pose},
        }

        result = retargeter.retarget(data)
        # Output: [left_wrist(7), right_wrist(7), joints(2)]
        self.assertEqual(result.shape, (16,))


if __name__ == "__main__":
    unittest.main()
