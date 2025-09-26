# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import os
import torch
import yaml
from scipy.spatial.transform import Rotation as R

import omni.log
from dex_retargeting.retargeting_config import RetargetingConfig

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path

# The index to map the OpenXR hand joints to the hand joints used
# in Dex-retargeting.
_HAND_JOINTS_INDEX = [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25]

# The transformation matrices to convert hand pose to canonical view.
_OPERATOR2MANO_RIGHT = np.array([
    [0, -1, 0],
    [-1, 0, 0],
    [0, 0, -1],
])

_OPERATOR2MANO_LEFT = np.array([
    [0, -1, 0],
    [-1, 0, 0],
    [0, 0, -1],
])

_LEFT_HAND_JOINT_NAMES = [
    "L_index_proximal_joint",
    "L_index_intermediate_joint",
    "L_middle_proximal_joint",
    "L_middle_intermediate_joint",
    "L_pinky_proximal_joint",
    "L_pinky_intermediate_joint",
    "L_ring_proximal_joint",
    "L_ring_intermediate_joint",
    "L_thumb_proximal_yaw_joint",
    "L_thumb_proximal_pitch_joint",
    "L_thumb_distal_joint",
]


_RIGHT_HAND_JOINT_NAMES = [
    "R_index_proximal_joint",
    "R_index_intermediate_joint",
    "R_middle_proximal_joint",
    "R_middle_intermediate_joint",
    "R_pinky_proximal_joint",
    "R_pinky_intermediate_joint",
    "R_ring_proximal_joint",
    "R_ring_intermediate_joint",
    "R_thumb_proximal_yaw_joint",
    "R_thumb_proximal_pitch_joint",
    "R_thumb_distal_joint",
]


class GR1TR2DexRetargeting:
    """A class for hand retargeting with GR1Fourier.

    Handles retargeting of OpenXRhand tracking data to GR1T2 robot hand joint angles.
    """

    def __init__(
        self,
        hand_joint_names: list[str],
        right_hand_config_filename: str = "fourier_hand_right_dexpilot.yml",
        left_hand_config_filename: str = "fourier_hand_left_dexpilot.yml",
        left_hand_urdf_path: str = f"{ISAACLAB_NUCLEUS_DIR}/Mimic/GR1T2_assets/GR1_T2_left_hand.urdf",
        right_hand_urdf_path: str = f"{ISAACLAB_NUCLEUS_DIR}/Mimic/GR1T2_assets/GR1_T2_right_hand.urdf",
        calibrate_scaling_factor: bool = False,
    ):
        """Initialize the hand retargeting.

        Args:
            hand_joint_names: Names of hand joints in the robot model
            right_hand_config_filename: Config file for right hand retargeting
            left_hand_config_filename: Config file for left hand retargeting
            calibrate_scaling_factor: If True, calibrate the scaling factor for the robot hands
        """
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "data/"))
        config_dir = os.path.join(data_dir, "configs/dex-retargeting")

        # Download urdf files from aws
        local_left_urdf_path = retrieve_file_path(left_hand_urdf_path, force_download=True)
        local_right_urdf_path = retrieve_file_path(right_hand_urdf_path, force_download=True)

        left_config_path = os.path.join(config_dir, left_hand_config_filename)
        right_config_path = os.path.join(config_dir, right_hand_config_filename)

        # Update the YAML files with the correct URDF paths
        self._update_yaml_with_urdf_path(left_config_path, local_left_urdf_path)
        self._update_yaml_with_urdf_path(right_config_path, local_right_urdf_path)

        self._dex_left_hand = RetargetingConfig.load_from_file(left_config_path).build()
        self._dex_right_hand = RetargetingConfig.load_from_file(right_config_path).build()

        self.left_dof_names = self._dex_left_hand.optimizer.robot.dof_joint_names
        self.right_dof_names = self._dex_right_hand.optimizer.robot.dof_joint_names
        self.dof_names = self.left_dof_names + self.right_dof_names
        self.isaac_lab_hand_joint_names = hand_joint_names

        # Hand length measurement
        self.hand_length = []
        self.max_measurements = 200
        self.scaling_factor_calibrated = False
        self.calibrate_scaling_factor = calibrate_scaling_factor

        omni.log.info("[GR1T2DexRetargeter] init done.")

    def _update_yaml_with_urdf_path(self, yaml_path: str, urdf_path: str):
        """Update YAML file with the correct URDF path.

        Args:
            yaml_path: Path to the YAML configuration file
            urdf_path: Path to the URDF file to use
        """
        try:
            # Read the YAML file
            with open(yaml_path) as file:
                config = yaml.safe_load(file)

            # Update the URDF path in the configuration
            if "retargeting" in config:
                config["retargeting"]["urdf_path"] = urdf_path
                omni.log.info(f"Updated URDF path in {yaml_path} to {urdf_path}")
            else:
                omni.log.warn(f"Unable to find 'retargeting' section in {yaml_path}")

            # Write the updated configuration back to the file
            with open(yaml_path, "w") as file:
                yaml.dump(config, file)

        except Exception as e:
            omni.log.error(f"Error updating YAML file {yaml_path}: {e}")

    def convert_hand_joints(
        self, joint_positions: np.ndarray, wrist: np.ndarray, operator2mano: np.ndarray
    ) -> np.ndarray:
        """Prepares the hand joints data for retargeting.

        Args:
            joint_positions: Array of joint positions from OpenXR
            wrist: Wrist pose [x, y, z, qw, qx, qy, qz]
            operator2mano: Transformation matrix to convert from operator to MANO frame

        Returns:
            Joint positions with shape (21, 3)
        """
        joint_position = joint_positions[_HAND_JOINTS_INDEX]
        # Convert hand pose to the canonical frame.
        joint_position -= joint_position[0:1, :]
        xr_wrist_quat = wrist[3:]
        # OpenXR hand uses w,x,y,z order for quaternions but scipy uses x,y,z,w order
        wrist_rot = R.from_quat([xr_wrist_quat[1], xr_wrist_quat[2], xr_wrist_quat[3], xr_wrist_quat[0]]).as_matrix()

        return joint_position @ wrist_rot @ operator2mano

    def compute_ref_value(self, joint_position: np.ndarray, indices: np.ndarray, retargeting_type: str) -> np.ndarray:
        """Computes reference value for retargeting.

        Args:
            joint_position: Joint positions array
            indices: Target link indices
            retargeting_type: Type of retargeting ("POSITION" or other)

        Returns:
            Reference value in cartesian space
        """
        if retargeting_type == "POSITION":
            return joint_position[indices, :]
        else:
            origin_indices = indices[0, :]
            task_indices = indices[1, :]
            ref_value = joint_position[task_indices, :] - joint_position[origin_indices, :]
            return ref_value

    def compute_one_hand(
        self, joint_positions: np.ndarray, wrist: np.ndarray, retargeting: RetargetingConfig, operator2mano: np.ndarray
    ) -> np.ndarray:
        """Computes retargeted joint angles for one hand.

        Args:
            joint_positions: Array of joint positions from OpenXR
            wrist: Wrist pose [x, y, z, qw, qx, qy, qz]
            retargeting: Retargeting configuration object
            operator2mano: Transformation matrix from operator to MANO frame

        Returns:
            Retargeted joint angles
        """
        joint_pos = self.convert_hand_joints(joint_positions, wrist, operator2mano)
        ref_value = self.compute_ref_value(
            joint_pos,
            indices=retargeting.optimizer.target_link_human_indices,
            retargeting_type=retargeting.optimizer.retargeting_type,
        )
        # Enable gradient calculation and inference mode in case some other script has disabled it
        # This is necessary for the retargeting to work since it uses gradient features that
        # are not available in inference mode
        with torch.enable_grad():
            with torch.inference_mode(False):
                return retargeting.retarget(ref_value)

    def get_joint_names(self) -> list[str]:
        """Returns list of all joint names."""
        return self.dof_names

    def get_left_joint_names(self) -> list[str]:
        """Returns list of left hand joint names."""
        return self.left_dof_names

    def get_right_joint_names(self) -> list[str]:
        """Returns list of right hand joint names."""
        return self.right_dof_names

    def get_hand_indices(self, robot) -> np.ndarray:
        """Gets indices of hand joints in robot's DOF array.

        Args:
            robot: Robot object containing DOF information

        Returns:
            Array of joint indices
        """
        return np.array([robot.dof_names.index(name) for name in self.dof_names], dtype=np.int64)

    def compute_left(self, left_joint_positions: np.ndarray, left_wrist: np.ndarray) -> np.ndarray:
        """Computes retargeted joints for left hand.

        Args:
            left_joint_positions: Array of left hand joint positions from OpenXR
            left_wrist: Left wrist pose [x, y, z, qw, qx, qy, qz]

        Returns:
            Retargeted joint angles for left hand
        """
        if left_joint_positions is not None and left_wrist is not None:
            # Collect hand length measurement for scaling factor calculation
            self.collect_hand_length_measurement(left_joint_positions, left_wrist)
            left_hand_q = self.compute_one_hand(
                left_joint_positions, left_wrist, self._dex_left_hand, _OPERATOR2MANO_LEFT
            )
        else:
            left_hand_q = np.zeros(len(_LEFT_HAND_JOINT_NAMES))
        return left_hand_q

    def compute_right(self, right_joint_positions: np.ndarray, right_wrist: np.ndarray) -> np.ndarray:
        """Computes retargeted joints for right hand.

        Args:
            right_joint_positions: Array of right hand joint positions from OpenXR
            right_wrist: Right wrist pose [x, y, z, qw, qx, qy, qz]

        Returns:
            Retargeted joint angles for right hand
        """
        if right_joint_positions is not None and right_wrist is not None:
            # Collect hand length measurement for scaling factor calculation
            self.collect_hand_length_measurement(right_joint_positions, right_wrist)
            right_hand_q = self.compute_one_hand(
                right_joint_positions, right_wrist, self._dex_right_hand, _OPERATOR2MANO_RIGHT
            )
        else:
            right_hand_q = np.zeros(len(_RIGHT_HAND_JOINT_NAMES))
        return right_hand_q

    def collect_hand_length_measurement(self, joint_positions: np.ndarray, wrist: np.ndarray):
        """Collect hand length measurement for scaling factor calculation.

        Args:
            joint_positions: Array of joint positions from OpenXR
            wrist: Wrist pose [x, y, z, qw, qx, qy, qz]
        """
        if not self.calibrate_scaling_factor or self.scaling_factor_calibrated:
            return
        if np.linalg.norm(wrist[:3]) == 0 or len(self.hand_length) >= self.max_measurements:
            return
        # Calculate hand length (distance from wrist to middle finger tip)
        palm_dir = (joint_positions[12] - wrist[:3]) / np.linalg.norm(joint_positions[12] - wrist[:3])
        middle_finger_dir = (joint_positions[15] - joint_positions[12]) / np.linalg.norm(
            joint_positions[15] - joint_positions[12]
        )
        is_hand_open = np.dot(palm_dir, middle_finger_dir) > 0.9
        hand_length = np.linalg.norm(wrist[:3] - joint_positions[15])
        if is_hand_open and 0.12 < hand_length < 0.27:
            self.hand_length.append(hand_length)
            if len(self.hand_length) >= self.max_measurements:
                self.calibrate_scaling_factors()

    def calibrate_scaling_factors(self, min_measurements: int = 50):
        """Update scaling factors directly in retargeting optimizers based on the collected hand length measurements.

        Args:
            min_measurements: Minimum number of measurements required before updating scaling factors
        """
        # Update hand scaling factor directly in optimizers
        if len(self.hand_length) >= min_measurements:
            hand_length_array = np.array(self.hand_length)
            q25 = np.percentile(hand_length_array, 25)
            q75 = np.percentile(hand_length_array, 75)
            filtered_data = hand_length_array[(hand_length_array >= q25) & (hand_length_array <= q75)]
            hand_length = float(np.mean(filtered_data))
            reference_hand_length = 0.19  # average adult hand length (meters)
            scaling_factor = reference_hand_length / hand_length

            # Update hand scaling factor
            try:
                if hasattr(self._dex_left_hand, "optimizer") and hasattr(self._dex_left_hand.optimizer, "scaling"):
                    self._dex_left_hand.optimizer.scaling *= scaling_factor
                if hasattr(self._dex_right_hand, "optimizer") and hasattr(self._dex_right_hand.optimizer, "scaling"):
                    self._dex_right_hand.optimizer.scaling *= scaling_factor
                    omni.log.info(f"Successfully updated hand scaling factor to {scaling_factor:.3f}")
                else:
                    omni.log.warn("Optimizer does not have 'scaling' attribute")
            except Exception as e:
                omni.log.warn(f"Failed to update scaling factor: {e}")

            self.scaling_factor_calibrated = True
            omni.log.info(
                f"Calibrated scaling factor to {scaling_factor:.3f} (hand length average: {hand_length:.3f}m)"
            )
