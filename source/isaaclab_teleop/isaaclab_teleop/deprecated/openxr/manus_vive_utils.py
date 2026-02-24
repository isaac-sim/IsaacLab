# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import logging
from time import time

import numpy as np

from isaacsim.core.utils.extensions import enable_extension

# For testing purposes, we need to mock the XRCore
XRCore, XRPoseValidityFlags = None, None

with contextlib.suppress(ModuleNotFoundError):
    from omni.kit.xr.core import XRCore, XRPoseValidityFlags

from pxr import Gf

# import logger
logger = logging.getLogger(__name__)

# Mapping from Manus joint index (0-24) to joint name. Palm (25) is calculated from middle metacarpal and proximal.
HAND_JOINT_MAP = {
    # Wrist
    0: "wrist",
    # Thumb
    1: "thumb_metacarpal",
    2: "thumb_proximal",
    3: "thumb_distal",
    4: "thumb_tip",
    # Index
    5: "index_metacarpal",
    6: "index_proximal",
    7: "index_intermediate",
    8: "index_distal",
    9: "index_tip",
    # Middle
    10: "middle_metacarpal",
    11: "middle_proximal",
    12: "middle_intermediate",
    13: "middle_distal",
    14: "middle_tip",
    # Ring
    15: "ring_metacarpal",
    16: "ring_proximal",
    17: "ring_intermediate",
    18: "ring_distal",
    19: "ring_tip",
    # Little
    20: "little_metacarpal",
    21: "little_proximal",
    22: "little_intermediate",
    23: "little_distal",
    24: "little_tip",
    # Palm
    25: "palm",
}


class ManusViveIntegration:
    def __init__(self):
        enable_extension("isaacsim.xr.input_devices")
        from isaacsim.xr.input_devices.impl.manus_vive_integration import get_manus_vive_integration

        _manus_vive_integration = get_manus_vive_integration()
        self.manus = _manus_vive_integration.manus_tracker
        self.vive_tracker = _manus_vive_integration.vive_tracker
        self.device_status = _manus_vive_integration.device_status
        self.default_pose = {"position": [0, 0, 0], "orientation": [0, 0, 0, 1]}
        # 90-degree ccw rotation on Y-axis and 90-degree ccw rotation on Z-axis
        self.rot_adjust = Gf.Matrix3d().SetRotate(Gf.Quatd(0.5, Gf.Vec3d(-0.5, 0.5, 0.5)))
        self.scene_T_lighthouse_static = None
        self._vive_left_id = None
        self._vive_right_id = None
        self._pairA_candidates = []  # Pair A: WM0->Left, WM1->Right
        self._pairB_candidates = []  # Pair B: WM1->Left, WM0->Right
        self._pairA_trans_errs = []
        self._pairA_rot_errs = []
        self._pairB_trans_errs = []
        self._pairB_rot_errs = []

    def get_all_device_data(self) -> dict:
        """Get all tracked device data in scene coordinates.

        Returns:
            Manus glove joint data and Vive tracker data.
            {
                'manus_gloves': {
                    '{left/right}_{joint_index}': {
                        'position': [x, y, z],
                        'orientation': [x, y, z, w]
                    },
                    ...
                },
                'vive_trackers': {
                    '{vive_tracker_id}': {
                        'position': [x, y, z],
                        'orientation': [x, y, z, w]
                    },
                    ...
                }
            }
        """
        self.update_manus()
        self.update_vive()
        # Get raw data from trackers
        manus_data = self.manus.get_data()
        vive_data = self.vive_tracker.get_data()
        vive_transformed = self._transform_vive_data(vive_data)
        scene_T_wrist = self._get_scene_T_wrist_matrix(vive_transformed)

        return {
            "manus_gloves": self._transform_manus_data(manus_data, scene_T_wrist),
            "vive_trackers": vive_transformed,
        }

    def get_device_status(self) -> dict:
        """Get connection and data freshness status for Manus gloves and Vive trackers.

        Returns:
            Dictionary containing connection flags and last-data timestamps.
        Format: {
            'manus_gloves': {'connected': bool, 'last_data_time': float},
            'vive_trackers': {'connected': bool, 'last_data_time': float},
            'left_hand_connected': bool,
            'right_hand_connected': bool
        }
        """
        return self.device_status

    def update_manus(self):
        """Update raw Manus glove data and status flags."""
        self.manus.update()
        self.device_status["manus_gloves"]["last_data_time"] = time()
        manus_data = self.manus.get_data()
        self.device_status["left_hand_connected"] = "left_0" in manus_data
        self.device_status["right_hand_connected"] = "right_0" in manus_data

    def update_vive(self):
        """Update raw Vive tracker data, and initialize coordinate transformation if it is the first data update."""
        self.vive_tracker.update()
        self.device_status["vive_trackers"]["last_data_time"] = time()
        try:
            # Initialize coordinate transformation from first Vive wrist position
            if self.scene_T_lighthouse_static is None:
                self._initialize_coordinate_transformation()
        except Exception as e:
            logger.error(f"Vive tracker update failed: {e}")

    def _initialize_coordinate_transformation(self):
        """Initialize the scene to lighthouse coordinate transformation.

        The coordinate transformation is used to transform the wrist pose from lighthouse
        coordinate system to isaac sim scene coordinate. It is computed from multiple
        frames of AVP/OpenXR wrist pose and Vive wrist pose samples at the beginning of the session.
        """
        min_frames = 6
        tolerance = 3.0
        vive_data = self.vive_tracker.get_data()
        wm0_id, wm1_id = get_vive_wrist_ids(vive_data)
        if wm0_id is None and wm1_id is None:
            return

        try:
            # Fetch OpenXR wrists
            L, R, gloves = None, None, []
            if self.device_status["left_hand_connected"]:
                gloves.append("left")
                L = get_openxr_wrist_matrix("left")
            if self.device_status["right_hand_connected"]:
                gloves.append("right")
                R = get_openxr_wrist_matrix("right")

            M0, M1, vives = None, None, []
            if wm0_id is not None:
                vives.append(wm0_id)
                M0 = pose_to_matrix(vive_data[wm0_id])
            if wm1_id is not None:
                vives.append(wm1_id)
                M1 = pose_to_matrix(vive_data[wm1_id])

            TL0, TL1, TR0, TR1 = None, None, None, None
            # Compute transforms for available pairs
            if wm0_id is not None and L is not None:
                TL0 = M0.GetInverse() * L
                self._pairA_candidates.append(TL0)
            if wm1_id is not None and L is not None:
                TL1 = M1.GetInverse() * L
                self._pairB_candidates.append(TL1)
            if wm1_id is not None and R is not None:
                TR1 = M1.GetInverse() * R
                self._pairA_candidates.append(TR1)
            if wm0_id is not None and R is not None:
                TR0 = M0.GetInverse() * R
                self._pairB_candidates.append(TR0)

            # Per-frame pairing error if both candidates present
            if TL0 is not None and TR1 is not None and TL1 is not None and TR0 is not None:
                eT, eR = compute_delta_errors(TL0, TR1)
                self._pairA_trans_errs.append(eT)
                self._pairA_rot_errs.append(eR)
                eT, eR = compute_delta_errors(TL1, TR0)
                self._pairB_trans_errs.append(eT)
                self._pairB_rot_errs.append(eR)

            # Choose a mapping
            choose_A = None
            if len(self._pairA_candidates) == 0 and len(self._pairB_candidates) >= min_frames:
                choose_A = False
            elif len(self._pairB_candidates) == 0 and len(self._pairA_candidates) >= min_frames:
                choose_A = True
            elif len(self._pairA_trans_errs) >= min_frames and len(self._pairB_trans_errs) >= min_frames:
                errA = get_pairing_error(self._pairA_trans_errs, self._pairA_rot_errs)
                errB = get_pairing_error(self._pairB_trans_errs, self._pairB_rot_errs)
                if errA < errB and errA < tolerance:
                    choose_A = True
                elif errB < errA and errB < tolerance:
                    choose_A = False
                elif len(self._pairA_trans_errs) % 10 == 0 or len(self._pairB_trans_errs) % 10 == 0:
                    print("Computing pairing of Vive trackers with wrists")
                    logger.info(
                        f"Pairing Vive trackers with wrists: error of pairing A: {errA}, error of pairing B: {errB}"
                    )
            if choose_A is None:
                return

            if choose_A:
                chosen_list = self._pairA_candidates
                self._vive_left_id, self._vive_right_id = wm0_id, wm1_id
            else:
                chosen_list = self._pairB_candidates
                self._vive_left_id, self._vive_right_id = wm1_id, wm0_id

            if len(chosen_list) >= min_frames:
                cluster = select_mode_cluster(chosen_list)
                if len(chosen_list) % 10 == 0:
                    print(
                        f"Computing wrist calibration: formed size {len(cluster)} cluster from"
                        f" {len(chosen_list)} samples"
                    )
                if len(cluster) >= min_frames // 2:
                    averaged = average_transforms(cluster)
                    self.scene_T_lighthouse_static = averaged
                    print(
                        f"Wrist calibration computed. Resolved mapping: {self._vive_left_id}->Left,"
                        f" {self._vive_right_id}->Right"
                    )

        except Exception as e:
            logger.error(f"Failed to initialize coordinate transformation: {e}")

    def _transform_vive_data(self, device_data: dict) -> dict:
        """Transform Vive tracker poses to scene coordinates.

        The returned data is in xyzw format.

        Args:
            device_data: raw vive tracker poses, with device id as keys.

        Returns:
            Vive tracker poses in scene coordinates, with device id as keys.
        """
        transformed_data = {}
        for joint_name, joint_data in device_data.items():
            transformed_pose = self.default_pose
            if self.scene_T_lighthouse_static is not None:
                transformed_matrix = pose_to_matrix(joint_data) * self.scene_T_lighthouse_static
                transformed_pose = matrix_to_pose(transformed_matrix)
            transformed_data[joint_name] = transformed_pose
        return transformed_data

    def _get_scene_T_wrist_matrix(self, vive_data: dict) -> dict:
        """Compute scene-frame wrist transforms for left and right hands.

        Args:
            vive_data: Vive tracker poses expressed in scene coordinates.

        Returns:
            Dictionary with 'left' and 'right' keys mapping to 4x4 transforms.
        """
        scene_T_wrist = {"left": Gf.Matrix4d().SetIdentity(), "right": Gf.Matrix4d().SetIdentity()}
        # 10 cm offset on Y-axis for change in vive tracker position after flipping the palm
        Rcorr = Gf.Matrix4d(self.rot_adjust, Gf.Vec3d(0, -0.1, 0))
        if self._vive_left_id is not None:
            scene_T_wrist["left"] = Rcorr * pose_to_matrix(vive_data[self._vive_left_id])
        if self._vive_right_id is not None:
            scene_T_wrist["right"] = Rcorr * pose_to_matrix(vive_data[self._vive_right_id])
        return scene_T_wrist

    def _transform_manus_data(self, manus_data: dict, scene_T_wrist: dict) -> dict:
        """Transform Manus glove joints from wrist-relative to scene coordinates.

        The returned data is in xyzw format.

        Args:
            manus_data: Raw Manus joint pose dictionary, wrist-relative.
            scene_T_wrist: Dictionary of scene transforms for left and right wrists.

        Returns:
            Dictionary of Manus joint poses in scene coordinates.
        """
        Rcorr = Gf.Matrix4d(self.rot_adjust, Gf.Vec3d(0, 0, 0)).GetInverse()
        transformed_data = {}
        for joint_name, joint_data in manus_data.items():
            hand, _ = joint_name.split("_")
            joint_mat = Rcorr * pose_to_matrix(joint_data) * scene_T_wrist[hand]
            transformed_data[joint_name] = matrix_to_pose(joint_mat)
        # Calculate palm with middle metacarpal and proximal
        transformed_data["left_25"] = self._get_palm(transformed_data, "left")
        transformed_data["right_25"] = self._get_palm(transformed_data, "right")
        return transformed_data

    def _get_palm(self, transformed_data: dict, hand: str) -> dict:
        """Compute palm pose from middle metacarpal and proximal joints.

        Args:
            transformed_data: Manus joint poses in scene coordinates.
            hand: The hand side, either 'left' or 'right'.

        Returns:
            Pose dictionary with 'position' and 'orientation'.
        """
        if f"{hand}_6" not in transformed_data or f"{hand}_7" not in transformed_data:
            # Joint data not arrived yet
            return self.default_pose
        metacarpal = transformed_data[f"{hand}_6"]
        proximal = transformed_data[f"{hand}_7"]
        pos = (np.array(metacarpal["position"]) + np.array(proximal["position"])) / 2.0
        return {"position": [pos[0], pos[1], pos[2]], "orientation": metacarpal["orientation"]}


def compute_delta_errors(a: Gf.Matrix4d, b: Gf.Matrix4d) -> tuple[float, float]:
    """Compute translation and rotation error between two transforms.

    Args:
        a: The first transform.
        b: The second transform.

    Returns:
        Tuple containing (translation_error_m, rotation_error_deg).
    """
    try:
        delta = a * b.GetInverse()
        t = delta.ExtractTranslation()
        trans_err = float(np.linalg.norm([t[0], t[1], t[2]]))
        q = delta.ExtractRotation().GetQuat()
        w = float(max(min(q.GetReal(), 1.0), -1.0))
        ang = 2.0 * float(np.arccos(w))
        ang_deg = float(np.degrees(ang))
        if ang_deg > 180.0:
            ang_deg = 360.0 - ang_deg
        return trans_err, ang_deg
    except Exception:
        return float("inf"), float("inf")


def average_transforms(mats: list[Gf.Matrix4d]) -> Gf.Matrix4d:
    """Average rigid transforms across translations and quaternions.

    Args:
        mats: The list of 4x4 transforms to average.

    Returns:
        Averaged 4x4 transform, or None if the list is empty.
    """
    if not mats:
        return None
    ref_quat = mats[0].ExtractRotation().GetQuat()
    ref = np.array([ref_quat.GetReal(), *ref_quat.GetImaginary()])
    acc_q = np.zeros(4, dtype=np.float64)
    acc_t = np.zeros(3, dtype=np.float64)
    for m in mats:
        t = m.ExtractTranslation()
        acc_t += np.array([t[0], t[1], t[2]], dtype=np.float64)
        q = m.ExtractRotation().GetQuat()
        qi = np.array([q.GetReal(), *q.GetImaginary()], dtype=np.float64)
        if np.dot(qi, ref) < 0.0:
            qi = -qi
        acc_q += qi
    mean_t = acc_t / float(len(mats))
    norm = np.linalg.norm(acc_q)
    if norm <= 1e-12:
        quat_avg = Gf.Quatd(1.0, Gf.Vec3d(0.0, 0.0, 0.0))
    else:
        qn = acc_q / norm
        quat_avg = Gf.Quatd(float(qn[0]), Gf.Vec3d(float(qn[1]), float(qn[2]), float(qn[3])))
    rot3 = Gf.Matrix3d().SetRotate(quat_avg)
    trans = Gf.Vec3d(float(mean_t[0]), float(mean_t[1]), float(mean_t[2]))
    return Gf.Matrix4d(rot3, trans)


def select_mode_cluster(
    mats: list[Gf.Matrix4d], trans_thresh_m: float = 0.03, rot_thresh_deg: float = 10.0
) -> list[Gf.Matrix4d]:
    """Select the largest cluster of transforms under proximity thresholds.

    Args:
        mats: The list of 4x4 transforms to cluster.
        trans_thresh_m: The translation threshold in meters.
        rot_thresh_deg: The rotation threshold in degrees.

    Returns:
        The largest cluster (mode) of transforms.
    """
    if not mats:
        return []
    best_cluster: list[Gf.Matrix4d] = []
    for center in mats:
        cluster: list[Gf.Matrix4d] = []
        for m in mats:
            trans_err, rot_err = compute_delta_errors(m, center)
            if trans_err <= trans_thresh_m and rot_err <= rot_thresh_deg:
                cluster.append(m)
        if len(cluster) > len(best_cluster):
            best_cluster = cluster
    return best_cluster


def get_openxr_wrist_matrix(hand: str) -> Gf.Matrix4d:
    """Get the OpenXR wrist matrix if valid.

    Args:
        hand: The hand side ('left' or 'right').

    Returns:
        4x4 transform for the wrist if valid, otherwise None.
    """
    hand = hand.lower()
    try:
        hand_device = XRCore.get_singleton().get_input_device(f"/user/hand/{hand}")
        if hand_device is None:
            return None
        joints = hand_device.get_all_virtual_world_poses()
        if "wrist" not in joints:
            return None
        joint = joints["wrist"]
        required = XRPoseValidityFlags.POSITION_VALID | XRPoseValidityFlags.ORIENTATION_VALID
        if (joint.validity_flags & required) != required:
            return None
        return joint.pose_matrix
    except Exception as e:
        logger.warning(f"OpenXR {hand} wrist fetch failed: {e}")
        return None


def get_vive_wrist_ids(vive_data: dict) -> tuple[str, str]:
    """Get the Vive wrist tracker IDs if available.

    Args:
        vive_data: The raw Vive data dictionary.

    Returns:
        (wm0_id, wm1_id) if available, otherwise None values.
    """
    wm_ids = [k for k in vive_data.keys() if len(k) >= 2 and k[:2] == "WM"]
    wm_ids.sort()
    if len(wm_ids) >= 2:  # Assumes the first two vive trackers are the wrist trackers
        return wm_ids[0], wm_ids[1]
    if len(wm_ids) == 1:
        return wm_ids[0], None
    return None, None


def pose_to_matrix(pose: dict) -> Gf.Matrix4d:
    """Convert a pose dictionary to a 4x4 transform matrix.

    pose is a dictionary with 'position' and 'orientation' fields.
    position is a tuple of 3 floats.
    orientation is a tuple of 4 floats. (x, y, z, w)

    Args:
        pose: The pose with 'position' and 'orientation' fields.

    Returns:
        A 4x4 transform representing the pose.
    """
    pos, ori = pose["position"], pose["orientation"]
    # ori is (x, y, z, w) - Gf.Quatd takes (real, imaginary_vec)
    quat = Gf.Quatd(ori[3], Gf.Vec3d(ori[0], ori[1], ori[2]))
    rot = Gf.Matrix3d().SetRotate(quat)
    trans = Gf.Vec3d(pos[0], pos[1], pos[2])
    return Gf.Matrix4d(rot, trans)


def matrix_to_pose(matrix: Gf.Matrix4d) -> dict:
    """Convert a 4x4 transform matrix to a pose dictionary.

    pose is a dictionary with 'position' and 'orientation' fields.
    position is a tuple of 3 floats.
    orientation is a tuple of 4 floats. (x, y, z, w)

    Args:
        matrix: The 4x4 transform matrix to convert.

    Returns:
        Pose dictionary with 'position' and 'orientation'.
    """
    pos = matrix.ExtractTranslation()
    rot = matrix.ExtractRotation()
    quat = rot.GetQuat()
    return {
        "position": [pos[0], pos[1], pos[2]],
        "orientation": [quat.GetImaginary()[0], quat.GetImaginary()[1], quat.GetImaginary()[2], quat.GetReal()],
    }


def get_pairing_error(trans_errs: list, rot_errs: list) -> float:
    """Compute a scalar pairing error from translation and rotation errors.

    Args:
        trans_errs: The list of translation errors across samples.
        rot_errs: The list of rotation errors across samples.

    Returns:
        The weighted sum of medians of translation and rotation errors.
    """

    def _median(values: list) -> float:
        try:
            return float(np.median(np.asarray(values, dtype=np.float64)))
        except Exception:
            return float("inf")

    return _median(trans_errs) + 0.01 * _median(rot_errs)
