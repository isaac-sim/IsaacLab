# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

import carb
from isaacsim.core.utils.extensions import enable_extension
from omni.kit.xr.core import XRCore, XRPoseValidityFlags
from pxr import Gf

# Mapping from Manus joint index (0-24) to joint name. Palm (25) is calculated from middle metacarpal and proximal.
HAND_JOINT_MAP = {
    # Palm
    25: "palm",
    # Wrist
    0: "wrist",
    # Thumb
    21: "thumb_metacarpal",
    22: "thumb_proximal",
    23: "thumb_distal",
    24: "thumb_tip",
    # Index
    1: "index_metacarpal",
    2: "index_proximal",
    3: "index_intermediate",
    4: "index_distal",
    5: "index_tip",
    # Middle
    6: "middle_metacarpal",
    7: "middle_proximal",
    8: "middle_intermediate",
    9: "middle_distal",
    10: "middle_tip",
    # Ring
    11: "ring_metacarpal",
    12: "ring_proximal",
    13: "ring_intermediate",
    14: "ring_distal",
    15: "ring_tip",
    # Little
    16: "little_metacarpal",
    17: "little_proximal",
    18: "little_intermediate",
    19: "little_distal",
    20: "little_tip",
}


class ManusViveIntegration:
    def __init__(self):
        enable_extension("isaacsim.xr.input_devices")
        from isaacsim.xr.input_devices.impl.manus_vive_integration import get_manus_vive_integration

        _manus_vive_integration = get_manus_vive_integration()
        self.manus = _manus_vive_integration.manus_tracker
        self.vive_tracker = _manus_vive_integration.vive_tracker
        self.device_status = _manus_vive_integration.device_status
        self.default_pose = {"position": [0, 0, 0], "orientation": [1, 0, 0, 0]}
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
        """
        Get all device data.
        Format: {
            'manus_gloves': {
                '{left/right}_{joint_index}': {
                    'position': [x, y, z],
                    'orientation': [w, x, y, z]
                },
                ...
            },
            'vive_trackers': {
                '{vive_tracker_id}': {
                    'position': [x, y, z],
                    'orientation': [w, x, y, z]
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
        """
        Get device status.
        Format: {
            'manus_gloves': {'connected': bool, 'last_data_time': float},
            'vive_trackers': {'connected': bool, 'last_data_time': float},
            'left_hand_connected': bool,
            'right_hand_connected': bool
        }
        """
        return self.device_status

    def update_manus(self):
        """Update raw Manus glove data."""
        self.manus.update()
        manus_data = self.manus.get_data()
        self.device_status["left_hand_connected"] = "left_0" in manus_data
        self.device_status["right_hand_connected"] = "right_0" in manus_data

    def update_vive(self):
        """Update raw Vive tracker data, and initialize coordinate transformation if it is the first data update."""
        self.vive_tracker.update()
        try:
            # Initialize coordinate transformation from first Vive wrist position
            if self.scene_T_lighthouse_static is None:
                self._initialize_coordinate_transformation()
        except Exception as e:
            carb.log_error(f"Vive tracker update failed: {e}")

    def _initialize_coordinate_transformation(self):
        """
        Initialize the scene to lighthouse coordinate transformation.
        The coordinate transformation is used to transform the wrist pose from lighthouse coordinate system to isaac sim scene coordinate.
        It is computed from multiple frames of AVP/OpenXR wrist pose and Vive wrist pose samples at the beginning of the session.
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
            if choose_A is None:
                carb.log_info(f"error A: {errA}, error B: {errB}")
                return

            if choose_A:
                chosen_list = self._pairA_candidates
                self._vive_left_id, self._vive_right_id = wm0_id, wm1_id
            else:
                chosen_list = self._pairB_candidates
                self._vive_left_id, self._vive_right_id = wm1_id, wm0_id

            if len(chosen_list) >= min_frames:
                cluster = select_mode_cluster(chosen_list)
                carb.log_info(f"Wrist calibration: formed size {len(cluster)} cluster from {len(chosen_list)} samples")
                if len(cluster) >= min_frames // 2:
                    averaged = average_transforms(cluster)
                    self.scene_T_lighthouse_static = averaged
                    carb.log_info(f"Resolved mapping: {self._vive_left_id}->Left, {self._vive_right_id}->Right")

        except Exception as e:
            carb.log_error(f"Failed to initialize coordinate transformation: {e}")

    def _transform_vive_data(self, device_data: dict) -> dict:
        """Transform all joints in vive data to scene coordinates."""
        transformed_data = {}
        for joint_name, joint_data in device_data.items():
            transformed_pose = self.default_pose
            if self.scene_T_lighthouse_static is not None:
                transformed_matrix = pose_to_matrix(joint_data) * self.scene_T_lighthouse_static
                transformed_pose = matrix_to_pose(transformed_matrix)
            transformed_data[joint_name] = transformed_pose
        return transformed_data

    def _get_scene_T_wrist_matrix(self, vive_data: dict) -> dict:
        """Get the wrist position and orientation from vive data."""
        scene_T_wrist = {"left": Gf.Matrix4d().SetIdentity(), "right": Gf.Matrix4d().SetIdentity()}
        # 10 cm offset on Y-axis for change in vive tracker position after flipping the palm
        Rcorr = Gf.Matrix4d(self.rot_adjust, Gf.Vec3d(0, -0.1, 0))
        if self._vive_left_id is not None:
            scene_T_wrist["left"] = Rcorr * pose_to_matrix(vive_data[self._vive_left_id])
        if self._vive_right_id is not None:
            scene_T_wrist["right"] = Rcorr * pose_to_matrix(vive_data[self._vive_right_id])
        return scene_T_wrist

    def _transform_manus_data(self, manus_data: dict, scene_T_wrist: dict) -> dict:
        """Transform Manus glove data: relative joint poses to wrist -> joint in scene coordinates"""
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
        """Get palm position and orientation from transformed data."""
        if f"{hand}_6" not in transformed_data or f"{hand}_7" not in transformed_data:
            carb.log_error(f"Joint data not found for {hand}")
            return self.default_pose
        metacarpal = transformed_data[f"{hand}_6"]
        proximal = transformed_data[f"{hand}_7"]
        pos = (np.array(metacarpal["position"]) + np.array(proximal["position"])) / 2.0
        return {"position": [pos[0], pos[1], pos[2]], "orientation": metacarpal["orientation"]}


def compute_delta_errors(a: Gf.Matrix4d, b: Gf.Matrix4d) -> tuple[float, float]:
    """Return (translation_error_m, rotation_error_deg) between transforms a and b."""
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
    """Average rigid transforms: mean translation and normalized mean quaternion (with sign alignment)."""
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
    """Return the largest cluster (mode) under proximity thresholds."""
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
    """Get OpenXR wrist world matrix if valid, else None."""
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
        carb.log_warn(f"OpenXR {hand} wrist fetch failed: {e}")
        return None


def get_vive_wrist_ids(vive_data: dict):
    """Return tuple (wm0_id, wm1_id) if available, else Nones. Sort by key for stability."""
    wm_ids = [k for k in vive_data.keys() if len(k) >= 2 and k[:2] == "WM"]
    wm_ids.sort()
    if len(wm_ids) >= 2:  # Assumes the first two vive trackers are the wrist trackers
        return wm_ids[0], wm_ids[1]
    if len(wm_ids) == 1:
        return wm_ids[0], None
    return None, None


def pose_to_matrix(pose: dict) -> Gf.Matrix4d:
    """Convert position and orientation to a 4x4 matrix."""
    pos, ori = pose["position"], pose["orientation"]
    quat = Gf.Quatd(ori[0], Gf.Vec3d(ori[1], ori[2], ori[3]))
    rot = Gf.Matrix3d().SetRotate(quat)
    trans = Gf.Vec3d(pos[0], pos[1], pos[2])
    return Gf.Matrix4d(rot, trans)


def matrix_to_pose(matrix: Gf.Matrix4d) -> dict:
    """Convert a 4x4 matrix to position and orientation."""
    pos = matrix.ExtractTranslation()
    rot = matrix.ExtractRotation()
    quat = rot.GetQuat()
    return {
        "position": [pos[0], pos[1], pos[2]],
        "orientation": [quat.GetReal(), quat.GetImaginary()[0], quat.GetImaginary()[1], quat.GetImaginary()[2]],
    }


def get_pairing_error(trans_errs: list, rot_errs: list) -> float:
    def _median(values: list) -> float:
        try:
            return float(np.median(np.asarray(values, dtype=np.float64)))
        except Exception:
            return float("inf")

    return _median(trans_errs) + 0.01 * _median(rot_errs)
