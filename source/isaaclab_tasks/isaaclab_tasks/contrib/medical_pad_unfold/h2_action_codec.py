# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-specific H2 FK and unified-v1 action encoding, independent of policy SDKs."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import MISSING
from pathlib import Path

import numpy as np

from isaaclab.utils import configclass

from isaaclab_neural.models.cosmos3 import JointMappingCfg

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
H2_URDF = Path(__file__).with_name("assets") / "resources" / "H2_with_sharpa_hands.urdf"
"""The bundled, mesh-free H2 description with Sharpa hands."""
SIDES = ("right", "left")
VALID = np.zeros(59, dtype=bool)
VALID[9:33] = True
VALID[34:58] = True
# Columns are canonical wrist axes expressed in the native Sharpa C_MC frame.
# X: thumb-to-pinky, Z: wrist-to-fingers, Y = Z cross X; determinant +1.
WRIST_ALIGNMENT = {
    "right": np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    "left": np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
}


def urdf_joint_name(name):
    match = re.fullmatch(r"k(Left|Right)(.+)", name)
    return (match[1].lower() + "_" + re.sub(r"(?<!^)(?=[A-Z])", "_", match[2]).lower() + "_joint") if match else name


def _rotation(axis, angles):
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    angles = np.asarray(angles, dtype=np.float64)
    c, s = np.cos(angles)[..., None, None], np.sin(angles)[..., None, None]
    return c * np.eye(3) + (1 - c) * np.outer(axis, axis) + s * cross


def _origin_transform(element):
    result = np.eye(4)
    if element is None:
        return result
    xyz = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
    r, p, y = np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
    result[:3, :3] = _rotation([0, 0, 1], y) @ _rotation([0, 1, 0], p) @ _rotation([1, 0, 0], r)
    result[:3, 3] = xyz
    return result


class _H2FK:
    def __init__(self, xml, joint_names, *, root="torso_link"):
        self.names = [urdf_joint_name(name) for name in joint_names]
        robot = ET.fromstring(xml)
        by_child = {j.find("child").get("link"): j for j in robot.findall("joint")}
        targets = [s + "_hand_C_MC" for s in SIDES] + [s + "_" + f + "_fingertip" for s in SIDES for f in FINGERS]
        needed = set()
        for target in targets:
            child = target
            while child != root:
                if child not in by_child:
                    raise ValueError(f"No kinematic path from {root} to {target}")
                joint = by_child[child]
                needed.add(joint.get("name"))
                child = joint.find("parent").get("link")
        selected = [j for j in robot.findall("joint") if j.get("name") in needed]
        self.joints, available = [], {root}
        while selected:
            ready = [j for j in selected if j.find("parent").get("link") in available]
            if not ready:
                raise ValueError("Cyclic or disconnected kinematics")
            for j in ready:
                if j.get("type") not in ("fixed", "revolute", "continuous"):
                    raise ValueError("Unsupported joint type")
                name = j.get("name")
                idx = self.names.index(name) if name in self.names else None
                if j.get("type") != "fixed" and idx is None:
                    raise ValueError(f"Unrecorded movable joint {name}; refusing arbitrary zero")
                axis = j.find("axis")
                self.joints.append(
                    (
                        name,
                        j.get("type"),
                        idx,
                        j.find("parent").get("link"),
                        j.find("child").get("link"),
                        _origin_transform(j.find("origin")),
                        np.fromstring(axis.get("xyz", "1 0 0") if axis is not None else "1 0 0", sep=" "),
                    )
                )
                available.add(j.find("child").get("link"))
                selected.remove(j)
        self.root = root

    def frames(self, q):
        q = np.asarray(q, dtype=np.float64)
        result = {self.root: np.broadcast_to(np.eye(4), (len(q), 4, 4))}
        for name, kind, idx, parent, child, origin, axis in self.joints:
            joint = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
            if kind != "fixed":
                joint[:, :3, :3] = _rotation(axis, q[:, idx])
            result[child] = result[parent] @ origin @ joint
        return result

    def wrist_and_tips(self, q):
        frames = self.frames(q)
        wrists, tips = [], []
        for side in SIDES:
            wrist = frames[side + "_hand_C_MC"].copy()
            wrist[:, :3, :3] = wrist[:, :3, :3] @ WRIST_ALIGNMENT[side]
            positions = np.stack([frames[side + "_" + finger + "_fingertip"][:, :3, 3] for finger in FINGERS], axis=1)
            local = np.einsum("nji,nkj->nki", wrist[:, :3, :3], positions - wrist[:, None, :3, 3])
            wrists.append(wrist)
            tips.append(local)
        return np.stack(wrists, axis=1), np.stack(tips, axis=1)


def _load_stats(path):
    stats = json.loads(Path(path).read_bytes())["global"]
    low, high = np.array(stats["q01"]), np.array(stats["q99"])
    return (low + high) / 2, np.maximum((high - low) / 2, 1e-8)


def _raw_window(anchor_wrists, target_wrists, target_tips):
    """Encode one 16-command window using an explicitly supplied past anchor.

    Anchor choice is recorded by the caller and must match training/inference.
    Future measured poses must never be supplied to this function.
    """
    anchor = np.asarray(anchor_wrists)
    target, fingertips = np.asarray(target_wrists), np.asarray(target_tips)
    raw = np.zeros((16, 59), dtype=np.float64)
    for side, wrist_start, tips_start in [(0, 9, 18), (1, 34, 43)]:
        a, t = anchor[side], target[:, side]
        relative_rotation = a[:3, :3].T @ t[:, :3, :3]
        relative_translation = (t[:, :3, 3] - a[:3, 3]) @ a[:3, :3]
        raw[:, wrist_start : wrist_start + 3] = relative_translation
        raw[:, wrist_start + 3 : wrist_start + 9] = relative_rotation[:, :, :2].transpose(0, 2, 1).reshape(16, 6)
        raw[:, tips_start : tips_start + 15] = fingertips[:, side].reshape(16, 15)
    return raw


def _normalize(raw, stats):
    offset, scale = stats
    out = np.arcsinh((raw - offset) / scale) / np.arcsinh(1.0)
    out[..., ~VALID] = 0
    return out


_ARM = ("ShoulderPitch", "ShoulderRoll", "ShoulderYaw", "Elbow", "WristRoll", "WristPitch", "WristYaw")
_HAND = (
    "thumb_CMC_FE",
    "thumb_CMC_AA",
    "thumb_MCP_FE",
    "thumb_MCP_AA",
    "thumb_IP",
    "index_MCP_FE",
    "index_MCP_AA",
    "index_PIP",
    "index_DIP",
    "middle_MCP_FE",
    "middle_MCP_AA",
    "middle_PIP",
    "middle_DIP",
    "ring_MCP_FE",
    "ring_MCP_AA",
    "ring_PIP",
    "ring_DIP",
    "pinky_CMC",
    "pinky_MCP_FE",
    "pinky_MCP_AA",
    "pinky_PIP",
    "pinky_DIP",
)
JOINT_NAMES = tuple(
    [f"k{s}{j}" for s in ("Left", "Right") for j in _ARM] + [f"{s}_{j}" for s in ("left", "right") for j in _HAND]
)


@configclass
class H2JointMappingCfg(JointMappingCfg):
    """Joint mapping of the H2 robot with Sharpa hands for Cosmos3."""

    class_type: type | str = "{DIR}.h2_action_codec:H2JointMapping"

    urdf_path: str = str(H2_URDF)
    """The mesh-free H2 description whose forward kinematics give the wrist and fingertip poses."""

    global_stats_path: str = MISSING
    """Normalization statistics of the Cosmos Framework (``global_asinh_v1_backward_anchored_gripper_metric.json``).
    The Framework runs in the service's environment, so ``prepare_medical_pad.py`` copies the file next to the
    checkpoints."""


class H2JointMapping:
    """Convert H2 joint targets [rad] to the normalized ``unified_v1`` controls of Cosmos3.

    The 58 joints are in the order of :data:`JOINT_NAMES`. The controls are the wrist and fingertip poses of the
    targets relative to the wrist poses at the start of the chunk, normalized with the Cosmos Framework's statistics.
    """

    def __init__(self, cfg: H2JointMappingCfg):
        self.fk = _H2FK(Path(cfg.urdf_path).read_text(), JOINT_NAMES)
        self.stats = _load_stats(cfg.global_stats_path)

    def encode(self, joint_targets: np.ndarray, start_joint_state: np.ndarray) -> np.ndarray:
        """Return normalized controls shaped ``(16, 59)`` for 16 joint targets ``(16, 58)``."""
        anchor, _ = self.fk.wrist_and_tips(start_joint_state[None])
        wrists, tips = self.fk.wrist_and_tips(joint_targets)
        return _normalize(_raw_window(anchor[0], wrists, tips), self.stats).astype(np.float32)
