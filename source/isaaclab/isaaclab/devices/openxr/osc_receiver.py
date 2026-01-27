# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OSC receiver for body tracking data from external sources (e.g., Meta Quest)."""

from __future__ import annotations

import threading

import numpy as np
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer

# Body tracking joint names compatible with VRChat OSC body tracking protocol
BODY_JOINT_NAMES: tuple[str, ...] = (
    "head",
    "hip",
    "chest",
    "left_foot",
    "right_foot",
    "left_knee",
    "right_knee",
    "left_elbow",
    "right_elbow",
)

NUM_BODY_JOINTS: int = len(BODY_JOINT_NAMES)
DOF_PER_JOINT: int = 7  # 3 position + 4 rotation (quaternion)


def _normalize(v: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normalize a vector, returning unchanged if norm is below epsilon."""
    norm = np.linalg.norm(v)
    return v if norm < eps else v / norm


def _rotation_matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to quaternion (x, y, z, w)."""
    trace = np.trace(R)
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    return _normalize(np.array([qx, qy, qz, qw], dtype=np.float32))


# Default world up vector for coordinate frame construction
_DEFAULT_UP_REF = np.array([0.0, 0.0, 1.0], dtype=np.float32)
_ALT_UP_REF = np.array([0.0, 1.0, 0.0], dtype=np.float32)


def _quat_from_forward_up(forward: np.ndarray, up_ref: np.ndarray = _DEFAULT_UP_REF) -> np.ndarray:
    """Build a quaternion (x, y, z, w) from a forward direction vector.

    Constructs an orthonormal basis where +X is forward, +Z is up, and +Y completes
    the right-handed coordinate system.

    Args:
        forward: 3D direction vector in world coordinates for the +X axis.
        up_ref: Reference up vector (world), defaults to [0, 0, 1].

    Returns:
        Quaternion as (x, y, z, w) numpy array.
    """
    f = _normalize(forward.astype(np.float32))
    if np.allclose(f, 0.0):
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    # Project up_ref onto plane orthogonal to forward
    u_ref = up_ref.astype(np.float32)
    up_proj = u_ref - np.dot(u_ref, f) * f
    if np.linalg.norm(up_proj) < 1e-6:
        # Forward is nearly parallel to up_ref; use alternate
        up_proj = _ALT_UP_REF - np.dot(_ALT_UP_REF, f) * f

    z_axis = _normalize(up_proj)
    x_axis = f
    y_axis = _normalize(np.cross(z_axis, x_axis))

    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    return _rotation_matrix_to_quat(R)


class BodyOscReceiver:
    """Receives body tracking data via OSC protocol.

    This class listens for OSC messages containing body joint positions and
    computes heuristic orientations based on joint relationships. Compatible
    with Meta Quest body tracking via OSC forwarding.

    The data format for each joint is [x, y, z, qx, qy, qz, qw].
    """

    # Mapping of joint pairs for heuristic rotation computation: (source, target)
    _ROTATION_PAIRS: tuple[tuple[str, str], ...] = (
        ("hip", "chest"),  # Hip forward: hip -> chest
        ("chest", "head"),  # Chest forward: chest -> head
        ("chest", "head"),  # Head forward: same as chest
        ("hip", "left_foot"),  # Left foot forward: hip -> left_foot
        ("hip", "right_foot"),  # Right foot forward: hip -> right_foot
        ("hip", "left_knee"),  # Left knee forward: hip -> left_knee
        ("hip", "right_knee"),  # Right knee forward: hip -> right_knee
        ("chest", "left_elbow"),  # Left elbow forward: chest -> left_elbow
        ("chest", "right_elbow"),  # Right elbow forward: chest -> right_elbow
    )

    def __init__(self, ip: str = "127.0.0.1", port: int = 9000):
        """Initialize the OSC body tracking receiver.

        Args:
            ip: IP address to listen on.
            port: UDP port to listen on.
        """
        self._joint_index = {name: i for i, name in enumerate(BODY_JOINT_NAMES)}
        self._data = np.zeros((NUM_BODY_JOINTS, DOF_PER_JOINT), dtype=np.float32)
        self._data[:, 3:] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self._lock = threading.Lock()

        dispatcher = Dispatcher()
        dispatcher.map("/tracking/trackers/*/position", self._on_position)

        self._server = ThreadingOSCUDPServer((ip, port), dispatcher)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        """Stop the OSC server and clean up resources."""
        if self._server is not None:
            self._server.shutdown()
            self._server = None

    def __del__(self):
        """Clean up resources when the object is destroyed."""
        self.shutdown()

    def _on_position(self, addr: str, *args) -> None:
        """Handle incoming OSC position messages.

        Args:
            addr: OSC address (e.g., /tracking/trackers/head/position)
            args: Position values (x, y, z)
        """
        if len(args) < 3:
            return

        parts = addr.split("/")
        if len(parts) < 5:
            return

        token = parts[3]
        idx = self._joint_index.get(token)
        if idx is None:
            try:
                idx = int(token)
            except ValueError:
                return

        if idx < 0 or idx >= NUM_BODY_JOINTS:
            return

        # Note: coordinate swizzle from OSC format (x, z, y) to internal (x, y, z)
        x, z, y = args[:3]
        with self._lock:
            self._data[idx, 0:3] = [x, y, z]

    def recompute_rotations(self) -> None:
        """Recompute heuristic rotations for all joints based on current positions."""
        with self._lock:
            self._recompute_rotations_locked()

    def _recompute_rotations_locked(self) -> None:
        """Recompute rotations without acquiring lock. Caller must hold self._lock."""
        pos = self._data[:, 0:3]

        # Joint rotation targets based on _ROTATION_PAIRS order
        target_joints = (
            "hip",
            "chest",
            "head",
            "left_foot",
            "right_foot",
            "left_knee",
            "right_knee",
            "left_elbow",
            "right_elbow",
        )

        for target_joint, (source, dest) in zip(target_joints, self._ROTATION_PAIRS):
            source_idx = self._joint_index.get(source)
            dest_idx = self._joint_index.get(dest)
            target_idx = self._joint_index.get(target_joint)

            if source_idx is not None and dest_idx is not None and target_idx is not None:
                forward = pos[dest_idx] - pos[source_idx]
                self._data[target_idx, 3:7] = _quat_from_forward_up(forward)

    def get_flat(self) -> np.ndarray:
        """Return all joint data as a flat array.

        Returns:
            Flat array of shape (NUM_BODY_JOINTS * 7,) with all joint poses.
        """
        with self._lock:
            return self._data.reshape(-1).copy()

    def get_matrix(self) -> np.ndarray:
        """Return all joint data as a matrix.

        Returns:
            Array of shape (NUM_BODY_JOINTS, 7) with all joint poses.
        """
        with self._lock:
            return self._data.copy()

    def get_position(self, joint_name: str) -> np.ndarray:
        """Get the position of a specific joint.

        Args:
            joint_name: Name of the joint (must be in BODY_JOINT_NAMES).

        Returns:
            Position array of shape (3,).

        Raises:
            ValueError: If joint_name is not found.
        """
        idx = self._joint_index.get(joint_name)
        if idx is None:
            raise ValueError(f"Unknown joint name: {joint_name}")
        with self._lock:
            return self._data[idx, 0:3].copy()

    def get_pose(self, joint_name: str) -> np.ndarray:
        """Get the full pose (position + orientation) of a specific joint.

        Args:
            joint_name: Name of the joint (must be in BODY_JOINT_NAMES).

        Returns:
            Pose array of shape (7,) as [x, y, z, qx, qy, qz, qw].

        Raises:
            ValueError: If joint_name is not found.
        """
        idx = self._joint_index.get(joint_name)
        if idx is None:
            raise ValueError(f"Unknown joint name: {joint_name}")
        with self._lock:
            return self._data[idx].copy()
