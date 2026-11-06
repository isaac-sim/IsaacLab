# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RGB axis markers at the controller aim poses for teleop debug visualization."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _aim_world_pose(tensor_group, world_T_anchor: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract a controller's aim pose and transform it to world frame.

    The chained ``ControllersSource`` outputs are raw, i.e. expressed in the
    XR anchor frame.  This helper applies the world-to-anchor transform.
    Quaternions are XYZW on both sides: OpenXR aim orientations and
    :meth:`~isaaclab.markers.VisualizationMarkers.visualize` use the same
    convention.

    Args:
        tensor_group: A controller ``OptionalTensorGroup`` following the
            ``ControllerInput`` schema, or ``None``.
        world_T_anchor: The (4, 4) anchor-to-world transform matrix.

    Returns:
        A ``(position, orientation)`` tuple with the world-frame aim
        position [m], shape [3], and XYZW quaternion, shape [4], or ``None``
        when the controller is absent or its aim pose is invalid.
    """
    from isaacteleop.retargeting_engine.tensor_types import ControllerInputIndex
    from scipy.spatial.transform import Rotation

    if tensor_group is None or tensor_group.is_none:
        return None
    if not bool(tensor_group[ControllerInputIndex.AIM_IS_VALID]):
        return None

    position = np.asarray(tensor_group[ControllerInputIndex.AIM_POSITION], dtype=np.float32).reshape(3)
    quat_xyzw = np.asarray(tensor_group[ControllerInputIndex.AIM_ORIENTATION], dtype=np.float32).reshape(4)
    transform = np.asarray(world_T_anchor, dtype=np.float32)

    position_world = transform[:3, :3] @ position + transform[:3, 3]
    rotation_world = transform[:3, :3] @ Rotation.from_quat(quat_xyzw).as_matrix()
    return position_world, Rotation.from_matrix(rotation_world).as_quat().astype(np.float32)


class ControllerAimVisualizer:
    """RGB axis markers (x=red, y=green, z=blue) at the controller aim poses.

    The OpenXR aim pose sits at the top of the controller with ``-Z``
    pointing out the front (the natural pointing direction), which makes it
    more intuitive to read than the grip pose.

    Created lazily by :class:`~isaaclab_teleop.IsaacTeleopDevice` on the
    first frame with controller data.  Call :meth:`update` each frame with
    the controller ``TensorGroup`` pair and the current anchor-to-world
    transform.
    """

    def __init__(self):
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/ControllerAimMarkers",
            markers={
                "frame": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                    scale=(0.05, 0.05, 0.05),
                ),
            },
        )
        self._markers = VisualizationMarkers(marker_cfg)
        self._markers.set_visibility(True)
        self._logged_error = False

    def update(self, left_controller, right_controller, world_T_anchor: np.ndarray) -> None:
        """Update the axis markers from the latest controller data.

        Args:
            left_controller: Left controller ``OptionalTensorGroup``, or ``None``.
            right_controller: Right controller ``OptionalTensorGroup``, or ``None``.
            world_T_anchor: The (4, 4) anchor-to-world transform matrix.
        """
        try:
            poses = [
                pose
                for pose in (
                    _aim_world_pose(left_controller, world_T_anchor),
                    _aim_world_pose(right_controller, world_T_anchor),
                )
                if pose is not None
            ]
        except Exception:
            if not self._logged_error:
                logger.exception("ControllerAimVisualizer: error extracting aim poses")
                self._logged_error = True
            return
        if not poses:
            return

        self._markers.visualize(
            translations=np.stack([position for position, _ in poses]),
            orientations=np.stack([orientation for _, orientation in poses]),
        )
