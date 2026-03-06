# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Red sphere markers at each OpenXR hand joint for teleop visualization."""

from __future__ import annotations

import logging

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from isaaclab_teleop.isaac_teleop_device import IsaacTeleopDevice
from isaaclab_teleop.session_lifecycle import TeleopSessionLifecycle

logger = logging.getLogger(__name__)


class HandJointVisualizer:
    """Red sphere markers at each OpenXR hand joint (26 per hand).

    Call :meth:`update` each frame; it reads last_step_result from the session lifecycle and updates
    VisualizationMarkers when hand_left and/or hand_right are present (single hand or both).
    """

    @staticmethod
    def supports(teleop_interface: object) -> bool:
        """Return True if this visualizer can use the given teleop interface.

        Use this before constructing to avoid adding a no-op visualizer when the
        interface has no session lifecycle (e.g. no hand pipeline).
        """

        if not isinstance(teleop_interface, IsaacTeleopDevice):
            return False
        lifecycle = getattr(teleop_interface, "_session_lifecycle", None)
        return lifecycle is not None and isinstance(lifecycle, TeleopSessionLifecycle)

    def __init__(self, teleop_device: IsaacTeleopDevice):
        """Initialize with the IsaacTeleop device to read hand data from."""
        self._session_lifecycle: TeleopSessionLifecycle | None = getattr(teleop_device, "_session_lifecycle", None)
        self._markers: VisualizationMarkers | None = None
        if self._session_lifecycle is not None:
            marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/HandJointMarkers",
                markers={
                    "joint": sim_utils.SphereCfg(
                        radius=0.005,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    ),
                },
            )
            self._markers = VisualizationMarkers(marker_cfg)
            self._markers.set_visibility(True)
        else:
            logger.debug("HandJointVisualizer: no session lifecycle on teleop device, skipping marker setup")

    def update(self) -> None:
        """Update marker positions from the latest pipeline result."""
        if self._session_lifecycle is None:
            return
        result = self._session_lifecycle.last_step_result
        if result is None:
            return
        has_left = "hand_left" in result
        has_right = "hand_right" in result
        if not has_left and not has_right:
            return
        try:
            from isaacteleop.retargeting_engine.tensor_types import HandInputIndex

            parts = []
            if has_left:
                left_pos = result["hand_left"][HandInputIndex.JOINT_POSITIONS]
                if not isinstance(left_pos, torch.Tensor):
                    left_pos = torch.as_tensor(left_pos, dtype=torch.float32)
                parts.append(left_pos)
            if has_right:
                right_pos = result["hand_right"][HandInputIndex.JOINT_POSITIONS]
                if not isinstance(right_pos, torch.Tensor):
                    right_pos = torch.as_tensor(right_pos, dtype=torch.float32)
                parts.append(right_pos)
            positions = torch.cat(parts, dim=0)
            if positions.dim() > 2:
                positions = positions.reshape(-1, 3)
        except Exception:
            if not getattr(self, "_logged_update_error", False):
                logger.exception("HandJointVisualizer: error updating markers")
                self._logged_update_error = True
            return

        if self._markers is not None:
            self._markers.visualize(translations=positions)
