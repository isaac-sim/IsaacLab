# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Red sphere markers at each OpenXR hand joint for teleop debug visualization."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

if TYPE_CHECKING:
    from isaacteleop.retargeting_engine.interface import OptionalTensorGroup

logger = logging.getLogger(__name__)


class HandJointVisualizer:
    """Red sphere markers at each OpenXR hand joint (26 per hand).

    Created lazily by :class:`~isaaclab_teleop.isaac_teleop_device.IsaacTeleopDevice`
    when the pipeline step result first contains ``hand_left`` or ``hand_right``.
    Call :meth:`update` each frame with the step result dict.
    """

    def __init__(self):
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
        self._logged_error = False

    def update(self, result: dict[str, OptionalTensorGroup]) -> None:
        """Update marker positions from a pipeline step result.

        Args:
            result: The pipeline output dict from the most recent step.
        """
        try:
            from isaacteleop.retargeting_engine.tensor_types import HandInputIndex

            parts: list[torch.Tensor] = []
            if "hand_left" in result and not result["hand_left"].is_none:
                left_pos = result["hand_left"][HandInputIndex.JOINT_POSITIONS]
                if not isinstance(left_pos, torch.Tensor):
                    left_pos = torch.as_tensor(left_pos, dtype=torch.float32)
                parts.append(left_pos)
            if "hand_right" in result and not result["hand_right"].is_none:
                right_pos = result["hand_right"][HandInputIndex.JOINT_POSITIONS]
                if not isinstance(right_pos, torch.Tensor):
                    right_pos = torch.as_tensor(right_pos, dtype=torch.float32)
                parts.append(right_pos)
            if not parts:
                return
            positions = torch.cat(parts, dim=0).reshape(-1, 3)
        except Exception:
            if not self._logged_error:
                logger.exception("HandJointVisualizer: error extracting hand joint positions")
                self._logged_error = True
            return

        self._markers.visualize(translations=positions)
