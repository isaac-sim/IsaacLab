# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Red sphere markers at each OpenXR hand joint for teleop debug visualization."""

from __future__ import annotations

import logging

import numpy as np
import torch

from ..session_lifecycle import TeleopSessionLifecycle

logger = logging.getLogger(__name__)


def _extract_world_joint_positions(result: dict, world_T_anchor: np.ndarray) -> torch.Tensor | None:
    """Extract hand joint positions from a step result and transform them to world frame.

    The chained ``HandsSource`` outputs are raw, i.e. expressed in the XR
    anchor frame.  This helper applies the world-to-anchor transform so the
    returned positions are world-frame regardless of any ``target_T_world``
    rebase active in the retargeting pipeline.

    Args:
        result: Pipeline step result dict, possibly containing
            :attr:`~isaaclab_teleop.session_lifecycle.TeleopSessionLifecycle.HAND_LEFT_KEY`
            and/or
            :attr:`~isaaclab_teleop.session_lifecycle.TeleopSessionLifecycle.HAND_RIGHT_KEY`.
        world_T_anchor: The (4, 4) anchor-to-world transform matrix.

    Returns:
        World-frame joint positions [m], shape [num_joints, 3], float, or
        ``None`` when no hand data is present.
    """
    from isaacteleop.retargeting_engine.tensor_types import HandInputIndex

    parts: list[torch.Tensor] = []
    for key in (TeleopSessionLifecycle.HAND_LEFT_KEY, TeleopSessionLifecycle.HAND_RIGHT_KEY):
        tensor_group = result.get(key)
        if tensor_group is None or tensor_group.is_none:
            continue
        positions = tensor_group[HandInputIndex.JOINT_POSITIONS]
        if not isinstance(positions, torch.Tensor):
            positions = torch.as_tensor(np.asarray(positions), dtype=torch.float32)
        parts.append(positions.reshape(-1, 3))
    if not parts:
        return None

    positions_anchor = torch.cat(parts, dim=0)
    transform = torch.as_tensor(np.asarray(world_T_anchor), dtype=torch.float32)
    return positions_anchor @ transform[:3, :3].T + transform[:3, 3]


class HandJointVisualizer:
    """Red sphere markers at each OpenXR hand joint (26 per hand).

    Created lazily by :class:`~isaaclab_teleop.IsaacTeleopDevice` on the
    first frame where the step result contains chained hand debug outputs.
    Call :meth:`update` each frame with the step result dict and the current
    anchor-to-world transform.
    """

    def __init__(self):
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

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

    def update(self, result: dict, world_T_anchor: np.ndarray) -> None:
        """Update marker positions from a pipeline step result.

        Args:
            result: The pipeline output dict from the most recent step.
            world_T_anchor: The (4, 4) anchor-to-world transform matrix.
        """
        try:
            positions = _extract_world_joint_positions(result, world_T_anchor)
        except Exception:
            if not self._logged_error:
                logger.exception("HandJointVisualizer: error extracting hand joint positions")
                self._logged_error = True
            return
        if positions is None:
            return

        self._markers.visualize(translations=positions)
