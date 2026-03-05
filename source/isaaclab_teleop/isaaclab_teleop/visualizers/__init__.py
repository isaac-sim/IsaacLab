# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualizers for teleop session data (e.g. hand joint markers)."""

import logging
from typing import cast

from isaaclab_teleop.isaac_teleop_device import IsaacTeleopDevice

from .hand_joint_visualizer import HandJointVisualizer

logger = logging.getLogger(__name__)


def get_hand_joint_visualizers(enable_visualization: bool, teleop_interface: object) -> list:
    """Return teleop visualizers for hand joint markers when enabled and supported.

    Use this from env config :meth:`get_teleop_visualizers` when the config has
    ``enable_visualization`` and uses a pipeline that exposes hand_left/hand_right.
    Call :meth:`~HandJointVisualizer.update` after each advance() for each returned
    visualizer.

    Args:
        enable_visualization: If False, returns an empty list.
        teleop_interface: The teleop device (e.g. IsaacTeleopDevice) to attach to.

    Returns:
        List of visualizer objects with an :meth:`update` method (e.g.
        :class:`HandJointVisualizer`). Empty if disabled or interface not supported.
    """
    if not enable_visualization:
        return []
    if HandJointVisualizer.supports(teleop_interface):
        return [HandJointVisualizer(cast(IsaacTeleopDevice, teleop_interface))]
    logger.error(
        "Hand joint visualization enabled but teleop interface is not supported by HandJointVisualizer "
        "(expected IsaacTeleopDevice with session lifecycle)"
    )
    return []


__all__ = ["HandJointVisualizer", "get_hand_joint_visualizers"]
