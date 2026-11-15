# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for shoelace finger-tail contact observations."""

from typing import TYPE_CHECKING

from isaaclab.sensors import SensorBaseCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .shoelace_contacts import FingerTailContactSensor


@configclass
class FingerTailContactSensorCfg(SensorBaseCfg):
    """Configuration for the task's four finger-tail contact observations."""

    class_type: type["FingerTailContactSensor"] | str = "{DIR}.shoelace_contacts:FingerTailContactSensor"

    robot_prim_names: tuple[str, str] = ("RobotLeft", "RobotRight")
    """Robot prim names in left/right arm order; both may name one articulation."""
    finger_body_names: tuple[tuple[str, str], tuple[str, str]] = (
        ("panda_leftfinger", "panda_rightfinger"),
        ("panda_leftfinger", "panda_rightfinger"),
    )
    """Two contacting finger body names per arm, in observation column order."""
