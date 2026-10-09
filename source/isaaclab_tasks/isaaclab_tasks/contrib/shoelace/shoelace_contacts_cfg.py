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
