# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from isaaclab.utils import configclass

from ..sensor_base_cfg import SensorBaseCfg

if TYPE_CHECKING:
    from .joint_wrench_sensor import JointWrenchSensor


@configclass
class JointWrenchSensorCfg(SensorBaseCfg):
    """Configuration for a joint reaction wrench sensor."""

    class_type: type[JointWrenchSensor] | str = "{DIR}.joint_wrench_sensor:JointWrenchSensor"

    convention: Literal["incoming_joint_frame"] = "incoming_joint_frame"
    """Coordinate convention for the reported wrench. Defaults to ``"incoming_joint_frame"``.

    - ``"incoming_joint_frame"`` — child-side joint frame, child-side joint anchor as reference point.
      Matches what a real 6-axis F/T sensor mounted at the joint would measure. This is the same
      as PhysX convention in IsaacLab2.3
    """
