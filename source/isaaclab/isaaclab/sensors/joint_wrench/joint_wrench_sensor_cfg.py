# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils import configclass

from ..sensor_base_cfg import SensorBaseCfg

if TYPE_CHECKING:
    from .joint_wrench_sensor import JointWrenchSensor


@configclass
class JointWrenchSensorCfg(SensorBaseCfg):
    """Configuration for a joint reaction wrench sensor.

    The sensor always exposes wrenches in the ``INCOMING_JOINT_FRAME``
    convention: child-side joint frame, child-side joint anchor as reference
    point. This matches what a real 6-axis F/T sensor mounted at the joint
    would measure. Backends convert to this convention internally so the
    public surface is backend-independent.
    """

    class_type: type[JointWrenchSensor] | str = "{DIR}.joint_wrench_sensor:JointWrenchSensor"
