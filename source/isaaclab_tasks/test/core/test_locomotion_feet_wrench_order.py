# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests that the feet-wrench observation of the manager-based Humanoid task keeps the configured foot order."""

import pytest

from isaaclab.sensors import BaseJointWrenchSensor

from isaaclab_tasks.core.locomotion.humanoid.humanoid_common import FEET_BODY_NAMES
from isaaclab_tasks.core.locomotion.humanoid.humanoid_manager_env_cfg import HumanoidEnvCfg

# Body order of the joint-wrench sensor of the Humanoid, as listed by each backend.
SENSOR_BODIES = {
    "physx": [
        "torso",
        "head",
        "lower_waist",
        "right_upper_arm",
        "left_upper_arm",
        "pelvis",
        "right_lower_arm",
        "left_lower_arm",
        "right_thigh",
        "left_thigh",
        "right_hand",
        "left_hand",
        "right_shin",
        "left_shin",
        "right_foot",
        "left_foot",
    ],
    "newton": [
        "head",
        "left_upper_arm",
        "left_lower_arm",
        "left_hand",
        "lower_waist",
        "pelvis",
        "left_thigh",
        "left_shin",
        "left_foot",
        "right_thigh",
        "right_shin",
        "right_foot",
        "right_upper_arm",
        "right_lower_arm",
        "right_hand",
    ],
}


class _JointWrenchSensor:
    find_bodies = BaseJointWrenchSensor.find_bodies

    def __init__(self, body_names):
        self.body_names = body_names
        self.num_bodies = len(body_names)


@pytest.mark.parametrize("backend", ["physx", "newton"])
def test_feet_wrench_observation_follows_configured_order(backend):
    sensor = _JointWrenchSensor(SENSOR_BODIES[backend])
    sensor_cfg = HumanoidEnvCfg().observations.policy.feet_body_forces.params["sensor_cfg"]
    sensor_cfg.resolve({"joint_wrench": sensor})
    assert [sensor.body_names[i] for i in sensor_cfg.body_ids] == FEET_BODY_NAMES
