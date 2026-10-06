# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests that the feet-wrench observation of the manager-based Humanoid task keeps the configured foot order."""

from isaaclab.sensors import BaseJointWrenchSensor

from isaaclab_tasks.core.locomotion.humanoid.humanoid_common import FEET_BODY_NAMES
from isaaclab_tasks.core.locomotion.humanoid.humanoid_manager_env_cfg import HumanoidEnvCfg


class _JointWrenchSensor:
    find_bodies = BaseJointWrenchSensor.find_bodies

    def __init__(self, body_names):
        self.body_names = body_names
        self.num_bodies = len(body_names)


def test_feet_wrench_observation_follows_configured_order():
    # The PhysX joint-wrench sensor of the Humanoid lists right_foot before left_foot.
    sensor = _JointWrenchSensor(["torso", "right_foot", "left_foot"])
    sensor_cfg = HumanoidEnvCfg().observations.policy.feet_body_forces.params["sensor_cfg"]
    sensor_cfg.resolve({"joint_wrench": sensor})
    assert [sensor.body_names[i] for i in sensor_cfg.body_ids] == FEET_BODY_NAMES
