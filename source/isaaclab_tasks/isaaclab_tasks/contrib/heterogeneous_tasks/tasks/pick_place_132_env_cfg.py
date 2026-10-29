# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import HeterogeneousTaskManager, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from .. import mdp
from ..heterogeneous_tasks_env_cfg import VLABenchmarkEnvCfg
from ..objects import library_object


@configclass
class PickPlace132EnvCfg(VLABenchmarkEnvCfg):
    """Put the crab pen holder in the decorative bowl, on a console table."""

    instruction = "put the crab pen holder in the decorative bowl"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_aftonconsole = library_object(
            "av_aftonconsole", (0.43695, 0.02955, -0.697), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.av_decorativebowl = library_object(
            "av_decorativebowl", (0.43452, -0.07578, 0.10904), (0.0, 0.0, 0.545585, 0.838056)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.crabbypenholder = library_object(
            "crabbypenholder", (0.28225, 0.1287, 0.10897), (0.0, 0.0, 0.166293, 0.986076)
        )
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.orange_juice_carton = library_object(
            "orange_juice_carton", (0.5586, 0.12409, 0.20524), (0.0, 0.0, -0.981919, 0.1893)
        )
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.frozen_vegetable_block = library_object(
            "frozen_vegetable_block", (0.40251, 0.1927, 0.109), (0.0, 0.0, -0.859303, 0.511467)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("crabbypenholder")],
                "container": SceneEntityCfg("av_decorativebowl"),
                "require_gripper_detached": True,
            },
        )
