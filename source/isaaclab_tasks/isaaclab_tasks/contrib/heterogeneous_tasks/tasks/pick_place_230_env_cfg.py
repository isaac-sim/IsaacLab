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
class PickPlace230EnvCfg(VLABenchmarkEnvCfg):
    """Place a block into the red bowl, on a storage cabinet."""

    instruction = "place a block into the red bowl"
    episode_length_s = 65.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_lockedstorage = library_object(
            "contemporary_lockedstorage", (0.5262, -0.11036, -0.697), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.bowl_ycb = library_object("bowl_ycb", (0.40424, 0.17852, -0.00352), (0.0, 0.0, -0.482229, 0.876045))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.cordless_drill = library_object(
            "cordless_drill", (0.47299, -0.00765, -0.00234), (0.0, 0.0, 0.86072, 0.509079)
        )
        self.scene.pineapple_slices_can = library_object(
            "pineapple_slices_can", (0.30435, 0.05985, -0.00219), (0.0, 0.0, 0.979931, 0.199337)
        )
        self.scene.blue_block = library_object(
            "blue_block", (0.48393, -0.18218, -0.0075), (0.0, 0.0, 0.707978, 0.706235)
        )
        self.scene.yellow_block = library_object(
            "yellow_block", (0.28983, -0.07168, -0.0075), (0.0, 0.0, -0.178003, 0.98403)
        )
        self.scene.coffee_can = library_object(
            "coffee_can", (0.39138, -0.22726, 0.03907), (0.0, 0.0, -0.992014, 0.126126)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("blue_block"), SceneEntityCfg("yellow_block")],
                "container": SceneEntityCfg("bowl_ycb"),
                "logical": "any",
                "require_gripper_detached": True,
            },
        )
