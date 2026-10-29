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
class PickPlace003EnvCfg(VLABenchmarkEnvCfg):
    """Place a can into the black bowl, on a dining table."""

    instruction = "place a can into the black bowl"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_desperes_table = library_object(
            "av_desperes_table", (0.56543, 0.02839, -0.697), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.blackandbrassbowl_small = library_object(
            "blackandbrassbowl_small", (0.45577, -0.01553, 0.07696), (0.0, 0.0, -0.978571, 0.20591)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.whitepackerbottle_a02 = library_object(
            "whitepackerbottle_a02", (0.47724, 0.23338, 0.077), (0.0, 0.0, 0.999972, 0.00751)
        )
        self.scene.canned_peaches = library_object(
            "canned_peaches", (0.41202, -0.2461, 0.10589), (0.0, 0.0, 0.101339, 0.994852)
        )
        self.scene.av_mustard_jar = library_object(
            "av_mustard_jar", (0.39327, 0.25957, 0.07708), (0.0, 0.0, -0.993944, 0.109888)
        )
        self.scene.tomato_sauce_can = library_object(
            "tomato_sauce_can", (0.29392, -0.04795, 0.11842), (0.0, 0.0, -0.797466, 0.603364)
        )
        self.scene.pitted_cherries = library_object(
            "pitted_cherries", (0.36515, 0.158, 0.10611), (0.0, 0.0, -0.928736, 0.370742)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [
                    SceneEntityCfg("pitted_cherries"),
                    SceneEntityCfg("tomato_sauce_can"),
                    SceneEntityCfg("canned_peaches"),
                ],
                "container": SceneEntityCfg("blackandbrassbowl_small"),
                "logical": "any",
                "require_gripper_detached": True,
            },
        )
