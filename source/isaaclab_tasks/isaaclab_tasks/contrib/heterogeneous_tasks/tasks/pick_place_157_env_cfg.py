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
class PickPlace157EnvCfg(VLABenchmarkEnvCfg):
    """Place the orange into the black bowl, on a conference table."""

    instruction = "place the orange into the black bowl"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.steam = library_object(
            "steam", (0.76521, 1.05568, -0.697), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.blackandbrassbowl_small = library_object(
            "blackandbrassbowl_small", (0.46422, -0.10036, 0.10896), (0.0, 0.0, -0.441817, 0.897105)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.orange_02 = library_object("orange_02", (0.41277, 0.17805, 0.10906), (0.0, 0.0, -0.490685, 0.871337))
        self.scene.raisin_box = library_object(
            "raisin_box", (0.55509, 0.10256, 0.17059), (0.0, 0.0, -0.22817, 0.973621)
        )
        self.scene.peas_and_carrots = library_object(
            "peas_and_carrots", (0.36582, -0.23651, 0.13826), (0.0, 0.0, -0.726154, 0.687532)
        )
        self.scene.av_eraser = library_object(
            "av_eraser", (0.29553, -0.02393, 0.11333), (0.0, 0.0, -0.133994, 0.990982)
        )
        self.scene.red_block = library_object("red_block", (0.29468, 0.08219, 0.1325), (0.0, 0.0, 0.405321, 0.914174))
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("orange_02")],
                "container": SceneEntityCfg("blackandbrassbowl_small"),
                "require_gripper_detached": True,
            },
        )
