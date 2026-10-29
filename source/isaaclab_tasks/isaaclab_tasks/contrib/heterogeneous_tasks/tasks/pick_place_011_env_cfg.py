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
class PickPlace011EnvCfg(VLABenchmarkEnvCfg):
    """Set the avocado on the book, on a table."""

    instruction = "set the avocado on the book"
    episode_length_s = 65.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_willow = library_object(
            "av_willow", (0.58698, -0.1273, -0.69705), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.av_book_66 = library_object("av_book_66", (0.40184, 0.09676, 0.069), (0.0, 0.0, 0.030231, 0.999543))
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.av_chess_queen = library_object(
            "av_chess_queen", (0.5559, -0.08634, 0.069), (0.0, 0.0, 0.141186, 0.989983)
        )
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.av_chess_bishop = library_object(
            "av_chess_bishop", (0.40241, -0.13814, 0.069), (0.0, 0.0, -0.831029, 0.556228)
        )
        self.scene.frozen_vegetable_block = library_object(
            "frozen_vegetable_block", (0.34248, -0.17731, 0.069), (0.0, 0.0, 0.383105, 0.923705)
        )
        self.scene.avocado01 = library_object(
            "avocado01", (0.51141, -0.16177, 0.06883), (0.0, 0.0, -0.445839, 0.895113)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("avocado01")],
                "reference": SceneEntityCfg("av_book_66"),
                "require_gripper_detached": True,
            },
        )
