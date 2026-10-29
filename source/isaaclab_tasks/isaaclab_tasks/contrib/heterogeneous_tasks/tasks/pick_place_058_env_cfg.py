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
class PickPlace058EnvCfg(VLABenchmarkEnvCfg):
    """Set the raisin box on the book, on a dining table."""

    instruction = "set the raisin box on the book"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_eastrural_table = library_object(
            "av_eastrural_table", (0.97879, 0.14361, -0.697), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.av_book_11 = library_object("av_book_11", (0.39814, -0.03795, 0.071), (0.0, 0.0, 0.739436, 0.673227))
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.whitepackerbottle_a04 = library_object(
            "whitepackerbottle_a04", (0.48718, -0.23552, 0.071), (0.0, 0.0, 0.999984, 0.005689)
        )
        self.scene.raisin_box = library_object(
            "raisin_box", (0.40392, 0.22039, 0.13259), (0.0, 0.0, -0.788014, 0.615658)
        )
        self.scene.storage_box = library_object(
            "storage_box", (0.47756, 0.14948, 0.07087), (0.0, 0.0, -0.236423, 0.97165)
        )
        self.scene.empty_7 = HeterogeneousTaskManager.placeholder(7)
        self.scene.red_block = library_object("red_block", (0.32272, 0.11665, 0.0945), (0.0, 0.0, -0.468191, 0.883627))
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("raisin_box")],
                "reference": SceneEntityCfg("av_book_11"),
                "require_gripper_detached": True,
            },
        )
