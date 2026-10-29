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
class PickPlace160EnvCfg(VLABenchmarkEnvCfg):
    """Put the tomato soup can upright on the book, on a storage unit."""

    instruction = "put the tomato soup can upright on the book"
    episode_length_s = 70.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.standard_halfunit = library_object(
            "standard_halfunit", (0.24043, -0.11306, -0.6376), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.av_book_66 = library_object("av_book_66", (0.44706, 0.0399, 0.156), (0.0, 0.0, 0.644915, 0.764254))
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.tomato_soup_can = library_object(
            "tomato_soup_can", (0.39453, -0.12339, 0.20702), (0.0, 0.0, -0.721213, 0.692714)
        )
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.dry_erase_marker = library_object(
            "dry_erase_marker", (0.53457, 0.17213, 0.16544), (0.0, 0.0, -0.80869, 0.588235)
        )
        self.scene.bbq_sauce_bottle_hot3d = library_object(
            "bbq_sauce_bottle_hot3d", (0.52984, -0.11446, 0.15606), (0.0, 0.0, -0.07787, 0.996963)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("tomato_soup_can")],
                "reference": SceneEntityCfg("av_book_66"),
                "require_gripper_detached": True,
                "require_upright": True,
            },
        )
