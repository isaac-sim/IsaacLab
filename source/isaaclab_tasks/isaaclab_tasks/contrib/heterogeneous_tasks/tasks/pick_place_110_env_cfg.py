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
class PickPlace110EnvCfg(VLABenchmarkEnvCfg):
    """Put the cream cheese on the book, on a desk."""

    instruction = "put the cream cheese on the book"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.desk_01 = library_object(
            "desk_01", (0.51612, -0.30698, -0.69646), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.av_book_77 = library_object("av_book_77", (0.42533, 0.0416, 0.09), (0.0, 0.0, -0.061728, 0.998093))
        self.scene.spatula_11 = library_object(
            "spatula_11", (0.41285, -0.21476, 0.09673), (0.0, 0.0, 0.82222, 0.569169)
        )
        self.scene.cream_cheese = library_object(
            "cream_cheese", (0.30922, -0.17639, 0.1166), (0.0, 0.0, -0.662798, 0.748798)
        )
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.av_p_glassware_short = library_object(
            "av_p_glassware_short", (0.50382, 0.25016, 0.09), (0.0, 0.0, -0.797407, 0.603442)
        )
        self.scene.tomato_sauce_can = library_object(
            "tomato_sauce_can", (0.31997, 0.24105, 0.13142), (0.0, 0.0, 0.905562, 0.424214)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("cream_cheese")],
                "reference": SceneEntityCfg("av_book_77"),
                "require_gripper_detached": True,
            },
        )
