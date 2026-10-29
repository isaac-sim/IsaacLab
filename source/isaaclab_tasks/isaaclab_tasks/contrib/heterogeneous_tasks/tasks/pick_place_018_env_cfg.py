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
class PickPlace018EnvCfg(VLABenchmarkEnvCfg):
    """Put the popcorn box on the book, on a storage unit."""

    instruction = "put the popcorn box on the book"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.standard_smallunit = library_object(
            "standard_smallunit", (0.62377, 0.21294, -0.6376), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.av_book_99 = library_object("av_book_99", (0.4389, -0.03071, 0.156), (0.0, 0.0, 0.677434, 0.735584))
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.popcorn_box = library_object(
            "popcorn_box", (0.36083, 0.16813, 0.19849), (0.0, 0.0, -0.337355, 0.941377)
        )
        self.scene.av_p_glassware_short = library_object(
            "av_p_glassware_short", (0.5123, 0.2383, 0.156), (0.0, 0.0, -0.930616, 0.365996)
        )
        self.scene.mug_ycb = library_object("mug_ycb", (0.3818, -0.2176, 0.19669), (0.0, 0.0, 0.221225, 0.975223))
        self.scene.brick = library_object("brick", (0.50889, 0.12645, 0.18157), (0.0, 0.0, -0.903931, 0.427678))
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("popcorn_box")],
                "reference": SceneEntityCfg("av_book_99"),
                "require_gripper_detached": True,
            },
        )
