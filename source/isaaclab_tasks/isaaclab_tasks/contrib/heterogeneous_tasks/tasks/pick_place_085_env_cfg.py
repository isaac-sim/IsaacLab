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
class PickPlace085EnvCfg(VLABenchmarkEnvCfg):
    """Place the popcorn box on top of the book, on a dining table."""

    instruction = "place the popcorn box on top of the book"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_eastrural_table = library_object(
            "av_eastrural_table", (0.96886, -0.06876, -0.697), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.av_book_11 = library_object("av_book_11", (0.43253, 0.03544, 0.071), (0.0, 0.0, 0.363894, 0.93144))
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.ranch_dressing = library_object(
            "ranch_dressing", (0.49981, 0.21931, 0.14472), (0.0, 0.0, -0.548091, 0.836419)
        )
        self.scene.popcorn_box = library_object(
            "popcorn_box", (0.36042, -0.10761, 0.11349), (0.0, 0.0, 0.385178, 0.922842)
        )
        self.scene.peas_and_carrots = library_object(
            "peas_and_carrots", (0.47094, -0.17831, 0.10026), (0.0, 0.0, 0.889555, 0.456828)
        )
        self.scene.naturalbostonroundbottle_a02 = library_object(
            "naturalbostonroundbottle_a02", (0.30386, 0.22556, 0.071), (0.0, 0.0, -0.732967, 0.680265)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("popcorn_box")],
                "reference": SceneEntityCfg("av_book_11"),
                "require_gripper_detached": True,
            },
        )
