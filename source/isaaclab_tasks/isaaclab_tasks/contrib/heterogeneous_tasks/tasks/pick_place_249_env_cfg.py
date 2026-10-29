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
class PickPlace249EnvCfg(VLABenchmarkEnvCfg):
    """Set the macaroni and cheese box on the book, on a console table."""

    instruction = "set the macaroni and cheese box on the book"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_ellisville = library_object(
            "av_ellisville", (0.40543, 0.08988, -0.69599), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.av_book_99 = library_object("av_book_99", (0.42845, 0.12239, 0.198), (0.0, 0.0, 0.994252, 0.107062))
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.whitepackerbottle_a01 = library_object(
            "whitepackerbottle_a01", (0.31242, -0.10464, 0.198), (0.0, 0.0, -0.999335, 0.036476)
        )
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.peas_and_carrots = library_object(
            "peas_and_carrots", (0.47805, -0.12138, 0.22726), (0.0, 0.0, 0.645293, 0.763935)
        )
        self.scene.av_woodenchsmall = library_object(
            "av_woodenchsmall", (0.50895, -0.27187, 0.198), (0.0, 0.0, 0.999996, 0.002729)
        )
        self.scene.macaroni_and_cheese = library_object(
            "macaroni_and_cheese", (0.3891, -0.19019, 0.28113), (0.0, 0.0, 0.450287, 0.892884)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("macaroni_and_cheese")],
                "reference": SceneEntityCfg("av_book_99"),
                "require_gripper_detached": True,
            },
        )
