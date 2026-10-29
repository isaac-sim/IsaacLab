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
class PickPlace061EnvCfg(VLABenchmarkEnvCfg):
    """Put the chocolate pudding mix on the book, on a console table."""

    instruction = "put the chocolate pudding mix on the book"
    episode_length_s = 65.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_eurekaconsole = library_object(
            "av_eurekaconsole", (0.41323, 0.07372, -0.697), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.av_book_51 = library_object("av_book_51", (0.39587, -0.06054, 0.069), (0.0, 0.0, 0.998974, 0.045279))
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.chocolate_pudding_mix = library_object(
            "chocolate_pudding_mix", (0.45666, 0.22109, 0.09374), (0.0, 0.0, -0.1583, 0.987391)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("chocolate_pudding_mix")],
                "reference": SceneEntityCfg("av_book_51"),
                "require_gripper_detached": True,
            },
        )
