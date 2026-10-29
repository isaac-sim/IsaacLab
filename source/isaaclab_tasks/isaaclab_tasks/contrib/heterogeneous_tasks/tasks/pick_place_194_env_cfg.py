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
class PickPlace194EnvCfg(VLABenchmarkEnvCfg):
    """Put the frozen vegetables on the book, on a console table."""

    instruction = "put the frozen vegetables on the book"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_eurekaconsole = library_object(
            "av_eurekaconsole", (0.414, 0.06633, -0.697), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.av_book_66 = library_object("av_book_66", (0.40055, 0.04554, 0.069), (0.0, 0.0, -0.211377, 0.977405))
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.av_chess_queen = library_object(
            "av_chess_queen", (0.53763, -0.02553, 0.069), (0.0, 0.0, 0.776785, 0.629766)
        )
        self.scene.pumpkinlarge = library_object(
            "pumpkinlarge", (0.48543, 0.22965, 0.06876), (0.0, 0.0, -0.163684, 0.986513)
        )
        self.scene.canned_mushrooms = library_object(
            "canned_mushrooms", (0.50594, -0.15259, 0.08566), (0.0, 0.0, 0.094054, 0.995567)
        )
        self.scene.frozen_vegetable_block = library_object(
            "frozen_vegetable_block", (0.41087, -0.20928, 0.069), (0.0, 0.0, -0.372201, 0.928152)
        )
        self.scene.pitted_cherries = library_object(
            "pitted_cherries", (0.31951, 0.25626, 0.09811), (0.0, 0.0, 0.999945, 0.010522)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("frozen_vegetable_block")],
                "reference": SceneEntityCfg("av_book_66"),
                "require_gripper_detached": True,
            },
        )
