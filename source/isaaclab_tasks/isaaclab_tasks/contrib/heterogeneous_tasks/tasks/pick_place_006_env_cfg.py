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
class PickPlace006EnvCfg(VLABenchmarkEnvCfg):
    """Place the eraser on top of the book, on a console table."""

    instruction = "place the eraser on top of the book"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_ellisville = library_object(
            "av_ellisville", (0.40516, 0.07129, -0.69599), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.av_book_77 = library_object(
            "av_book_77", (0.41784, -0.11617, 0.198), (0.0, 0.0, -0.087316, 0.996181)
        )
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.ranch_dressing = library_object(
            "ranch_dressing", (0.5155, 0.0789, 0.27172), (0.0, 0.0, 0.892025, 0.451986)
        )
        self.scene.smartphone = library_object(
            "smartphone", (0.36231, 0.14669, 0.20365), (0.0, 0.0, 0.391908, 0.920004)
        )
        self.scene.av_chess_king = library_object(
            "av_chess_king", (0.28133, 0.06535, 0.198), (0.0, 0.0, 0.696726, 0.717337)
        )
        self.scene.av_eraser = library_object("av_eraser", (0.38553, 0.25128, 0.20233), (0.0, 0.0, 0.723138, 0.690704))
        self.scene.bbq_sauce_bottle = library_object(
            "bbq_sauce_bottle", (0.49945, 0.20977, 0.27216), (0.0, 0.0, -0.925907, 0.377753)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("av_eraser")],
                "reference": SceneEntityCfg("av_book_77"),
                "require_gripper_detached": True,
            },
        )
