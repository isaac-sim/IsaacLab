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
class PickPlace037EnvCfg(VLABenchmarkEnvCfg):
    """Place the chess pawn upright on top of the book, on a table."""

    instruction = "place the chess pawn upright on top of the book"
    episode_length_s = 75.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.table_bamboo = library_object(
            "table_bamboo", (0.50555, 0.12847, -0.347), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.av_book_51 = library_object("av_book_51", (0.40642, 0.02472, 0.009), (0.0, 0.0, -0.058924, 0.998262))
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.av_chess_pawn = library_object(
            "av_chess_pawn", (0.56535, -0.05444, 0.009), (0.0, 0.0, 0.910559, 0.413379)
        )
        self.scene.tomato_soup_can = library_object(
            "tomato_soup_can", (0.37684, -0.20023, 0.06002), (0.0, 0.0, 0.988827, 0.149065)
        )
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.av_chess_rook = library_object(
            "av_chess_rook", (0.48712, 0.26939, 0.009), (0.0, 0.0, -0.201832, 0.97942)
        )
        self.scene.mustard_bottle = library_object(
            "mustard_bottle", (0.48208, -0.21956, 0.08903), (0.0, 0.0, -0.965828, 0.259185)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("av_chess_pawn")],
                "reference": SceneEntityCfg("av_book_51"),
                "require_gripper_detached": True,
                "require_upright": True,
            },
        )
