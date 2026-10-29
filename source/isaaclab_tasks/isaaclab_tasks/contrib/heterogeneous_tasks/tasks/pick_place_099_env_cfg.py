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
class PickPlace099EnvCfg(VLABenchmarkEnvCfg):
    """Place a chess piece into the red bowl, on a dining table."""

    instruction = "place a chess piece into the red bowl"
    episode_length_s = 65.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_desperes_table = library_object(
            "av_desperes_table", (0.64514, 0.44175, -0.697), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.bowl = library_object("bowl", (0.48312, 0.02898, 0.10448), (0.0, 0.0, 0.542976, 0.839748))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.av_chess_pawn = library_object(
            "av_chess_pawn", (0.33077, 0.18361, 0.077), (0.0, 0.0, -0.330384, 0.943847)
        )
        self.scene.mug = library_object("mug", (0.32626, -0.16933, 0.07718), (0.0, 0.0, 0.743664, 0.668554))
        self.scene.av_chess_bishop = library_object(
            "av_chess_bishop", (0.44956, -0.25801, 0.077), (0.0, 0.0, -0.275551, 0.961287)
        )
        self.scene.av_chess_rook = library_object(
            "av_chess_rook", (0.49245, 0.27415, 0.077), (0.0, 0.0, 0.996505, 0.083533)
        )
        self.scene.bbq_sauce_bottle_hot3d = library_object(
            "bbq_sauce_bottle_hot3d", (0.46911, -0.17747, 0.07706), (0.0, 0.0, 0.687002, 0.726656)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [
                    SceneEntityCfg("av_chess_rook"),
                    SceneEntityCfg("av_chess_bishop"),
                    SceneEntityCfg("av_chess_pawn"),
                ],
                "container": SceneEntityCfg("bowl"),
                "logical": "any",
                "require_gripper_detached": True,
            },
        )
