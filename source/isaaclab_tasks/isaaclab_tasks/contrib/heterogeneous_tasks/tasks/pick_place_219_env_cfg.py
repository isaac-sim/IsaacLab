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
class PickPlace219EnvCfg(VLABenchmarkEnvCfg):
    """Put a chess piece in the black bowl, on a dining table."""

    instruction = "put a chess piece in the black bowl"
    episode_length_s = 65.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_desperes_table = library_object(
            "av_desperes_table", (0.5758, -0.47788, -0.697), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.blackandbrassbowl_large = library_object(
            "blackandbrassbowl_large", (0.42491, -0.06642, 0.07694), (0.0, 0.0, -0.017548, 0.999846)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.av_chess_pawn = library_object(
            "av_chess_pawn", (0.38546, -0.25564, 0.077), (0.0, 0.0, 0.832083, 0.554651)
        )
        self.scene.milkjug_a02 = library_object("milkjug_a02", (0.4279, 0.18024, 0.077), (0.0, 0.0, 0.914581, 0.404404))
        self.scene.av_chess_king = library_object(
            "av_chess_king", (0.30412, 0.15623, 0.077), (0.0, 0.0, -0.965439, 0.260629)
        )
        self.scene.av_chess_rook = library_object(
            "av_chess_rook", (0.55879, 0.20851, 0.077), (0.0, 0.0, -0.613189, 0.789936)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [
                    SceneEntityCfg("av_chess_pawn"),
                    SceneEntityCfg("av_chess_king"),
                    SceneEntityCfg("av_chess_rook"),
                ],
                "container": SceneEntityCfg("blackandbrassbowl_large"),
                "logical": "any",
                "require_gripper_detached": True,
            },
        )
