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
class PickPlace001EnvCfg(VLABenchmarkEnvCfg):
    """Put a chess piece in the red bowl, on a storage cabinet."""

    instruction = "put a chess piece in the red bowl"
    episode_length_s = 65.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_lockedstorage = library_object(
            "contemporary_lockedstorage", (0.44445, -0.06131, -0.697), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.bowl_ycb = library_object("bowl_ycb", (0.42061, 0.13864, -0.00352), (0.0, 0.0, 0.763881, 0.645357))
        self.scene.av_coaster_hexagon = library_object(
            "av_coaster_hexagon", (0.33901, -0.19627, -0.031), (0.0, 0.0, -0.919043, 0.394156)
        )
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.av_chess_pawn = library_object(
            "av_chess_pawn", (0.55422, 0.06806, -0.031), (0.0, 0.0, -0.213145, 0.977021)
        )
        self.scene.milkjug_a01 = library_object(
            "milkjug_a01", (0.29936, 0.04439, -0.031), (0.0, 0.0, 0.788763, 0.614697)
        )
        self.scene.av_chess_king = library_object(
            "av_chess_king", (0.42411, -0.04901, -0.031), (0.0, 0.0, -0.32347, 0.946238)
        )
        self.scene.av_chess_rook = library_object(
            "av_chess_rook", (0.54345, -0.11546, -0.031), (0.0, 0.0, -0.606052, 0.795425)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [
                    SceneEntityCfg("av_chess_rook"),
                    SceneEntityCfg("av_chess_pawn"),
                    SceneEntityCfg("av_chess_king"),
                ],
                "container": SceneEntityCfg("bowl_ycb"),
                "logical": "any",
                "require_gripper_detached": True,
            },
        )
