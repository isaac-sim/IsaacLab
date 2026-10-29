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
class PickPlace125EnvCfg(VLABenchmarkEnvCfg):
    """Place a chess piece into the white bowl, on an open shelf."""

    instruction = "place a chess piece into the white bowl"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_openstorage = library_object(
            "contemporary_openstorage", (0.41503, -0.03699, -0.697), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.av_bowl_plate = library_object(
            "av_bowl_plate", (0.43816, 0.0969, -0.0312), (0.0, 0.0, 0.889687, 0.45657)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.av_chess_queen = library_object(
            "av_chess_queen", (0.33967, -0.20446, -0.031), (0.0, 0.0, -0.754442, 0.656366)
        )
        self.scene.smartphone = library_object(
            "smartphone", (0.49126, -0.13639, -0.02535), (0.0, 0.0, -0.982519, 0.186162)
        )
        self.scene.av_chess_bishop = library_object(
            "av_chess_bishop", (0.30205, 0.24969, -0.031), (0.0, 0.0, 0.779859, 0.625956)
        )
        self.scene.av_chess_rook = library_object(
            "av_chess_rook", (0.38819, -0.14225, -0.031), (0.0, 0.0, 0.99029, 0.139017)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [
                    SceneEntityCfg("av_chess_queen"),
                    SceneEntityCfg("av_chess_rook"),
                    SceneEntityCfg("av_chess_bishop"),
                ],
                "container": SceneEntityCfg("av_bowl_plate"),
                "logical": "any",
                "require_gripper_detached": True,
            },
        )
