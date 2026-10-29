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
class PickPlace095EnvCfg(VLABenchmarkEnvCfg):
    """Place a bottle into the salt bowl, on a storage cabinet."""

    instruction = "place a bottle into the salt bowl"
    episode_length_s = 65.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_lockedstorage = library_object(
            "contemporary_lockedstorage", (0.52727, 0.02256, -0.697), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.av_salt_bowl = library_object(
            "av_salt_bowl", (0.48508, -0.05707, -0.03094), (0.0, 0.0, -0.766631, 0.642088)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.spatula_13 = library_object("spatula_13", (0.42448, 0.15707, -0.02733), (0.0, 0.0, 0.87752, 0.47954))
        self.scene.ranch_dressing = library_object(
            "ranch_dressing", (0.34948, -0.01345, 0.04272), (0.0, 0.0, 0.623166, 0.782089)
        )
        self.scene.milkjug_a01 = library_object(
            "milkjug_a01", (0.47972, -0.23984, -0.031), (0.0, 0.0, -0.913852, 0.406047)
        )
        self.scene.av_chess_king = library_object(
            "av_chess_king", (0.55695, 0.07206, -0.031), (0.0, 0.0, 0.517474, 0.855699)
        )
        self.scene.naturalbostonroundbottle_a02 = library_object(
            "naturalbostonroundbottle_a02", (0.31575, -0.24422, -0.031), (0.0, 0.0, 0.063492, 0.997982)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [
                    SceneEntityCfg("milkjug_a01"),
                    SceneEntityCfg("naturalbostonroundbottle_a02"),
                    SceneEntityCfg("ranch_dressing"),
                ],
                "container": SceneEntityCfg("av_salt_bowl"),
                "logical": "any",
                "require_gripper_detached": True,
            },
        )
