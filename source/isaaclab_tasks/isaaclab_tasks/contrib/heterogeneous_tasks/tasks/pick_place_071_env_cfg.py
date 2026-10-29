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
class PickPlace071EnvCfg(VLABenchmarkEnvCfg):
    """Put the chess queen and the can of green beans in the black bowl, on an open shelf."""

    instruction = "put the chess queen and the can of green beans in the black bowl"
    episode_length_s = 110.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_openstorage = library_object(
            "contemporary_openstorage", (0.44638, -0.12242, -0.697), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.blackandbrassbowl_small = library_object(
            "blackandbrassbowl_small", (0.38334, -0.13273, -0.03104), (0.0, 0.0, -0.024206, 0.999707)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.spatula_08 = library_object(
            "spatula_08", (0.43832, 0.11856, -0.02427), (0.0, 0.0, -0.71949, 0.694503)
        )
        self.scene.av_chess_queen = library_object(
            "av_chess_queen", (0.58031, 0.01349, -0.031), (0.0, 0.0, 0.80347, 0.595345)
        )
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.av_knife = library_object("av_knife", (0.51832, -0.02532, -0.02994), (0.0, 0.0, -0.999929, 0.011936))
        self.scene.green_beans_can = library_object(
            "green_beans_can", (0.32166, 0.03822, -0.00221), (0.0, 0.0, -0.814614, 0.580003)
        )
        self.scene.macaroni_and_cheese = library_object(
            "macaroni_and_cheese", (0.31608, 0.11773, 0.05213), (0.0, 0.0, -0.773563, 0.633719)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("av_chess_queen"), SceneEntityCfg("green_beans_can")],
                "container": SceneEntityCfg("blackandbrassbowl_small"),
                "require_gripper_detached": True,
            },
        )
