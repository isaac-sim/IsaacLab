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
class PickPlace180EnvCfg(VLABenchmarkEnvCfg):
    """Put the chess queen in the metal bowl, on a desk."""

    instruction = "put the chess queen in the metal bowl"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.desk_01 = library_object(
            "desk_01", (0.45692, 0.52505, -0.69646), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.serving_bowl = library_object(
            "serving_bowl", (0.45301, -0.17118, 0.09025), (0.0, 0.0, -0.766799, 0.641887)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.av_chess_queen = library_object(
            "av_chess_queen", (0.43068, 0.21701, 0.09), (0.0, 0.0, -0.663878, 0.747841)
        )
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.av_knife = library_object("av_knife", (0.30202, -0.16563, 0.09106), (0.0, 0.0, 0.170162, 0.985416))
        self.scene.green_beans_can = library_object(
            "green_beans_can", (0.32908, 0.10256, 0.11879), (0.0, 0.0, -0.908113, 0.418725)
        )
        self.scene.dumbbell = library_object("dumbbell", (0.40311, -0.00868, 0.12722), (0.0, 0.0, -0.536387, 0.843972))
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("av_chess_queen")],
                "container": SceneEntityCfg("serving_bowl"),
                "require_gripper_detached": True,
            },
        )
