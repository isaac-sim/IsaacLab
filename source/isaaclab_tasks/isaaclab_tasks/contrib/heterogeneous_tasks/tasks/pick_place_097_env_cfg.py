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
class PickPlace097EnvCfg(VLABenchmarkEnvCfg):
    """Move the chess pawn into the wooden bowl, on an open shelf."""

    instruction = "move the chess pawn into the wooden bowl"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_openstorage = library_object(
            "contemporary_openstorage", (0.466, -0.0773, -0.697), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.wooden_bowl = library_object(
            "wooden_bowl", (0.44146, -0.01051, -0.03108), (0.0, 0.0, -0.082748, 0.996571)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.spatula_10 = library_object(
            "spatula_10", (0.41808, 0.20677, -0.02427), (0.0, 0.0, 0.029905, 0.999553)
        )
        self.scene.av_chess_pawn = library_object(
            "av_chess_pawn", (0.51892, -0.23098, -0.031), (0.0, 0.0, -0.836859, 0.547419)
        )
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.naturalbostonroundbottle_a03 = library_object(
            "naturalbostonroundbottle_a03", (0.4104, -0.23965, -0.031), (0.0, 0.0, 0.998023, 0.062842)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("av_chess_pawn")],
                "container": SceneEntityCfg("wooden_bowl"),
                "require_gripper_detached": True,
            },
        )
