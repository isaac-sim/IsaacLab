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
class PickPlace135EnvCfg(VLABenchmarkEnvCfg):
    """Put the can of mushrooms and the chess queen in the red bowl, on an end table."""

    instruction = "put the can of mushrooms and the chess queen in the red bowl"
    episode_length_s = 110.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.appleseed_endtable = library_object(
            "appleseed_endtable", (0.43094, -0.02306, -0.697), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.bowl_ycb = library_object("bowl_ycb", (0.402, -0.06305, -0.02852), (0.0, 0.0, -0.207436, 0.978249))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.av_chess_queen = library_object(
            "av_chess_queen", (0.44153, 0.20015, -0.056), (0.0, 0.0, -0.999892, 0.014684)
        )
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.canned_mushrooms = library_object(
            "canned_mushrooms", (0.47693, 0.10181, -0.03934), (0.0, 0.0, 0.202706, 0.97924)
        )
        self.scene.naturalbostonroundbottle_a03 = library_object(
            "naturalbostonroundbottle_a03", (0.34514, -0.22309, -0.056), (0.0, 0.0, 0.251457, 0.967869)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("canned_mushrooms"), SceneEntityCfg("av_chess_queen")],
                "container": SceneEntityCfg("bowl_ycb"),
                "require_gripper_detached": True,
            },
        )
