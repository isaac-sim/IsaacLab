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
class PickPlace217EnvCfg(VLABenchmarkEnvCfg):
    """Place both the chess bishop and the eraser into the white bowl, on a sofa table."""

    instruction = "place both the chess bishop and the eraser into the white bowl"
    episode_length_s = 110.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.appleseed_sofatable = library_object(
            "appleseed_sofatable", (0.44704, 0.2156, -0.69682), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.av_bowl_plate = library_object(
            "av_bowl_plate", (0.44545, 0.04254, 0.0578), (0.0, 0.0, -0.986609, 0.163102)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.whitepackerbottle_a02 = library_object(
            "whitepackerbottle_a02", (0.5169, -0.21691, 0.058), (0.0, 0.0, 0.980294, 0.197544)
        )
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.av_chess_bishop = library_object(
            "av_chess_bishop", (0.38181, 0.22153, 0.058), (0.0, 0.0, 0.968147, 0.250384)
        )
        self.scene.av_eraser = library_object(
            "av_eraser", (0.55635, -0.11532, 0.06233), (0.0, 0.0, -0.991909, 0.126947)
        )
        self.scene.pitcher = library_object("pitcher", (0.36401, -0.2229, 0.05771), (0.0, 0.0, -0.94903, 0.315186))
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("av_chess_bishop"), SceneEntityCfg("av_eraser")],
                "container": SceneEntityCfg("av_bowl_plate"),
                "require_gripper_detached": True,
            },
        )
