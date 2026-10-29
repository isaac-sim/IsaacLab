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
class PickPlace139EnvCfg(VLABenchmarkEnvCfg):
    """Place the lime on top of the clay plate, on a dining table."""

    instruction = "place the lime on top of the clay plate"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_eastrural_table = library_object(
            "av_eastrural_table", (0.66258, -0.39869, -0.697), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.clay_plates = library_object(
            "clay_plates", (0.41348, -0.01916, 0.07083), (0.0, 0.0, -0.65629, 0.754509)
        )
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.av_chess_queen = library_object(
            "av_chess_queen", (0.28801, -0.27089, 0.071), (0.0, 0.0, 0.991211, 0.132287)
        )
        self.scene.tomato_soup_can = library_object(
            "tomato_soup_can", (0.52048, 0.23189, 0.12202), (0.0, 0.0, 0.49005, 0.871694)
        )
        self.scene.av_mustard_jar = library_object(
            "av_mustard_jar", (0.37525, 0.22432, 0.07108), (0.0, 0.0, 0.027764, 0.999615)
        )
        self.scene.empty_7 = HeterogeneousTaskManager.placeholder(7)
        self.scene.lime01 = library_object("lime01", (0.4176, -0.22683, 0.07122), (0.0, 0.0, 0.979956, 0.199213))
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("lime01")],
                "reference": SceneEntityCfg("clay_plates"),
                "require_gripper_detached": True,
            },
        )
