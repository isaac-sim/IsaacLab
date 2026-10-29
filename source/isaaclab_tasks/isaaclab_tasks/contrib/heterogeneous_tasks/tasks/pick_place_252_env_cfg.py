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
class PickPlace252EnvCfg(VLABenchmarkEnvCfg):
    """Remove the eraser from the metal bin, on a dining table."""

    instruction = "remove the eraser from the metal bin"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_jennings_table = library_object(
            "av_jennings_table", (0.59529, -0.32853, -0.697), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.bin_a03 = library_object("bin_a03", (0.50648, -0.11917, 0.071), (0.0, 0.0, 0.953403, 0.301699))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.av_chess_pawn = library_object(
            "av_chess_pawn", (0.29617, 0.20274, 0.071), (0.0, 0.0, 0.994246, 0.107117)
        )
        self.scene.milkjug_a02 = library_object("milkjug_a02", (0.3509, -0.042, 0.071), (0.0, 0.0, 0.862375, 0.506271))
        self.scene.av_apple = library_object("av_apple", (0.34035, -0.19989, 0.07196), (0.0, 0.0, 0.887198, 0.461388))
        self.scene.av_eraser = library_object(
            "av_eraser", (0.50648, -0.11917, 0.08933), (0.0, 0.0, -0.960835, 0.277122)
        )
        self.scene.birdhouse = library_object("birdhouse", (0.46865, 0.12394, 0.07105), (0.0, 0.0, -0.355426, 0.934704))
        self.terminations.success = DoneTerm(
            func=mdp.object_outside_of,
            params={
                "objects": [SceneEntityCfg("av_eraser")],
                "container": SceneEntityCfg("bin_a03"),
                "require_gripper_detached": True,
            },
        )
