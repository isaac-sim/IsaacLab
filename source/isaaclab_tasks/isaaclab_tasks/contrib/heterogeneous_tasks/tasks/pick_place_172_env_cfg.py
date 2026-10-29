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
class PickPlace172EnvCfg(VLABenchmarkEnvCfg):
    """Put the chess bishop and the chess pawn in the black bowl, on a table."""

    instruction = "put the chess bishop and the chess pawn in the black bowl"
    episode_length_s = 105.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_whittershins = library_object(
            "av_whittershins", (0.87751, 0.06047, -0.697), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.blackandbrassbowl_small = library_object(
            "blackandbrassbowl_small", (0.46007, -0.05064, -0.05804), (0.0, 0.0, 0.69016, 0.723657)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.av_chess_pawn = library_object(
            "av_chess_pawn", (0.35383, 0.16974, -0.058), (0.0, 0.0, 0.1628, 0.986659)
        )
        self.scene.granola_bars = library_object(
            "granola_bars", (0.28787, -0.10958, 0.004), (0.0, 0.0, 0.018505, 0.999829)
        )
        self.scene.av_chess_bishop = library_object(
            "av_chess_bishop", (0.45958, 0.08514, -0.058), (0.0, 0.0, -0.945947, 0.324321)
        )
        self.scene.empty_7 = HeterogeneousTaskManager.placeholder(7)
        self.scene.coffee_can = library_object("coffee_can", (0.48286, 0.19733, 0.01207), (0.0, 0.0, 0.999574, 0.02918))
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("av_chess_bishop"), SceneEntityCfg("av_chess_pawn")],
                "container": SceneEntityCfg("blackandbrassbowl_small"),
                "require_gripper_detached": True,
            },
        )
