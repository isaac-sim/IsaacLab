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
class PickPlace196EnvCfg(VLABenchmarkEnvCfg):
    """Pick up the tomato sauce can and put it in the red bowl, on a storage unit."""

    instruction = "pick up the tomato sauce can and put it in the red bowl"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.standard_halfunit = library_object(
            "standard_halfunit", (0.55316, -0.16297, -0.6376), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.bowl_ycb = library_object("bowl_ycb", (0.47635, -0.00206, 0.18348), (0.0, 0.0, 0.586619, 0.809863))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.av_chess_queen = library_object(
            "av_chess_queen", (0.27946, 0.1894, 0.156), (0.0, 0.0, -0.607286, 0.794484)
        )
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.blue_block = library_object("blue_block", (0.43871, 0.18675, 0.1795), (0.0, 0.0, 0.599898, 0.800077))
        self.scene.tomato_sauce_can = library_object(
            "tomato_sauce_can", (0.35249, 0.1125, 0.19742), (0.0, 0.0, -0.447811, 0.894128)
        )
        self.scene.red_block = library_object("red_block", (0.57001, 0.13991, 0.1795), (0.0, 0.0, 0.0387, 0.999251))
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("tomato_sauce_can")],
                "container": SceneEntityCfg("bowl_ycb"),
                "require_gripper_detached": True,
            },
        )
