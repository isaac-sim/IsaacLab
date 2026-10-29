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
class PickPlace086EnvCfg(VLABenchmarkEnvCfg):
    """Place the can of mushrooms into the salt bowl, on an end table."""

    instruction = "place the can of mushrooms into the salt bowl"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_ellendale = library_object(
            "av_ellendale", (0.46401, 0.00717, -0.697), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.av_salt_bowl = library_object(
            "av_salt_bowl", (0.45578, 0.16706, -0.00094), (0.0, 0.0, 0.999887, 0.015022)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.raisin_box = library_object(
            "raisin_box", (0.56738, -0.15058, 0.06059), (0.0, 0.0, 0.963046, 0.269336)
        )
        self.scene.canned_mushrooms = library_object(
            "canned_mushrooms", (0.47931, -0.04383, 0.01566), (0.0, 0.0, 0.831734, 0.555174)
        )
        self.scene.mug_ycb = library_object("mug_ycb", (0.33429, -0.09673, 0.03969), (0.0, 0.0, 0.993051, 0.117685))
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("canned_mushrooms")],
                "container": SceneEntityCfg("av_salt_bowl"),
                "require_gripper_detached": True,
            },
        )
