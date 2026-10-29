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
class PickPlace083EnvCfg(VLABenchmarkEnvCfg):
    """Set the blue block on the book, on a console table."""

    instruction = "set the blue block on the book"
    episode_length_s = 65.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_aftonconsole = library_object(
            "av_aftonconsole", (0.42561, -0.09907, -0.697), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.av_book_11 = library_object("av_book_11", (0.41526, 0.05126, 0.109), (0.0, 0.0, 0.045392, 0.998969))
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.blue_block = library_object(
            "blue_block", (0.46179, -0.16217, 0.1325), (0.0, 0.0, -0.987418, 0.158131)
        )
        self.scene.av_woodenchsmall = library_object(
            "av_woodenchsmall", (0.56583, -0.08067, 0.109), (0.0, 0.0, -0.792058, 0.610446)
        )
        self.scene.pitted_cherries = library_object(
            "pitted_cherries", (0.48526, -0.2554, 0.13811), (0.0, 0.0, 0.186722, 0.982413)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("blue_block")],
                "reference": SceneEntityCfg("av_book_11"),
                "require_gripper_detached": True,
            },
        )
