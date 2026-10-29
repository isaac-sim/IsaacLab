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
class PickPlace069EnvCfg(VLABenchmarkEnvCfg):
    """Place the orange on top of the book, on an office desk."""

    instruction = "place the orange on top of the book"
    episode_length_s = 65.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_lowfulldesk = library_object(
            "contemporary_lowfulldesk", (0.4274, 0.56358, -0.697), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.av_book_01 = library_object("av_book_01", (0.46266, 0.09801, 0.007), (0.0, 0.0, -0.996804, 0.079885))
        self.scene.spatula_11 = library_object(
            "spatula_11", (0.36722, -0.1668, 0.01373), (0.0, 0.0, -0.945741, 0.32492)
        )
        self.scene.orange_02 = library_object("orange_02", (0.46963, -0.15569, 0.00706), (0.0, 0.0, -0.92817, 0.372158))
        self.scene.soup_can = library_object("soup_can", (0.28453, 0.09276, 0.00706), (0.0, 0.0, -0.691743, 0.722144))
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.empty_7 = HeterogeneousTaskManager.placeholder(7)
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("orange_02")],
                "reference": SceneEntityCfg("av_book_01"),
                "require_gripper_detached": True,
            },
        )
