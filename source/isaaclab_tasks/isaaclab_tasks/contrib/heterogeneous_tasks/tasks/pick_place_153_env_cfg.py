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
class PickPlace153EnvCfg(VLABenchmarkEnvCfg):
    """Put the phone on the book, on a desk."""

    instruction = "put the phone on the book"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.desk_01 = library_object(
            "desk_01", (0.52583, 0.43409, -0.69646), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.av_book_21 = library_object("av_book_21", (0.41851, -0.10044, 0.09), (0.0, 0.0, -0.647684, 0.761909))
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.whitepackerbottle_a02 = library_object(
            "whitepackerbottle_a02", (0.54377, 0.21952, 0.09), (0.0, 0.0, 0.021901, 0.99976)
        )
        self.scene.smartphone = library_object("smartphone", (0.4027, 0.18813, 0.09565), (0.0, 0.0, 0.686326, 0.727294))
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.empty_7 = HeterogeneousTaskManager.placeholder(7)
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("smartphone")],
                "reference": SceneEntityCfg("av_book_21"),
                "require_gripper_detached": True,
            },
        )
