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
class PickPlace044EnvCfg(VLABenchmarkEnvCfg):
    """Put the chocolate pudding mix on the book, on a dining table."""

    instruction = "put the chocolate pudding mix on the book"
    episode_length_s = 65.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_jennings_table = library_object(
            "av_jennings_table", (0.6041, 0.14157, -0.697), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.av_book_77 = library_object("av_book_77", (0.4577, 0.05045, 0.071), (0.0, 0.0, 0.05728, 0.998358))
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.parmesan_cheese_canister = library_object(
            "parmesan_cheese_canister", (0.3408, 0.24909, 0.12243), (0.0, 0.0, 0.933439, 0.358736)
        )
        self.scene.av_hippoorganizer = library_object(
            "av_hippoorganizer", (0.3698, -0.17003, 0.071), (0.0, 0.0, -0.808241, 0.588851)
        )
        self.scene.av_chess_bishop = library_object(
            "av_chess_bishop", (0.43106, 0.22868, 0.071), (0.0, 0.0, -0.983702, 0.179804)
        )
        self.scene.chocolate_pudding_mix = library_object(
            "chocolate_pudding_mix", (0.55096, -0.19654, 0.09574), (0.0, 0.0, 0.997268, 0.073862)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("chocolate_pudding_mix")],
                "reference": SceneEntityCfg("av_book_77"),
                "require_gripper_detached": True,
            },
        )
