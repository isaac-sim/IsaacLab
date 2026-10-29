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
class PickPlace171EnvCfg(VLABenchmarkEnvCfg):
    """Put the raisin box on the book, on a dining table."""

    instruction = "put the raisin box on the book"
    episode_length_s = 65.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_jennings_table = library_object(
            "av_jennings_table", (0.59613, 0.60145, -0.697), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.bin_a02 = library_object("bin_a02", (0.367, 0.2241, 0.071), (0.0, 0.0, -0.98908, 0.147379))
        self.scene.av_book_21 = library_object("av_book_21", (0.4353, -0.03893, 0.071), (0.0, 0.0, 0.790785, 0.612094))
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.whitepackerbottle_a03 = library_object(
            "whitepackerbottle_a03", (0.28981, -0.2323, 0.071), (0.0, 0.0, 0.993907, 0.11022)
        )
        self.scene.raisin_box = library_object(
            "raisin_box", (0.46803, -0.23466, 0.13259), (0.0, 0.0, 0.859247, 0.511562)
        )
        self.scene.av_knife = library_object("av_knife", (0.43327, 0.11361, 0.07206), (0.0, 0.0, 0.779801, 0.626028))
        self.scene.empty_7 = HeterogeneousTaskManager.placeholder(7)
        self.scene.potato_masher = library_object(
            "potato_masher", (0.5149, 0.20924, 0.07142), (0.0, 0.0, -0.61483, 0.78866)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("raisin_box")],
                "reference": SceneEntityCfg("av_book_21"),
                "require_gripper_detached": True,
            },
        )
