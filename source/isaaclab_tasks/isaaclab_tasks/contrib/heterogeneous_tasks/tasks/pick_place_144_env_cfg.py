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
class PickPlace144EnvCfg(VLABenchmarkEnvCfg):
    """Place the spoon on top of the book, on a dining table."""

    instruction = "place the spoon on top of the book"
    episode_length_s = 65.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_jennings_table = library_object(
            "av_jennings_table", (0.56402, 0.76774, -0.697), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.av_book_33 = library_object("av_book_33", (0.45364, 0.07767, 0.071), (0.0, 0.0, -0.998613, 0.052644))
        self.scene.sugar_box = library_object("sugar_box", (0.40737, 0.26266, 0.15901), (0.0, 0.0, -0.684557, 0.72896))
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.orange_juice_carton = library_object(
            "orange_juice_carton", (0.43314, -0.24402, 0.16724), (0.0, 0.0, 0.993394, 0.114751)
        )
        self.scene.spoon = library_object("spoon", (0.51149, -0.20563, 0.07978), (0.0, 0.0, 0.847746, 0.530402))
        self.scene.coffee_pot = library_object(
            "coffee_pot", (0.31151, -0.19599, 0.07096), (0.0, 0.0, -0.774332, 0.63278)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("spoon")],
                "reference": SceneEntityCfg("av_book_33"),
                "require_gripper_detached": True,
            },
        )
