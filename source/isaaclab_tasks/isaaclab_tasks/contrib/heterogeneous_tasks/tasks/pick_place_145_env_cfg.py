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
class PickPlace145EnvCfg(VLABenchmarkEnvCfg):
    """Place the wooden box on top of the book, on a table."""

    instruction = "place the wooden box on top of the book"
    episode_length_s = 65.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_whittershins = library_object(
            "av_whittershins", (0.87849, 0.00658, -0.697), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.av_book_33 = library_object("av_book_33", (0.4491, -0.09127, -0.058), (0.0, 0.0, 0.10664, 0.994298))
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.ranch_dressing = library_object(
            "ranch_dressing", (0.32439, 0.10676, 0.01572), (0.0, 0.0, 0.925993, 0.37754)
        )
        self.scene.corn_can = library_object("corn_can", (0.31044, -0.16532, -0.02898), (0.0, 0.0, 0.085782, 0.996314))
        self.scene.storage_box = library_object(
            "storage_box", (0.51827, 0.1592, -0.05813), (0.0, 0.0, 0.170608, 0.985339)
        )
        self.scene.empty_7 = HeterogeneousTaskManager.placeholder(7)
        self.scene.lime01 = library_object("lime01", (0.41552, 0.23898, -0.05778), (0.0, 0.0, -0.879986, 0.475))
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("storage_box")],
                "reference": SceneEntityCfg("av_book_33"),
                "require_gripper_detached": True,
            },
        )
