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
class PickPlace111EnvCfg(VLABenchmarkEnvCfg):
    """Place the sugar box on top of the book, on a dining table."""

    instruction = "place the sugar box on top of the book"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_desperes_table = library_object(
            "av_desperes_table", (0.6049, -0.45986, -0.697), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.av_book_51 = library_object(
            "av_book_51", (0.40362, -0.10539, 0.077), (0.0, 0.0, -0.997887, 0.064978)
        )
        self.scene.sugar_box = library_object("sugar_box", (0.39366, 0.15185, 0.16501), (0.0, 0.0, -0.99538, 0.09601))
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.canned_peaches = library_object(
            "canned_peaches", (0.56102, 0.09254, 0.10589), (0.0, 0.0, -0.960108, 0.279629)
        )
        self.scene.av_apple = library_object("av_apple", (0.50556, 0.1968, 0.07796), (0.0, 0.0, 0.099494, 0.995038))
        self.scene.naturalbostonroundbottle_a01 = library_object(
            "naturalbostonroundbottle_a01", (0.29569, 0.2305, 0.077), (0.0, 0.0, -0.99817, 0.060477)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("sugar_box")],
                "reference": SceneEntityCfg("av_book_51"),
                "require_gripper_detached": True,
            },
        )
