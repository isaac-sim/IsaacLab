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
class PickPlace163EnvCfg(VLABenchmarkEnvCfg):
    """Place the macaroni and cheese box on top of the book, on a sofa table."""

    instruction = "place the macaroni and cheese box on top of the book"
    episode_length_s = 65.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.appleseed_sofatable = library_object(
            "appleseed_sofatable", (0.4419, -0.2432, -0.69682), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.av_book_01 = library_object("av_book_01", (0.41171, 0.09989, 0.058), (0.0, 0.0, 0.993763, 0.111515))
        self.scene.frozen_waffles = library_object(
            "frozen_waffles", (0.4256, -0.24699, 0.05798), (0.0, 0.0, 0.688558, 0.725182)
        )
        self.scene.whitepackerbottle_a01 = library_object(
            "whitepackerbottle_a01", (0.44434, -0.14169, 0.058), (0.0, 0.0, -0.959938, 0.280214)
        )
        self.scene.milkjug_a01 = library_object(
            "milkjug_a01", (0.29482, -0.1456, 0.058), (0.0, 0.0, 0.999141, 0.041451)
        )
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.av_eraser = library_object("av_eraser", (0.2888, -0.23813, 0.06233), (0.0, 0.0, -0.182313, 0.98324))
        self.scene.macaroni_and_cheese = library_object(
            "macaroni_and_cheese", (0.53096, -0.15144, 0.14113), (0.0, 0.0, -0.459394, 0.888233)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("macaroni_and_cheese")],
                "reference": SceneEntityCfg("av_book_01"),
                "require_gripper_detached": True,
            },
        )
