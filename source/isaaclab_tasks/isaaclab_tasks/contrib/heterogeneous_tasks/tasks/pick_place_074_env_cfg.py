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
class PickPlace074EnvCfg(VLABenchmarkEnvCfg):
    """Put the milk carton upright on the book, on an end table."""

    instruction = "put the milk carton upright on the book"
    episode_length_s = 70.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_ellendale = library_object(
            "av_ellendale", (0.40334, -0.01236, -0.697), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.av_book_21 = library_object(
            "av_book_21", (0.43111, -0.06866, -0.001), (0.0, 0.0, -0.999926, 0.01217)
        )
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.pumpkinlarge = library_object(
            "pumpkinlarge", (0.52183, 0.14729, -0.00124), (0.0, 0.0, -0.713153, 0.701009)
        )
        self.scene.milk_carton = library_object(
            "milk_carton", (0.32108, 0.16106, 0.09418), (0.0, 0.0, 0.434141, 0.900845)
        )
        self.scene.naturalbostonroundbottle_a01 = library_object(
            "naturalbostonroundbottle_a01", (0.4118, 0.13582, -0.001), (0.0, 0.0, 0.602133, 0.798396)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("milk_carton")],
                "reference": SceneEntityCfg("av_book_21"),
                "require_gripper_detached": True,
                "require_upright": True,
            },
        )
