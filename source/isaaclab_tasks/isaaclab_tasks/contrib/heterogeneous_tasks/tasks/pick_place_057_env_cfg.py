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
class PickPlace057EnvCfg(VLABenchmarkEnvCfg):
    """Set the blue block on the book, on a dining table."""

    instruction = "set the blue block on the book"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_desperes_table = library_object(
            "av_desperes_table", (0.53574, -0.03183, -0.697), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.av_book_21 = library_object("av_book_21", (0.45596, -0.02376, 0.077), (0.0, 0.0, 0.999732, 0.023161))
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.parmesan_cheese_canister = library_object(
            "parmesan_cheese_canister", (0.39741, -0.23214, 0.12843), (0.0, 0.0, 0.800756, 0.598991)
        )
        self.scene.tuna_can = library_object("tuna_can", (0.43453, 0.20701, 0.09376), (0.0, 0.0, 0.400373, 0.916352))
        self.scene.blue_block = library_object("blue_block", (0.3355, 0.25122, 0.1005), (0.0, 0.0, -0.944563, 0.328332))
        self.scene.measuring_spoon = library_object(
            "measuring_spoon", (0.28629, -0.17252, 0.09134), (0.0, 0.0, 0.712315, 0.70186)
        )
        self.scene.mustard_bottle = library_object(
            "mustard_bottle", (0.52041, -0.25887, 0.15703), (0.0, 0.0, 0.581977, 0.813205)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("blue_block")],
                "reference": SceneEntityCfg("av_book_21"),
                "require_gripper_detached": True,
            },
        )
