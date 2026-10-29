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
class PickPlace004EnvCfg(VLABenchmarkEnvCfg):
    """Put a box in the black bowl, on a dining table."""

    instruction = "put a box in the black bowl"
    episode_length_s = 65.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_eastrural_table = library_object(
            "av_eastrural_table", (0.59233, -0.40532, -0.697), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.blackandbrassbowl_small = library_object(
            "blackandbrassbowl_small", (0.48008, -0.14738, 0.07096), (0.0, 0.0, -0.711185, 0.703005)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.spatula_07 = library_object(
            "spatula_07", (0.31522, -0.15653, 0.07414), (0.0, 0.0, 0.698943, 0.715178)
        )
        self.scene.parmesan_cheese_canister = library_object(
            "parmesan_cheese_canister", (0.45045, 0.22182, 0.12243), (0.0, 0.0, 0.860687, 0.509135)
        )
        self.scene.raisin_box = library_object(
            "raisin_box", (0.32572, 0.21769, 0.13259), (0.0, 0.0, -0.43192, 0.901912)
        )
        self.scene.av_chess_bishop = library_object(
            "av_chess_bishop", (0.30827, 0.07762, 0.071), (0.0, 0.0, 0.451788, 0.892125)
        )
        self.scene.empty_7 = HeterogeneousTaskManager.placeholder(7)
        self.scene.macaroni_and_cheese = library_object(
            "macaroni_and_cheese", (0.48348, 0.03854, 0.15413), (0.0, 0.0, -0.000918, 1.0)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("raisin_box"), SceneEntityCfg("macaroni_and_cheese")],
                "container": SceneEntityCfg("blackandbrassbowl_small"),
                "logical": "any",
                "require_gripper_detached": True,
            },
        )
