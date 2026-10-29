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
class PickPlace176EnvCfg(VLABenchmarkEnvCfg):
    """Pick up the frozen vegetables and put it in the black bowl, on a storage cabinet."""

    instruction = "pick up the frozen vegetables and put it in the black bowl"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_lockedstorage = library_object(
            "contemporary_lockedstorage", (0.53624, -0.08208, -0.697), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.blackandbrassbowl_small = library_object(
            "blackandbrassbowl_small", (0.38167, -0.13701, -0.03104), (0.0, 0.0, 0.669137, 0.743139)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.canned_peaches = library_object(
            "canned_peaches", (0.53546, -0.1871, -0.00211), (0.0, 0.0, -0.050124, 0.998743)
        )
        self.scene.av_p_glassware_short = library_object(
            "av_p_glassware_short", (0.49101, 0.22244, -0.031), (0.0, 0.0, 0.980546, 0.19629)
        )
        self.scene.frozen_vegetable_block = library_object(
            "frozen_vegetable_block", (0.55731, 0.01732, -0.031), (0.0, 0.0, -0.789226, 0.614103)
        )
        self.scene.chocolate_pudding = library_object(
            "chocolate_pudding", (0.37263, 0.06104, -0.01159), (0.0, 0.0, 0.987653, 0.15666)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("frozen_vegetable_block")],
                "container": SceneEntityCfg("blackandbrassbowl_small"),
                "require_gripper_detached": True,
            },
        )
