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
class PickPlace119EnvCfg(VLABenchmarkEnvCfg):
    """Move the can of mushrooms into the metal bin, on a table."""

    instruction = "move the can of mushrooms into the metal bin"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_willow = library_object(
            "av_willow", (0.55591, 0.04381, -0.69705), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.bin_a05 = library_object("bin_a05", (0.43255, 0.02465, 0.069), (0.0, 0.0, -0.753265, 0.657718))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.whitepackerbottle_a04 = library_object(
            "whitepackerbottle_a04", (0.4759, -0.21581, 0.069), (0.0, 0.0, -0.936059, 0.351844)
        )
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.canned_mushrooms = library_object(
            "canned_mushrooms", (0.3577, -0.18997, 0.08566), (0.0, 0.0, 0.833965, 0.551818)
        )
        self.scene.frozen_vegetable_block = library_object(
            "frozen_vegetable_block", (0.57567, -0.13119, 0.069), (0.0, 0.0, -0.130863, 0.9914)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("canned_mushrooms")],
                "container": SceneEntityCfg("bin_a05"),
                "require_gripper_detached": True,
            },
        )
