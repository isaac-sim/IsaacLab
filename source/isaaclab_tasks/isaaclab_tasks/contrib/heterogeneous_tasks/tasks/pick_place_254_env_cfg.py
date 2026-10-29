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
class PickPlace254EnvCfg(VLABenchmarkEnvCfg):
    """Place the phone into the black bowl, on a table."""

    instruction = "place the phone into the black bowl"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_willow = library_object(
            "av_willow", (0.53625, -0.13093, -0.69705), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.blackandbrassbowl_large = library_object(
            "blackandbrassbowl_large", (0.42975, -0.07853, 0.06894), (0.0, 0.0, -0.060939, 0.998142)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.sugar_box = library_object("sugar_box", (0.2834, 0.15636, 0.15701), (0.0, 0.0, 0.999177, 0.040555))
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.smartphone = library_object(
            "smartphone", (0.47346, 0.19466, 0.07465), (0.0, 0.0, 0.570351, 0.821401)
        )
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.tomato_sauce_can = library_object(
            "tomato_sauce_can", (0.36313, 0.13503, 0.11042), (0.0, 0.0, 0.675515, 0.737346)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("smartphone")],
                "container": SceneEntityCfg("blackandbrassbowl_large"),
                "require_gripper_detached": True,
            },
        )
