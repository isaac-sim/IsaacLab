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
class PickPlace077EnvCfg(VLABenchmarkEnvCfg):
    """Move the orange into the salt bowl, on a table."""

    instruction = "move the orange into the salt bowl"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.table_bamboo = library_object(
            "table_bamboo", (0.54512, -0.02786, -0.347), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.av_salt_bowl = library_object(
            "av_salt_bowl", (0.36888, -0.1961, 0.00906), (0.0, 0.0, -0.735623, 0.677391)
        )
        self.scene.av_woodentray_medium = library_object(
            "av_woodentray_medium", (0.42782, 0.09363, 0.009), (0.0, 0.0, 0.718942, 0.695071)
        )
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.orange_02 = library_object("orange_02", (0.5268, -0.1588, 0.00906), (0.0, 0.0, 0.822779, 0.568361))
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.empty_7 = HeterogeneousTaskManager.placeholder(7)
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("orange_02")],
                "container": SceneEntityCfg("av_salt_bowl"),
                "require_gripper_detached": True,
            },
        )
