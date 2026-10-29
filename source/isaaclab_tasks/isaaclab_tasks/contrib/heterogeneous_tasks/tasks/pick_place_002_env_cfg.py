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
class PickPlace002EnvCfg(VLABenchmarkEnvCfg):
    """Put the mayonnaise bottle in the wooden bowl, on a console table."""

    instruction = "put the mayonnaise bottle in the wooden bowl"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_aftonconsole = library_object(
            "av_aftonconsole", (0.437, 0.21564, -0.697), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.wooden_bowl = library_object(
            "wooden_bowl", (0.4415, -0.04025, 0.10892), (0.0, 0.0, 0.990727, 0.135868)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.storage_box = library_object(
            "storage_box", (0.32239, 0.2552, 0.10887), (0.0, 0.0, -0.710787, 0.703408)
        )
        self.scene.mayonnaise_bottle = library_object(
            "mayonnaise_bottle", (0.43489, 0.19939, 0.18295), (0.0, 0.0, -0.725997, 0.687698)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("mayonnaise_bottle")],
                "container": SceneEntityCfg("wooden_bowl"),
                "require_gripper_detached": True,
            },
        )
