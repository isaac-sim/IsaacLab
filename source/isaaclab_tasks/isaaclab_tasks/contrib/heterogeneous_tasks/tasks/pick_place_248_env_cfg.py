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
class PickPlace248EnvCfg(VLABenchmarkEnvCfg):
    """Put the mayonnaise bottle in the wooden bowl, on an end table."""

    instruction = "put the mayonnaise bottle in the wooden bowl"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_ellendale = library_object(
            "av_ellendale", (0.38958, 0.00637, -0.697), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.wooden_bowl = library_object(
            "wooden_bowl", (0.41401, -0.00942, -0.00108), (0.0, 0.0, 0.999936, 0.01132)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.parmesan_cheese_canister = library_object(
            "parmesan_cheese_canister", (0.29372, 0.22294, 0.05043), (0.0, 0.0, 0.740173, 0.672417)
        )
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.mayonnaise_bottle = library_object(
            "mayonnaise_bottle", (0.3898, 0.21013, 0.07295), (0.0, 0.0, 0.054798, 0.998497)
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
